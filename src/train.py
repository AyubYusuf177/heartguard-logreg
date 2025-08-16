"""
Training script:
- Reads a CSV with a binary target
- Splits into train/val/test (stratified)
- Preprocesses (scale numerics, one-hot encode categoricals)
- Trains from-scratch logistic regression via GD
- Evaluates on val/test, saves metrics and plots
- Writes artifacts needed by the Gradio apps
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .model_scratch import train_logreg_gd, predict_proba


# ----- small metric container for easy JSON dump -----
@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    """Compute common classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = None
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc,
    )


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_prefix: str) -> None:
    """Save ROC and PR curves to <out_prefix>_roc.png and <out_prefix>_pr.png."""
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png", dpi=160)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr.png", dpi=160)
    plt.close()


def build_preprocessor(df: pd.DataFrame, target: str) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a ColumnTransformer that:
    - standardizes numeric columns
    - one-hot encodes categoricals (ignores unknowns at inference)
    """
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Handle sklearn version differences: sparse_output introduced ~1.2
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers: list[tuple[str, object, list[str]]] = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", ohe, cat_cols))

    if not transformers:
        raise ValueError("No features found after excluding target column.")

    pre = ColumnTransformer(transformers)
    return pre, num_cols, cat_cols


def main() -> None:
    # ---- CLI args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with a binary target column")
    ap.add_argument("--target", required=True, help="Name of the binary target column (0/1)")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    # ---- Load data ----
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise KeyError(f"target '{args.target}' not in columns: {list(df.columns)}")

    # ---- Split data ----
    # First split into train and temp (val+test), stratified to preserve class balance
    df_train, df_temp = train_test_split(
        df, test_size=args.val_size + args.test_size, random_state=42, stratify=df[args.target]
    )
    # Then split temp into val and test, keeping proportions
    relative_test = args.test_size / (args.val_size + args.test_size)
    df_val, df_test = train_test_split(
        df_temp, test_size=relative_test, random_state=42, stratify=df_temp[args.target]
    )

    # ---- Preprocess (fit on train; transform val/test) ----
    pre, num_cols, cat_cols = build_preprocessor(df_train, args.target)
    X_train = pre.fit_transform(df_train.drop(columns=[args.target]))
    X_val   = pre.transform(df_val.drop(columns=[args.target]))
    X_test  = pre.transform(df_test.drop(columns=[args.target]))

    y_train = df_train[args.target].to_numpy().astype(int)
    y_val   = df_val[args.target].to_numpy().astype(int)
    y_test  = df_test[args.target].to_numpy().astype(int)

    # ---- Train from-scratch LR ----
    W, b, hist = train_logreg_gd(
        X_train, y_train, lam=args.lam, lr=args.lr, epochs=args.epochs, tol=1e-7, verbose=True
    )

    # ---- Evaluate ----
    y_val_prob  = predict_proba(W, b, X_val)
    y_test_prob = predict_proba(W, b, X_test)
    val_metrics  = compute_metrics(y_val, y_val_prob, args.threshold)
    test_metrics = compute_metrics(y_test, y_test_prob, args.threshold)

    print("\nValidation:", asdict(val_metrics))
    print("Test:      ", asdict(test_metrics))

    # ---- Save artifacts for the app ----
    # 1) Preprocessor (scaler + OHE) so inference uses identical transforms
    with open(os.path.join(args.artifacts, "preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)
    # 2) Weights vector
    np.save(os.path.join(args.artifacts, "weights.npy"), W)

    # 3) Meta (bias, threshold, feature names, metrics)
    # Try to get transformed feature names
    feat_names: list[str] | None
    if hasattr(pre, "get_feature_names_out"):
        feat_names = pre.get_feature_names_out().tolist()
    else:
        feat_names = None  # fallback below if needed

    if feat_names is None:
        feat_names = []
        if num_cols:
            feat_names.extend(num_cols)
        if cat_cols:
            enc = pre.named_transformers_.get("cat")
            if enc is not None and hasattr(enc, "get_feature_names_out"):
                feat_names.extend(enc.get_feature_names_out(cat_cols).tolist())

    meta = {
        "bias": float(b),
        "threshold": float(args.threshold),
        "feature_names": feat_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "metrics": {
            "val": asdict(val_metrics),
            "test": asdict(test_metrics),
        },
    }
    with open(os.path.join(args.artifacts, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Plots (loss + ROC/PR) ----
    # Training loss over epochs (this is your "cost" curve)
    plt.figure()
    plt.plot(hist["loss"])
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "loss.png"), dpi=160)
    plt.close()

    # ROC/PR for val and test
    plot_curves(y_val,  y_val_prob,  os.path.join(args.artifacts, "val"))
    plot_curves(y_test, y_test_prob, os.path.join(args.artifacts, "test"))


if __name__ == "__main__":
    main()
