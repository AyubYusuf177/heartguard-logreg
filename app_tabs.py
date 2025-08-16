"""
app_tabs.py
-----------
Gradio UI for vectorized Logistic Regression (from scratch).

What you can do:
1) Train on Upload
   - Upload ANY CSV with a **binary** target column.
   - Auto preprocessing: impute (median/most-frequent), scale numerics,
     one-hot encode categoricals (handle_unknown='ignore').
   - Train our from-scratch LR (gradient descent + optional L2).
   - View loss curve + ROC/PR + metrics, and persist session predictions.

2) Batch CSV Predictions
   - Score another CSV using either:
       a) the **session-trained** model you just fit, OR
       b) previously saved artifacts created by `src/train.py`.
   - Optionally supply a target column to compute metrics and plots.

3) Saved Training Plots
   - If you have offline artifacts (from `python -m src.train ...`),
     view loss/ROC/PR images here.

Notes:
- Target must have exactly two unique labels. If labels aren’t {0,1},
  we map them deterministically to {0,1} for training/metrics and show the mapping.
- This UI uses only standard sklearn tools for preprocessing; the LR core
  is our own vectorized implementation in `src/model_scratch.py`.

Dependencies (pinned in requirements.txt):
  numpy==2.3.2
  pandas==2.3.1
  scikit-learn==1.7.1
  gradio==5.42.0
  matplotlib==3.10.5
"""

from __future__ import annotations

import os
import json
import pickle
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
)

# ---- Import our vectorized logistic regression trainer (from scratch) ----
# Expected signature:
#   train_logreg_gd(X, y, lam: float, lr: float, epochs: int, tol: float, verbose: bool)
# Returns:
#   W (ndarray shape [D]), b (float), hist (dict with "loss": list[float])
from src.model_scratch import train_logreg_gd  # type: ignore


# =============================================================================
# Utility functions
# =============================================================================

def _ensure_artifacts_dir() -> None:
    """Create artifacts/ if needed (used before saving files)."""
    os.makedirs("artifacts", exist_ok=True)


def _read_csv_file(file_obj: Any) -> pd.DataFrame:
    """
    Gradio's File component can return different types (path str, object with .name, etc.).
    This helper robustly extracts a usable path and reads the CSV.
    """
    if file_obj is None:
        raise ValueError("No file provided.")
    if isinstance(file_obj, str):
        path = file_obj
    elif hasattr(file_obj, "name"):
        path = file_obj.name
    else:
        # Fallback: try to treat it as a path-like
        path = str(file_obj)
    return pd.read_csv(path)


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute common binary classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc),
    }


def save_plot(fig, path: str) -> str:
    """Save a Matplotlib figure to a file path and close it."""
    _ensure_artifacts_dir()
    fig.savefig(path, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def try_ohe() -> OneHotEncoder:
    """
    Handle sklearn's API change for OneHotEncoder (sparse_output vs sparse).
    We always want a dense output at the end of preprocessing.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor_with_imputation(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - imputes numerics with median and scales them
      - imputes categoricals with most_frequent and one-hot encodes them
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", try_ohe()),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("No usable features found (after excluding the target).")
    return ColumnTransformer(transformers)


def ensure_binary_target(y_series: pd.Series) -> Tuple[np.ndarray, Optional[Dict[Any, int]], str]:
    """
    Ensure the target is binary {0,1}. If labels are something else (e.g., {"Y","N"}),
    map them deterministically to {0,1} (sorted by string representation).

    Returns:
        y: np.ndarray[int] with values in {0,1}
        mapping: dict[original_label -> {0,1}] or None if already {0,1}
        mapping_note: human-readable note about how mapping was handled
    """
    uniq = pd.unique(y_series.dropna())
    if len(uniq) != 2:
        raise ValueError(f"Target must have exactly two unique values, found {len(uniq)}: {uniq}")

    # Already 0/1?
    if set(pd.Series(uniq).astype(str)) <= {"0", "1"}:
        y = y_series.astype(int).to_numpy()
        return y, None, "Target already in {0,1}."

    # Map deterministically: sort by string representation
    uniq_sorted = sorted(uniq, key=lambda x: str(x))
    mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    y = y_series.map(mapping).astype(int).to_numpy()
    note = f"Mapped target labels {uniq_sorted[0]}→0, {uniq_sorted[1]}→1."
    return y, mapping, note


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, prefix: str) -> tuple[str, str]:
    """Create and save ROC and PR curves; return the two file paths."""
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = save_plot(fig, os.path.join("artifacts", f"{prefix}_roc.png"))

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(rec, prec)
    fig2 = plt.figure()
    plt.plot(rec, prec, label=f"AUC = {pr_auc_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_path = save_plot(fig2, os.path.join("artifacts", f"{prefix}_pr.png"))

    return roc_path, pr_path


# =============================================================================
# Load previously saved artifacts (optional)
# =============================================================================

# These are produced by running:  python -m src.train --csv ... --target ...
artifact_meta_path = os.path.join("artifacts", "meta.json")
artifact_preproc_path = os.path.join("artifacts", "preprocessor.pkl")
artifact_weights_path = os.path.join("artifacts", "weights.npy")

_artifacts_available = (
    os.path.exists(artifact_meta_path)
    and os.path.exists(artifact_preproc_path)
    and os.path.exists(artifact_weights_path)
)

if _artifacts_available:
    with open(artifact_meta_path) as f:
        meta_saved = json.load(f)
    num_cols_saved: list[str] = meta_saved.get("num_cols", [])
    cat_cols_saved: list[str] = meta_saved.get("cat_cols", [])
    threshold_default = float(meta_saved.get("threshold", 0.5))
    with open(artifact_preproc_path, "rb") as f:
        pre_saved: ColumnTransformer = pickle.load(f)
    W_saved = np.load(artifact_weights_path)
    b_saved = float(meta_saved.get("bias", 0.0))
else:
    meta_saved = {}
    num_cols_saved = []
    cat_cols_saved = []
    threshold_default = 0.5
    pre_saved = None  # type: ignore
    W_saved = None    # type: ignore
    b_saved = 0.0


# =============================================================================
# Gradio handlers
# =============================================================================

def inspect_csv_for_training(file) -> tuple[Any, pd.DataFrame, str]:
    """
    When a CSV is uploaded in 'Train on Upload':
    - fill the Target dropdown (default = last column)
    - show a preview (head)
    - and display column info text.
    NOTE: The first tuple item is a gr.update(...) result (type-hinted as Any for Gradio 5).
    """
    if file is None:
        return gr.update(choices=[], value=None), pd.DataFrame(), "Upload a CSV to begin."
    df = _read_csv_file(file)
    cols = df.columns.tolist()
    suggested = cols[-1] if cols else None
    info = f"Detected {len(cols)} columns.\n\nColumns: {cols}"
    preview = df.head(10)
    return gr.update(choices=cols, value=suggested), preview, info


def train_on_upload(
    file,
    target_name: str,
    val_size: float,
    test_size: float,
    lr: float,
    lam: float,
    epochs: int,
    threshold: float,
) -> tuple[str, Optional[str], Optional[str], Optional[str], dict, str]:
    """
    Train LR-from-scratch on an uploaded CSV.

    Returns:
      - metrics markdown (str)
      - loss plot path (or None)
      - test ROC plot path (or None)
      - test PR plot path (or None)
      - model_state dict (for use in Batch tab via gr.State)
      - status string
    """
    if file is None:
        return "Please upload a CSV.", None, None, None, {}, "No file uploaded."

    df = _read_csv_file(file)

    if target_name not in df.columns:
        return f"Target '{target_name}' not found in columns.", None, None, None, {}, "Target missing."

    # Build y and ensure it's binary
    y_raw = df[target_name]
    if y_raw.isna().any():
        return "Target contains missing values; please clean or choose another target.", None, None, None, {}, "Target has NaN."

    try:
        y, mapping, mapping_note = ensure_binary_target(y_raw)
    except Exception as e:
        return f"Error: {e}", None, None, None, {}, "Target not binary."

    # Build X and auto-detect feature types
    X = df.drop(columns=[target_name])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if not num_cols and not cat_cols:
        return "No usable features after excluding the target.", None, None, None, {}, "No features."

    # Split train / val / test (stratified)
    from sklearn.model_selection import train_test_split
    df_train, df_temp, y_train, y_temp = train_test_split(
        X, y, test_size=float(val_size + test_size), random_state=42, stratify=y
    )
    # proportion of test within the temp split
    relative_test = float(test_size) / float(val_size + test_size) if (val_size + test_size) > 0 else 0.0
    X_val, X_test, y_val, y_test = train_test_split(
        df_temp, y_temp, test_size=relative_test, random_state=42, stratify=y_temp
    )

    # Preprocess with imputation + OHE + scaling
    pre = build_preprocessor_with_imputation(num_cols, cat_cols)
    X_train_t = pre.fit_transform(df_train)
    X_val_t   = pre.transform(X_val)
    X_test_t  = pre.transform(X_test)

    # Train our from-scratch Logistic Regression (vectorized GD)
    W, b, hist = train_logreg_gd(
        X_train_t,
        y_train,
        lam=float(lam),
        lr=float(lr),
        epochs=int(epochs),
        tol=1e-7,
        verbose=True,
    )

    # Evaluate on val/test
    y_val_prob  = sigmoid(X_val_t @ W + b).astype(float)
    y_test_prob = sigmoid(X_test_t @ W + b).astype(float)

    val_metrics  = compute_metrics(y_val,  y_val_prob,  float(threshold))
    test_metrics = compute_metrics(y_test, y_test_prob, float(threshold))

    # Plots (loss + ROC/PR)
    loss_path = None
    roc_path  = None
    pr_path   = None
    try:
        fig = plt.figure()
        plt.plot(hist.get("loss", []))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training Loss")
        loss_path = save_plot(fig, os.path.join("artifacts", "session_loss.png"))
    except Exception:
        pass
    try:
        roc_path, pr_path = plot_roc_pr(y_test, y_test_prob, prefix="session_test")
    except Exception:
        pass

    # Build metrics markdown
    md = [
        "### Session Model Metrics",
        f"- **Target**: `{target_name}`",
        f"- **Mapping**: {mapping_note}" if mapping else "- **Mapping**: Target already in {0,1}.",
        "",
        "**Validation:**",
        f"- Accuracy: {val_metrics['accuracy']:.3f} | Precision: {val_metrics['precision']:.3f} | "
        f"Recall: {val_metrics['recall']:.3f} | F1: {val_metrics['f1']:.3f} | ROC-AUC: {val_metrics['roc_auc']:.3f}",
        "**Test:**",
        f"- Accuracy: {test_metrics['accuracy']:.3f} | Precision: {test_metrics['precision']:.3f} | "
        f"Recall: {test_metrics['recall']:.3f} | F1: {test_metrics['f1']:.3f} | ROC-AUC: {test_metrics['roc_auc']:.3f}",
    ]
    metrics_md = "\n".join(md)

    # Persist full predictions on the entire uploaded CSV (convenience)
    try:
        _ensure_artifacts_dir()
        full_X = pre.transform(X)
        full_probs = sigmoid(full_X @ W + b).astype(float)
        full_preds = (full_probs >= float(threshold)).astype(int)
        out = X.copy()
        out[target_name] = y  # mapped target
        out["y_prob"] = full_probs
        out["y_pred"] = full_preds
        out.to_csv(os.path.join("artifacts", "session_full_predictions.csv"), index=False)
    except Exception:
        pass

    # Prepare state for reuse in Batch Predictions tab
    model_state = {
        "pre": pre,
        "W": W,
        "b": float(b),
        "threshold": float(threshold),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_name": target_name,
        "label_mapping": mapping or {},  # original_label -> {0,1}
    }
    status = (
        f"Session model READY. Features: {len(num_cols)} numeric, {len(cat_cols)} categorical. "
        f"Target: {target_name}."
    )
    return metrics_md, loss_path, roc_path, pr_path, model_state, status


def batch_infer_dynamic(
    file,
    target_name: Optional[str],
    threshold: float,
    model_source: str,
    model_state: Optional[dict],
) -> tuple[Optional[pd.DataFrame], str, Optional[str], Optional[str]]:
    """
    Batch predictions using either:
      - the session-trained model (if available), or
      - the saved artifacts (preprocessor + weights) from offline training.

    Returns:
      preview DataFrame (first 100 rows), metrics markdown (may be empty),
      ROC path (optional), PR path (optional).
    """
    if file is None:
        return None, "Upload a CSV.", None, None

    # Choose model source
    use_session = (model_source == "Session-trained") and (model_state is not None) and ("pre" in model_state)
    if use_session:
        pre = model_state["pre"]
        W = model_state["W"]
        b = float(model_state["b"])
        num_cols = model_state["num_cols"]
        cat_cols = model_state["cat_cols"]
    else:
        if not _artifacts_available:
            return None, "No session model and no saved artifacts available.", None, None
        pre = pre_saved
        W = W_saved
        b = b_saved
        num_cols = num_cols_saved
        cat_cols = cat_cols_saved

    df = _read_csv_file(file)

    # Ensure required columns exist
    required_cols = (num_cols + cat_cols)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}", None, None

    # Transform and predict
    X = pre.transform(df[required_cols])
    probs = sigmoid(X @ W + b).astype(float)
    preds = (probs >= float(threshold)).astype(int)

    out = df.copy()
    out["y_prob"] = probs
    out["y_pred"] = preds

    # Metrics & plots (if user provided a target)
    metrics_md = ""
    roc_path = None
    pr_path = None
    if target_name and target_name in df.columns:
        try:
            y_series = df[target_name]
            if y_series.isna().any():
                raise ValueError("Target contains missing values; cannot compute metrics.")
            uniq = pd.unique(y_series)
            if len(uniq) == 2 and set(pd.Series(uniq).astype(str)) <= {"0", "1"}:
                y_true = y_series.astype(int).to_numpy()
            elif len(uniq) == 2:
                uniq_sorted = sorted(uniq, key=lambda x: str(x))
                mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
                y_true = y_series.map(mapping).astype(int).to_numpy()
                metrics_md += f"(Mapped target {uniq_sorted[0]}→0, {uniq_sorted[1]}→1)\n\n"
            else:
                raise ValueError(f"Target must be binary; found {len(uniq)} unique values.")

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1v = f1_score(y_true, preds, zero_division=0)
            try:
                roc_v = roc_auc_score(y_true, probs)
            except Exception:
                roc_v = float("nan")

            metrics_md += (
                f"**Metrics**\n\n"
                f"Accuracy: {acc:.3f}  |  Precision: {prec:.3f}  |  "
                f"Recall: {rec:.3f}  |  F1: {f1v:.3f}  |  ROC-AUC: {roc_v:.3f}\n"
            )

            # ROC
            fpr, tpr, _ = roc_curve(y_true, probs)
            fig = plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve (uploaded CSV)")
            roc_path = save_plot(fig, os.path.join("artifacts", "uploaded_roc.png"))

            # PR
            precs, recs, _ = precision_recall_curve(y_true, probs)
            fig2 = plt.figure()
            plt.plot(recs, precs)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("PR Curve (uploaded CSV)")
            pr_path = save_plot(fig2, os.path.join("artifacts", "uploaded_pr.png"))
        except Exception as e:
            metrics_md = f"Could not compute metrics: {e}"

    # Preview (first 100 rows) and save full predictions
    preview = out.head(100)
    try:
        _ensure_artifacts_dir()
        out.to_csv(os.path.join("artifacts", "batch_predictions.csv"), index=False)
    except Exception:
        pass

    return preview, metrics_md, roc_path, pr_path


# =============================================================================
# Build the Gradio UI
# =============================================================================

with gr.Blocks(title="Logistic Regression — Train on Upload") as demo:
    gr.Markdown(
        "## Logistic Regression — From Scratch\n"
        "Upload a CSV and pick a **binary** target to train in-session, or use saved artifacts."
    )

    # Shared decision threshold (used in both Train & Batch tabs)
    threshold_slider = gr.Slider(0, 1, value=threshold_default, step=0.01, label="Decision threshold")

    # Keep the session-trained model in memory between interactions
    model_state = gr.State(value=None)  # dict with pre, W, b, num_cols, cat_cols, etc.

    with gr.Tabs():
        # ----------------- Train on Upload -----------------
        with gr.TabItem("Train on Upload"):
            gr.Markdown("Upload any CSV, choose a **binary** target, set hyperparams, then click **Train**.")

            file_train = gr.File(file_types=[".csv"], label="CSV to train")
            with gr.Row():
                target_dropdown = gr.Dropdown(choices=[], label="Target column (binary)")
                info_md = gr.Markdown()  # shows column list and counts

            preview_df = gr.Dataframe(label="CSV Preview (head)")

            # When user uploads a CSV, populate the target dropdown and show preview/info
            file_train.change(
                fn=inspect_csv_for_training,
                inputs=[file_train],
                outputs=[target_dropdown, preview_df, info_md],
            )

            with gr.Accordion("Hyperparameters", open=False):
                with gr.Row():
                    val_size = gr.Number(value=0.15, label="Validation size (0-0.5)")
                    test_size = gr.Number(value=0.15, label="Test size (0-0.5)")
                with gr.Row():
                    lr = gr.Number(value=0.1, label="Learning rate")
                    lam = gr.Number(value=0.0, label="L2 lambda (regularization)")
                    epochs = gr.Number(value=1000, label="Epochs", precision=0)

            train_btn = gr.Button("Train")
            metrics_md = gr.Markdown()
            loss_img = gr.Image(label="Training Loss", type="filepath")
            roc_img = gr.Image(label="Test ROC", type="filepath")
            pr_img = gr.Image(label="Test PR", type="filepath")
            session_status = gr.Markdown()

            train_btn.click(
                fn=train_on_upload,
                inputs=[file_train, target_dropdown, val_size, test_size, lr, lam, epochs, threshold_slider],
                outputs=[metrics_md, loss_img, roc_img, pr_img, model_state, session_status],
            )

        # ----------------- Batch CSV Predictions -----------------
        with gr.TabItem("Batch CSV Predictions"):
            gr.Markdown(
                "Run batch predictions using either the **session-trained** model or the **saved artifacts** "
                "(from `python -m src.train ...`)."
            )
            model_source = gr.Radio(
                choices=["Session-trained", "Saved artifacts"],
                value="Session-trained" if _artifacts_available is False else "Saved artifacts",
                label="Model source",
            )
            csv_file = gr.File(file_types=[".csv"], label="CSV to score")
            target_name = gr.Textbox(label="Target column (optional — for metrics)")

            preview = gr.Dataframe(label="Preview (first 100 rows with predictions)")
            metrics = gr.Markdown()
            roc_plot = gr.Image(label="ROC Curve", type="filepath")
            pr_plot = gr.Image(label="PR Curve", type="filepath")
            run = gr.Button("Run Batch Inference")

            run.click(
                fn=batch_infer_dynamic,
                inputs=[csv_file, target_name, threshold_slider, model_source, model_state],
                outputs=[preview, metrics, roc_plot, pr_plot],
            )

        # ----------------- Saved Training Plots -----------------
        with gr.TabItem("Saved Training Plots"):
            gr.Markdown("Plots produced by the offline training step in `artifacts/` (via `src.train`).")
            loss_img_saved = os.path.join("artifacts", "loss.png")
            val_roc = os.path.join("artifacts", "val_roc.png")
            val_pr  = os.path.join("artifacts", "val_pr.png")
            test_roc = os.path.join("artifacts", "test_roc.png")
            test_pr  = os.path.join("artifacts", "test_pr.png")
            # Display each image if it exists; otherwise omit gracefully
            if os.path.exists(loss_img_saved):
                gr.Image(loss_img_saved, label="Training Loss (offline)")
            if os.path.exists(val_roc):
                gr.Image(val_roc, label="Validation ROC (offline)")
            if os.path.exists(val_pr):
                gr.Image(val_pr, label="Validation PR (offline)")
            if os.path.exists(test_roc):
                gr.Image(test_roc, label="Test ROC (offline)")
            if os.path.exists(test_pr):
                gr.Image(test_pr, label="Test PR (offline)")


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    # Allow overriding the port via env var if you want (optional).
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_port=port)
