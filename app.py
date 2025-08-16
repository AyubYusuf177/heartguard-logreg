"""
Minimal Gradio UI for single-sample inference.

- Loads artifacts saved by src/train.py
- Builds form inputs for raw numeric/categorical columns
- On Predict: raw dict -> DataFrame -> preprocessor.transform -> sigmoid(XW+b)
- Displays label + probability
"""

from __future__ import annotations
import json
import pickle
import numpy as np
import gradio as gr


# ----- Load artifacts produced by training -----
with open("artifacts/meta.json") as f:
    meta = json.load(f)

threshold = float(meta.get("threshold", 0.5))
num_cols = meta.get("num_cols", [])
cat_cols = meta.get("cat_cols", [])

with open("artifacts/preprocessor.pkl", "rb") as f:
    pre = pickle.load(f)

W = np.load("artifacts/weights.npy")
b = float(meta["bias"])


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_prob(raw_inputs: dict) -> float:
    """Transform raw inputs exactly like training, then compute probability."""
    import pandas as pd
    df = pd.DataFrame([raw_inputs])                    # 1-row DataFrame
    X = pre.transform(df)                              # match training transforms
    prob = float(sigmoid(X @ W + b))                   # X: (1,n), W: (n,)
    return prob


def infer(threshold_val, *vals):
    """Gradio handler: collects UI inputs, runs predict_prob, formats outputs."""
    cols = num_cols + cat_cols
    raw = {c: vals[i] for i, c in enumerate(cols)}     # map ordered inputs -> dict
    p = predict_prob(raw)
    label = "Positive" if p >= float(threshold_val) else "Negative"
    return f"**{label}**", round(p, 4)


# ----- Build Gradio UI -----
threshold_slider = gr.Slider(0, 1, value=threshold, step=0.01, label="Decision threshold")

inputs = []
for c in num_cols:
    inputs.append(gr.Number(label=c))                  # numeric inputs
for c in cat_cols:
    inputs.append(gr.Textbox(label=c))                 # free-text categories

iface = gr.Interface(
    fn=lambda *vals: infer(vals[0], *vals[1:]),
    inputs=[threshold_slider] + inputs,
    outputs=[gr.Markdown(), gr.Number(label="Probability")],
    title="Logistic Regression — From Scratch",
    description="Enter values, adjust threshold, and predict.",
)

if __name__ == "__main__":
    iface.launch()
