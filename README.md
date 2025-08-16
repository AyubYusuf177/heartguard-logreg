````markdown
# HeartGuard — Logistic Regression From Scratch (Gradio)

A plug-and-play web app for **binary classification on any CSV**.  
Upload a CSV, pick a target column, and the app will:
- auto-detect numeric/categorical features
- impute missing values (median for numeric, most-frequent for categorical)
- one-hot encode categoricals (`handle_unknown='ignore'`)
- **train a vectorized logistic regression from scratch** (gradient descent, optional L2)
- evaluate (Accuracy, Precision, Recall, F1, ROC-AUC)
- plot Loss / ROC / PR
- let you do **batch predictions** on another CSV with the trained model

> **Note:** Target must be **binary**. If labels aren’t 0/1 (e.g., “Yes/No”), the app maps them to `{0,1}` and shows the mapping.

---

## Quickstart

```bash
git clone <your-repo-url>
cd heartguard-logreg

python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (cmd):        .venv\Scripts\activate.bat

pip install -r requirements.txt
python app_tabs.py
# open the shown local URL (default http://127.0.0.1:7860)
````

---

## How to Use

### Train on Upload

1. Upload any CSV.
2. Choose the **Target** column (must be binary).
3. (Optional) tweak Validation/Test split, Learning rate, L2 lambda, Epochs.
4. Click **Train**.
5. See metrics and plots. Full session predictions are saved to:

   * `artifacts/session_full_predictions.csv`
   * plots under `artifacts/` (loss / ROC / PR)

### Batch CSV Predictions

1. Upload a CSV to score.
2. Choose model source:

   * **Session-trained** (what you just trained), or
   * **Saved artifacts** (from an earlier `src.train` run)
3. (Optional) provide a target column to compute metrics on that CSV.
4. Download predictions from `artifacts/batch_predictions.csv`.

### Notes & Constraints

* Training in-session does **not** require your CSV to match any previous schema.
* For batch predictions using a **saved** model, the CSV **must** include the same feature columns that model was trained on.
* Unseen categorical levels at inference are ignored safely (`handle_unknown='ignore'`).

---

## Project Layout

```
heartguard-logreg/
├─ app_tabs.py                 # Gradio UI (Train on Upload + Batch Predicts)
├─ src/
│  ├─ model_scratch.py         # Vectorized logistic regression (GD)
│  └─ train.py                 # Offline training CLI (saves artifacts/)
├─ artifacts/                  # Plots & predictions (git-ignored except .gitkeep)
├─ data/                       # Local CSVs (git-ignored except .gitkeep)
├─ tests/
│  └─ grad_check.py            # (optional) gradient checks for sanity
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## Troubleshooting

**Port already in use**
Kill the old server and re-run:

```bash
lsof -ti :7860 | xargs kill -9
python app_tabs.py
```

Or launch on a different port:

```bash
python -c 'import app_tabs; app_tabs.demo.launch(server_port=7861, inbrowser=True)'
```

**App opens but no “Train on Upload” tab**
You’re likely running an old file. Check which file Python is importing:

```bash
python - <<'PY'
import app_tabs, os, time
print("USING FILE:", app_tabs.__file__)
print("LAST MOD  :", time.ctime(os.path.getmtime(app_tabs.__file__)))
PY
```

**Binary target check fails**
Ensure the target column has exactly **two** unique labels (e.g., `{0,1}` or `Yes/No`).

---

## Reproducible CLI training (optional)

Train offline and save artifacts that the UI can use:

```bash
python -m src.train --csv data/your.csv --target <target_col> --epochs 800 --lr 0.1
```

This writes:

* `artifacts/preprocessor.pkl`
* `artifacts/weights.npy`
* `artifacts/meta.json` (bias, threshold, column types)
* plots under `artifacts/`

---

## (Optional) Make the app port configurable

Change the very last lines of `app_tabs.py` to:

```python
if __name__ == "__main__":
    import os
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        inbrowser=True,
    )
```

```
```
