# HeartGuard — Logistic Regression From Scratch (Gradio)

A plug-and-play web app for **binary classification on any CSV**.

> **Why your README links looked "broken"**
>
> Links like `http://127.0.0.1:7891/` (or `http://localhost:7891/`) only open **on the machine that is currently running the app**. If someone reads your README on GitHub, clicking that will try to open *their* computer’s localhost (which isn’t running your app), so it appears broken. Below, the README keeps localhost as a **clickable link** for your own use, and clarifies that others must run the app locally or you must deploy a public URL.

## Local URL (when running)

👉 [http://localhost:7891/](http://localhost:7891/)
(Equivalent: [http://127.0.0.1:7891/](http://127.0.0.1:7891/))

---

## Quickstart

Clone and set up the environment:

```bash
git clone https://github.com/<your-username>/heartguard-logreg.git
cd heartguard-logreg

python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (cmd):        .venv\Scripts\activate.bat

pip install -r requirements.txt
```

### Option A) Run in foreground (simple)

```bash
python app_tabs.py
```

Then open: [http://localhost:7860/](http://localhost:7860/)

### Option B) Run in background (recommended)

We include helper scripts that start/stop the app detached on **port 7891**:

```bash
# start (defaults to port 7891)
./start_server.sh
```

Open the app: [http://localhost:7891/](http://localhost:7891/)

```bash
# stop (pass the port if you changed it)
./stop_server.sh 7891
```

> **Tip:** If you prefer a different port, run `PORT=9000 ./start_server.sh` and open [http://localhost:9000/](http://localhost:9000/).

---

## How to Use

### Train on Upload

1. Upload any CSV.
2. Choose the **Target** column (must be binary). If labels aren’t 0/1 (e.g., “Yes/No”), the app maps them to `{0,1}` and shows the mapping.
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

* Target must be **binary** (exactly two unique labels).
* Training in-session does **not** require your CSV to match any previous schema.
* For batch predictions using a **saved** model, the CSV **must** include the same feature columns that model was trained on.
* Unseen categorical levels at inference are ignored safely (`handle_unknown='ignore'`).

---

## Project Layout

```
heartguard-logreg/
├─ app_tabs.py                 # Gradio UI (Train on Upload + Batch Predicts)
├─ serve.py                    # Tiny launcher used by start_server.sh
├─ start_server.sh             # Start detached (defaults to port 7891)
├─ stop_server.sh              # Stop a running server by port
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

**Quick links inside this repo:**

* [`app_tabs.py`](app_tabs.py)
* [`serve.py`](serve.py)
* [`start_server.sh`](start_server.sh) · [`stop_server.sh`](stop_server.sh)
* [`src/model_scratch.py`](src/model_scratch.py) · [`src/train.py`](src/train.py)
* [`tests/grad_check.py`](tests/grad_check.py)

---

## Troubleshooting

**Port already in use**

* If you ran foreground mode (7860):

  ```bash
  lsof -ti :7860 | xargs kill -9
  python app_tabs.py
  ```

* If you used the background script (7891):

  ```bash
  ./stop_server.sh 7891
  ./start_server.sh 7891
  # then open
  xdg-open http://localhost:7891/   # Linux
  open http://localhost:7891/       # macOS
  start http://localhost:7891/      # Windows
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

## (Optional) Make the app port configurable in code

If you prefer to bake the port into `app_tabs.py`, adjust the final lines to:

```python
if __name__ == "__main__":
    import os
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7891")),  # default 7891
        inbrowser=True,
    )
```

*(Note: `server_name` should be just the host, e.g. `"127.0.0.1"`, not `"127.0.0.1:7891"`.)*

---

# Start / Stop / Verify UI

**Start the app (background; survives closing the terminal)**

```bash
./start_server.sh
```

**Open the UI in your browser**

* macOS: `open http://localhost:7891/`
* Linux: `xdg-open http://localhost:7891/`
* Windows: `start http://localhost:7891/`

Or just click: [http://localhost:7891/](http://localhost:7891/)

**Verify it’s running / listening on the port**

```bash
lsof -iTCP:7891 -sTCP:LISTEN -nP      # should show a Python process bound to 127.0.0.1:7891
```

**Check recent logs if the page says "site can't be reached"**

```bash
tail -n 80 gradio-7891.log
```

**Stop the app cleanly**

```bash
./stop_server.sh
```

**Notes**

* After a reboot (or if you stop it), just run `./start_server.sh` again.
* Logs live at `gradio-7891.log`; the background process ID is stored in `gradio-7891.pid`.
* If 7891 is busy, run `PORT=9000 ./start_server.sh` and open [http://localhost:9000/](http://localhost:9000/).

---

## Sharing with others

* Localhost URLs only work for **you**, while the app is running on your machine.
* To share publicly, deploy to a hosting service (e.g., Hugging Face Spaces, Render, Fly.io, Railway) and update the README with that public URL.
