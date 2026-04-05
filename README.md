# deeplearning-co3133

Course work: **image** classification (CIFAR-100) and **text** classification (news topics), with a small live demo.

## Repository layout

| Path | Role |
|------|------|
| `ImageClassification/` | Training, checkpoints, CIFAR-100 data dir |
| `TextClassification/` | Notebook, Python package `tc/`, `models/*.pth`, demo samples |
| `demo-api/` | Unified Flask server (`app.py`) |
| `demo-web/` | Vite + React UI (tabs: image + text) |
| `assignments/` | Assignment HTML |

## Run the live demo

**1. Python API** (default port `5000`):

```bash
pip install -r requirements.txt
python demo-api/app.py
```

(`demo-api/requirements.txt`, `ImageClassification/requirements.txt`, and `TextClassification/requirements.txt` all include the same root file.)

**2. Frontend** (from repo root; installs dependencies under `demo-web/`):

```bash
npm install --prefix demo-web
npm run dev
```

Open the URL Vite prints (usually `http://127.0.0.1:5173`). The dev server proxies `/api` to the Flask app.

### Environment

- Image checkpoints: `ImageClassification/checkpoint/*.pth` (see `ImageClassification/src/config.py`).
- Text checkpoints: `TextClassification/models/*.pth`.
- Optional: `PORT`, `MODEL_CHECKPOINT`, `TEXT_MODELS_DIR`, `VITE_PROXY_API` (see `demo-web/vite.config.js`).

## License / course

CO3133 — HCMUT.
