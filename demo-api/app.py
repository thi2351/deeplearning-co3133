"""
Flask API for CIFAR-100 image classification demo.

Run from repo root or from this folder (paths resolve to ImageClassification/).
Requires: ``pip install -r ImageClassification/requirements.txt`` (torch/timm) and
``pip install -r demo-api/requirements.txt`` (flask only).
"""
import io
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IC_ROOT = REPO_ROOT / "ImageClassification"
if not IC_ROOT.is_dir():
    raise RuntimeError(f"Expected ImageClassification at {IC_ROOT}")

sys.path.insert(0, str(IC_ROOT))

from flask import Flask, jsonify, make_response, request, send_file
from PIL import UnidentifiedImageError
from torchvision.datasets import CIFAR100

from src.config import CHECKPOINT_DIR, CIFAR100_CLASS_NAMES, DATA_DIR
from src.inference import (
    _infer_arch_from_checkpoint_path,
    load_model_for_inference,
    predict_image_bytes,
)

app = Flask(__name__)


@app.after_request
def _cors_headers(response):
    """Allow browser demos without extra ``flask-cors`` install."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.before_request
def _cors_options():
    if request.method == "OPTIONS" and request.path.startswith("/api"):
        return make_response("", 204)

_model_cache = {}
_cifar_test = None


def get_cifar_test_ds():
    """Official CIFAR-100 test split (PIL images, 32×32). Downloads to ImageClassification/data on first use."""
    global _cifar_test
    if _cifar_test is None:
        _cifar_test = CIFAR100(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=None,
        )
    return _cifar_test


def _default_checkpoint() -> Path:
    override = os.environ.get("MODEL_CHECKPOINT")
    if override:
        return Path(override)
    return CHECKPOINT_DIR / f'{os.environ.get("MODEL_ARCH", "resnet50")}_best.pth'


def _safe_checkpoint_dir() -> Path:
    return CHECKPOINT_DIR.resolve()


def list_checkpoint_files():
    """``*.pth`` files directly under ``checkpoint/`` (no recursion)."""
    base = CHECKPOINT_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*.pth") if p.is_file())


def resolve_checkpoint(model_id: str | None) -> Path:
    """
    Resolve demo model selection to an absolute checkpoint path under ``CHECKPOINT_DIR``.
    Empty / None → default from env / ``resnet50_best.pth``.
    ``model_id`` may be stem (``resnet50_best``) or filename (``resnet50_best.pth``).
    """
    if not model_id or not str(model_id).strip():
        ckpt = _default_checkpoint().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}. Set MODEL_CHECKPOINT or place .pth in "
                f"{CHECKPOINT_DIR}"
            )
        return ckpt

    mid = str(model_id).strip()
    if ".." in mid or "/" in mid or "\\" in mid:
        raise ValueError("Invalid model id")
    name = mid if mid.endswith(".pth") else f"{mid}.pth"
    cand = (CHECKPOINT_DIR / name).resolve()
    root = _safe_checkpoint_dir()
    if not str(cand).startswith(str(root)):
        raise ValueError("Invalid model path")
    if not cand.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {name}")
    return cand


def get_model(ckpt_path: Path):
    """Load and cache one model per resolved checkpoint path."""
    key = str(ckpt_path.resolve())
    if key not in _model_cache:
        # arch=None: use metadata inside checkpoint + filename heuristics (multi-checkpoint safe).
        model, _ = load_model_for_inference(ckpt_path, arch=None)
        device = next(model.parameters()).device
        _model_cache[key] = (model, device)
    return _model_cache[key]


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/models")
def models_list():
    """Checkpoints available for the demo (``ImageClassification/checkpoint/*.pth``)."""
    try:
        files = list_checkpoint_files()
        items = []
        for p in files:
            arch_guess = _infer_arch_from_checkpoint_path(p)
            items.append(
                {
                    "id": p.stem,
                    "file": p.name,
                    "arch": arch_guess,
                }
            )
        default_path = _default_checkpoint()
        default_id = default_path.stem if default_path.is_file() else None
        return jsonify(default_id=default_id, models=items)
    except Exception as e:
        return jsonify(error=f"models list failed: {e}"), 500


@app.get("/api/dataset-samples")
def dataset_samples():
    """Random indices from CIFAR-100 test set with ground-truth label names."""
    try:
        count = min(24, max(1, int(request.args.get("count", 9))))
    except ValueError:
        return jsonify(error="bad count"), 400
    try:
        seed = int(request.args.get("seed", 42))
    except ValueError:
        seed = 42
    try:
        ds = get_cifar_test_ds()
        rng = random.Random(seed)
        n = len(ds)
        indices = rng.sample(range(n), min(count, n))
        samples = []
        for i in indices:
            _, y = ds[i]
            yi = int(y)
            samples.append(
                {
                    "index": i,
                    "label_id": yi,
                    "label": CIFAR100_CLASS_NAMES[yi],
                }
            )
        return jsonify(split="test", total=n, samples=samples)
    except Exception as e:
        return jsonify(error=f"CIFAR load failed: {e}"), 500


@app.get("/api/dataset-label")
def dataset_label():
    """Ground-truth label for one CIFAR-100 test index (for predicted vs actual UI)."""
    try:
        idx = int(request.args.get("index", 0))
    except ValueError:
        return jsonify(error="bad index"), 400
    try:
        ds = get_cifar_test_ds()
        if idx < 0 or idx >= len(ds):
            return jsonify(error="index out of range"), 400
        _, y = ds[idx]
        yi = int(y)
        return jsonify(
            index=idx,
            label_id=yi,
            label=CIFAR100_CLASS_NAMES[yi],
        )
    except Exception as e:
        return jsonify(error=f"dataset label failed: {e}"), 500


@app.get("/api/dataset-image")
def dataset_image():
    """PNG of one CIFAR-100 test image (native 32×32, upscaled by browser CSS)."""
    try:
        idx = int(request.args.get("index", 0))
    except ValueError:
        return jsonify(error="bad index"), 400
    try:
        ds = get_cifar_test_ds()
        if idx < 0 or idx >= len(ds):
            return jsonify(error="index out of range"), 400
        pil_img, _ = ds[idx]
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify(error=f"CIFAR image failed: {e}"), 500


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="Missing form field 'image'"), 400
    file = request.files["image"]
    if file.filename in (None, ""):
        return jsonify(error="Empty filename"), 400

    try:
        topk = max(1, min(20, int(request.args.get("topk", 5))))
    except ValueError:
        return jsonify(error="Invalid topk"), 400

    model_id = request.args.get("model") or request.form.get("model")

    try:
        ckpt_path = resolve_checkpoint(model_id)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    try:
        model, device = get_model(ckpt_path)
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except Exception as e:
        return jsonify(error=f"Model load failed: {e}"), 503

    raw = file.read()
    if not raw:
        return jsonify(error="Empty file body"), 400

    try:
        result = predict_image_bytes(model, raw, device, topk=topk)
    except UnidentifiedImageError:
        return jsonify(
            error="Not a valid image (try PNG, JPEG, WebP, GIF)."
        ), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {e}"), 500

    return jsonify(
        {
            "top1_id": result["top1_id"],
            "top1_name": result["top1_name"],
            "top1_prob": result["top1_prob"],
            "topk": result["topk"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
