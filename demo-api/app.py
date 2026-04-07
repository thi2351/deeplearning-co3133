"""
Single-process Flask API: CIFAR-100 image demo plus optional text classification.

Install (from repo root)::

    pip install -r requirements.txt
    python demo-api/app.py

Routes: image ``/api/*``; text ``/api/text/*``.

Image code lives under ``ImageClassification/src`` (imported as ``src``).
Text code lives under ``TextClassification/tc`` (imported as ``tc``) so both
trees can stay on ``sys.path`` without import hacks.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import quote

from flask import Blueprint, Flask, jsonify, make_response, request, send_file

REPO_ROOT = Path(__file__).resolve().parent.parent
IC_ROOT = REPO_ROOT / "ImageClassification"
TC_ROOT = REPO_ROOT / "TextClassification"
MM_ROOT = REPO_ROOT / "MultimodalClassification"

if not IC_ROOT.is_dir():
    raise RuntimeError(f"Expected ImageClassification at {IC_ROOT}")

# Keep existing imports working (`src` from ImageClassification, `tc` from TextClassification)
# and add repo root for namespaced imports (`MultimodalClassification.*`).
sys.path.insert(0, str(IC_ROOT))
if TC_ROOT.is_dir():
    sys.path.insert(0, str(TC_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, UnidentifiedImageError
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
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.before_request
def _cors_options():
    if request.method == "OPTIONS" and request.path.startswith("/api"):
        return make_response("", 204)


# --- Image (CIFAR-100) ---
_model_cache = {}
_cifar_test = None


def get_cifar_test_ds():
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
    base = CHECKPOINT_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*.pth") if p.is_file())


def resolve_checkpoint(model_id: str | None) -> Path:
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
    key = str(ckpt_path.resolve())
    if key not in _model_cache:
        model, _ = load_model_for_inference(ckpt_path, arch=None)
        device = next(model.parameters()).device
        _model_cache[key] = (model, device)
    return _model_cache[key]


@app.get("/api/health")
def health():
    out = {"status": "ok", "image": True}
    if TC_ROOT.is_dir():
        out["text"] = True
        out["text_prefix"] = "/api/text"
    else:
        out["text"] = False

    out["multimodal"] = MM_ROOT.is_dir()
    if out["multimodal"]:
        out["multimodal_prefix"] = "/api/mm"
    return jsonify(out)


@app.get("/api/models")
def models_list():
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


# --- Multimodal (CLIP + head under MultimodalClassification/) ---
def _register_multimodal_routes() -> None:
    if not MM_ROOT.is_dir():
        return

    try:
        from MultimodalClassification.src.config import (
            DATA_DIR as MM_DATA_DIR,
            DEFAULT_CHECKPOINT,
            DEFAULT_CLASSES_JSON,
            DEMO_SAMPLES_PATH as MM_DEMO_SAMPLES_PATH,
        )
        from MultimodalClassification.src.inference import (
            load_classes_from_path,
            load_demo_samples as load_mm_demo_samples,
        )
        from MultimodalClassification.src.reporting import (
            predict_few_shot,
            predict_zero_shot,
        )
        from MultimodalClassification.src.service import MultimodalFoodClassifierService
    except Exception as e:
        warnings.warn(f"Multimodal routes not registered: {e}", stacklevel=1)
        return

    mm_bp = Blueprint("multimodal_demo", __name__, url_prefix="/api/mm")
    mm_cache: dict = {}

    def _mm_checkpoint_path() -> Path:
        override = os.environ.get("MM_CHECKPOINT")
        return Path(override).resolve() if override else Path(DEFAULT_CHECKPOINT).resolve()

    def _mm_hf_model() -> str:
        return os.environ.get("MM_HF_MODEL", "openai/clip-vit-base-patch32")

    def _mm_classes() -> list[str]:
        inline = os.environ.get("MM_CLASSES_JSON", "").strip()
        if inline:
            try:
                parsed = json.loads(inline)
            except Exception as e:
                raise ValueError(f"MM_CLASSES_JSON is invalid JSON: {e}") from e
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError("MM_CLASSES_JSON must be a JSON list of strings")
            classes = [x.strip() for x in parsed if x.strip()]
            if not classes:
                raise ValueError("MM_CLASSES_JSON is empty")
            return classes

        classes_path = Path(os.environ.get("MM_CLASSES_PATH", str(DEFAULT_CLASSES_JSON))).resolve()
        return load_classes_from_path(classes_path)

    def _get_mm_service() -> MultimodalFoodClassifierService:
        ckpt = _mm_checkpoint_path()
        classes = _mm_classes()
        hf_model = _mm_hf_model()
        key = (str(ckpt), tuple(classes), hf_model)
        if key not in mm_cache:
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"Multimodal checkpoint not found: {ckpt}. Set MM_CHECKPOINT env var."
                )
            mm_cache[key] = MultimodalFoodClassifierService(
                classes=classes,
                checkpoint_path=str(ckpt),
                hf_model_name=hf_model,
            )
        return mm_cache[key]

    def _mm_dataset_root() -> Path:
        env_root = os.environ.get("MM_DATASET_ROOT", "").strip()
        if env_root:
            return Path(env_root).expanduser().resolve()

<<<<<<< HEAD
        local_test_root = (Path(MM_DATA_DIR).resolve() / "test").resolve()
        if local_test_root.is_dir():
            return local_test_root

        # Fallback to a common kagglehub cache location when the repo-local
        # sample tree is unavailable.
=======
        local_root = (Path(MM_DATA_DIR) / "images" / "test").resolve()
        if local_root.is_dir():
            return local_root

        # Default to common kagglehub cache location.
>>>>>>> 68c96d6 (Modify multimodal)
        return (
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "gianmarco96"
            / "upmcfood101"
            / "versions"
            / "1"
            / "images"
            / "test"
        ).resolve()

    @mm_bp.get("/health")
    def mm_health():
        try:
            classes = _mm_classes()
            checkpoint = _mm_checkpoint_path()
            return jsonify(
                status="ok",
                demo="multimodal",
                classes_count=len(classes),
                checkpoint_exists=checkpoint.is_file(),
                checkpoint=str(checkpoint),
            )
        except Exception as e:
            return jsonify(status="error", demo="multimodal", error=str(e)), 503

    @mm_bp.post("/predict")
    def mm_predict():
        if "image" not in request.files:
            return jsonify(error="Missing form field 'image'"), 400

        file = request.files["image"]
        if file.filename in (None, ""):
            return jsonify(error="Empty filename"), 400

        text = request.form.get("text", request.args.get("text", ""))

        try:
            topk = max(1, min(20, int(request.args.get("topk", request.form.get("topk", 5)))))
        except ValueError:
            return jsonify(error="Invalid topk"), 400

        raw = file.read()
        if not raw:
            return jsonify(error="Empty file body"), 400

        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            return jsonify(error="Not a valid image (try PNG, JPEG, WebP, GIF)."), 400

        try:
            svc = _get_mm_service()
        except FileNotFoundError as e:
            return jsonify(error=str(e)), 503
        except Exception as e:
            return jsonify(error=f"Multimodal service init failed: {e}"), 503

        try:
            preds = svc.predict(image=image, text=text, top_k=topk)
        except Exception as e:
            return jsonify(error=f"Prediction failed: {e}"), 500

        return jsonify(
            {
                "predictions": [
                    {"class_name": row.class_name, "score": row.score}
                    for row in preds
                ]
            }
        )

    @mm_bp.post("/report")
    def mm_report():
        if "image" not in request.files:
            return jsonify(error="Missing form field 'image'"), 400

        file = request.files["image"]
        if file.filename in (None, ""):
            return jsonify(error="Empty filename"), 400

        text = request.form.get("text", request.args.get("text", ""))

        try:
            topk = max(1, min(20, int(request.args.get("topk", request.form.get("topk", 5)))))
        except ValueError:
            return jsonify(error="Invalid topk"), 400

        try:
            shots = max(1, min(8, int(request.args.get("shots", request.form.get("shots", 1)))))
        except ValueError:
            return jsonify(error="Invalid shots"), 400

        raw = file.read()
        if not raw:
            return jsonify(error="Empty file body"), 400

        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            return jsonify(error="Not a valid image (try PNG, JPEG, WebP, GIF)."), 400

        try:
            svc = _get_mm_service()
            classes = _mm_classes()
            dataset_root = _mm_dataset_root()
            if not dataset_root.is_dir():
                return jsonify(
                    error=(
                        f"Few-shot dataset root not found: {dataset_root}. "
                        "Set MM_DATASET_ROOT to a Food-101 class-folder root, or keep "
                        "MultimodalClassification/data/test available for the default fallback."
                    )
                ), 503
        except Exception as e:
            return jsonify(error=f"Multimodal service init failed: {e}"), 503

        try:
            t0 = time.perf_counter()
            zero = predict_zero_shot(
                clip=svc.clip,
                classes=classes,
                image=image,
                text=text,
                top_k=topk,
            )
            t_zero = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            few = predict_few_shot(
                clip=svc.clip,
                classes=classes,
                image=image,
                text=text,
                dataset_root=dataset_root,
                shots=shots,
                top_k=topk,
            )
            t_few = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            full = svc.predict(image=image, text=text, top_k=topk)
            t_full = (time.perf_counter() - t2) * 1000.0
        except Exception as e:
            return jsonify(error=f"Report run failed: {e}"), 500

        def _rows(rows):
            return [{"class_name": r.class_name, "score": float(r.score)} for r in rows]

        zero_rows = _rows(zero.rows)
        few_rows = _rows(few.rows)
        full_rows = _rows(full)

        return jsonify(
            {
                "shots": shots,
                "dataset_root": str(dataset_root),
                "methods": {
                    "zero_shot": {
                        "top1": zero_rows[0] if zero_rows else None,
                        "topk": zero_rows,
                        "latency_ms": t_zero,
                    },
                    "few_shot": {
                        "top1": few_rows[0] if few_rows else None,
                        "topk": few_rows,
                        "latency_ms": t_few,
                    },
                    "full_model": {
                        "top1": full_rows[0] if full_rows else None,
                        "topk": full_rows,
                        "latency_ms": t_full,
                    },
                },
            }
        )

    @mm_bp.get("/dataset-samples")
    def mm_dataset_samples():
        try:
            count = min(512, max(1, int(request.args.get("count", 9))))
        except ValueError:
            return jsonify(error="bad count"), 400

        index_arg = request.args.get("index", "").strip()

        try:
            rows = load_mm_demo_samples(MM_DEMO_SAMPLES_PATH, strict_text=True, min_text_len=3)
        except Exception as e:
            return jsonify(error=f"Cannot read multimodal samples: {e}"), 500

        if not rows:
            return jsonify(samples=[], warning="No multimodal demo samples found.")

        def _format_item(row: dict, idx: int) -> dict:
            item = dict(row)
            item["index"] = idx
            image_rel = str(item.get("image_rel", "")).strip()
            if image_rel:
                item["image_url"] = f"/api/mm/sample-image?path={quote(image_rel)}"
            return item

        if index_arg:
            try:
                idx = int(index_arg)
            except ValueError:
                return jsonify(error="bad index"), 400
            if idx < 0 or idx >= len(rows):
                return jsonify(error="index out of range"), 400
            return jsonify(sample=_format_item(rows[idx], idx), total=len(rows))

        items = []
        for idx, row in enumerate(rows[:count]):
            items.append(_format_item(row, idx))

        return jsonify(samples=items, total=len(rows))

    @mm_bp.get("/sample-image")
    def mm_sample_image():
        rel = request.args.get("path", "").strip()
        if not rel:
            return jsonify(error="missing path"), 400

        base = Path(MM_DATA_DIR).resolve()
        cand = (base / rel).resolve()
        if not cand.is_relative_to(base):
            return jsonify(error="invalid path"), 400
        if not cand.is_file():
            return jsonify(error=f"sample image not found: {rel}"), 404

        return send_file(cand)

    app.register_blueprint(mm_bp)


_register_multimodal_routes()


# --- Text (transformer checkpoints under TextClassification/) ---
_text_model_cache: dict = {}


def _register_text_routes() -> None:
    if not TC_ROOT.is_dir():
        return

    try:
        from tc.config import DEMO_SAMPLES_PATH, MODELS_DIR
        from tc.inference import (
            checkpoint_stem_to_pretrained_id,
            is_allowed_text_checkpoint_stem,
            load_demo_samples,
            load_text_checkpoint_bundle,
            predict_text,
        )
    except Exception as e:
        warnings.warn(f"Text demo routes not registered: {e}", stacklevel=1)
        return

    text_bp = Blueprint("text_demo", __name__, url_prefix="/api/text")

    def models_dir_resolved() -> Path:
        return Path(os.environ.get("TEXT_MODELS_DIR", str(MODELS_DIR))).resolve()

    def default_text_ckpt() -> Path:
        override = os.environ.get("TEXT_MODEL_CHECKPOINT")
        if override:
            return Path(override)
        env_arch = os.environ.get("TEXT_MODEL_NAME", "roberta-base")
        return (models_dir_resolved() / f"{env_arch}.pth").resolve()

    def list_text_checkpoints():
        base = models_dir_resolved()
        if not base.is_dir():
            return []
        out = []
        for p in sorted(base.glob("*.pth")):
            if p.is_file() and is_allowed_text_checkpoint_stem(p.stem):
                out.append(p)
        return out

    def resolve_text_checkpoint(model_id: str | None) -> Path:
        root = models_dir_resolved()
        if not model_id or not str(model_id).strip():
            ckpt = default_text_ckpt()
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"No default checkpoint at {ckpt}. Set TEXT_MODEL_CHECKPOINT or add .pth under {root}"
                )
            return ckpt
        mid = str(model_id).strip()
        if ".." in mid or "/" in mid or "\\" in mid:
            raise ValueError("Invalid model id")
        name = mid if mid.endswith(".pth") else f"{mid}.pth"
        cand = (root / name).resolve()
        if not str(cand).startswith(str(root)):
            raise ValueError("Invalid model path")
        if not cand.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {name}")
        return cand

    def get_text_bundle(ckpt_path: Path):
        key = str(ckpt_path.resolve())
        if key not in _text_model_cache:
            model, tokenizer, device, names, is_custom = load_text_checkpoint_bundle(ckpt_path)
            _text_model_cache[key] = (model, tokenizer, device, names, is_custom)
        return _text_model_cache[key]

    @text_bp.get("/health")
    def text_health():
        return jsonify(status="ok", demo="text")

    @text_bp.get("/models")
    def text_models_list():
        try:
            files = list_text_checkpoints()
            items = []
            for p in files:
                items.append(
                    {
                        "id": p.stem,
                        "file": p.name,
                        "arch": checkpoint_stem_to_pretrained_id(p.stem),
                    }
                )
            dp = default_text_ckpt()
            default_id = dp.stem if dp.is_file() else None
            return jsonify(default_id=default_id, models=items)
        except Exception as e:
            return jsonify(error=f"models list failed: {e}"), 500

    @text_bp.get("/dataset-samples")
    def text_dataset_samples():
        try:
            count = min(24, max(1, int(request.args.get("count", 9))))
        except ValueError:
            return jsonify(error="bad count"), 400
        all_rows = load_demo_samples(DEMO_SAMPLES_PATH)
        if not all_rows:
            return jsonify(
                warning=f"No demo samples at {TC_ROOT / 'data' / 'demo_samples.json'}",
                samples=[],
                total=0,
            )
        slice_rows = all_rows[:count]
        samples = []
        for row in slice_rows:
            tid = int(row.get("id", len(samples)))
            text = str(row.get("text", ""))
            preview = text if len(text) <= 140 else text[:137] + "…"
            samples.append(
                {
                    "id": tid,
                    "index": tid,
                    "text": text,
                    "preview": preview,
                    "label": row.get("label"),
                }
            )
        return jsonify(split="demo", total=len(all_rows), samples=samples)

    @text_bp.get("/dataset-label")
    def text_dataset_label():
        try:
            sid = int(request.args.get("id", request.args.get("index", -1)))
        except ValueError:
            return jsonify(error="bad id"), 400
        rows = load_demo_samples(DEMO_SAMPLES_PATH)
        for row in rows:
            if int(row.get("id", -1)) == sid:
                return jsonify(id=sid, label=row.get("label"), label_id=None)
        return jsonify(error="unknown id"), 404

    @text_bp.post("/predict")
    def text_predict():
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") if isinstance(payload, dict) else None) or request.form.get(
            "text", ""
        )
        text = str(text).strip()
        if not text:
            return jsonify(error="Missing or empty 'text' (JSON body or form field)."), 400
        try:
            topk = max(1, min(20, int(request.args.get("topk", 5))))
        except ValueError:
            return jsonify(error="Invalid topk"), 400
        model_id = request.args.get("model") or (
            payload.get("model") if isinstance(payload, dict) else None
        )
        try:
            ckpt_path = resolve_text_checkpoint(model_id)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except FileNotFoundError as e:
            return jsonify(error=str(e)), 503
        try:
            model, tokenizer, device, names, is_custom = get_text_bundle(ckpt_path)
        except Exception as e:
            return jsonify(error=f"Model load failed: {e}"), 503
        try:
            result = predict_text(
                model,
                tokenizer,
                text,
                device,
                names,
                topk=topk,
                is_custom=is_custom,
            )
        except ValueError as e:
            return jsonify(error=str(e)), 400
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

    app.register_blueprint(text_bp)


_register_text_routes()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
