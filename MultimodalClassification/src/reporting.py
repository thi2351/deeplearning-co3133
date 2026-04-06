from __future__ import annotations

"""Reporting helpers for side-by-side multimodal comparisons.

Methods included:
- zero-shot CLIP over class text prompts
- few-shot CLIP prototype classifier from support images
- full model (handled by service.py)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

from .clip_utils import CLIPBundle, encode_pil_images, encode_texts
from .models import PredictionRecord

_DEFAULT_PROMPT = "a photo of {}"

# Caches to avoid recomputing prompt embeddings and support prototypes.
_CLASS_TEXT_CACHE: Dict[Tuple[str, Tuple[str, ...]], torch.Tensor] = {}
_FEWSHOT_PROTO_CACHE: Dict[Tuple[str, str, int, Tuple[str, ...]], torch.Tensor] = {}


@dataclass
class ReportScores:
    rows: List[PredictionRecord]


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _class_text_features(clip: CLIPBundle, classes: Sequence[str]) -> torch.Tensor:
    key = (clip.model_name, tuple(classes))
    cached = _CLASS_TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    prompts = [_DEFAULT_PROMPT.format(c.replace("_", " ")) for c in classes]
    feats = encode_texts(prompts, clip, normalize=True)
    _CLASS_TEXT_CACHE[key] = feats
    return feats


def _to_records(scores: torch.Tensor, classes: Sequence[str], top_k: int) -> List[PredictionRecord]:
    probs = torch.softmax(scores, dim=-1)
    k = min(max(1, int(top_k)), len(classes))
    vals, idx = probs.topk(k, dim=-1)
    out: List[PredictionRecord] = []
    for score, i in zip(vals[0].tolist(), idx[0].tolist()):
        out.append(PredictionRecord(class_name=classes[i], score=float(score)))
    return out


def predict_zero_shot(
    *,
    clip: CLIPBundle,
    classes: Sequence[str],
    image: Image.Image,
    text: str,
    top_k: int = 5,
    image_weight: float = 0.8,
    text_weight: float = 0.2,
) -> ReportScores:
    image_feat = encode_pil_images([image.convert("RGB")], clip, normalize=True)
    query = (text or "").strip()
    if len(query) <= 2:
        query = "a photo of food"
    text_feat = encode_texts([query], clip, normalize=True)

    fused = _normalize(image_weight * image_feat + text_weight * text_feat)
    class_feats = _class_text_features(clip, classes)
    scores = fused @ class_feats.T
    return ReportScores(rows=_to_records(scores, classes, top_k))


def _load_support_images(dataset_root: Path, class_name: str, shots: int) -> List[Image.Image]:
    class_dir = dataset_root / class_name
    if not class_dir.is_dir():
        return []

    files = sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    out: List[Image.Image] = []
    for p in files[:shots]:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


def _fewshot_prototypes(
    *,
    clip: CLIPBundle,
    classes: Sequence[str],
    dataset_root: Path,
    shots: int,
) -> torch.Tensor:
    key = (clip.model_name, str(dataset_root.resolve()), int(shots), tuple(classes))
    cached = _FEWSHOT_PROTO_CACHE.get(key)
    if cached is not None:
        return cached

    class_text = _class_text_features(clip, classes)
    protos: List[torch.Tensor] = []

    for i, cls in enumerate(classes):
        images = _load_support_images(dataset_root, cls, shots=shots)
        if not images:
            # Fallback to text prompt embedding if class folder is unavailable.
            protos.append(class_text[i : i + 1])
            continue

        feats = encode_pil_images(images, clip, normalize=True)
        proto = _normalize(feats.mean(dim=0, keepdim=True))
        protos.append(proto)

    stacked = torch.cat(protos, dim=0)
    _FEWSHOT_PROTO_CACHE[key] = stacked
    return stacked


def predict_few_shot(
    *,
    clip: CLIPBundle,
    classes: Sequence[str],
    image: Image.Image,
    text: str,
    dataset_root: Path,
    shots: int = 1,
    top_k: int = 5,
    image_weight: float = 0.85,
    text_weight: float = 0.15,
) -> ReportScores:
    image_feat = encode_pil_images([image.convert("RGB")], clip, normalize=True)
    query = (text or "").strip()
    if len(query) <= 2:
        query = "a photo of food"
    text_feat = encode_texts([query], clip, normalize=True)

    fused = _normalize(image_weight * image_feat + text_weight * text_feat)
    protos = _fewshot_prototypes(
        clip=clip,
        classes=classes,
        dataset_root=Path(dataset_root),
        shots=max(1, int(shots)),
    )
    scores = fused @ protos.T
    return ReportScores(rows=_to_records(scores, classes, top_k))
