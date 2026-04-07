"""Inference helpers for multimodal service integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .clip_utils import HF_CLIP_MODEL
from .config import DEFAULT_CHECKPOINT, DEMO_SAMPLES_PATH
from .service import MultimodalFoodClassifierService


def load_classes_from_path(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Classes file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError(f"Expected JSON list[str] at {p}")
        classes = [x.strip() for x in data if x.strip()]
    else:
        classes = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not classes:
        raise ValueError(f"No classes found in {p}")
    return classes


def build_service(
    classes: Sequence[str],
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    hf_model_name: str = HF_CLIP_MODEL,
) -> MultimodalFoodClassifierService:
    return MultimodalFoodClassifierService(
        classes=list(classes),
        checkpoint_path=str(checkpoint_path),
        hf_model_name=hf_model_name,
    )


def load_demo_samples(
    path: str | Path = DEMO_SAMPLES_PATH,
    strict_text: bool = True,
    min_text_len: int = 3,
) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        image_url = str(row.get("image_url", "")).strip()
        image_rel = str(row.get("image_rel", row.get("image_file", ""))).strip()
        text = str(row.get("text", "")).strip()
        if (not image_url and not image_rel):
            continue
        if strict_text and len(text) < min_text_len:
            continue
        if not strict_text and not text:
            continue

        sample: Dict[str, Any] = {
            "id": str(row.get("id", "")) or f"sample-{len(out)}",
                "text": text,
            "label": str(row.get("label", "")).strip(),
        }
        if image_url:
            sample["image_url"] = image_url
        if image_rel:
            sample["image_rel"] = image_rel

        out.append(
            sample
        )
    return out
