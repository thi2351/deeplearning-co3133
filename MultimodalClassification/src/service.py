from __future__ import annotations

"""Web-friendly inference service wrapper around frozen CLIP + trained head."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image

from .clip_utils import HF_CLIP_MODEL, encode_pil_images, encode_texts, get_device, load_frozen_clip
from .models import MultimodalClassificationHead, PredictionRecord
from .trainer import load_head


@dataclass
class ServiceConfig:
    classes: Sequence[str]
    checkpoint_path: str
    hf_model_name: str = HF_CLIP_MODEL
    device: torch.device | None = None


class MultimodalFoodClassifierService:
    """Simple service object meant to be reused inside FastAPI/Gradio/Flask.

    Example
    -------
    service = MultimodalFoodClassifierService(classes, checkpoint_path)
    preds = service.predict(pil_image, text="spicy noodle soup", top_k=5)
    """

    def __init__(
        self,
        classes: Sequence[str],
        checkpoint_path: str,
        hf_model_name: str = HF_CLIP_MODEL,
        device: torch.device | None = None,
    ) -> None:
        self.classes = list(classes)
        self.device = device or get_device()
        self.clip = load_frozen_clip(hf_model_name, device=self.device)
        self.head = load_head(
            num_classes=len(self.classes),
            checkpoint_path=checkpoint_path,
            clip_embed_dim=int(self.clip.model.config.projection_dim),
            device=self.device,
        )
        self.head.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        text: str,
        top_k: int = 5,
    ) -> List[PredictionRecord]:
        image = image.convert("RGB")
        text = (text or "").strip()
        if len(text) <= 2:
            text = "a photo of food"

        image_features = encode_pil_images([image], self.clip, normalize=True)
        text_features = encode_texts([text], self.clip, normalize=True)
        logits = self.head(image_features, text_features)
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, len(self.classes))
        scores, indices = probs.topk(k, dim=-1)

        results: List[PredictionRecord] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            results.append(PredictionRecord(class_name=self.classes[idx], score=float(score)))
        return results
