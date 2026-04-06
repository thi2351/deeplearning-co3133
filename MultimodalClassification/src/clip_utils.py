from __future__ import annotations

"""Shared CLIP helpers for training, baselines, and web inference."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

HF_CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class CLIPBundle:
    model: CLIPModel
    processor: CLIPProcessor
    device: torch.device
    model_name: str = HF_CLIP_MODEL


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def load_frozen_clip(
    model_name: str = HF_CLIP_MODEL,
    device: torch.device | None = None,
) -> CLIPBundle:
    if device is None:
        device = get_device()
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return CLIPBundle(model=model, processor=processor, device=device, model_name=model_name)



def _coerce_feature_tensor(output, clip_model, modality: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.float()

    embed_attr = "image_embeds" if modality == "image" else "text_embeds"
    if hasattr(output, embed_attr):
        embeds = getattr(output, embed_attr)
        if embeds is not None:
            return embeds.float()

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        pooled = output.pooler_output
        projection = clip_model.visual_projection if modality == "image" else clip_model.text_projection
        if projection is not None:
            return projection(pooled).float()
        return pooled.float()

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        pooled = output.last_hidden_state[:, 0]
        projection = clip_model.visual_projection if modality == "image" else clip_model.text_projection
        if projection is not None:
            return projection(pooled).float()
        return pooled.float()

    raise TypeError(
        f"Unsupported {modality} feature output type: {type(output)!r}."
    )


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    clip: CLIPBundle,
    normalize: bool = True,
    max_length: int = 77,
) -> torch.Tensor:
    inputs = clip.processor(
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(clip.device) for k, v in inputs.items()}
    # safest path across transformers versions
    out = clip.model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    feats = _coerce_feature_tensor(out, clip.model, modality="text")
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_tokenized_text_inputs(
    text_inputs: Dict[str, torch.Tensor],
    clip: CLIPBundle,
    normalize: bool = True,
) -> torch.Tensor:
    text_inputs = {k: v.to(clip.device) for k, v in text_inputs.items()}
    out = clip.model.text_model(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
    )
    feats = _coerce_feature_tensor(out, clip.model, modality="text")
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_pil_images(
    images: Sequence[Image.Image],
    clip: CLIPBundle,
    normalize: bool = True,
) -> torch.Tensor:
    inputs = clip.processor(images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(clip.device)
    out = clip.model.vision_model(pixel_values=pixel_values)
    feats = _coerce_feature_tensor(out, clip.model, modality="image")
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_pixel_values(
    pixel_values: torch.Tensor,
    clip: CLIPBundle,
    normalize: bool = True,
) -> torch.Tensor:
    pixel_values = pixel_values.to(clip.device)
    out = clip.model.vision_model(pixel_values=pixel_values)
    feats = _coerce_feature_tensor(out, clip.model, modality="image")
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def extract_batch_features(
    images: torch.Tensor,
    text_inputs: Dict[str, torch.Tensor],
    clip: CLIPBundle,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_feats = encode_pixel_values(images, clip, normalize=True)
    text_feats = encode_tokenized_text_inputs(text_inputs, clip, normalize=True)
    return image_feats, text_feats
