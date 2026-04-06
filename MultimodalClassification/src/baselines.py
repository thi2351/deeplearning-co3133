from __future__ import annotations

"""Research-style baselines carried over from Assignment 1, but fully modular."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .clip_utils import HF_CLIP_MODEL, CLIPBundle, encode_texts, extract_batch_features, get_device, load_frozen_clip
from .data_loader import build_dataloader_from_dataframe

CAPTION_TEMPLATE = "A photo of {label}, a delicious food"


@dataclass
class FewShotResult:
    k: int
    accuracy: float
    macro_f1: float
    predictions: np.ndarray


@torch.no_grad()
def evaluate_multimodal_zero_shot(
    loader: DataLoader,
    classes: Sequence[str],
    alpha: float = 0.5,
    hf_model_name: str = HF_CLIP_MODEL,
    clip: CLIPBundle | None = None,
    caption_template: str = CAPTION_TEMPLATE,
    device: torch.device | None = None,
) -> Tuple[List[int], List[int], Dict[str, float]]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if clip is None:
        if device is None:
            device = get_device()
        clip = load_frozen_clip(hf_model_name, device=device)

    class_prompts = [caption_template.format(label=c.replace("_", " ")) for c in classes]
    class_text_features = encode_texts(class_prompts, clip, normalize=True)
    logit_scale = clip.model.logit_scale.exp()

    all_preds: List[int] = []
    all_labels: List[int] = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels, text_inputs in tqdm(loader, desc="Zero-shot multimodal eval"):
        labels = labels.to(clip.device)
        img_feats, txt_feats = extract_batch_features(images, text_inputs, clip)
        img_logits = img_feats @ class_text_features.T
        txt_logits = txt_feats @ class_text_features.T
        fused_logits = logit_scale * (alpha * img_logits + (1.0 - alpha) * txt_logits)

        preds = fused_logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        correct_top1 += (preds == labels).sum().item()

        k = min(5, len(classes))
        top5 = fused_logits.topk(k, dim=-1).indices
        correct_top5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
        total += labels.size(0)

    metrics = {
        "accuracy": correct_top1 / total,
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "top5_accuracy": correct_top5 / total,
        "alpha": alpha,
    }
    return all_preds, all_labels, metrics


@torch.no_grad()
def extract_multimodal_embeddings(
    loader: DataLoader,
    hf_model_name: str = HF_CLIP_MODEL,
    clip: CLIPBundle | None = None,
    device: torch.device | None = None,
    desc: str = "Extracting embeddings",
) -> Tuple[np.ndarray, np.ndarray]:
    if clip is None:
        if device is None:
            device = get_device()
        clip = load_frozen_clip(hf_model_name, device=device)

    all_embs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for images, labels, text_inputs in tqdm(loader, desc=desc):
        img_feats, txt_feats = extract_batch_features(images, text_inputs, clip)
        embs = torch.cat([img_feats, txt_feats], dim=-1)
        all_embs.append(embs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)



def run_few_shot_logreg(
    train_df: pd.DataFrame,
    test_loader: DataLoader,
    processor,
    k_values: Sequence[int] = (1, 5, 10),
    seed: int = 42,
    hf_model_name: str = HF_CLIP_MODEL,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> Dict[int, FewShotResult]:
    if device is None:
        device = get_device()

    clip = load_frozen_clip(hf_model_name, device=device)
    x_test, y_test = extract_multimodal_embeddings(
        test_loader,
        clip=clip,
        desc="Test embeddings",
    )

    results: Dict[int, FewShotResult] = {}
    for k in k_values:
        few_shot_df = (
            train_df.groupby("label")
            .sample(n=k, random_state=seed, replace=False)
            .reset_index(drop=True)
        )
        few_shot_loader = build_dataloader_from_dataframe(
            few_shot_df,
            processor=processor,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        x_train_k, y_train_k = extract_multimodal_embeddings(
            few_shot_loader,
            clip=clip,
            desc=f"Train embeddings K={k}",
        )

        clf = LogisticRegression(random_state=seed, max_iter=1000, C=1.0)
        clf.fit(x_train_k, y_train_k)
        preds = clf.predict(x_test)
        results[k] = FewShotResult(
            k=k,
            accuracy=accuracy_score(y_test, preds),
            macro_f1=f1_score(y_test, preds, average="macro"),
            predictions=preds,
        )
    return results



def build_results_table(
    zero_shot_metrics: Dict[str, float],
    few_shot_results: Dict[int, FewShotResult],
    final_model_metrics: Dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = [
        {
            "Method": f"Zero-Shot CLIP (multimodal, α={zero_shot_metrics['alpha']:.2f})",
            "Accuracy": zero_shot_metrics["accuracy"],
            "Macro-F1": zero_shot_metrics["macro_f1"],
        }
    ]
    for k in sorted(few_shot_results):
        r = few_shot_results[k]
        rows.append(
            {
                "Method": f"Few-Shot CLIP K={k} + LogReg",
                "Accuracy": r.accuracy,
                "Macro-F1": r.macro_f1,
            }
        )
    if final_model_metrics is not None:
        rows.append(
            {
                "Method": "Fine-tuned Multimodal Head",
                "Accuracy": final_model_metrics["accuracy"],
                "Macro-F1": final_model_metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)
