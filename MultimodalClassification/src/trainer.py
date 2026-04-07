from __future__ import annotations

"""Training utilities for the final modular multimodal project.

Key optimization added here:
Because CLIP is frozen, we can precompute image/text features once and then train
only the MLP head on cached embeddings. This makes training much faster and much
more stable than repeatedly re-encoding the same batches every epoch.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .clip_utils import HF_CLIP_MODEL, CLIPBundle, extract_batch_features, get_device, load_frozen_clip
from .models import MultimodalClassificationHead

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
CHECKPOINT_PATH = WEIGHTS_DIR / "food101_head.pth"


@dataclass
class CachedFeatures:
    image_features: torch.Tensor
    text_features: torch.Tensor
    labels: torch.Tensor

    def to_dataset(self) -> TensorDataset:
        return TensorDataset(self.image_features, self.text_features, self.labels)


@dataclass
class TrainingArtifacts:
    head: MultimodalClassificationHead
    history: Dict[str, List[float]]
    checkpoint_path: str


@torch.no_grad()
def cache_loader_features(
    loader: DataLoader,
    clip: CLIPBundle,
    desc: str = "Caching features",
) -> CachedFeatures:
    image_parts: List[torch.Tensor] = []
    text_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []

    for images, labels, text_inputs in tqdm(loader, desc=desc):
        img_f, txt_f = extract_batch_features(images, text_inputs, clip)
        image_parts.append(img_f.cpu())
        text_parts.append(txt_f.cpu())
        label_parts.append(labels.cpu())

    return CachedFeatures(
        image_features=torch.cat(image_parts, dim=0),
        text_features=torch.cat(text_parts, dim=0),
        labels=torch.cat(label_parts, dim=0),
    )



def build_cached_feature_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    clip: CLIPBundle,
    batch_size: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    cached_train = cache_loader_features(train_loader, clip, desc="Caching train")
    cached_val = cache_loader_features(val_loader, clip, desc="Caching val")

    batch_size = batch_size or train_loader.batch_size or 32
    train_cached_loader = DataLoader(
        cached_train.to_dataset(), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_cached_loader = DataLoader(
        cached_val.to_dataset(), batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_cached_loader, val_cached_loader



def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()



def train_head_on_cached_features(
    train_cached_loader: DataLoader,
    val_cached_loader: DataLoader,
    num_classes: int,
    clip_embed_dim: int = 512,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> TrainingArtifacts:
    if device is None:
        device = get_device()

    ckpt_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    head = MultimodalClassificationHead(
        num_classes=num_classes,
        clip_embed_dim=clip_embed_dim,
        use_text=True,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = float("-inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        head.train()
        run_loss = run_acc = n = 0
        for img_f, txt_f, labels in tqdm(
            train_cached_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False
        ):
            img_f = img_f.to(device, non_blocking=True)
            txt_f = txt_f.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = head(img_f, txt_f)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item()
            run_acc += _accuracy(logits.detach(), labels)
            n += 1

        scheduler.step()
        train_loss = run_loss / max(n, 1)
        train_acc = run_acc / max(n, 1)

        head.eval()
        vl = va = nv = 0
        with torch.no_grad():
            for img_f, txt_f, labels in tqdm(
                val_cached_loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=False
            ):
                img_f = img_f.to(device, non_blocking=True)
                txt_f = txt_f.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = head(img_f, txt_f)
                vl += criterion(logits, labels).item()
                va += _accuracy(logits, labels)
                nv += 1

        val_loss = vl / max(nv, 1)
        val_acc = va / max(nv, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        print(
            f"Epoch {epoch:>3}/{num_epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}" + ("  ← best" if is_best else "")
        )
        if is_best:
            best_val_acc = val_acc
            torch.save(head.state_dict(), ckpt_path)
            print(f"[trainer] Checkpoint → {ckpt_path} (val_acc={best_val_acc:.4f})")

    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    head.eval()
    return TrainingArtifacts(head=head, history=history, checkpoint_path=str(ckpt_path))



def train_model(
    hf_model_name: str = HF_CLIP_MODEL,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    classes: List[str] | None = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_cached_features: bool = True,
) -> Tuple[MultimodalClassificationHead, Dict[str, List[float]]]:
    """Public training entrypoint used by notebook and future apps.

    By default it caches CLIP features once, then trains only on those features.
    This keeps the module web-friendly and much faster for iterative experiments.
    """
    if train_loader is None or val_loader is None:
        raise ValueError("Both train_loader and val_loader must be provided")
    if classes is None:
        raise ValueError("classes must be provided")
    if device is None:
        device = get_device()

    print(f"[trainer] Using device: {device}")
    clip = load_frozen_clip(hf_model_name, device=device)
    clip_embed_dim = int(clip.model.config.projection_dim)

    if use_cached_features:
        train_cached_loader, val_cached_loader = build_cached_feature_loaders(
            train_loader=train_loader,
            val_loader=val_loader,
            clip=clip,
            batch_size=train_loader.batch_size,
        )
    else:
        # fallback path: cache anyway because CLIP is frozen and repeated extraction is wasteful
        train_cached_loader, val_cached_loader = build_cached_feature_loaders(
            train_loader=train_loader,
            val_loader=val_loader,
            clip=clip,
            batch_size=train_loader.batch_size,
        )

    artifacts = train_head_on_cached_features(
        train_cached_loader=train_cached_loader,
        val_cached_loader=val_cached_loader,
        num_classes=len(classes),
        clip_embed_dim=clip_embed_dim,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return artifacts.head, artifacts.history


@torch.no_grad()
def evaluate_head(
    head: MultimodalClassificationHead,
    loader: DataLoader,
    hf_model_name: str = HF_CLIP_MODEL,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], List[int]]:
    if device is None:
        device = get_device()
    clip = load_frozen_clip(hf_model_name, device=device)
    head = head.to(device).eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels, text_inputs in tqdm(loader, desc="Evaluating head"):
        img_f, txt_f = extract_batch_features(images, text_inputs, clip)
        preds = head(img_f, txt_f).argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels



def load_head(
    num_classes: int,
    checkpoint_path: Optional[str] = None,
    clip_embed_dim: int = 512,
    device: Optional[torch.device] = None,
) -> MultimodalClassificationHead:
    if device is None:
        device = get_device()
    ckpt = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt}'. Run train_model() first.")
    head = MultimodalClassificationHead(num_classes=num_classes, clip_embed_dim=clip_embed_dim).to(device)
    head.load_state_dict(torch.load(ckpt, map_location=device))
    head.eval()
    return head
