import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score

from src.config import (
    ARCH,
    BACKBONE_LR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    HEAD_LR,
    IMG_SIZE,
    LABEL_SMOOTHING,
    PRETRAINED,
    VAL_SPLIT,
    WEIGHT_DECAY,
    epochs_for_arch,
)
from src.model import get_optimizer_groups


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        acc.update(preds, labels)
        f1.update(preds, labels)
    return acc.compute().item(), f1.compute().item()


def _save_checkpoint(path: Path, arch: str, num_classes: int, model: nn.Module):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
    }
    torch.save(payload, path)


def train(
    model,
    train_loader,
    val_loader,
    num_classes,
    arch="model",
    epochs=None,
    head_lr=None,
    backbone_lr=None,
    label_smoothing=None,
    weight_decay=None,
    checkpoint_dir=None,
    min_delta=1e-4,
):
    """Train for exactly ``epochs`` passes; no early stopping. Keeps best val_acc checkpoint."""
    if epochs is None:
        epochs = epochs_for_arch(arch)
    head_lr = HEAD_LR if head_lr is None else head_lr
    backbone_lr = BACKBONE_LR if backbone_lr is None else backbone_lr
    label_smoothing = LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay

    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(
        get_optimizer_groups(model, head_lr, backbone_lr), weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n{'─' * 55}")
    print(f"  Training : {arch}  |  device: {device}")
    print(f"  Fixed epochs: {epochs} (no early stopping)")
    print(f"  head_lr: {head_lr}  |  backbone_lr: {backbone_lr}")
    print(f"{'─' * 55}")

    best_path = checkpoint_dir / f"{arch}_best.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, num_classes, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        improved = val_acc > best_acc + min_delta
        marker = " (new best)" if improved else ""
        print(
            f"  Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | {elapsed:.0f}s{marker}"
        )

        if improved:
            best_acc = val_acc
            _save_checkpoint(best_path, arch, num_classes, model)

    print(f"{'─' * 55}")
    print(f"  Best val_acc: {best_acc:.4f}  |  saved to {best_path}")
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-100: fixed 15 epochs (CNN) or 10 (ViT); no early stopping."
    )
    parser.add_argument(
        "--arch",
        default=ARCH,
        choices=[
            "resnet50",
            "efficientnet_b3",
            "swin_tiny_patch4_window7_224",
            "vit_base_patch16_224",
        ],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Ghi đè số epoch cố định (mặc định: 15 cho CNN, 10 cho ViT).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--head-lr", type=float, default=HEAD_LR)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument(
        "--val-split", type=float, default=VAL_SPLIT, help="Fraction of train for validation."
    )
    parser.add_argument("--data-dir", default=None, help="CIFAR-100 root (default: project data/)")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Where to write .pth (default: project checkpoint/)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="only train the head (freeze the backbone).",
    )
    args = parser.parse_args()

    from src.dataset import get_dataloaders
    from src.model import build_model, freeze_backbone

    train_loader, val_loader, _, num_classes, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    model = build_model(
        args.arch, num_classes=num_classes, pretrained=PRETRAINED
    )
    if args.freeze_backbone:
        freeze_backbone(model)
    n_epochs = args.epochs if args.epochs is not None else epochs_for_arch(args.arch)
    train(
        model,
        train_loader,
        val_loader,
        num_classes,
        arch=args.arch,
        epochs=n_epochs,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
