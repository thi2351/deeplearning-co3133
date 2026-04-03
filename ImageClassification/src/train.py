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
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    HEAD_LR,
    IMG_SIZE,
    LABEL_SMOOTHING,
    PRETRAINED,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from src.model import get_optimizer_groups

# ── Training loop ────────────────────────────────────────────────────────────


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


# ── Main train function ──────────────────────────────────────────────────────


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
    early_stop_patience=None,
    early_stop_min_delta=None,
):
    epochs = EPOCHS if epochs is None else epochs
    head_lr = HEAD_LR if head_lr is None else head_lr
    backbone_lr = BACKBONE_LR if backbone_lr is None else backbone_lr
    label_smoothing = LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    stop_patience = (
        EARLY_STOP_PATIENCE if early_stop_patience is None else early_stop_patience
    )
    min_delta = (
        EARLY_STOP_MIN_DELTA
        if early_stop_min_delta is None
        else early_stop_min_delta
    )

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
    epochs_no_improve = 0
    stopped_early = False

    print(f"\n{'─' * 55}")
    print(f"  Training : {arch}  |  device: {device}")
    print(
        f"  Epochs (max): {epochs}  |  early_stop patience: {stop_patience}"
        if stop_patience > 0
        else f"  Epochs: {epochs}  |  early stopping: off"
    )
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
            epochs_no_improve = 0
            _save_checkpoint(best_path, arch, num_classes, model)
        else:
            epochs_no_improve += 1
            if stop_patience > 0 and epochs_no_improve >= stop_patience:
                print(
                    f"  Early stopping: val_acc không vượt best ({best_acc:.4f}) "
                    f"sau {stop_patience} epoch liên tiếp."
                )
                stopped_early = True
                break

    print(f"{'─' * 55}")
    stop_note = " (early stop)" if stopped_early else ""
    print(
        f"  Best val_acc: {best_acc:.4f}{stop_note}  |  saved to {best_path}"
    )
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-100 classifiers. Defaults come from src/config.py."
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
        default=EPOCHS,
        help="Số epoch tối đa; có thể dừng sớm nếu bật early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="Dừng nếu val_acc không cải thiện trong nhiều epoch liên tiếp; 0 = tắt.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=EARLY_STOP_MIN_DELTA,
        help="Ngưỡng cải thiện val_acc tối thiểu để coi là epoch tốt hơn.",
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
    args = parser.parse_args()

    from src.dataset import get_dataloaders
    from src.model import build_model

    train_loader, val_loader, _, num_classes, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    model = build_model(
        args.arch, num_classes=num_classes, pretrained=PRETRAINED
    )
    train(
        model,
        train_loader,
        val_loader,
        num_classes,
        arch=args.arch,
        epochs=args.epochs,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        checkpoint_dir=args.checkpoint_dir,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )


if __name__ == "__main__":
    main()
