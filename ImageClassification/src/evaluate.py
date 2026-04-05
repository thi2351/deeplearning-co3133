from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassConfusionMatrix

from src.config import OUTPUTS_DIR



@torch.no_grad()
def evaluate_test(model, loader, num_classes, class_names, device):
    model.eval()
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    f1pc = F1Score(
        task="multiclass", num_classes=num_classes, average="none"
    ).to(device)
    cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        acc.update(preds, labels)
        f1.update(preds, labels)
        f1pc.update(preds, labels)
        cm.update(preds, labels)

    results = {
        "accuracy": acc.compute().item(),
        "f1_macro": f1.compute().item(),
        "f1_per_class": f1pc.compute().cpu().numpy(),
        "confusion_matrix": cm.compute().cpu().numpy(),
    }
    return results


def plot_history(histories: dict, save_path=None):
    """histories = {"resnet50": history_dict, "efficientnet_b3": history_dict, ...}"""
    save_path = Path(save_path or (OUTPUTS_DIR / "training_curves.png"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1D9E75", "#E8593C", "#3B8BD4", "#BA7517"]

    for (arch, h), color in zip(histories.items(), colors):
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], label=arch, color=color)
        axes[1].plot(epochs, h["val_acc"], label=arch, color=color)

    axes[0].set_title("Training loss", fontsize=13)
    axes[1].set_title("Validation accuracy", fontsize=13)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {save_path}")


def plot_confusion_matrix(cm, class_names, arch, save_path=None, show: bool = True):
    """Plot a readable confusion matrix (shows top-20 classes for readability)."""
    per_class_errors = cm.sum(axis=1) - np.diag(cm)
    top20_idx = np.argsort(per_class_errors)[-20:][::-1]
    cm_sub = cm[np.ix_(top20_idx, top20_idx)]
    labels = [class_names[i] for i in top20_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_sub, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(20),
        yticks=range(20),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion matrix — {arch} (top-20 most confused classes)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()

    path = Path(save_path or (OUTPUTS_DIR / f"cm_{arch}.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved -> {path}")


def print_results_table(all_results: dict):
    print(f"\n{'─' * 55}")
    print(f"  {'Model':<35} {'Accuracy':>8}  {'F1 macro':>8}")
    print(f"{'─' * 55}")
    for arch, r in all_results.items():
        print(f"  {arch:<35} {r['accuracy']:>8.4f}  {r['f1_macro']:>8.4f}")
    print(f"{'─' * 55}\n")
