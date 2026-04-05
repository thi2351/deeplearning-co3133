"""Assignment-style figures from a 100×100 test confusion matrix."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.cifar100_superclass import SUPERCLASSES, SUPER_NAMES, SUPER_TO_CLASSES
from src.config import CIFAR100_CLASS_NAMES


def plot_inter_superclass(cm100: np.ndarray, arch: str, save_path: Path) -> tuple[np.ndarray, np.ndarray]:
    super_cm = np.zeros((20, 20))
    for si in range(20):
        for sj in range(20):
            ri = SUPER_TO_CLASSES[si]
            ci = SUPER_TO_CLASSES[sj]
            super_cm[si, sj] = cm100[np.ix_(ri, ci)].sum()
    row_sum = super_cm.sum(axis=1, keepdims=True)
    super_cm_norm = np.divide(
        super_cm, row_sum, out=np.zeros_like(super_cm, dtype=float), where=row_sum > 0
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(super_cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Proportion")
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels(SUPER_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(SUPER_NAMES, fontsize=9)
    ax.set_xlabel("Predicted superclass", fontsize=11)
    ax.set_ylabel("True superclass", fontsize=11)
    ax.set_title(
        f"Inter-superclass confusion — {arch}\n"
        f"Diagonal = correct rate  |  Off-diagonal = wrong superclass rate",
        fontsize=12,
    )
    for i in range(20):
        for j in range(20):
            val = super_cm_norm[i, j]
            if val > 0.01:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if val > 0.5 else "black",
                )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return cm100, super_cm


def plot_intra_superclass(
    cm100: np.ndarray, super_cm: np.ndarray, arch: str, save_path: Path
) -> None:
    intra_error = np.zeros(20)
    for si in range(20):
        ri = SUPER_TO_CLASSES[si]
        intra_correct_super = super_cm[si, si]
        intra_correct_class = cm100[np.ix_(ri, ri)].diagonal().sum()
        if intra_correct_super > 0:
            intra_error[si] = 1 - intra_correct_class / intra_correct_super

    colors = [
        "#E24B4A" if e > 0.15 else "#EF9F27" if e > 0.08 else "#1D9E75"
        for e in intra_error
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(20), intra_error, color=colors, alpha=0.85)
    ax.set_xticks(range(20))
    ax.set_xticklabels(SUPER_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Intra-class error rate", fontsize=11)
    ax.set_ylim(0, 0.65)
    ax.set_title(
        f"Intra-superclass confusion — {arch}\n"
        f"= correct superclass but wrong subclass rate",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2, axis="y")
    for bar, val in zip(bars, intra_error):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(
        handles=[
            Patch(color="#E24B4A", label="> 15% high confusion"),
            Patch(color="#EF9F27", label="8–15% medium"),
            Patch(color="#1D9E75", label="< 8% low"),
        ],
        fontsize=9,
        loc="upper right",
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_subclass_accuracy(cm100: np.ndarray, arch: str, save_path: Path) -> None:
    row_sum = cm100.sum(axis=1)
    acc_per_class = np.divide(
        np.diag(cm100).astype(float), row_sum, out=np.zeros(100), where=row_sum > 0
    )
    class_names = list(CIFAR100_CLASS_NAMES)

    x_vals, acc_vals, colors_sub, xtick_labels = [], [], [], []
    x = 0.0
    for si, (_sname, _subs) in enumerate(SUPERCLASSES.items()):
        for cls_idx in SUPER_TO_CLASSES[si]:
            x_vals.append(x)
            acc_vals.append(acc_per_class[cls_idx])
            colors_sub.append("#7F77DD" if si % 2 == 0 else "#378ADD")
            xtick_labels.append(class_names[cls_idx])
            x += 1
        x += 1.5

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x_vals, acc_vals, color=colors_sub, alpha=0.85, width=0.8)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Per-subclass accuracy — {arch}\nGrouped by superclass (alternating colors)",
        fontsize=12,
    )
    ax.axhline(
        y=float(acc_per_class.mean()),
        color="#E24B4A",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {acc_per_class.mean():.3f}",
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=9)

    x = 0.0
    for si, (sname, _) in enumerate(SUPERCLASSES.items()):
        center = float(np.mean([x + j for j in range(5)]))
        ax.text(
            center,
            1.08,
            sname,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=30,
            color="#444441",
        )
        x += 5 + 1.5

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
