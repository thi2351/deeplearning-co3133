from __future__ import annotations

"""EDA helpers extracted from Assignment 1 into reusable functions."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix


def dataset_overview(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float | int | str]:
    class_counts = train_df["class_name"].value_counts()
    return {
        "num_classes": int(train_df["class_name"].nunique()),
        "num_train_samples": int(len(train_df)),
        "num_test_samples": int(len(test_df)),
        "avg_train_samples_per_class": float(class_counts.mean()),
        "min_class": str(class_counts.idxmin()),
        "min_class_count": int(class_counts.min()),
        "max_class": str(class_counts.idxmax()),
        "max_class_count": int(class_counts.max()),
    }



def add_text_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_char_len"] = out["text"].fillna("").astype(str).apply(len)
    out["text_word_len"] = out["text"].fillna("").astype(str).apply(lambda x: len(x.split()))
    return out



def multimodal_text_health_report(train_df: pd.DataFrame) -> Dict[str, float | int]:
    df = add_text_length_columns(train_df)
    short_mask = df["text_word_len"] <= 2
    return {
        "short_text_samples": int(short_mask.sum()),
        "short_text_ratio_pct": float(short_mask.mean() * 100.0),
        "char_mean": float(df["text_char_len"].mean()),
        "char_median": float(df["text_char_len"].median()),
        "word_mean": float(df["text_word_len"].mean()),
        "word_median": float(df["text_word_len"].median()),
    }



def plot_class_distribution(
    train_df: pd.DataFrame,
    save_path: str | None = None,
    top_n: int = 20,
):
    class_counts = train_df["class_name"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_classes = class_counts.head(top_n)
    axes[0].barh(top_classes.index[::-1], top_classes.values[::-1])
    axes[0].set_title(f"Top {top_n} classes with most samples")
    axes[0].set_xlabel("Number of samples")

    axes[1].hist(class_counts.values, bins=20, edgecolor="black")
    axes[1].set_title("Sample distribution across classes")
    axes[1].set_xlabel("Samples per class")
    axes[1].set_ylabel("Number of classes")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig



def plot_text_length_histograms(train_df: pd.DataFrame, save_path: str | None = None):
    df = add_text_length_columns(train_df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(df["text_char_len"], bins=50, edgecolor="black")
    axes[0].set_title("Text length distribution (characters)")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Samples")

    axes[1].hist(df["text_word_len"], bins=50, edgecolor="black")
    axes[1].set_title("Text length distribution (words)")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Samples")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig



def plot_sample_gallery(
    df: pd.DataFrame,
    class_names: Sequence[str],
    save_path: str | None = None,
    n_samples: int = 6,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        row = df.iloc[int(idx)]
        try:
            image = Image.open(row["img_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        text = str(row["text"])
        if len(text) > 120:
            text = text[:117] + "..."
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Class: {row['class_name']}\nText: {text}", fontsize=10, loc="left")

    for ax in axes[len(idxs):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig



def top_confused_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    top_n: int = 15,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)
    rows: List[Tuple[int, str, str]] = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                rows.append((int(cm[i, j]), class_names[i], class_names[j]))
    rows.sort(reverse=True)
    return pd.DataFrame(rows[:top_n], columns=["Count", "True Class", "Predicted As"])
