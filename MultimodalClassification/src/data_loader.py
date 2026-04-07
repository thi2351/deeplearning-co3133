from __future__ import annotations

"""
data_loader.py
--------------
Module-first data utilities for the UPMC Food-101 multimodal project.

This loader combines the strengths of both notebooks:
- keeps the Assignment-1 CSV semantics (real sample text, official train/test)
- preserves a trainer-friendly batch format for the supervised head pipeline
- exposes metadata and dataset builders so baselines / notebook / future web code
  can all reuse the same data contract
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPProcessor

HF_CLIP_MODEL = "openai/clip-vit-base-patch32"
KAGGLE_DATASET_HANDLE = "gianmarco96/upmcfood101"
BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_SEED = 42
MAX_TEXT_LENGTH = 500
DEFAULT_VAL_RATIO = 0.10
CAPTION_TEMPLATE = "a photo of {label}, a type of food"

SampleTuple = Tuple[str, int, str, str]  # (img_path, label_idx, text, class_name)


@dataclass
class DatasetMetadata:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    classes: List[str]
    class_to_idx: Dict[str, int]
    dataset_root: str
    layout: str

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {idx: cls for cls, idx in self.class_to_idx.items()}


class Food101MultimodalDataset(Dataset):
    """Dataset returning (pixel_values, label_idx, text).

    The returned format is optimized for module reuse:
    - `pixel_values`: already preprocessed image tensor for CLIP
    - `label_idx`    : integer class label
    - `text`         : cleaned raw text, later tokenized in collate
    """

    def __init__(
        self,
        samples: Sequence[SampleTuple],
        image_processor,
        max_text_length: int = MAX_TEXT_LENGTH,
        fallback_template: str = CAPTION_TEMPLATE,
    ) -> None:
        self.samples = list(samples)
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.fallback_template = fallback_template
        if not self.samples:
            raise RuntimeError("Food101MultimodalDataset received no samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label_idx, text, class_name = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        cleaned_text = str(text).strip()[: self.max_text_length]
        if len(cleaned_text) <= 2:
            cleaned_text = self.fallback_template.format(
                label=class_name.replace("_", " ")
            )

        return pixel_values, label_idx, cleaned_text


class CollateWithProcessor:
    """Picklable collator for CLIP-friendly batched outputs."""

    def __init__(self, processor: CLIPProcessor, max_length: int = 77) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        images, labels, texts = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        text_inputs: Dict[str, torch.Tensor] = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return images, labels, text_inputs


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def _resolve_dataset_root(dataset_root: Optional[str]) -> Path:
    if dataset_root is None:
        import kagglehub

        print(f"[data_loader] Downloading '{KAGGLE_DATASET_HANDLE}' via kagglehub …")
        dataset_root = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        print(f"[data_loader] Dataset cached at: {dataset_root}")

    root_path = Path(dataset_root)
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not any(root_path.glob("*/*.jpg")):
        root_path = subdirs[0]
        print(f"[data_loader] Using inner directory: {root_path}")
    return root_path


def _read_upmc_csv_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, names=["filename", "text", "class_name"])
    df = df.dropna(subset=["class_name"]).copy()

    if len(df) > 0 and str(df.iloc[0]["filename"]).strip().lower() == "filename":
        df = df.iloc[1:].reset_index(drop=True)

    df["filename"] = df["filename"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df["class_name"] = df["class_name"].astype(str).str.strip()
    return df


def load_food101_metadata(
    dataset_root: Optional[str] = None,
    num_classes: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> DatasetMetadata:
    """Load UPMC metadata without constructing DataLoaders.

    This is the central source of truth for EDA, few-shot sampling, and label
    mapping. It preserves the Assignment-1 semantics when CSV files exist.
    """
    root_path = _resolve_dataset_root(dataset_root)

    csv_layout = (
        (root_path / "texts" / "train_titles.csv").exists()
        and (root_path / "texts" / "test_titles.csv").exists()
        and (root_path / "images" / "train").exists()
        and (root_path / "images" / "test").exists()
    )

    rng = random.Random(seed)

    if csv_layout:
        train_df = _read_upmc_csv_dataframe(root_path / "texts" / "train_titles.csv")
        test_df = _read_upmc_csv_dataframe(root_path / "texts" / "test_titles.csv")
        all_classes = sorted(train_df["class_name"].unique().tolist())
        layout = "assignment1_csv"
    else:
        all_classes = sorted(d.name for d in root_path.iterdir() if d.is_dir())
        layout = "folder_only"
        train_rows = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        for cls in all_classes:
            cls_dir = root_path / cls
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    train_rows.append(
                        {
                            "filename": img_path.name,
                            "text": CAPTION_TEMPLATE.format(label=cls.replace("_", " ")),
                            "class_name": cls,
                            "img_path": str(img_path),
                        }
                    )
        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(columns=train_df.columns)

    if num_classes is not None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive or None, got {num_classes}")
        if num_classes > len(all_classes):
            raise RuntimeError(
                f"Requested {num_classes} classes, but only {len(all_classes)} available"
            )
        selected_classes = sorted(rng.sample(all_classes, num_classes))
    else:
        selected_classes = all_classes

    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    train_df = train_df[train_df["class_name"].isin(selected_classes)].reset_index(drop=True)
    test_df = test_df[test_df["class_name"].isin(selected_classes)].reset_index(drop=True)

    # Attach resolved image paths for downstream EDA / few-shot sampling.
    if layout == "assignment1_csv":
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["img_path"] = train_df.apply(
            lambda row: str(root_path / "images" / "train" / row["class_name"] / row["filename"]),
            axis=1,
        )
        test_df["img_path"] = test_df.apply(
            lambda row: str(root_path / "images" / "test" / row["class_name"] / row["filename"]),
            axis=1,
        )

    train_df["label"] = train_df["class_name"].map(class_to_idx)
    if len(test_df) > 0:
        test_df["label"] = test_df["class_name"].map(class_to_idx)

    return DatasetMetadata(
        train_df=train_df,
        test_df=test_df,
        classes=selected_classes,
        class_to_idx=class_to_idx,
        dataset_root=str(root_path),
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Dataset / DataLoader builders
# ---------------------------------------------------------------------------


def dataframe_to_samples(df: pd.DataFrame) -> List[SampleTuple]:
    required_cols = {"img_path", "label", "text", "class_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    return [
        (
            str(row.img_path),
            int(row.label),
            str(row.text),
            str(row.class_name),
        )
        for row in df.itertuples(index=False)
    ]



def build_dataset_from_dataframe(
    df: pd.DataFrame,
    processor: CLIPProcessor,
) -> Food101MultimodalDataset:
    return Food101MultimodalDataset(
        samples=dataframe_to_samples(df),
        image_processor=processor.image_processor,
    )



def build_dataloader_from_dataframe(
    df: pd.DataFrame,
    processor: CLIPProcessor,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    shuffle: bool = False,
    seed: int = RANDOM_SEED,
) -> DataLoader:
    dataset = build_dataset_from_dataframe(df, processor)
    collate_fn = CollateWithProcessor(processor)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator,
    )



def split_train_val_dataframe(
    train_df: pd.DataFrame,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    indices = list(range(len(train_df)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    train_size = len(indices) - val_size
    if train_size <= 0:
        raise RuntimeError("val_ratio is too large; no samples left for training.")
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)



def get_food101_dataloaders(
    num_classes: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    hf_model_name: str = HF_CLIP_MODEL,
    dataset_root: Optional[str] = None,
    processor: Optional[CLIPProcessor] = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    return_test_loader: bool = True,
    return_metadata: bool = False,
):
    """Build the train / val / test loaders used by the final project.

    Returns either:
      train_loader, val_loader, test_loader, classes, class_to_idx
    or, when `return_metadata=True`:
      train_loader, val_loader, test_loader, classes, class_to_idx, metadata
    """
    metadata = load_food101_metadata(
        dataset_root=dataset_root,
        num_classes=num_classes,
        seed=seed,
    )

    if processor is None:
        print(f"[data_loader] Loading CLIPProcessor from '{hf_model_name}' …")
        processor = CLIPProcessor.from_pretrained(hf_model_name)
    else:
        print("[data_loader] Using pre-loaded CLIPProcessor.")

    train_df, val_df = split_train_val_dataframe(metadata.train_df, val_ratio=val_ratio, seed=seed)
    train_loader = build_dataloader_from_dataframe(
        train_df,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
    )
    val_loader = build_dataloader_from_dataframe(
        val_df,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
    )

    if len(metadata.test_df) > 0:
        test_loader = build_dataloader_from_dataframe(
            metadata.test_df,
            processor=processor,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=seed,
        )
    else:
        test_loader = val_loader

    print(
        f"[data_loader] layout={metadata.layout} | "
        f"train={len(train_df)} | val={len(val_df)} | test={len(metadata.test_df)} | "
        f"classes={len(metadata.classes)}"
    )

    if return_test_loader:
        outputs = [
            train_loader,
            val_loader,
            test_loader,
            metadata.classes,
            metadata.class_to_idx,
        ]
        if return_metadata:
            outputs.append(metadata)
        return tuple(outputs)

    outputs = [train_loader, val_loader, metadata.classes, metadata.class_to_idx]
    if return_metadata:
        outputs.append(metadata)
    return tuple(outputs)
