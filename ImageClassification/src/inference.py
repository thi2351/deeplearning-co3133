"""Load checkpoints saved by `train` and run predictions (deployment-friendly)."""
import argparse
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision import datasets

from src.config import CHECKPOINT_DIR, DATA_DIR
from src.dataset import VAL_TRANSFORMS, get_cifar100_class_names
from src.model import SUPPORTED_MODELS, build_model


def _infer_arch_from_checkpoint_path(path: Path) -> Optional[str]:
    """Recover timm arch name from filenames like ``resnet50_best.pth``."""
    stem = path.stem.removesuffix("_best")
    if stem in SUPPORTED_MODELS:
        return stem
    # Longest key first so e.g. swin_tiny... wins over shorter partial matches
    for arch in sorted(SUPPORTED_MODELS.keys(), key=len, reverse=True):
        if arch in path.name:
            return arch
    return None


def load_checkpoint_dict(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    # Legacy: raw state_dict only
    return {"state_dict": ckpt, "arch": None, "num_classes": 100, "img_size": None}


def load_model_for_inference(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    pretrained_backbone: bool = False,
    arch: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Build model from checkpoint metadata and load weights.
    Use ``pretrained_backbone=False`` for deployment (weights come from checkpoint).
    For older checkpoints saved as raw ``state_dict`` only, pass ``arch`` (and optionally ``num_classes``).
    """
    checkpoint_path = Path(checkpoint_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_checkpoint_dict(checkpoint_path)
    resolved_arch = arch or meta.get("arch") or _infer_arch_from_checkpoint_path(
        checkpoint_path
    )
    resolved_num = int(num_classes or meta.get("num_classes", 100))
    if not resolved_arch:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'arch' metadata and filename "
            f"does not match a known model. Pass arch=... explicitly, or save checkpoints "
            f"with the updated train.py (includes arch in the file)."
        )
    model = build_model(
        resolved_arch, num_classes=resolved_num, pretrained=pretrained_backbone
    )
    model.load_state_dict(meta["state_dict"], strict=True)
    model.to(device)
    model.eval()
    meta_out = {**meta, "arch": resolved_arch, "num_classes": resolved_num}
    return model, meta_out


@torch.inference_mode()
def predict_logits(model: torch.nn.Module, images: torch.Tensor, device: torch.device):
    images = images.to(device)
    return model(images)


@torch.inference_mode()
def predict_class_ids(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    logits = predict_logits(model, images, device)
    return logits.argmax(dim=1)


def default_checkpoint_path(arch: str) -> Path:
    return CHECKPOINT_DIR / f"{arch}_best.pth"


def load_image_for_model(path: Union[str, Path]) -> torch.Tensor:
    """Load an RGB image file and return a batch ``[1, 3, H, W]`` (same preprocessing as val)."""
    with Image.open(path) as img:
        x = VAL_TRANSFORMS(img.convert("RGB"))
    return x.unsqueeze(0)


def load_image_tensor_from_bytes(raw: bytes) -> torch.Tensor:
    """Decode image bytes to batch ``[1, 3, H, W]`` (avoids temp files; better on Windows)."""
    with Image.open(BytesIO(raw)) as img:
        x = VAL_TRANSFORMS(img.convert("RGB"))
    return x.unsqueeze(0)


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image_path: Union[str, Path],
    device: torch.device,
    class_names: Optional[List[str]] = None,
    topk: int = 5,
) -> Dict[str, Any]:
    """
    Classify a single image file. Uses CIFAR-100 label names unless ``class_names`` is set.

    The model was trained on CIFAR-100 (32×32, same normalize); arbitrary photos are resized
    to ``IMG_SIZE`` but predictions are only meaningful for CIFAR-like categories.
    """
    image_path = Path(image_path)
    if class_names is None:
        class_names = get_cifar100_class_names()
    num_classes = len(class_names)
    k = max(1, min(topk, num_classes))

    x = load_image_for_model(image_path)
    logits = predict_logits(model, x, device)[0]
    probs = logits.softmax(dim=0)
    topk_prob, topk_idx = probs.topk(k)

    top1_i = int(topk_idx[0].item())
    out: Dict[str, Any] = {
        "path": str(image_path.resolve()),
        "top1_id": top1_i,
        "top1_name": class_names[top1_i],
        "top1_prob": float(topk_prob[0].item()),
        "topk": [
            {"id": int(i), "name": class_names[int(i)], "prob": float(p)}
            for i, p in zip(topk_idx.tolist(), topk_prob.tolist())
        ],
    }
    return out


@torch.inference_mode()
def predict_image_bytes(
    model: torch.nn.Module,
    raw: bytes,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    topk: int = 5,
) -> Dict[str, Any]:
    """Same as :func:`predict_image` but takes raw file bytes (e.g. HTTP upload)."""
    if class_names is None:
        class_names = get_cifar100_class_names()
    num_classes = len(class_names)
    k = max(1, min(topk, num_classes))
    x = load_image_tensor_from_bytes(raw)
    logits = predict_logits(model, x, device)[0]
    probs = logits.softmax(dim=0)
    topk_prob, topk_idx = probs.topk(k)
    top1_i = int(topk_idx[0].item())
    return {
        "top1_id": top1_i,
        "top1_name": class_names[top1_i],
        "top1_prob": float(topk_prob[0].item()),
        "topk": [
            {"id": int(i), "name": class_names[int(i)], "prob": float(p)}
            for i, p in zip(topk_idx.tolist(), topk_prob.tolist())
        ],
    }


def predict_one_image_file(
    checkpoint_path: Union[str, Path],
    image_path: Union[str, Path],
    topk: int = 5,
    arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Load checkpoint + model, then run :func:`predict_image`."""
    model, _meta = load_model_for_inference(checkpoint_path, arch=arch)
    device = next(model.parameters()).device
    return predict_image(model, image_path, device, topk=topk)


def _parse_cifar_indices(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def get_cifar100_tensor_dataset(
    split: str = "test",
    data_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
):
    """CIFAR-100 with ``VAL_TRANSFORMS`` (same as eval). ``split`` is ``train`` or ``test``."""
    root = Path(data_dir or DATA_DIR)
    train = split == "train"
    return datasets.CIFAR100(
        root=root, train=train, download=download, transform=VAL_TRANSFORMS
    )


@torch.inference_mode()
def _print_cifar_one(
    idx: int,
    true_y: int,
    logits: torch.Tensor,
    class_names: Sequence[str],
    topk: int,
) -> None:
    probs = logits[0].softmax(dim=0)
    k = max(1, min(topk, len(probs)))
    topv, topi = probs.topk(k)
    pred = int(topi[0].item())
    ok = "yes" if pred == true_y else "no"
    print(
        f"index {idx:5d}  true: [{true_y:3d}] {class_names[true_y]:20s}  "
        f"pred: [{pred:3d}] {class_names[pred]:20s}  correct: {ok}"
    )
    for rank, (p, i) in enumerate(zip(topv.tolist(), topi.tolist()), 1):
        print(f"           top-{rank}: {p:.2%}  [{i:3d}] {class_names[i]}")


def predict_cifar_indices(
    checkpoint_path: Union[str, Path],
    indices: Sequence[int],
    *,
    arch: Optional[str] = None,
    split: str = "test",
    data_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
    topk: int = 5,
    ds: Optional[Any] = None,
) -> None:
    """
    Load checkpoint and CIFAR-100 (unless ``ds`` is provided), then print predictions for ``indices``.
    """
    if ds is None:
        ds = get_cifar100_tensor_dataset(
            split=split, data_dir=data_dir, download=download
        )
    class_names = get_cifar100_class_names()
    ckpt = Path(checkpoint_path)
    model, _ = load_model_for_inference(ckpt, arch=arch)
    device = next(model.parameters()).device

    print(f"checkpoint : {ckpt}")
    print(f"split      : {split}  ({len(ds)} images)")
    shown = list(indices) if len(indices) <= 20 else f"{len(indices)} samples"
    print(f"indices    : {shown}")
    print()

    for idx in indices:
        if idx < 0 or idx >= len(ds):
            raise IndexError(f"index {idx} out of range for split (0 .. {len(ds) - 1})")
        img, y = ds[idx]
        logits = predict_logits(model, img.unsqueeze(0), device)
        _print_cifar_one(idx, int(y), logits, class_names, topk)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify with a saved .pth: either an image file, or CIFAR-100 samples by index."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="Path to an image (omit if using --cifar-indices / --cifar-sample).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Path to .pth (default: {CHECKPOINT_DIR}/<arch>_best.pth)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Override architecture or default checkpoint (default file: resnet50).",
    )
    parser.add_argument("--topk", type=int, default=5)

    cifar = parser.add_argument_group("CIFAR-100 (official split, val preprocessing)")
    cg = cifar.add_mutually_exclusive_group()
    cg.add_argument(
        "--cifar-indices",
        type=str,
        metavar="0,1,42",
        help="Comma-separated indices (use --split test or train).",
    )
    cg.add_argument(
        "--cifar-sample",
        type=int,
        metavar="N",
        help="Random N images from the split (uses --seed).",
    )
    cifar.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
    )
    cifar.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Dataset root (default: {DATA_DIR})",
    )
    cifar.add_argument("--seed", type=int, default=42)
    cifar.add_argument(
        "--no-download",
        action="store_true",
        help="Require CIFAR-100 already on disk (no fetch).",
    )

    args = parser.parse_args()

    cifar_mode = args.cifar_indices is not None or args.cifar_sample is not None
    if cifar_mode and args.image is not None:
        parser.error("Pass either an image path or --cifar-indices / --cifar-sample, not both.")

    arch_fallback = args.arch or "resnet50"
    ckpt = args.checkpoint or default_checkpoint_path(arch_fallback)

    if cifar_mode:
        download = not args.no_download
        ds = get_cifar100_tensor_dataset(
            split=args.split,
            data_dir=args.data_dir,
            download=download,
        )
        if args.cifar_indices is not None:
            indices = _parse_cifar_indices(args.cifar_indices)
        else:
            rng = random.Random(args.seed)
            assert args.cifar_sample is not None
            indices = rng.sample(range(len(ds)), min(args.cifar_sample, len(ds)))
        predict_cifar_indices(
            ckpt,
            indices,
            arch=args.arch,
            split=args.split,
            data_dir=args.data_dir,
            download=download,
            topk=args.topk,
            ds=ds,
        )
        return

    if args.image is None:
        parser.error(
            "Pass an image path (e.g. python -m src.inference photo.jpg) "
            "or CIFAR flags (--cifar-indices / --cifar-sample)."
        )

    result = predict_one_image_file(ckpt, args.image, topk=args.topk, arch=args.arch)
    print(f"File       : {result['path']}")
    print(f"Top-1      : {result['top1_name']} ({result['top1_prob']:.2%})")
    print("Top-k      :")
    for row in result["topk"]:
        print(f"  {row['prob']:.2%}  [{row['id']:3d}] {row['name']}")


if __name__ == "__main__":
    main()
