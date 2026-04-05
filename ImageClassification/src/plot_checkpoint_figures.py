"""Generate assignment figures from ``checkpoint/*_best.pth`` (CIFAR-100 test set)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import CHECKPOINT_DIR, ROOT
from src.dataset import get_dataloaders
from src.evaluate import evaluate_test, plot_confusion_matrix, print_results_table
from src.inference import load_model_for_inference
from src.model import SUPPORTED_MODELS
from src.report_plots import plot_inter_superclass, plot_intra_superclass, plot_subclass_accuracy


def default_figure_dir() -> Path:
    return ROOT.parent / "assignments" / "img_cls_figures" / "img_cls"


def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    return sorted(p for p in checkpoint_dir.glob("*_best.pth") if p.is_file())


def infer_arch_from_path(ckpt_path: Path) -> str | None:
    for name in sorted(SUPPORTED_MODELS.keys(), key=len, reverse=True):
        if name in ckpt_path.name:
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot report figures from checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cm-output-dir", type=Path, default=None)
    parser.add_argument("--skip-top20-cm", action="store_true")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir.resolve()
    out_dir = (args.output_dir or default_figure_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = discover_checkpoints(ckpt_dir)
    if not ckpts:
        print(f"No checkpoints in {ckpt_dir} (*_best.pth).")
        return

    _, _, test_loader, num_classes, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results: dict = {}

    for ckpt_path in ckpts:
        arch_hint = infer_arch_from_path(ckpt_path)
        print(f"\n{'─' * 55}\n  {ckpt_path.name}  device={device}")
        model, meta = load_model_for_inference(
            ckpt_path,
            device=device,
            pretrained_backbone=False,
            arch=arch_hint,
        )
        arch = meta["arch"]
        results = evaluate_test(model, test_loader, num_classes, class_names, device)
        all_results[arch] = results

        cm = results["confusion_matrix"]
        _, super_cm = plot_inter_superclass(cm, arch, out_dir / f"1_inter_superclass_{arch}.png")
        plot_intra_superclass(cm, super_cm, arch, out_dir / f"2_intra_superclass_{arch}.png")
        plot_subclass_accuracy(cm, arch, out_dir / f"3_subclass_accuracy_{arch}.png")

        if not args.skip_top20_cm:
            plot_confusion_matrix(
                cm,
                class_names,
                arch,
                save_path=(args.cm_output_dir / f"cm_{arch}.png")
                if args.cm_output_dir
                else None,
                show=False,
            )

    print_results_table(all_results)
    print(f"\nFigures -> {out_dir}")


if __name__ == "__main__":
    main()
