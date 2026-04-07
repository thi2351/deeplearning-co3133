from __future__ import annotations

"""Modular multimodal Food-101 project package.

Keep package exports lazy so lightweight runtime consumers such as the Flask API
can import `MultimodalClassification.src.<module>` without eagerly pulling in
optional notebook/baseline dependencies like `scikit-learn`.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CLIPBundle": (".clip_utils", "CLIPBundle"),
    "DatasetMetadata": (".data_loader", "DatasetMetadata"),
    "MultimodalClassificationHead": (".models", "MultimodalClassificationHead"),
    "MultimodalFoodClassifierService": (".service", "MultimodalFoodClassifierService"),
    "CHECKPOINT_PATH": (".trainer", "CHECKPOINT_PATH"),
    "build_service": (".inference", "build_service"),
    "build_results_table": (".baselines", "build_results_table"),
    "classification_metrics": (".metrics", "classification_metrics"),
    "evaluate_head": (".trainer", "evaluate_head"),
    "evaluate_multimodal_zero_shot": (".baselines", "evaluate_multimodal_zero_shot"),
    "full_classification_report": (".metrics", "full_classification_report"),
    "get_device": (".clip_utils", "get_device"),
    "get_food101_dataloaders": (".data_loader", "get_food101_dataloaders"),
    "load_classes_from_path": (".inference", "load_classes_from_path"),
    "load_demo_samples": (".inference", "load_demo_samples"),
    "load_food101_metadata": (".data_loader", "load_food101_metadata"),
    "load_frozen_clip": (".clip_utils", "load_frozen_clip"),
    "load_head": (".trainer", "load_head"),
    "predict_few_shot": (".reporting", "predict_few_shot"),
    "predict_zero_shot": (".reporting", "predict_zero_shot"),
    "run_few_shot_logreg": (".baselines", "run_few_shot_logreg"),
    "train_model": (".trainer", "train_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            raise ModuleNotFoundError(
                f"Optional dependency 'scikit-learn' is required for {name!r}. "
                "Install it to use baseline and metrics helpers."
            ) from exc
        raise

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
