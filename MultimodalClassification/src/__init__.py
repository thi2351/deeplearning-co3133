"""Modular multimodal Food-101 project package."""

from .baselines import build_results_table, evaluate_multimodal_zero_shot, run_few_shot_logreg
from .clip_utils import CLIPBundle, get_device, load_frozen_clip
from .data_loader import DatasetMetadata, get_food101_dataloaders, load_food101_metadata
from .inference import build_service, load_classes_from_path, load_demo_samples
from .metrics import classification_metrics, full_classification_report
from .models import MultimodalClassificationHead
from .reporting import predict_few_shot, predict_zero_shot
from .service import MultimodalFoodClassifierService
from .trainer import CHECKPOINT_PATH, evaluate_head, load_head, train_model

__all__ = [
    "CLIPBundle",
    "DatasetMetadata",
    "MultimodalClassificationHead",
    "MultimodalFoodClassifierService",
    "CHECKPOINT_PATH",
    "build_service",
    "build_results_table",
    "classification_metrics",
    "evaluate_head",
    "evaluate_multimodal_zero_shot",
    "full_classification_report",
    "get_device",
    "get_food101_dataloaders",
    "load_classes_from_path",
    "load_demo_samples",
    "load_food101_metadata",
    "load_frozen_clip",
    "load_head",
    "predict_few_shot",
    "predict_zero_shot",
    "run_few_shot_logreg",
    "train_model",
]
