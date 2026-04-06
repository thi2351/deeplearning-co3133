"""Configuration for multimodal classification module."""

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = MODULE_ROOT / "data"
DATA_TEST_DIR = DATA_DIR / "test"
WEIGHTS_DIR = MODULE_ROOT / "weights"
DEMO_SAMPLES_PATH = DATA_DIR / "demo_samples.json"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "food101_head.pth"
DEFAULT_CLASSES_JSON = WEIGHTS_DIR / "classes.json"
