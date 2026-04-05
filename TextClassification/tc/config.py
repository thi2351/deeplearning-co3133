"""Paths and class-name order aligned with ``DL252_1_text_pth.ipynb``."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DEMO_SAMPLES_PATH = DATA_DIR / "demo_samples.json"

MAX_LEN = 256

# ``classes = np.unique(df['category'])`` → lexicographic order on the seven target categories.
DEFAULT_CLASS_NAMES = [
    "ENTERTAINMENT",
    "HEALTHY LIVING",
    "PARENTING",
    "POLITICS",
    "STYLE & BEAUTY",
    "TRAVEL",
    "WELLNESS",
]
