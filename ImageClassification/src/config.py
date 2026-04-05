"""Project paths, training hyperparameters, and shared constants for deployment."""
from pathlib import Path

# ImageClassification/ (directory that contains `src/` and `checkpoint/`)
ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoint"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

IMG_SIZE = 96

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# --- Training / data (chỉnh tập trung ở đây; train CLI có thể ghi đè vài giá trị) ---
ARCH = "resnet50"
PRETRAINED = True
BATCH_SIZE = 64
VAL_SPLIT = 0.1
NUM_WORKERS = 0
PIN_MEMORY = True
SPLIT_SEED = 42
TRAIN_RANDOM_CROP_PADDING = 12
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
# Số epoch cố định theo kiến trúc (khớp báo cáo: CNN 15, ViT 10). Không early stopping.
EPOCHS_BY_ARCH = {
    "resnet50": 15,
    "efficientnet_b3": 15,
    "swin_tiny_patch4_window7_224": 10,
    "vit_base_patch16_224": 10,
}


def epochs_for_arch(arch: str) -> int:
    if arch not in EPOCHS_BY_ARCH:
        raise ValueError(f"Unknown arch {arch!r}; add to EPOCHS_BY_ARCH in config.py")
    return EPOCHS_BY_ARCH[arch]

# torchvision.datasets.CIFAR100 fine labels, index 0..99 — no download needed for inference
CIFAR100_CLASS_NAMES = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)
