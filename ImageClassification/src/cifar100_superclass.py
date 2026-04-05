"""CIFAR-100 fine label → 20 superclass grouping (torchvision fine-class order)."""
from __future__ import annotations

from src.config import CIFAR100_CLASS_NAMES

SUPERCLASSES: dict[str, tuple[str, ...]] = {
    "aquatic mammals": ("beaver", "dolphin", "otter", "seal", "whale"),
    "fish": ("aquarium_fish", "flatfish", "ray", "shark", "trout"),
    "flowers": ("orchid", "poppy", "rose", "sunflower", "tulip"),
    "food containers": ("bottle", "bowl", "can", "cup", "plate"),
    "fruit and vegetables": ("apple", "mushroom", "orange", "pear", "sweet_pepper"),
    "household electrical": ("clock", "keyboard", "lamp", "telephone", "television"),
    "household furniture": ("bed", "chair", "couch", "table", "wardrobe"),
    "insects": ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
    "large carnivores": ("bear", "leopard", "lion", "tiger", "wolf"),
    "large outdoor man-made": ("bridge", "castle", "house", "road", "skyscraper"),
    "large outdoor natural": ("cloud", "forest", "mountain", "plain", "sea"),
    "large omnivores/herbivores": ("camel", "cattle", "chimpanzee", "elephant", "kangaroo"),
    "medium mammals": ("fox", "porcupine", "possum", "raccoon", "skunk"),
    "invertebrates": ("crab", "lobster", "snail", "spider", "worm"),
    "people": ("baby", "boy", "girl", "man", "woman"),
    "reptiles": ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
    "small mammals": ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
    "trees": ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"),
    "vehicles 1": ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
    "vehicles 2": ("lawn_mower", "rocket", "streetcar", "tank", "tractor"),
}

_super_to_classes: list[list[int]] = []
for subs in SUPERCLASSES.values():
    _super_to_classes.append([CIFAR100_CLASS_NAMES.index(name) for name in subs])

SUPER_NAMES = tuple(SUPERCLASSES.keys())
SUPER_TO_CLASSES = _super_to_classes

_flat = [i for g in _super_to_classes for i in g]
assert len(_flat) == len(set(_flat)) == 100, "Superclass map must cover 100 classes once"
