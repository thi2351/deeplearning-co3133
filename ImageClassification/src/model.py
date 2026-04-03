import timm

from src.config import IMG_SIZE

SUPPORTED_MODELS = {
    "resnet50": "CNN",
    "efficientnet_b3": "CNN",
    "swin_tiny_patch4_window7_224": "ViT",
    "vit_base_patch16_224": "ViT",
}


def build_model(arch: str, num_classes: int = 100, pretrained: bool = True):
    assert (
        arch in SUPPORTED_MODELS
    ), f"Unknown arch '{arch}'. Choose from: {list(SUPPORTED_MODELS.keys())}"
    is_vit = SUPPORTED_MODELS[arch] == "ViT"
    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
        **({"img_size": IMG_SIZE} if is_vit else {}),
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Built {arch} ({SUPPORTED_MODELS[arch]}) — {params:.1f}M params")
    return model


def get_optimizer_groups(model, head_lr=1e-4, backbone_lr=1e-5):
    head_keys = ["head", "classifier", "fc"]
    head_params = [p for n, p in model.named_parameters() if any(k in n for k in head_keys)]
    backbone_params = [
        p for n, p in model.named_parameters() if not any(k in n for k in head_keys)
    ]
    return [
        {"params": head_params, "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]


def freeze_backbone(model):
    head_keys = ["head", "classifier", "fc"]
    for name, param in model.named_parameters():
        if not any(k in name for k in head_keys):
            param.requires_grad = False
    print("Backbone frozen — only head is trainable")
    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")
    return model
