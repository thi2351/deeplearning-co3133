# MultimodalClassification

This directory contains the multimodal pipeline (image + text fusion).

## Structure

- `src/`: model, CLIP helpers, service, and inference utilities.
- `weights/`: model checkpoints (`*.pth`) and class list.
- `requirements.txt`: Python dependencies for this module.

## API integration

`demo-api/app.py` already registers multimodal routes:

- `GET /api/mm/health`
- `POST /api/mm/predict`

### Required runtime config

Set one of the following for class names:

1. `MM_CLASSES_PATH=/abs/path/to/classes.json` where JSON is `list[str]`
2. `MM_CLASSES_JSON='["class_a", "class_b"]'`

Optional:

- `MM_CHECKPOINT=/abs/path/to/food101_head.pth`
- `MM_HF_MODEL=openai/clip-vit-base-patch32`
