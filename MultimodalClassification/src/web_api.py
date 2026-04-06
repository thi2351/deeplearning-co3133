from __future__ import annotations

"""Example FastAPI wrapper for the modular service.

This file is intentionally small. It shows how the project can be exposed as a
web service once a checkpoint and class list are available.
"""

from io import BytesIO
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from .service import MultimodalFoodClassifierService


def create_app(service: MultimodalFoodClassifierService) -> FastAPI:
    app = FastAPI(title="UPMC Food-101 Multimodal Classifier")

    @app.get("/health")
    def health():
        return {"status": "ok", "num_classes": len(service.classes)}

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        text: str = Form(""),
        top_k: int = Form(5),
    ):
        payload = await image.read()
        pil_image = Image.open(BytesIO(payload)).convert("RGB")
        preds = service.predict(pil_image, text=text, top_k=top_k)
        return {
            "predictions": [
                {"class_name": item.class_name, "score": item.score}
                for item in preds
            ]
        }

    return app
