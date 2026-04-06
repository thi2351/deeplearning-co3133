from __future__ import annotations

"""Classification heads used on top of frozen CLIP features."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class MultimodalClassificationHead(nn.Module):
    """MLP head over concatenated CLIP image/text embeddings.

    Design choice:
    - keep CLIP frozen for stability and lower GPU cost
    - learn only a compact fusion/classification head
    - module is simple enough to deploy behind a web endpoint
    """

    def __init__(
        self,
        num_classes: int,
        clip_embed_dim: int = 512,
        use_text: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.30,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_text = use_text
        input_dim = clip_embed_dim * 2 if use_text else clip_embed_dim

        norm1 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
        norm2 = nn.BatchNorm1d(hidden_dim // 2) if use_batchnorm else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm1,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            norm2,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_text:
            if text_features is None:
                raise ValueError("use_text=True but text_features was not provided")
            x = torch.cat([image_features, text_features], dim=-1)
        else:
            x = image_features
        return self.classifier(x)


@dataclass
class PredictionRecord:
    class_name: str
    score: float


def build_head(num_classes: int, **kwargs) -> MultimodalClassificationHead:
    return MultimodalClassificationHead(num_classes=num_classes, **kwargs)
