"""LSTM / CNN / RCNN from ``DL252_1_text_pth.ipynb`` — same tokenization as training (bert-base-uncased)."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text: torch.Tensor, attention_mask=None):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden)


class CNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_filters: int,
        filter_sizes: List[int],
        output_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n_filters, (fs, embed_dim)) for fs in filter_sizes]
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

    def forward(self, text: torch.Tensor, attention_mask=None):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)


class RCNNModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 + embed_dim, output_dim)

    def forward(self, text: torch.Tensor, attention_mask=None):
        embedded = self.embedding(text)
        out, _ = self.lstm(embedded)
        out = torch.cat((out, embedded), dim=2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.shape[2]).squeeze(2)
        return self.fc(out)


CUSTOM_TOKENIZER_ID = "bert-base-uncased"


def is_custom_checkpoint_stem(stem: str) -> bool:
    return stem.upper() in ("LSTM", "CNN", "RCNN")


def _cnn_params_from_state_dict(sd: Dict[str, torch.Tensor]) -> tuple[int, List[int]]:
    sizes: List[int] = []
    i = 0
    while f"convs.{i}.weight" in sd:
        sizes.append(int(sd[f"convs.{i}.weight"].shape[2]))
        i += 1
    if not sizes:
        raise ValueError("CNN checkpoint missing convs.*.weight")
    n_filters = int(sd["convs.0.weight"].shape[0])
    return n_filters, sizes


def build_custom_model_from_state_dict(stem: str, sd: Dict[str, torch.Tensor]) -> nn.Module:
    if "embedding.weight" not in sd or "fc.weight" not in sd:
        raise ValueError("Not a valid custom text checkpoint (missing embedding/fc).")
    vocab_size, embed_dim = int(sd["embedding.weight"].shape[0]), int(sd["embedding.weight"].shape[1])
    num_classes = int(sd["fc.weight"].shape[0])
    key = stem.upper()

    if key == "LSTM":
        fc_in = int(sd["fc.weight"].shape[1])
        hidden_dim = fc_in // 2
        m = LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes)
    elif key == "CNN":
        n_filters, filter_sizes = _cnn_params_from_state_dict(sd)
        m = CNNModel(vocab_size, embed_dim, n_filters, filter_sizes, num_classes)
    elif key == "RCNN":
        fc_in = int(sd["fc.weight"].shape[1])
        hidden_dim = (fc_in - embed_dim) // 2
        if hidden_dim < 1:
            raise ValueError("Cannot infer RCNN hidden_dim from checkpoint.")
        m = RCNNModel(vocab_size, embed_dim, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown custom model stem: {stem!r}")

    m.load_state_dict(sd, strict=True)
    return m
