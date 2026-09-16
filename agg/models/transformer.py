"""A deliberately small baseline; no AGG imports or hooks are required."""

from typing import Any

import torch
from torch import Tensor, nn


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        classes: int,
        *,
        width: int = 32,
        layers: int = 2,
        heads: int = 4,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        if min(vocab_size, classes, width, layers, heads, max_length) < 1 or width % heads:
            raise ValueError("positive dimensions and width divisible by heads required")
        self.width = width
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    heads,
                    dim_feedforward=width * 2,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, classes)

    def forward(self, tokens: Tensor, *, return_hidden: bool = False) -> Any:
        if tokens.ndim != 2 or tokens.shape[1] > self.position.num_embeddings:
            raise ValueError("tokens must be [batch,length] within configured max_length")
        length = tokens.shape[1]
        h = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        hidden = [h]
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        for block in self.blocks:
            h = block(h, src_mask=mask)
            hidden.append(h)
        logits = self.readout(self.norm(h[:, -1]))
        return (logits, hidden) if return_hidden else logits
