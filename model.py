"""Small causal Transformer language model for Prototype."""

from __future__ import annotations

import math

import torch
from torch import nn


class PrototypeLanguageModel(nn.Module):
    """Decoder-style Transformer trained for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 256,
        hidden_size: int = 1024,
        context_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if context_size < 1:
            raise ValueError("context_size must be positive")
        if embedding_size % num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_size, embedding_size)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False)

        # Weight tying reduces parameters and keeps input/output token spaces
        # aligned, which is a useful property for a small language model.
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits shaped (batch, sequence, vocabulary)."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.shape[1] > self.context_size:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds context size "
                f"{self.context_size}"
            )

        positions = torch.arange(
            token_ids.shape[1],
            device=token_ids.device,
        ).unsqueeze(0)

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)
        x = self.blocks(x, mask=self._causal_mask(token_ids.shape[1], token_ids.device))
        x = self.final_norm(x)
        return self.lm_head(x) / math.sqrt(self.embedding_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )

    @torch.no_grad()
    def predict_next(self, token_ids: torch.Tensor) -> int:
        self.eval()
        logits = self(token_ids[:, -self.context_size:])
        return int(torch.argmax(logits[:, -1, :], dim=-1)[0].item())


def create_model(vocab_size: int, context_size: int = 128) -> PrototypeLanguageModel:
    return PrototypeLanguageModel(vocab_size=vocab_size, context_size=context_size)
