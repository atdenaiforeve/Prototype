"""Prototype's first trainable neural language model.

This is intentionally small and from-scratch at the model level. It predicts
the next token from a fixed context window. Training code will be added next.
"""

from __future__ import annotations

import torch
from torch import nn


class PrototypeLanguageModel(nn.Module):
    """Tiny next-token predictor for Prototype.

    The model uses token embeddings followed by a small feed-forward network.
    It is not a transformer yet; this gives Prototype a simple neural learning
    foundation before we add attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
        context_size: int = 8,
    ) -> None:
        super().__init__()

        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if context_size < 1:
            raise ValueError("context_size must be positive")

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.network = nn.Sequential(
            nn.Linear(context_size * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for the next token.

        token_ids shape: (batch, context_size)
        output shape:    (batch, vocab_size)
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, context_size)")
        if token_ids.shape[1] != self.context_size:
            raise ValueError(
                f"expected context size {self.context_size}, "
                f"got {token_ids.shape[1]}"
            )

        embedded = self.embedding(token_ids)
        flattened = embedded.reshape(embedded.shape[0], -1)
        return self.network(flattened)

    def predict_next(self, token_ids: torch.Tensor) -> int:
        """Return the highest-scoring next-token ID."""
        self.eval()
        with torch.no_grad():
            logits = self(token_ids)
            return int(torch.argmax(logits, dim=-1)[0].item())


def create_model(vocab_size: int, context_size: int = 8) -> PrototypeLanguageModel:
    """Create a fresh randomly initialized Prototype model."""
    return PrototypeLanguageModel(
        vocab_size=vocab_size,
        context_size=context_size,
    )
