"""Train Prototype's first neural language model.

This keeps training small and explicit:
1. build next-token examples from text,
2. run the model,
3. calculate prediction loss,
4. backpropagate,
5. update the weights,
6. save a checkpoint.

The training data is intentionally supplied by the caller; this script does
not download or silently import an external pretrained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import PrototypeLanguageModel
from tokenizer import Tokenizer


def make_examples(token_ids: list[int], context_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create (context, next_token) training examples from token IDs."""
    if len(token_ids) <= context_size:
        raise ValueError("training text is too short for the chosen context size")

    contexts = []
    targets = []

    for index in range(context_size, len(token_ids)):
        contexts.append(token_ids[index - context_size:index])
        targets.append(token_ids[index])

    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train(
    model: PrototypeLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 0.001,
) -> list[float]:
    """Train the model and return the loss after each epoch."""
    if epochs < 1:
        raise ValueError("epochs must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history: list[float] = []

    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = loss_function(logits, targets)

        loss.backward()
        optimizer.step()

        value = float(loss.item())
        history.append(value)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d} | loss {value:.4f}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prototype's neural language model.")
    parser.add_argument("text_file", type=Path, help="UTF-8 text file containing training text")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=Path("prototype_model.pt"))
    args = parser.parse_args()

    tokenizer = Tokenizer()
    text = args.text_file.read_text(encoding="utf-8")
    tokenizer.learn(text)
    tokenizer.save()

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    inputs, targets = make_examples(token_ids, args.context_size)

    model = PrototypeLanguageModel(
        vocab_size=tokenizer.vocabulary_size,
        context_size=args.context_size,
    )

    history = train(
        model,
        inputs,
        targets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocabulary_size,
            "context_size": args.context_size,
            "embedding_size": model.embedding_size,
            "hidden_size": model.hidden_size,
            "final_loss": history[-1],
        },
        args.checkpoint,
    )

    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
