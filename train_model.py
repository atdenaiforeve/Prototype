"""Train Prototype's neural language model from the whole training-data folder.

By default, every .txt file under data/training/ is loaded recursively and
combined into one training corpus. Add new lessons by dropping UTF-8 .txt files
into that folder; no code changes are required.

This script does not download or silently import an external pretrained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import PrototypeLanguageModel
from tokenizer import Tokenizer


def load_training_folder(data_dir: Path) -> tuple[str, list[Path]]:
    """Load every UTF-8 .txt file below data_dir, in stable path order."""
    if not data_dir.exists():
        raise FileNotFoundError(f"training directory does not exist: {data_dir}")

    files = sorted(path for path in data_dir.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"no .txt training files found in {data_dir}")

    parts: list[str] = []
    for path in files:
        parts.append(path.read_text(encoding="utf-8").strip())

    text = "\n\n".join(part for part in parts if part)
    if not text:
        raise ValueError(f"training files in {data_dir} are empty")

    return text, files


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
    parser = argparse.ArgumentParser(
        description="Train Prototype from every .txt file in data/training/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/training"),
        help="directory containing training .txt files (default: data/training)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prototype_model.pt"),
    )
    args = parser.parse_args()

    text, files = load_training_folder(args.data_dir)

    print(f"loaded {len(files)} training file(s):")
    for path in files:
        print(f"  - {path}")

    tokenizer = Tokenizer()
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
            "training_files": [str(path) for path in files],
        },
        args.checkpoint,
    )

    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
