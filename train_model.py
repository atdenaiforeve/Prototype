"""Train Prototype's causal Transformer on the complete training curriculum."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn

from model import PrototypeLanguageModel
from tokenizer import Tokenizer


# Use the CPU capacity actually available to the Codespace. Keep inter-op
# parallelism bounded so the two CPU cores are not oversubscribed.
_cpu_count = max(1, os.cpu_count() or 1)
torch.set_num_threads(_cpu_count)
torch.set_num_interop_threads(max(1, min(2, _cpu_count)))


def load_training_folder(data_dir: Path) -> tuple[str, list[Path]]:
    """Load every UTF-8 .txt file below data_dir, in stable path order."""
    if not data_dir.exists():
        raise FileNotFoundError(f"training directory does not exist: {data_dir}")

    files = sorted(path for path in data_dir.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"no .txt training files found in {data_dir}")

    parts = [path.read_text(encoding="utf-8").strip() for path in files]
    text = "\n\n".join(part for part in parts if part)
    if not text:
        raise ValueError(f"training files in {data_dir} are empty")
    return text, files


def make_windows(
    token_ids: list[int],
    context_size: int,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sequence inputs and next-token targets efficiently."""
    if context_size < 1:
        raise ValueError("context_size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if len(token_ids) <= context_size:
        raise ValueError("training text is too short for the chosen context size")

    tokens = torch.tensor(token_ids, dtype=torch.long)
    inputs = tokens[:-1].unfold(0, context_size, stride)
    targets = tokens[1:].unfold(0, context_size, stride)
    return inputs.contiguous(), targets.contiguous()


def split_token_ids(
    token_ids: list[int],
    context_size: int,
    validation_fraction: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Split raw tokens before windowing so validation text is truly unseen."""
    if context_size < 1:
        raise ValueError("context_size must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    minimum_segment = context_size + 1
    if len(token_ids) < minimum_segment * 2:
        raise ValueError(
            "training text is too short to create independent training and validation sets"
        )

    split = int(len(token_ids) * (1.0 - validation_fraction))
    split = max(minimum_segment, split)
    split = min(split, len(token_ids) - minimum_segment)
    return token_ids[:split], token_ids[split:]


def run_epoch(
    model: PrototypeLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(inputs)) if training else torch.arange(len(inputs))

    if inputs.shape != targets.shape:
        raise ValueError("inputs and targets must have the same shape")
    if inputs.numel() == 0:
        raise ValueError("training epoch received no samples")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]
        x = inputs.index_select(0, batch_indices).to(device)
        y = targets.index_select(0, batch_indices).to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            y.reshape(-1),
        )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        token_count = y.numel()
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Prototype from every .txt file in data/training/."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--vocab-limit", type=int, default=8192)
    parser.add_argument("--checkpoint", type=Path, default=Path("prototype_model.pt"))
    args = parser.parse_args()

    if args.context_size < 8:
        raise ValueError("context-size should be at least 8")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    text, files = load_training_folder(args.data_dir)
    print(f"loaded {len(files)} training file(s):")
    for path in files:
        print(f"  - {path}")

    tokenizer = Tokenizer(vocab_limit=args.vocab_limit)
    tokenizer.learn(text)
    tokenizer.save()

    token_ids = tokenizer.encode(text, add_boundaries=True)
    train_tokens, val_tokens = split_token_ids(
        token_ids,
        args.context_size,
        args.validation_fraction,
    )
    train_x, train_y = make_windows(train_tokens, args.context_size)
    val_x, val_y = make_windows(val_tokens, args.context_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"vocabulary: {tokenizer.vocabulary_size}")
    print(f"tokens: {len(token_ids)} | device: {device}")
    print(f"cpu threads: {torch.get_num_threads()} | inter-op threads: {torch.get_num_interop_threads()}")

    model = PrototypeLanguageModel(
        vocab_size=tokenizer.vocabulary_size,
        context_size=args.context_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(
            model, train_x, train_y, optimizer, args.batch_size, device
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, val_x, val_y, None, args.batch_size, device
            )

        history.append({"train_loss": train_loss, "validation_loss": val_loss})
        elapsed = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
            f"validation loss {val_loss:.4f} | time {elapsed:.2f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": 2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": tokenizer.vocabulary_size,
                    "context_size": args.context_size,
                    "embedding_size": model.embedding_size,
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                    "num_heads": model.num_heads,
                    "dropout": model.dropout,
                    "vocab_limit": args.vocab_limit,
                    "best_validation_loss": best_val,
                    "training_files": [str(path) for path in files],
                    "history": history,
                },
                args.checkpoint,
            )

    print(f"best validation loss: {best_val:.4f}")
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
