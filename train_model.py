"""Train Prototype's causal Transformer on the complete training curriculum.

The default behavior trains on 100% of every .txt file under data/training/.
Validation is optional and must be explicitly enabled. A checkpoint is saved
after every epoch. If a checkpoint already exists, training resumes from the
latest completed epoch by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from model import PrototypeLanguageModel
from tokenizer import Tokenizer


_cpu_count = max(1, os.cpu_count() or 1)
torch.set_num_threads(_cpu_count)
torch.set_num_interop_threads(max(1, min(2, _cpu_count)))


def load_training_folder(data_dir: Path) -> tuple[str, list[Path]]:
    """Load every UTF-8 .txt file below data_dir in stable path order."""
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



def token_cache_fingerprint(text: str, tokenizer: Tokenizer) -> str:
    """Return a stable key for the exact text + tokenizer state."""
    payload = {
        "text": text,
        "token_to_id": tokenizer.token_to_id,
        "merges": tokenizer.merges,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_build_token_ids(
    text: str,
    tokenizer: Tokenizer,
    cache_path: Path,
) -> list[int]:
    """Reuse token IDs when the training text and tokenizer are unchanged."""
    fingerprint = token_cache_fingerprint(text, tokenizer)

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint:
                token_ids = data.get("token_ids")
                if isinstance(token_ids, list) and token_ids:
                    print(f"token cache: loaded {len(token_ids)} tokens from {cache_path}")
                    return [int(token_id) for token_id in token_ids]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    token_ids = load_or_build_token_ids(text, tokenizer, args.token_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"version": 1, "fingerprint": fingerprint, "token_ids": token_ids}),
        encoding="utf-8",
    )
    print(f"token cache: built {len(token_ids)} tokens at {cache_path}")
    return token_ids


def make_windows(
    token_ids: list[int],
    context_size: int,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create next-token training windows."""
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
    """Split raw tokens before windowing for optional unseen validation."""
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

    order = (
        torch.randperm(len(inputs))
        if training
        else torch.arange(len(inputs))
    )

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


def save_checkpoint(
    path: Path,
    model: PrototypeLanguageModel,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    files: list[Path],
    args: argparse.Namespace,
    history: list[dict[str, float]],
    *,
    epoch: int,
    validation_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 4,
            "epoch": epoch,
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
            "training_files": [str(path) for path in files],
            "training_file_count": len(files),
            "training_fraction": (
                1.0 if args.validation_fraction == 0.0
                else 1.0 - args.validation_fraction
            ),
            "validation_loss": validation_loss,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: PrototypeLanguageModel,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
) -> tuple[int, list[dict[str, float]]]:
    """Restore the latest checkpoint and return its completed epoch/history."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    expected = {
        "vocab_size": tokenizer.vocabulary_size,
        "context_size": args.context_size,
        "embedding_size": model.embedding_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "vocab_limit": args.vocab_limit,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"checkpoint is incompatible for {key}: "
                f"saved={checkpoint.get(key)!r}, current={value!r}. "
                "Use a new checkpoint path or --no-resume for a new model architecture."
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        history = []

    # Older format-3 checkpoints did not store the epoch number.\n    # Their history contains one entry per completed epoch, so recover it safely.\n    saved_epoch = checkpoint.get("epoch")\n    if not isinstance(saved_epoch, int) or saved_epoch < 0:\n        saved_epoch = len(history)\n        if checkpoint.get("format_version") == 3:\n            print(\n                f"legacy checkpoint detected: recovering completed epoch {saved_epoch} " \n                "from training history"\n            )\n\n    return saved_epoch, history\n

def build_data(
    token_ids: list[int],
    context_size: int,
    validation_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Build training data and optional validation data."""
    if validation_fraction == 0.0:
        train_x, train_y = make_windows(token_ids, context_size)
        return train_x, train_y, None, None

    train_tokens, val_tokens = split_token_ids(
        token_ids,
        context_size,
        validation_fraction,
    )
    train_x, train_y = make_windows(train_tokens, context_size)
    val_x, val_y = make_windows(val_tokens, context_size)
    return train_x, train_y, val_x, val_y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Prototype on every .txt file in data/training/."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="complete passes over the training curriculum",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="samples per optimizer step; 32 improves CPU throughput when RAM allows",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="0 trains on 100%% of the text; set >0 only for held-out validation",
    )
    parser.add_argument("--vocab-limit", type=int, default=8192)
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=Path("data/cache/training_tokens.json"),
        help="cache encoded training tokens to avoid repeating tokenizer work",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("prototype_model.pt"),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing checkpoint and start training from scratch",
    )
    args = parser.parse_args()

    if args.context_size < 8:
        raise ValueError("context-size should be at least 8")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1")

    text, files = load_training_folder(args.data_dir)

    print(f"loaded {len(files)} training file(s):")
    for path in files:
        print(f"  - {path}")

    tokenizer = Tokenizer(
        model_path="vocabulary.json",
        vocab_limit=args.vocab_limit,
    )

    # Reuse the tokenizer and encoded tokens when the training text has not changed.
    token_ids: list[int] | None = None
    if args.token_cache.exists():
        try:
            data = json.loads(args.token_cache.read_text(encoding="utf-8"))
            if data.get("fingerprint") == token_cache_fingerprint(text, tokenizer):
                cached = data.get("token_ids")
                if isinstance(cached, list) and cached:
                    token_ids = [int(token_id) for token_id in cached]
                    print(f"token cache: loaded {len(token_ids)} tokens from {args.token_cache}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            token_ids = None

    if token_ids is None:
        tokenizer.learn(text)
        tokenizer.save()
        token_ids = load_or_build_token_ids(text, tokenizer, args.token_cache)
    train_x, train_y, val_x, val_y = build_data(
        token_ids,
        args.context_size,
        args.validation_fraction,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"vocabulary: {tokenizer.vocabulary_size}")
    print(f"tokens: {len(token_ids)} | device: {device}")
    print(
        f"training tokens: {train_y.numel()} | "
        f"validation: {0 if val_y is None else val_y.numel()}"
    )
    print(
        f"cpu threads: {torch.get_num_threads()} | "
        f"inter-op threads: {torch.get_num_interop_threads()}"
    )

    model = PrototypeLanguageModel(
        vocab_size=tokenizer.vocabulary_size,
        context_size=args.context_size,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    history: list[dict[str, float]] = []
    start_epoch = 0

    if args.checkpoint.exists() and not args.no_resume:
        print(f"loading checkpoint: {args.checkpoint}")
        start_epoch, history = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            tokenizer,
            args,
        )
        print(f"resuming after epoch {start_epoch}")
        if start_epoch >= args.epochs:
            print(
                f"checkpoint already reached epoch {start_epoch}; "
                f"use --epochs {start_epoch + 1} or higher to continue"
            )
            return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = run_epoch(
            model,
            train_x,
            train_y,
            optimizer,
            args.batch_size,
            device,
        )

        validation_loss = None

        if val_x is not None and val_y is not None:
            with torch.no_grad():
                validation_loss = run_epoch(
                    model,
                    val_x,
                    val_y,
                    None,
                    args.batch_size,
                    device,
                )

        row: dict[str, float] = {
            "train_loss": train_loss,
        }

        if validation_loss is not None:
            row["validation_loss"] = validation_loss

        history.append(row)

        elapsed = time.perf_counter() - epoch_start

        if validation_loss is None:
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
                f"time {elapsed:.2f}s"
            )
        else:
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
                f"validation loss {validation_loss:.4f} | "
                f"time {elapsed:.2f}s"
            )

        # Save the latest completed epoch so an interruption can be resumed.
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            tokenizer,
            files,
            args,
            history,
            epoch=epoch,
            validation_loss=validation_loss,
        )

    final_loss = history[-1]["train_loss"]
    print()
    print("TRAINING COMPLETE")
    print(f"files trained: {len(files)}")
    print("training coverage: 100%")
    print(f"final training loss: {final_loss:.4f}")
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
