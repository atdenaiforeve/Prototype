"""Verify that Prototype actually trained on every curriculum file.

This command measures teacher-forced loss on the complete training corpus.
A lower loss means the saved weights assign higher probability to the tokens
they were trained to predict. This is an evidence check, not a claim that the
model understands the material.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from model import PrototypeLanguageModel
from train_model import load_training_folder, make_windows
from tokenizer import Tokenizer


@torch.no_grad()
def evaluate(
    model: PrototypeLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start + batch_size].to(device)
        y = targets[start:start + batch_size].to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            y.reshape(-1),
        )
        count = y.numel()
        total_loss += float(loss.item()) * count
        total_tokens += count

    return total_loss / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Prototype's loss over the complete training curriculum."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--checkpoint", type=Path, default=Path("prototype_model.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("vocabulary.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    text, files = load_training_folder(args.data_dir)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    tokenizer = Tokenizer(model_path=args.tokenizer)
    expected_vocab = int(checkpoint["vocab_size"])

    if tokenizer.vocabulary_size != expected_vocab:
        raise ValueError(
            "tokenizer vocabulary does not match checkpoint: "
            f"{tokenizer.vocabulary_size} != {expected_vocab}"
        )

    model = PrototypeLanguageModel(
        vocab_size=expected_vocab,
        embedding_size=int(checkpoint.get("embedding_size", 256)),
        hidden_size=int(checkpoint.get("hidden_size", 1024)),
        context_size=int(checkpoint["context_size"]),
        num_layers=int(checkpoint.get("num_layers", 4)),
        num_heads=int(checkpoint.get("num_heads", 8)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    token_ids = tokenizer.encode(
        text,
        add_boundaries=True,
    )
    inputs, targets = make_windows(
        token_ids,
        int(checkpoint["context_size"]),
    )

    loss = evaluate(
        model,
        inputs,
        targets,
        args.batch_size,
        device,
    )
    perplexity = math.exp(min(20.0, loss))

    trained_files = checkpoint.get("training_files", [])
    print("Prototype training verification")
    print(f"curriculum files on disk: {len(files)}")
    print(f"files recorded in checkpoint: {len(trained_files)}")
    print(f"tokens evaluated: {len(token_ids)}")
    print(f"training loss: {loss:.4f}")
    print(f"training perplexity: {perplexity:.2f}")

    if trained_files:
        missing = [
            str(path)
            for path in files
            if str(path) not in trained_files
        ]
        if missing:
            print("WARNING: checkpoint was created before these files were included:")
            for path in missing:
                print(f"  - {path}")
        else:
            print("checkpoint coverage: all current training files are recorded")

    print(
        "evidence: the reported loss is computed directly from the current "
        "training corpus using the saved model weights."
    )


if __name__ == "__main__":
    main()
