"""Inference utilities for Prototype with optional long-term memory."""

from __future__ import annotations

from pathlib import Path

import torch

from learning import LearningLoop
from memory_context import build_prompt, format_memory_context
from model import PrototypeLanguageModel
from monitor import ReasoningMonitor, entropy
from reasoning import ReasoningWorkspace
from reasoning_engine import ReasoningResult, reason
from tokenizer import Tokenizer


def load_checkpoint(
    checkpoint_path: Path | str,
    tokenizer_path: Path | str = "vocabulary.json",
) -> tuple[PrototypeLanguageModel, Tokenizer, torch.device]:
    """Load a trained Prototype checkpoint and its matching tokenizer."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    vocab_size = int(checkpoint["vocab_size"])
    context_size = int(checkpoint["context_size"])

    if tokenizer.vocabulary_size != vocab_size:
        raise ValueError(
            "tokenizer vocabulary does not match checkpoint: "
            f"{tokenizer.vocabulary_size} != {vocab_size}"
        )

    model = PrototypeLanguageModel(
        vocab_size=vocab_size,
        embedding_size=int(checkpoint.get("embedding_size", 256)),
        hidden_size=int(checkpoint.get("hidden_size", 1024)),
        context_size=context_size,
        num_layers=int(checkpoint.get("num_layers", 4)),
        num_heads=int(checkpoint.get("num_heads", 8)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(
    model: PrototypeLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 20,
    monitor: ReasoningMonitor | None = None,
) -> str:
    """Generate a short continuation from a prompt."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    ids = tokenizer.encode(prompt, add_boundaries=False)
    if not ids:
        ids = [2]

    generated = list(ids)

    for _ in range(max_new_tokens):
        context = generated[-model.context_size:]
        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        logits = model(input_ids)[0, -1, :] / temperature

        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        probabilities = torch.softmax(values, dim=-1)
        next_index = torch.multinomial(probabilities, num_samples=1)
        next_token = int(indices[next_index].item())
        generated.append(next_token)

        if monitor is not None:
            candidate_limit = min(monitor.top_k, k)
            candidates = [
                {
                    "token_id": int(indices[i].item()),
                    "text": tokenizer.decode([int(indices[i].item())]),
                    "probability": float(probabilities[i].item()),
                }
                for i in range(candidate_limit)
            ]
            monitor.record(
                step=len(generated) - len(ids) - 1,
                context_length=len(context),
                token_id=next_token,
                probability=float(probabilities[next_index].item()),
                entropy=entropy(probabilities.tolist()),
                top_candidates=candidates,
                stopped=next_token == 3,
            )

        if next_token == 3:
            break

    return tokenizer.decode(generated)


def reason_generate(
    model: PrototypeLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    candidates: int = 3,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 20,
    memory: LearningLoop | None = None,
    memory_limit: int = 3,
    memory_max_chars: int = 400,
    learn: bool = False,
) -> ReasoningResult:
    """Reason with relevant long-term memories and optionally record the result."""
    if memory_limit < 0:
        raise ValueError("memory_limit must not be negative")
    learning = memory or LearningLoop()
    memories = learning.retrieve(prompt, limit=memory_limit)
    memory_context = format_memory_context(
        memories,
        max_chars=memory_max_chars,
        max_memories=memory_limit,
    )
    candidate_prompt = build_prompt(prompt, memory_context)
    workspace = ReasoningWorkspace(max_hypotheses=candidates)
    result = reason(
        candidate_prompt,
        lambda candidate_prompt, **kwargs: generate(
            model,
            tokenizer,
            candidate_prompt,
            **kwargs,
        ),
        candidates=candidates,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        workspace=workspace,
    )
    result.workspace["memory"] = {
        "retrieved": len(memories),
        "memories": [
            {
                "id": memory_item.get("id"),
                "type": memory_item.get("memory_type"),
                "relevance_score": memory_item.get("relevance_score", 0.0),
            }
            for memory_item in memories
        ],
    }
    if learn:
        learning.record_result(prompt, result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with Prototype.")
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, default=Path("prototype_model.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("vocabulary.json"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--monitor", type=Path, default=None)
    parser.add_argument("--reason", action="store_true", help="Explore bounded candidates before choosing an answer.")
    parser.add_argument("--reasoning-candidates", type=int, default=3)
    parser.add_argument("--memory", action="store_true", help="Use relevant long-term memories during reasoning.")
    parser.add_argument("--learn", action="store_true", help="Record the selected result as an experience.")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.tokenizer)

    if args.reason:
        result = reason_generate(
            model,
            tokenizer,
            args.prompt,
            candidates=args.reasoning_candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            memory=LearningLoop() if args.memory or args.learn else None,
            learn=args.learn,
        )
        print(result.output)
        print("\nReasoning decision:")
        print(result.workspace)
    else:
        monitor = ReasoningMonitor(tokenizer.decode, path=args.monitor) if args.monitor else None
        print(
            generate(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                monitor=monitor,
            )
        )
