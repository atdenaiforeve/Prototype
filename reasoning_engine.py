"""Bounded reasoning controller for Prototype.

Prototype's reasoning is explicit structured state: it explores a small number
of candidate continuations, records evidence from inference telemetry, scores
them, and commits to one. This is not hidden human-like chain-of-thought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from monitor import ReasoningMonitor
from reasoning import Hypothesis, ReasoningWorkspace


@dataclass
class ReasoningResult:
    output: str
    chosen: Hypothesis | None
    workspace: dict[str, object]


def _score_monitor(monitor: ReasoningMonitor) -> float:
    """Turn token confidence into a bounded candidate score."""
    if not monitor.events:
        return 0.0
    log_probability = sum(
        math.log(max(event.probability, 1e-12)) for event in monitor.events
    )
    return max(0.0, min(1.0, math.exp(log_probability / len(monitor.events))))


def reason(
    prompt: str,
    generate_fn: Callable[..., str],
    *,
    candidates: int = 3,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 20,
    workspace: ReasoningWorkspace | None = None,
) -> ReasoningResult:
    """Explore bounded candidate continuations and choose one by model confidence."""
    if candidates < 1:
        raise ValueError("candidates must be positive")
    if candidates > 8:
        raise ValueError("candidates must be at most 8")

    workspace = workspace or ReasoningWorkspace(max_hypotheses=candidates)
    workspace.clear()

    for index in range(candidates):
        monitor = ReasoningMonitor(lambda ids: str(ids[0]), top_k=top_k)
        output = generate_fn(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            monitor=monitor,
        )
        score = _score_monitor(monitor)
        hypothesis = workspace.add_hypothesis(output, score)
        summary = monitor.summary()
        workspace.add_evidence(
            hypothesis,
            f"candidate {index + 1}: average token confidence "
            f"{summary['average_probability']:.4f}; "
            f"{summary['steps']} generated steps",
        )

    chosen = workspace.choose()
    return ReasoningResult(
        output=chosen.text if chosen is not None else "",
        chosen=chosen,
        workspace=workspace.snapshot(),
    )
