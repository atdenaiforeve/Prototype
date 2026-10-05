"""Observable reasoning monitor for Prototype.

This module records model-visible inference signals, not private human-like
thought. Prototype can use these events as its own debugging/reasoning trace.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ReasoningEvent:
    step: int
    context_length: int
    token_id: int
    token_text: str
    probability: float
    entropy: float
    top_candidates: list[dict[str, object]]
    elapsed_ms: float
    stopped: bool = False


class ReasoningMonitor:
    """Collect and optionally persist structured generation telemetry."""

    def __init__(
        self,
        token_decoder: Callable[[list[int]], str],
        path: Path | str | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.token_decoder = token_decoder
        self.path = Path(path) if path is not None else None
        self.top_k = top_k
        self.events: list[ReasoningEvent] = []
        self._started_at = time.perf_counter()

    def record(
        self,
        *,
        step: int,
        context_length: int,
        token_id: int,
        probability: float,
        entropy: float,
        top_candidates: list[dict[str, object]],
        stopped: bool = False,
    ) -> ReasoningEvent:
        token_text = self.token_decoder([token_id])
        event = ReasoningEvent(
            step=step,
            context_length=context_length,
            token_id=token_id,
            token_text=token_text,
            probability=float(probability),
            entropy=float(entropy),
            top_candidates=top_candidates,
            elapsed_ms=(time.perf_counter() - self._started_at) * 1000.0,
            stopped=stopped,
        )
        self.events.append(event)

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        return event

    def summary(self) -> dict[str, object]:
        if not self.events:
            return {
                "steps": 0,
                "average_probability": 0.0,
                "average_entropy": 0.0,
                "elapsed_ms": 0.0,
            }
        return {
            "steps": len(self.events),
            "average_probability": sum(e.probability for e in self.events) / len(self.events),
            "average_entropy": sum(e.entropy for e in self.events) / len(self.events),
            "elapsed_ms": self.events[-1].elapsed_ms,
        }

    def clear(self) -> None:
        self.events.clear()
        self._started_at = time.perf_counter()


def entropy(probabilities) -> float:
    """Return Shannon entropy for a probability distribution."""
    value = 0.0
    for probability in probabilities:
        p = float(probability)
        if p > 0.0:
            value -= p * math.log(p)
    return value
