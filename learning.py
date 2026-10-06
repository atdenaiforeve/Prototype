"""Persistent evaluation and feedback loop for Prototype.

This layer does not secretly modify neural weights. It records outcomes and
turns explicit feedback into durable memories and corrections that future
systems can retrieve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memory import MemoryEngine, get_memory
from reasoning_engine import ReasoningResult


@dataclass
class LearningRecord:
    prompt: str
    output: str
    score: float
    feedback: str | None = None
    memory_id: int | None = None


class LearningLoop:
    """Record inference outcomes and learn from explicit feedback."""

    def __init__(self, memory: MemoryEngine | None = None) -> None:
        self.memory = memory or get_memory()

    def record_result(self, prompt: str, result: ReasoningResult, *, source: str = "inference") -> LearningRecord:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        score = float(result.chosen.confidence) if result.chosen is not None else 0.0
        payload = {
            "prompt": prompt.strip(),
            "output": result.output,
            "confidence": score,
            "reasoning": result.workspace,
        }
        memory_id = self.memory.remember(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            memory_type="experience",
            importance=min(1.0, 0.4 + score * 0.5),
            confidence=score,
            source=source,
            tags=["inference", "learning"],
        )
        return LearningRecord(prompt.strip(), result.output, score, None, memory_id)

    def learn_from_user(self, message: str, *, conversation_id: str | None = None) -> int:
        """Store what a person explicitly says so future conversations can use it.

        This records the user's words as an experience; it does not guess hidden
        traits or turn an inference into a fact. The normal memory system decides
        later whether the experience is useful enough to retrieve.
        """
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        if len(message) > 8_000:
            raise ValueError("message is too long")

        metadata: dict[str, Any] = {"conversation_id": conversation_id} if conversation_id else {}
        return self.memory.remember(
            f"User explicitly said: {message}",
            memory_type="experience",
            importance=0.7,
            confidence=0.9,
            source="user-conversation",
            tags=["conversation", "user-shared", "learning"],
            metadata=metadata,
        )

    def give_feedback(self, record: LearningRecord, *, good: bool, feedback: str = "") -> LearningRecord:
        """Apply explicit feedback to the stored experience."""
        if record.memory_id is None:
            raise ValueError("record must be persisted before feedback")
        feedback = feedback.strip()
        if good:
            self.memory.reinforce(record.memory_id, amount=0.10, reason=feedback or "positive inference feedback")
            score = min(1.0, record.score + 0.10)
        else:
            self.memory.weaken(record.memory_id, amount=0.20, reason=feedback or "negative inference feedback")
            score = max(0.0, record.score - 0.20)
            if feedback:
                self.memory.remember(
                    f"Correction for prompt '{record.prompt}': {feedback}",
                    memory_type="correction",
                    importance=0.8,
                    confidence=0.9,
                    source="inference-feedback",
                    tags=["learning", "correction"],
                )
        return LearningRecord(record.prompt, record.output, score, feedback or None, record.memory_id)

    def retrieve(self, prompt: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.memory.recall(prompt, limit=limit)
"