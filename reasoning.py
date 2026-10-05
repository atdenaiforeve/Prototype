"""Structured reasoning workspace for Prototype.

This is an explicit internal workspace, not a claim of hidden human-like
thought. It lets Prototype maintain hypotheses, evidence, confidence, and
decisions before producing an answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Hypothesis:
    text: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    status: str = "active"

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


class ReasoningWorkspace:
    """Bounded scratch space for explicit model reasoning state."""

    def __init__(self, max_hypotheses: int = 8, max_evidence: int = 8) -> None:
        if max_hypotheses < 1 or max_evidence < 1:
            raise ValueError("workspace limits must be positive")
        self.max_hypotheses = max_hypotheses
        self.max_evidence = max_evidence
        self.hypotheses: list[Hypothesis] = []
        self.decisions: list[str] = []

    def add_hypothesis(self, text: str, confidence: float = 0.5) -> Hypothesis:
        text = str(text).strip()
        if not text:
            raise ValueError("hypothesis text must not be empty")
        hypothesis = Hypothesis(text=text, confidence=confidence)
        self.hypotheses.append(hypothesis)
        self.hypotheses = self.hypotheses[-self.max_hypotheses :]
        return hypothesis

    def add_evidence(self, hypothesis: Hypothesis, evidence: str) -> None:
        evidence = str(evidence).strip()
        if not evidence:
            raise ValueError("evidence must not be empty")
        hypothesis.evidence.append(evidence)
        hypothesis.evidence = hypothesis.evidence[-self.max_evidence :]

    def update_confidence(self, hypothesis: Hypothesis, confidence: float) -> None:
        hypothesis.confidence = max(0.0, min(1.0, float(confidence)))

    def choose(self) -> Hypothesis | None:
        active = [h for h in self.hypotheses if h.status == "active"]
        if not active:
            return None
        chosen = max(active, key=lambda h: h.confidence)
        for hypothesis in active:
            hypothesis.status = "chosen" if hypothesis is chosen else "rejected"
        self.decisions.append(chosen.text)
        self.decisions = self.decisions[-self.max_hypotheses :]
        return chosen

    def snapshot(self) -> dict[str, object]:
        return {
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "decisions": list(self.decisions),
        }

    def clear(self) -> None:
        self.hypotheses.clear()
        self.decisions.clear()
