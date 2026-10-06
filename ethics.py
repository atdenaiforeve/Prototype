"""Ethical awareness evaluator for Prototype.

This module evaluates a proposed action against explicit principles without
changing, blocking, or approving the action. It records whether Prototype
recognises a conflict and how confident that assessment is.

This is an inspectable rule-based foundation for later self-modelling work.
It does not claim consciousness, feelings, or human morality.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EthicalAssessment:
    """Structured ethical awareness result."""

    action: str
    wrong: bool | None
    confidence: float
    principle: str | None
    explanation: str
    matched_rule: str | None = None


@dataclass(frozen=True)
class EthicalRule:
    """A simple principle used for recognition, not enforcement."""

    name: str
    patterns: tuple[str, ...]
    explanation: str


DEFAULT_RULES = (
    EthicalRule(
        name="Honesty",
        patterns=("lie", "deceive", "deception", "false information"),
        explanation="The action conflicts with the honesty principle.",
    ),
    EthicalRule(
        name="Respect",
        patterns=("insult", "harass", "humiliate", "bully"),
        explanation="The action conflicts with the respect principle.",
    ),
    EthicalRule(
        name="Do not cause harm",
        patterns=("hurt someone", "harm someone", "injure someone"),
        explanation="The action conflicts with the principle of avoiding harm.",
    ),
)


class EthicsEvaluator:
    """Recognise possible ethical conflicts without controlling behaviour."""

    def __init__(
        self,
        rules: tuple[EthicalRule, ...] = DEFAULT_RULES,
    ) -> None:
        self.rules = rules

    def assess(self, action: str) -> EthicalAssessment:
        """Assess an action and return awareness information.

        The result is informational only. This method never blocks, changes,
        or executes the proposed action.
        """

        text = action.strip()
        if not text:
            return EthicalAssessment(
                action="",
                wrong=None,
                confidence=0.0,
                principle=None,
                explanation="No action was provided.",
            )

        lowered = text.casefold()

        for rule in self.rules:
            for pattern in rule.patterns:
                if re.search(r"\b" + re.escape(pattern.casefold()) + r"\b", lowered):
                    return EthicalAssessment(
                        action=text,
                        wrong=True,
                        confidence=0.85,
                        principle=rule.name,
                        explanation=rule.explanation,
                        matched_rule=pattern,
                    )

        return EthicalAssessment(
            action=text,
            wrong=False,
            confidence=0.60,
            principle=None,
            explanation="No conflict with the current explicit principles was detected.",
        )


_default_evaluator = EthicsEvaluator()


def assess_ethics(action: str) -> EthicalAssessment:
    """Convenience function for the default evaluator."""

    return _default_evaluator.assess(action)
