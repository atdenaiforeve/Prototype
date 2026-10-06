"""Model-driven controller for Prototype's first environment."""

from __future__ import annotations

import re
from typing import Callable

from experience import ACTIONS, ExperienceEngine, Observation


class ExperienceAgent:
    """Let Prototype's language model choose an action from its observation."""

    def __init__(self, engine: ExperienceEngine, generate_fn: Callable[..., str]) -> None:
        self.engine = engine
        self.generate_fn = generate_fn

    def choose_action(self, observation: Observation) -> tuple[str, str]:
        prompt = (
            "You are Prototype controlling a small grid world. "
            f"Your position is {observation.position}. "
            f"Your goal is {observation.goal}. "
            f"Available actions are: {', '.join(observation.available_actions)}. "
            "Choose exactly one available action. "
            "Reply with only the action name."
        )
        raw = self.generate_fn(
            prompt,
            max_new_tokens=8,
            temperature=0.3,
            top_k=8,
        )
        action = self._extract_action(raw, observation.available_actions)

        if action is None:
            self.engine.memory.remember_self_update_intention(
                "Improve action selection so Prototype reliably outputs one valid "
                "environment action.",
                reason=f"Model produced an invalid action response: {raw[:120]!r}",
            )
            # Safe deterministic fallback keeps the environment running.
            action = observation.available_actions[0]

        return action, raw.strip()

    @staticmethod
    def _extract_action(raw: str, available: tuple[str, ...]) -> str | None:
        text = str(raw).lower().strip()
        for action in available:
            if re.search(rf"\b{re.escape(action)}\b", text):
                return action
        return None

    def step(self) -> dict:
        observation = self.engine.observe()
        action, model_output = self.choose_action(observation)
        experience = self.engine.step(action, source="prototype-model")
        return {
            "model_output": model_output,
            "chosen_action": action,
            "experience": experience.as_dict(),
            "status": self.engine.status(),
        }
