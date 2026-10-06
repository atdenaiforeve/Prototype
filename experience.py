"""Controlled environment and experience loop for Prototype.

This is the first environment Prototype can interact with without connecting it
to a real-world system. Experiences are explicit records: observation, action,
result, and outcome. Failed or surprising outcomes can become fresh
self-update intentions for the existing bounded self-update system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from memory import MemoryEngine, get_memory


ACTIONS = ("north", "south", "east", "west", "wait")


@dataclass(frozen=True)
class Observation:
    position: tuple[int, int]
    goal: tuple[int, int]
    available_actions: tuple[str, ...]


@dataclass(frozen=True)
class Experience:
    observation: Observation
    action: str
    result: Observation
    moved: bool
    reached_goal: bool

    def as_dict(self) -> dict:
        return asdict(self)


class GridWorld:
    """Small deterministic 2-D environment for safe first experiments."""

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("width and height must both be at least 2")
        self.width = int(width)
        self.height = int(height)
        self.start = self._validate_position(start)
        self.goal = self._validate_position(
            goal if goal is not None else (self.width - 1, self.height - 1)
        )
        self.reset()

    def _validate_position(self, position: tuple[int, int]) -> tuple[int, int]:
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not all(isinstance(value, int) for value in position)
        ):
            raise ValueError("position must be a pair of integers")
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("position is outside the grid")
        return position

    def reset(self) -> Observation:
        self.position = self.start
        return self.observe()

    def observe(self) -> Observation:
        x, y = self.position
        available: list[str] = ["wait"]
        if y < self.height - 1:
            available.append("north")
        if y > 0:
            available.append("south")
        if x < self.width - 1:
            available.append("east")
        if x > 0:
            available.append("west")
        return Observation(
            position=self.position,
            goal=self.goal,
            available_actions=tuple(available),
        )

    def step(self, action: str) -> tuple[Observation, bool, bool]:
        action = str(action).strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        before = self.position
        x, y = before

        if action == "north" and y < self.height - 1:
            y += 1
        elif action == "south" and y > 0:
            y -= 1
        elif action == "east" and x < self.width - 1:
            x += 1
        elif action == "west" and x > 0:
            x -= 1

        self.position = (x, y)
        moved = self.position != before
        reached_goal = self.position == self.goal
        return self.observe(), moved, reached_goal


class ExperienceEngine:
    """Connect an environment to Prototype's persistent memory."""

    def __init__(
        self,
        environment: GridWorld | None = None,
        memory: MemoryEngine | None = None,
    ) -> None:
        self.environment = environment or GridWorld()
        self.memory = memory or get_memory()
        self.steps = 0

    def observe(self) -> Observation:
        return self.environment.observe()

    def step(self, action: str, *, source: str = "experience-engine") -> Experience:
        observation = self.environment.observe()
        result, moved, reached_goal = self.environment.step(action)

        experience = Experience(
            observation=observation,
            action=str(action).strip().lower(),
            result=result,
            moved=moved,
            reached_goal=reached_goal,
        )
        self.steps += 1

        self.memory.remember(
            self._memory_text(experience),
            memory_type="experience",
            importance=0.8 if reached_goal else 0.55,
            confidence=1.0,
            source=source,
            tags=["experience", "environment", "grid-world"],
            metadata={"experience": experience.as_dict()},
        )

        if not moved and experience.action != "wait":
            self.memory.remember_self_update_intention(
                "Improve environment action handling so Prototype can avoid "
                "repeating an action that failed to move it.",
                reason=f"Action '{experience.action}' did not move Prototype.",
            )

        return experience

    def run(
        self,
        action_selector: Callable[[Observation], str],
        *,
        max_steps: int = 20,
    ) -> list[Experience]:
        if max_steps < 1 or max_steps > 100:
            raise ValueError("max_steps must be between 1 and 100")

        experiences: list[Experience] = []
        for _ in range(max_steps):
            observation = self.observe()
            action = action_selector(observation)
            experience = self.step(action)
            experiences.append(experience)
            if experience.reached_goal:
                break
        return experiences

    @staticmethod
    def _memory_text(experience: Experience) -> str:
        return (
            f"Prototype experienced action={experience.action}; "
            f"position={experience.observation.position}; "
            f"result={experience.result.position}; "
            f"moved={experience.moved}; "
            f"reached_goal={experience.reached_goal}."
        )

    def status(self) -> dict:
        observation = self.observe()
        return {
            "environment": "grid_world",
            "size": [self.environment.width, self.environment.height],
            "position": list(observation.position),
            "goal": list(observation.goal),
            "available_actions": list(observation.available_actions),
            "steps": self.steps,
        }
