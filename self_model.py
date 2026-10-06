"""Observable self-model for Prototype.

The self-model describes what the running system can observe about itself.
It is deliberately descriptive: it does not approve, reject, or veto actions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SelfObservation:
    generation: int
    model: dict[str, Any]
    memory: dict[str, Any]
    code: dict[str, Any]
    state: dict[str, Any]


class SelfModel:
    """Build and persist a compact, observable model of Prototype itself."""

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)
        self.identity = {
            "name": "Prototype",
            "type": "experimental self-modelling language system",
        }
        self.capabilities = [
            "language generation",
            "memory retrieval",
            "bounded reasoning",
            "learning from recorded feedback",
            "self-observation",
            "self-update intention planning",
            "bounded self-update validation",
        ]
        self.limitations = [
            "observations depend on files and runtime metadata being available",
            "self-model does not imply consciousness or human-like self-awareness",
        ]
        self.current_state: dict[str, Any] = {}
        self.environment: dict[str, Any] = {}

    def _generation(self) -> int:
        path = self.root / "generation.json"
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("generation", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0

    def _code_snapshot(self) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*.py")):
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            try:
                content = path.read_bytes()
            except OSError:
                continue
            files.append({
                "path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            })
        return {
            "python_files": files,
            "python_file_count": len(files),
        }

    def observe(
        self,
        *,
        model: Any | None = None,
        memory: Any | None = None,
        state: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> SelfObservation:
        model_info: dict[str, Any] = {}
        if model is not None:
            model_info = {
                "class": type(model).__name__,
                "parameters": sum(
                    int(parameter.numel())
                    for parameter in model.parameters()
                    if hasattr(parameter, "numel")
                ),
                "training": bool(getattr(model, "training", False)),
            }
            for name in (
                "vocab_size",
                "embedding_size",
                "hidden_size",
                "context_size",
                "num_layers",
                "num_heads",
                "dropout",
            ):
                if hasattr(model, name):
                    model_info[name] = getattr(model, name)

        memory_info: dict[str, Any] = {}
        if memory is not None and hasattr(memory, "stats"):
            try:
                memory_info = dict(memory.stats())
            except Exception:
                memory_info = {"available": False}

        self.current_state = dict(state or {})
        self.environment = dict(environment or {})

        return SelfObservation(
            generation=self._generation(),
            model=model_info,
            memory=memory_info,
            code=self._code_snapshot(),
            state=self.current_state,
        )

    def record_self_update_intention(
        self,
        goal: str,
        *,
        reason: str = "",
        freshness_seconds: int = 24 * 60 * 60,
    ) -> int:
        """Immediately write a self-update intention to Prototype memory."""
        from memory import get_memory
        return get_memory().remember_self_update_intention(
            goal,
            reason=reason,
            freshness_seconds=freshness_seconds,
        )

    def latest_self_update_intention(self) -> dict[str, Any] | None:
        """Expose the newest fresh self-update intention to the self-model."""
        try:
            from memory import get_memory
            return get_memory().latest_self_update_intention()
        except Exception:
            return None

    def snapshot(self, observation: SelfObservation) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
            "observation": asdict(observation),
            "environment": dict(self.environment),
        }

    def save(self, observation: SelfObservation, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.snapshot(observation), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
