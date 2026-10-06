"""Bounded memory context formatting for Prototype.

Retrieves only relevant memories and formats them for the reasoning prompt.
This keeps long-term memory separate from the neural model's weights.
"""

from __future__ import annotations

from typing import Any


def format_memory_context(
    memories: list[dict[str, Any]],
    *,
    max_chars: int = 400,
    max_memories: int = 3,
    max_memory_chars: int = 120,
) -> str:
    """Format a small, bounded memory section for a model prompt."""
    max_chars = max(0, int(max_chars))
    max_memories = max(0, int(max_memories))
    max_memory_chars = max(1, int(max_memory_chars))
    if max_chars == 0 or max_memories == 0 or not memories:
        return ""

    lines: list[str] = []
    used = len("[MEMORY CONTEXT]\n")
    for memory in memories[:max_memories]:
        content = " ".join(str(memory.get("content", "")).split())
        if not content:
            continue
        content = content[:max_memory_chars].rstrip()
        memory_type = str(memory.get("memory_type", "note"))
        confidence = float(memory.get("confidence", 0.0))
        line = f"- [{memory_type}; confidence {confidence:.2f}] {content}"
        extra = len(line) + (1 if lines else 0)
        if used + extra > max_chars:
            break
        lines.append(line)
        used += extra

    return "[MEMORY CONTEXT]\n" + "\n".join(lines) if lines else ""


def build_prompt(prompt: str, memory_context: str = "") -> str:
    """Build a bounded prompt with memory clearly separated from user input."""
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if not memory_context.strip():
        return prompt
    return f"{memory_context.strip()}\n[CURRENT INPUT]\n{prompt}"
