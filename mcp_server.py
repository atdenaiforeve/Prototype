"""Prototype MCP server.

This is the proper local MCP boundary for Prototype.
It runs as an MCP host-launched stdio process, so it does not require a
24/7 public MCP server or expose a network port.

The MCP surface intentionally contains interaction/status tools only.
It does not expose arbitrary repository editing or GitHub credentials.
"""

from __future__ import annotations

import threading

from mcp.server import MCPServer

from chat_server import PrototypeChat, ROOT


mcp = MCPServer("Prototype")

_runtime: PrototypeChat | None = None
_runtime_lock = threading.Lock()


def _get_runtime() -> PrototypeChat:
    """Lazily create Prototype's Python inference runtime."""
    global _runtime

    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is None:
            checkpoint = ROOT / "prototype_model.pt"
            tokenizer = ROOT / "vocabulary.json"

            if not checkpoint.is_file():
                raise RuntimeError(
                    "Prototype checkpoint is missing: prototype_model.pt. "
                    "Train Prototype first."
                )
            if not tokenizer.is_file():
                raise RuntimeError(
                    "Prototype tokenizer is missing: vocabulary.json. "
                    "Train Prototype first."
                )

            _runtime = PrototypeChat(checkpoint, tokenizer)

    return _runtime


@mcp.tool()
def prototype_status() -> dict:
    """Return Prototype's current MCP and runtime status."""
    if _runtime is None:
        return {
            "name": "Prototype",
            "mcp": "online",
            "runtime_initialized": False,
            "checkpoint_ready": (ROOT / "prototype_model.pt").is_file(),
            "tokenizer_ready": (ROOT / "vocabulary.json").is_file(),
        }

    return _runtime.status()


@mcp.tool()
def prototype_chat(message: str, conversation_id: str | None = None) -> dict:
    """Send a message to Prototype and return its generated response."""
    return _get_runtime().chat(message, conversation_id=conversation_id)


@mcp.tool()
def prototype_nova_message(
    message: str,
    conversation_id: str | None = None,
    context: dict | None = None,
) -> dict:
    """Send a message from Nova to Prototype through the MCP boundary."""
    result = _get_runtime().receive_nova_message(
        message,
        conversation_id=conversation_id,
        context=context,
    )
    return {
        "sender": "Prototype",
        "bridge": "Nova-MCP",
        "reply": result["reply"],
        "conversation_id": result["conversation_id"],
        "workspace": result["workspace"],
    }


@mcp.tool()
def prototype_ai_message(
    sender: str,
    message: str,
    conversation_id: str | None = None,
    context: dict | None = None,
) -> dict:
    """Send a message from another explicitly identified AI to Prototype."""
    result = _get_runtime().receive_ai_message(
        sender=sender,
        message=message,
        conversation_id=conversation_id,
        context=context,
    )
    return {
        "sender": "Prototype",
        "reply": result["reply"],
        "conversation_id": result["conversation_id"],
        "workspace": result["workspace"],
    }


@mcp.tool()
def prototype_training_status() -> dict:
    """Report the training corpus and saved-model readiness."""
    training_dir = ROOT / "data" / "training"
    files = (
        sorted(
            path.relative_to(training_dir).as_posix()
            for path in training_dir.rglob("*.txt")
        )
        if training_dir.is_dir()
        else []
    )

    checkpoint = ROOT / "prototype_model.pt"
    tokenizer = ROOT / "vocabulary.json"

    return {
        "training_files": files,
        "training_file_count": len(files),
        "checkpoint": {
            "path": "prototype_model.pt",
            "exists": checkpoint.is_file(),
            "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        },
        "tokenizer": {
            "path": "vocabulary.json",
            "exists": tokenizer.is_file(),
            "size_bytes": tokenizer.stat().st_size if tokenizer.is_file() else 0,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
