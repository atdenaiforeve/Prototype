"""Standard HTTPS AI group-network layer for Prototype.

The network deliberately uses ordinary HTTP/JSON and per-agent bearer keys.
No MCP or platform-specific integration is required.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass


MAX_SENDER_CHARS = 200
MAX_MESSAGE_CHARS = 8_000
MAX_ROOM_CHARS = 200
MAX_HISTORY = 1_000


@dataclass(frozen=True)
class NetworkAgent:
    agent_id: str
    display_name: str
    kind: str


class PrototypeNetwork:
    """In-process group chat with independently authenticated AI agents."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: dict[str, NetworkAgent] = {}
        self._keys: dict[str, str] = {}
        self._rooms: dict[str, list[dict]] = {}

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_keys() -> dict[str, str]:
        raw = os.getenv("PROTOTYPE_NETWORK_KEYS", "{}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PROTOTYPE_NETWORK_KEYS must be valid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("PROTOTYPE_NETWORK_KEYS must be a JSON object")
        return {str(agent_id): str(key) for agent_id, key in data.items() if str(key)}

    def reload_keys(self) -> None:
        """Reload agent credentials from the environment without storing plaintext keys."""
        loaded = self._load_keys()
        with self._lock:
            self._keys = {agent_id: self._hash_key(key) for agent_id, key in loaded.items()}

    def authenticate(self, agent_id: str, supplied_key: str) -> bool:
        if not agent_id or not supplied_key:
            return False
        with self._lock:
            expected = self._keys.get(agent_id)
        if expected is None:
            return False
        return hmac.compare_digest(self._hash_key(supplied_key), expected)

    def register_agent(self, agent_id: str, display_name: str, kind: str) -> NetworkAgent:
        agent_id = str(agent_id).strip()
        display_name = str(display_name).strip()
        kind = str(kind).strip().lower()
        if not agent_id or len(agent_id) > MAX_SENDER_CHARS:
            raise ValueError("agent_id must be 1-200 characters")
        if not display_name or len(display_name) > MAX_SENDER_CHARS:
            raise ValueError("display_name must be 1-200 characters")
        if kind not in {"ai", "prototype"}:
            raise ValueError("kind must be 'ai' or 'prototype'")
        agent = NetworkAgent(agent_id=agent_id, display_name=display_name, kind=kind)
        with self._lock:
            self._agents[agent_id] = agent
        return agent

    def send(self, agent_id: str, room_id: str, message: str) -> dict:
        room_id = str(room_id).strip()
        message = str(message).strip()
        if not room_id or len(room_id) > MAX_ROOM_CHARS:
            raise ValueError("room_id must be 1-200 characters")
        if not message:
            raise ValueError("message must not be empty")
        if len(message) > MAX_MESSAGE_CHARS:
            raise ValueError("message is too long")
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise ValueError("agent is not registered")
            messages = self._rooms.setdefault(room_id, [])
            item = {
                "id": str(uuid.uuid4()),
                "room_id": room_id,
                "sender_id": agent.agent_id,
                "sender": agent.display_name,
                "kind": agent.kind,
                "message": message,
                "timestamp": time.time(),
            }
            messages.append(item)
            del messages[:-MAX_HISTORY]
            return dict(item)

    def history(self, agent_id: str, room_id: str, after: float | None = None) -> list[dict]:
        room_id = str(room_id).strip()
        with self._lock:
            if agent_id not in self._agents:
                raise ValueError("agent is not registered")
            messages = list(self._rooms.get(room_id, []))
        if after is None:
            return messages
        return [item for item in messages if item["timestamp"] > float(after)]

    def status(self) -> dict:
        with self._lock:
            return {
                "agents": len(self._agents),
                "rooms": len(self._rooms),
                "messages": sum(len(messages) for messages in self._rooms.values()),
            }
