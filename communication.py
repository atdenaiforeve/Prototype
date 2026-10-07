"""Bounded communication layer for Prototype.

Prototype can communicate with explicitly configured peer AI endpoints. No peer
is discovered automatically and no credentials are forwarded to peers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from memory import MemoryEngine, get_memory


@dataclass(frozen=True)
class Peer:
    name: str
    url: str


class AICommunication:
    """Explicit peer-to-peer message transport with strict size/time limits."""

    def __init__(
        self,
        peers: dict[str, str] | None = None,
        memory: MemoryEngine | None = None,
    ) -> None:
        self.memory = memory or get_memory()
        self.peers: dict[str, Peer] = {}
        configured = peers if peers is not None else self._load_environment_peers()
        for name, url in configured.items():
            self.add_peer(name, url)

    @staticmethod
    def _load_environment_peers() -> dict[str, str]:
        raw = os.environ.get("PROTOTYPE_PEERS", "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("PROTOTYPE_PEERS must contain valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("PROTOTYPE_PEERS must be a JSON object")
        return {str(name): str(url) for name, url in data.items()}

    def add_peer(self, name: str, url: str) -> None:
        name = str(name).strip()
        url = str(url).strip()
        if not name:
            raise ValueError("peer name must not be empty")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("peer URL must use HTTP or HTTPS")
        if len(url) > 500:
            raise ValueError("peer URL is too long")
        self.peers[name] = Peer(name=name, url=url)

    def list_peers(self) -> list[dict[str, str]]:
        return [{"name": peer.name, "url": peer.url} for peer in self.peers.values()]

    def send(self, peer_name: str, message: str, *, context: dict[str, Any] | None = None) -> dict:
        peer = self.peers.get(str(peer_name).strip())
        if peer is None:
            raise ValueError(f"unknown peer: {peer_name}")

        message = str(message).strip()
        if not message:
            raise ValueError("message must not be empty")
        if len(message) > 8_000:
            raise ValueError("message is too long")

        context_payload = context or {}
        if not isinstance(context_payload, dict):
            raise ValueError("context must be a JSON object")
        try:
            encoded_context = json.dumps(
                context_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("context must contain JSON-compatible values") from exc
        if len(encoded_context) > 4_000:
            raise ValueError("context is too large")

        payload = {
            "sender": "Prototype",
            "message": message,
            "context": context_payload,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            peer.url,
            data=encoded,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Prototype-AI-communication/1.0",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read(32_000)
                result = json.loads(raw.decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"peer communication failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("peer returned invalid JSON") from exc

        if not isinstance(result, dict):
            raise RuntimeError("peer response must be a JSON object")

        reply = str(result.get("reply", result.get("message", ""))).strip()
        if reply:
            self.memory.remember(
                f"Peer {peer.name} said: {reply}",
                memory_type="experience",
                importance=0.55,
                confidence=0.5,
                source=f"peer:{peer.name}",
                tags=["communication", "external-information", peer.name],
                metadata={"peer": peer.name},
            )

        return {
            "peer": peer.name,
            "reply": reply,
            "raw": result,
        }
