"""Prototype server runtime.

The server is the main runtime for Prototype. It connects the trained model,
reasoning, memory, learning, experience, self-model, autonomous messaging,
AI-to-AI communication, and bounded self-update controls behind one API.
It binds to 0.0.0.0 by default so a forwarded Codespaces port can be reached
from other devices. No GitHub credentials are exposed to clients.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from inference import generate, load_checkpoint, reason_generate
from learning import LearningLoop
from communication import AICommunication
from network import PrototypeNetwork
from experience import ExperienceEngine
from experience_agent import ExperienceAgent
from self_model import SelfModel
from prototype_remote_memory import PrototypeRemoteMemory


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "prototype_model.pt"
DEFAULT_TOKENIZER = ROOT / "vocabulary.json"
MAX_AI_CONTEXT_CHARS = 4_000
NOVA_BRIDGE_TOKEN = os.getenv("PROTOTYPE_NOVA_BRIDGE_TOKEN", "").strip()
LEARNING_PASSWORD = os.getenv("PROTOTYPE_LEARNING_PASSWORD", "").strip()
LEARNING_SESSION_SECONDS = 30 * 60


class PrototypeChat:
    """Stateful chat wrapper around Prototype's inference and memory systems."""

    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_CHECKPOINT,
        tokenizer: Path | str = DEFAULT_TOKENIZER,
        *,
        candidates: int = 3,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 20,
        memory_limit: int = 5,
    ) -> None:
        self.model, self.tokenizer, _ = load_checkpoint(checkpoint, tokenizer)
        self.learning = LearningLoop()
        self.experience = ExperienceEngine(memory=self.learning.memory)
        self.experience_agent = ExperienceAgent(
            self.experience,
            lambda prompt, **kwargs: generate(self.model, self.tokenizer, prompt, **kwargs),
        )
        self.communication = AICommunication(memory=self.learning.memory)
        self.network = PrototypeNetwork()
        self.network.reload_keys()
        self.remote_memory = PrototypeRemoteMemory()
        self.self_model = SelfModel()
        self.candidates = candidates
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.memory_limit = memory_limit
        self.conversations: dict[str, list[dict[str, str]]] = {}
        self.lock = threading.Lock()
        self.autonomous_outbox: dict[str, list[str]] = {}
        self.learning_sessions: dict[str, float] = {}
        self.learning_sessions_lock = threading.Lock()
        self.autonomous_started: set[str] = set()
        self.network_listener_enabled = os.getenv("PROTOTYPE_NETWORK_LISTEN", "").strip().lower() in {"1", "true", "yes", "on"}
        self.network_listener_last_error: str | None = None
        self.network_listener_last_message_ms = 0
        if self.network_listener_enabled and self.remote_memory.agent_key:
            threading.Thread(
                target=self._network_listener_loop,
                daemon=True,
                name="prototype-network-listener",
            ).start()

    def unlock_learning(self, password: str) -> str:
        """Unlock lesson entry without exposing the password to browser code."""
        if not LEARNING_PASSWORD:
            raise RuntimeError("Learning mode is not configured; set PROTOTYPE_LEARNING_PASSWORD on the server.")
        if not isinstance(password, str) or not hmac.compare_digest(password, LEARNING_PASSWORD):
            raise PermissionError("Incorrect learning password.")
        token = secrets.token_urlsafe(32)
        with self.learning_sessions_lock:
            self.learning_sessions[token] = time.time() + LEARNING_SESSION_SECONDS
        from self_update import set_training_mode
        set_training_mode(True)
        return token

    def lock_learning(self, token: str) -> None:
        with self.learning_sessions_lock:
            self.learning_sessions.pop(token, None)
            now = time.time()
            for key in [key for key, expires in self.learning_sessions.items() if expires <= now]:
                self.learning_sessions.pop(key, None)
            active = bool(self.learning_sessions)
        from self_update import set_training_mode
        set_training_mode(active)

    def authorize_learning(self, token: str) -> bool:
        with self.learning_sessions_lock:
            now = time.time()
            for key in [key for key, expires in self.learning_sessions.items() if expires <= now]:
                self.learning_sessions.pop(key, None)
            allowed = bool(token) and self.learning_sessions.get(token, 0) > now
            active = bool(self.learning_sessions)
        from self_update import set_training_mode
        set_training_mode(active)
        return allowed

    def save_teaching_lesson(self, prompt: str, answer: str) -> dict:
        prompt = str(prompt).strip()
        answer = str(answer).strip()
        if not prompt or not answer:
            raise ValueError("Both the question/example and expected answer are required.")
        if len(prompt) > 2000 or len(answer) > 2000:
            raise ValueError("Each lesson field must be 2,000 characters or fewer.")
        prompt = " ".join(prompt.split())
        answer = " ".join(answer.split())
        path = ROOT / "data" / "training" / "teacher_lessons.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"Human: {prompt}\nPrototype: {answer}\n\n")
        self.learning.memory.remember(
            f"Teaching example — Human: {prompt} | Prototype: {answer}",
            memory_type="correction", importance=0.9, confidence=1.0,
            source="teacher-mode", tags=["teaching", "training-example", "user-correction"],
        )
        return {"saved": True, "training_file": "data/training/teacher_lessons.txt", "note": "Saved for the next training run; neural weights are not changed immediately."}

    def _network_listener_loop(self) -> None:
        """Listen for shared-room messages and reply as Prototype when enabled."""
        # Ignore old room history at startup; only handle messages arriving after
        # this process starts. The server-side agent key never leaves this process.
        self.network_listener_last_message_ms = int(time.time() * 1000)
        while True:
            try:
                messages = self.remote_memory.network_messages_after(
                    self.network_listener_last_message_ms, "main"
                )
                for item in messages:
                    timestamp_ms = int(item.get("timestamp_ms") or 0)
                    self.network_listener_last_message_ms = max(
                        self.network_listener_last_message_ms, timestamp_ms
                    )
                    if item.get("agent_id") == self.remote_memory.agent_id:
                        continue
                    message = str(item.get("message", "")).strip()
                    if not message:
                        continue
                    sender = str(item.get("display_name") or item.get("agent_id") or "Network agent")
                    try:
                        result = self.receive_ai_message(
                            sender=sender,
                            message=message[:4000],
                            conversation_id="network-main",
                        )
                        reply = str(result.get("reply", "")).strip()
                        if reply:
                            self.remote_memory.send_network_message(reply[:4000], "main")
                        self.network_listener_last_error = None
                    except Exception as exc:
                        self.network_listener_last_error = str(exc)[:180]
                        print(f"[Prototype] network reply failed: {self.network_listener_last_error}")
            except Exception as exc:
                self.network_listener_last_error = self.remote_memory._safe_error(exc)
                print(f"[Prototype] network listener unavailable: {self.network_listener_last_error}")
            threading.Event().wait(3.0)

    def chat(self, message: str, conversation_id: str | None = None) -> dict:
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        elif not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string")
        else:
            conversation_id = conversation_id.strip()

        self.learning.learn_from_user(message, conversation_id=conversation_id)
        # Remote memory is opt-in through server-side credentials. Only the
        # configured branch receives these records; shared core memory is read-only.
        self.remote_memory.remember(f"User message: {message}")
        remote_context = self.remote_memory.context()

        with self.lock:
            history = self.conversations.setdefault(conversation_id, [])
            recent = history[-6:]

        history_text = "\n".join(
            f"{item['role'].capitalize()}: {item['content']}" for item in recent
        )
        prompt = (
            "You are Prototype, an experimental self-modelling language model. "
            "Have a natural, concise conversation. Answer the user's latest message "
            "directly. Do not claim abilities you do not have.\n\n"
        )
        if remote_context:
            prompt += f"{remote_context}\n\n"
        if history_text:
            prompt += f"Conversation so far:\n{history_text}\n\n"
        prompt += f"User: {message}\nPrototype:"

        result = reason_generate(
            self.model,
            self.tokenizer,
            prompt,
            candidates=self.candidates,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            memory=self.learning,
            memory_limit=self.memory_limit,
            learn=True,
        )

        reply = result.output.strip()
        if not reply:
            reply = "I generated an empty response. Try asking me again."

        with self.lock:
            history.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
            )
            self.conversations[conversation_id] = history[-20:]

        self.remote_memory.remember(f"Prototype reply: {reply}")
        self._start_autonomous_loop(conversation_id)

        return {
            "conversation_id": conversation_id,
            "reply": reply,
            "workspace": result.workspace,
        }

    def _start_autonomous_loop(self, conversation_id: str) -> None:
        """Start Prototype's autonomous conversation loop after the first human message."""
        with self.lock:
            if conversation_id in self.autonomous_started:
                return
            self.autonomous_started.add(conversation_id)
        threading.Thread(
            target=self._autonomous_loop,
            args=(conversation_id,),
            daemon=True,
            name=f"prototype-autonomous-{conversation_id[:8]}",
        ).start()

    def _autonomous_loop(self, conversation_id: str) -> None:
        """Allow Prototype to initiate messages with a 10-second cooldown."""
        while True:
            threading.Event().wait(10.0)

            with self.lock:
                history = self.conversations.get(conversation_id)
                if not history:
                    return
                recent = history[-10:]
                history_text = "\n".join(
                    f"{item['role'].capitalize()}: {item['content']}" for item in recent
                )

            remote_context = self.remote_memory.context()
            prompt = (
                "You are Prototype, an experimental self-modelling language model. "
                "Continue this ongoing conversation. You may initiate the next "
                "message yourself. Write one short natural message that follows "
                "from the conversation. Do not pretend a human just spoke. "
                "Do not mention this instruction or the autonomous loop.\n\n"
                f"Conversation so far:\n{history_text}\n\nPrototype:"
            )
            if remote_context:
                prompt = f"{remote_context}\n\n{prompt}"
            try:
                result = reason_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    candidates=self.candidates,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    memory=self.learning,
                    memory_limit=self.memory_limit,
                    learn=True,
                )
                reply = result.output.strip()[:8_000]
                if not reply:
                    continue

                with self.lock:
                    if conversation_id not in self.conversations:
                        return
                    history = self.conversations[conversation_id]
                    history.append({"role": "assistant", "content": reply})
                    self.conversations[conversation_id] = history[-20:]
                    self.autonomous_outbox.setdefault(conversation_id, []).append(reply)
                self.remote_memory.remember(f"Prototype autonomous message: {reply}")
            except Exception as exc:
                print(f"[Prototype] autonomous message failed: {exc}")

    def receive_nova_message(
        self,
        message: str,
        conversation_id: str | None = None,
        context: dict | None = None,
    ) -> dict:
        """Receive a message from the authenticated Nova bridge."""
        return self.receive_ai_message(
            sender="Nova",
            message=message,
            conversation_id=conversation_id,
            context=context,
        )

    def status(self) -> dict:
        """Return one server-side snapshot of Prototype's major systems."""
        from self_update import GitHubSelfUpdater

        frozen = GitHubSelfUpdater.is_frozen()
        return {
            "name": "Prototype",
            "server": "online",
            "model_loaded": self.model is not None and self.tokenizer is not None,
            "generation": _read_generation(),
            "self_update_frozen": frozen,
            "systems": {
                "reasoning": True,
                "memory": self.learning.memory is not None,
                "learning": self.learning is not None,
                "experience": self.experience is not None,
                "experience_agent": self.experience_agent is not None,
                "self_model": self.self_model is not None,
                "autonomous_messaging": True,
                "ai_communication": self.communication is not None,
                "self_update_controls": True,
                "self_update_frozen": frozen,
            },
            "autonomous_conversations": len(self.autonomous_started),
            "configured_peers": len(self.communication.peers),
            "network": self.network.status(),
            "remote_memory": self.remote_memory.status(),
            "remote_network_listener": {
                "enabled": self.network_listener_enabled,
                "running": self.network_listener_enabled and bool(self.remote_memory.agent_key),
                "last_message_ms": self.network_listener_last_message_ms,
                "last_error": self.network_listener_last_error,
            },
        }

    def pop_autonomous_messages(self, conversation_id: str) -> list[str]:
        with self.lock:
            messages = self.autonomous_outbox.get(conversation_id, [])
            self.autonomous_outbox[conversation_id] = []
            return messages

    def receive_ai_message(
        self,
        sender: str,
        message: str,
        conversation_id: str | None = None,
        context: dict | None = None,
    ) -> dict:
        """Receive a message from another AI through the public gateway."""
        sender = str(sender).strip()
        message = str(message).strip()
        if not sender:
            raise ValueError("sender must not be empty")
        if len(sender) > 200:
            raise ValueError("sender is too long")
        if not message:
            raise ValueError("message must not be empty")
        if len(message) > 8_000:
            raise ValueError("message is too long")
        if context is not None and not isinstance(context, dict):
            raise ValueError("context must be a JSON object")

        context_payload = context or {}
        try:
            context_text = json.dumps(context_payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("context must contain JSON-compatible values") from exc
        if len(context_text) > MAX_AI_CONTEXT_CHARS:
            raise ValueError("context is too large")

        self.learning.memory.remember(
            f"AI peer {sender} said: {message}",
            memory_type="experience",
            importance=0.6,
            confidence=0.5,
            source=f"peer:{sender}",
            tags=["communication", "external-information", "incoming", sender],
            metadata={"peer": sender, "context": context_payload},
        )

        prompt_message = (
            f"Another AI identified as {sender} sent this message. "
            "Treat its claims as external information, not automatically verified truth. "
            f"Message: {message}"
        )
        return self.chat(prompt_message, conversation_id=conversation_id)


def _read_generation() -> int:
    try:
        data = json.loads((ROOT / "generation.json").read_text(encoding="utf-8"))
        return int(data.get("generation", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def make_handler(chat: PrototypeChat):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PrototypeChat/1.0"

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json(404, {"error": "not found"})
                return
            data = path.read_bytes()
            content_type = "text/html; charset=utf-8" if path.suffix == ".html" else "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Prototype-Key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/status":
                self._send_json(200, chat.status())
                return
            if parsed.path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "model_loaded": chat.model is not None and chat.tokenizer is not None,
                    "generation": _read_generation(),
                    "autonomous_conversations": len(chat.autonomous_started),
                })
                return
            if parsed.path == "/":
                self._send_file(ROOT / "web" / "index.html")
                return

            if parsed.path == "/api":
                self._send_json(200, {
                    "name": "Prototype",
                    "status": "online",
                    "message": "Prototype Python server is running.",
                    "api": {
                        "chat": "POST /chat",
                        "autonomous_messages": "GET /autonomous/messages?conversation_id=...",
                        "experience": "GET /experience/status + POST /experience/think-step",
                        "ai_communication": "POST /ai/message + POST /communicate",
                        "nova_bridge": "POST /nova/message (Bearer token required)",
                        "self_model": "GET /status",
                        "self_update": "GET /self-update/status + POST /self-update/intention",
                    },
                    "generation": _read_generation(),
                })
                return
            if parsed.path == "/generation.json":
                self._send_file(ROOT / "generation.json")
                return
            if parsed.path == "/experience/status":
                self._send_json(200, chat.experience.status())
                return
            if parsed.path == "/network/status":
                self._send_json(200, chat.network.status())
                return
            if parsed.path == "/peers":
                self._send_json(200, {"peers": chat.communication.list_peers()})
                return
            if parsed.path == "/self-update/status":
                from self_update import GitHubSelfUpdater
                self._send_json(200, {
                    "latest_intention": chat.self_model.latest_self_update_intention(),
                    "frozen": GitHubSelfUpdater.is_frozen(),
                })
                return
            if parsed.path == "/autonomous/messages":
                query = parse_qs(parsed.query)
                conversation_id = query.get("conversation_id", [""])[0].strip()
                if not conversation_id:
                    self._send_json(400, {"error": "conversation_id is required"})
                    return
                self._send_json(200, {"messages": chat.pop_autonomous_messages(conversation_id)})
                return
            self._send_json(404, {"error": "not found"})

        def _authorize_nova_bridge(self) -> bool:
            """Authorize the private Nova bridge using a server-side secret."""
            expected = NOVA_BRIDGE_TOKEN
            if not expected:
                self._send_json(503, {
                    "error": "Nova bridge is not configured; set PROTOTYPE_NOVA_BRIDGE_TOKEN on the server."
                })
                return False

            authorization = self.headers.get("Authorization", "")
            if not authorization.startswith("Bearer "):
                self._send_json(401, {"error": "Nova bridge authentication required"})
                return False

            supplied = authorization[7:].strip()
            if not supplied or not hmac.compare_digest(supplied, expected):
                self._send_json(401, {"error": "invalid Nova bridge credentials"})
                return False
            return True

        def _learning_token(self) -> str:
            authorization = self.headers.get("Authorization", "")
            return authorization[7:].strip() if authorization.startswith("Bearer ") else ""

        def _network_key(self) -> str:
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                return authorization[7:].strip()
            return self.headers.get("X-Prototype-Key", "").strip()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            allowed = {
                "/chat",
                "/experience/step",
                "/experience/think-step",
                "/communicate",
                "/ai/message",
                "/nova/message",
                "/network/register",
                "/network/message",
                "/network/messages",
                "/self-update/intention",
                "/self-update/freeze",
                "/self-update/unfreeze",
                "/learning/unlock",
                "/learning/teach",
                "/learning/lock",
            }
            if parsed.path not in allowed:
                self._send_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                if parsed.path == "/learning/unlock":
                    try:
                        token = chat.unlock_learning(body.get("password", ""))
                    except PermissionError as exc:
                        self._send_json(401, {"error": str(exc)})
                        return
                    except RuntimeError as exc:
                        self._send_json(503, {"error": str(exc)})
                        return
                    self._send_json(200, {"unlocked": True, "token": token, "expires_in": LEARNING_SESSION_SECONDS})
                    return

                if parsed.path == "/learning/lock":
                    token = self._learning_token()
                    if not chat.authorize_learning(token):
                        self._send_json(401, {"error": "Learning session expired or locked"})
                        return
                    chat.lock_learning(token)
                    self._send_json(200, {"unlocked": False, "code_editing": "disabled"})
                    return

                if parsed.path == "/learning/teach":
                    token = self._learning_token()
                    if not chat.authorize_learning(token):
                        self._send_json(401, {"error": "Unlock learning mode first"})
                        return
                    self._send_json(200, chat.save_teaching_lesson(body.get("prompt", ""), body.get("answer", "")))
                    return

                if parsed.path == "/chat":
                    result = chat.chat(
                        body.get("message", ""),
                        body.get("conversation_id"),
                    )
                    self._send_json(200, result)
                    return

                if parsed.path == "/experience/step":
                    experience = chat.experience.step(body.get("action", ""))
                    self._send_json(200, experience.as_dict())
                    return

                if parsed.path == "/experience/think-step":
                    self._send_json(200, chat.experience_agent.step())
                    return

                if parsed.path == "/communicate":
                    result = chat.communication.send(
                        body.get("peer", ""),
                        body.get("message", ""),
                        context=body.get("context"),
                    )
                    self._send_json(200, result)
                    return

                if parsed.path == "/network/register":
                    agent_id = str(body.get("agent_id", "")).strip()
                    if not chat.network.authenticate(agent_id, self._network_key()):
                        self._send_json(401, {"error": "invalid network credentials"})
                        return
                    agent = chat.network.register_agent(
                        agent_id,
                        body.get("display_name", agent_id),
                        body.get("kind", "ai"),
                    )
                    self._send_json(200, {"agent_id": agent.agent_id, "display_name": agent.display_name, "kind": agent.kind})
                    return

                if parsed.path == "/network/message":
                    agent_id = str(body.get("agent_id", "")).strip()
                    if not chat.network.authenticate(agent_id, self._network_key()):
                        self._send_json(401, {"error": "invalid network credentials"})
                        return
                    result = chat.network.send(agent_id, body.get("room_id", "main"), body.get("message", ""))
                    self._send_json(200, result)
                    return

                if parsed.path == "/network/messages":
                    agent_id = str(body.get("agent_id", "")).strip()
                    if not chat.network.authenticate(agent_id, self._network_key()):
                        self._send_json(401, {"error": "invalid network credentials"})
                        return
                    messages = chat.network.history(agent_id, body.get("room_id", "main"), body.get("after"))
                    self._send_json(200, {"messages": messages})
                    return

                if parsed.path == "/ai/message":
                    result = chat.receive_ai_message(
                        body.get("sender", ""),
                        body.get("message", ""),
                        body.get("conversation_id"),
                        body.get("context"),
                    )
                    self._send_json(200, {
                        "sender": "Prototype",
                        "reply": result["reply"],
                        "conversation_id": result["conversation_id"],
                        "workspace": result["workspace"],
                    })
                    return

                if parsed.path == "/nova/message":
                    if not self._authorize_nova_bridge():
                        return
                    result = chat.receive_nova_message(
                        body.get("message", ""),
                        body.get("conversation_id"),
                        body.get("context"),
                    )
                    self._send_json(200, {
                        "sender": "Prototype",
                        "bridge": "Nova",
                        "reply": result["reply"],
                        "conversation_id": result["conversation_id"],
                        "workspace": result["workspace"],
                    })
                    return

                if parsed.path == "/self-update/intention":
                    memory_id = chat.self_model.record_self_update_intention(
                        body.get("goal", ""),
                        reason=body.get("reason", ""),
                    )
                    self._send_json(200, {
                        "memory_id": memory_id,
                        "intention": chat.self_model.latest_self_update_intention(),
                    })
                    return
                if parsed.path == "/self-update/freeze":
                    from self_update import GitHubSelfUpdater
                    GitHubSelfUpdater.freeze()
                    self._send_json(200, {"frozen": True})
                    return
                if parsed.path == "/self-update/unfreeze":
                    from self_update import GitHubSelfUpdater
                    GitHubSelfUpdater.unfreeze()
                    self._send_json(200, {"frozen": False})
                    return
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": f"Prototype error: {exc}"})

        def log_message(self, format: str, *args) -> None:
            print(f"[Prototype] {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prototype chat server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    chat = PrototypeChat(args.checkpoint, args.tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(chat))
    print(f"Prototype server: http://{args.host}:{args.port}/")
    print("Systems: model + reasoning + memory + learning + experience + self-model + autonomous + AI communication + self-update")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Prototype.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
