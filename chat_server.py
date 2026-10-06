"""Local chat runtime for Prototype.

Runs the actual Prototype model behind a small HTTP API and serves the web UI.
The server is intentionally local by default; it does not expose model or GitHub
credentials to browser JavaScript.
"""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from inference import load_checkpoint, reason_generate
from learning import LearningLoop


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "prototype_model.pt"
DEFAULT_TOKENIZER = ROOT / "vocabulary.json"


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
        self.candidates = candidates
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.memory_limit = memory_limit
        self.conversations: dict[str, list[dict[str, str]]] = {}
        self.lock = threading.Lock()

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

        return {
            "conversation_id": conversation_id,
            "reply": reply,
            "workspace": result.workspace,
        }


def make_handler(chat: PrototypeChat):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PrototypeChat/1.0"

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
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
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(200, {"status": "ok", "model_loaded": True})
                return
            if parsed.path == "/":
                self._send_file(ROOT / "index.html")
                return
            if parsed.path == "/generation.json":
                self._send_file(ROOT / "generation.json")
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                result = chat.chat(
                    body.get("message", ""),
                    body.get("conversation_id"),
                )
                self._send_json(200, result)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": f"Prototype error: {exc}"})

        def log_message(self, format: str, *args) -> None:
            print(f"[Prototype] {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prototype chat server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    chat = PrototypeChat(args.checkpoint, args.tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(chat))
    print(f"Prototype chat: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Prototype.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
