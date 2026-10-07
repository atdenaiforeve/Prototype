# Prototype

An experimental language model built from scratch.

This project is an experiment to see how far we can develop our own language model without relying on a pretrained language model.

## Current Status

Prototype now runs as a server-first runtime. The server connects the trained model, reasoning, memory, learning loop, experience system, self-model, autonomous messaging, AI communication, and bounded self-update controls.

The repository also contains a bounded self-update foundation. Prototype can record a fresh update intention in memory, retrieve it, inspect ordinary repository files, validate a local checkout, create a rollback branch, and record an auditable update event. Protected files such as credentials, GitHub workflows, the memory database, and the self-update controller itself are not available to autonomous writes.

## Install

Install the Python dependency set before training or running the neural model:

```bash
python -m pip install -r requirements.txt
```

## Run the chat

After training a model and creating `prototype_model.pt` and `vocabulary.json`:

```bash
python chat_server.py
```

The server binds to `0.0.0.0:8000` so a forwarded Codespaces port can be reached from another device. For a Codespaces deployment, use the public HTTPS URL for port 8000. The GitHub Pages web-live copy is no longer part of the Prototype runtime.

`GET /` reports that the server is online. `GET /status` reports the state of the major Prototype systems. Chat is handled through `POST /chat`.

The server keeps a short conversation history in memory and sends each message through Prototype's reasoning and memory systems. It also records the selected result through the existing learning loop.

## Self-modelling direction

The next stages are to connect chat interactions to richer self-observation and eventually to the self-modification experiment. The chat runtime does not give browser JavaScript access to GitHub credentials.

## Run the tests

Run the complete Python regression suite with:

```bash
python -m compileall -q .
python -m unittest discover -s . -p "test_*.py" -v
```

## Experience loop

Prototype now has a controlled GridWorld environment and a model-driven experience controller.

With a trained checkpoint running, the local server exposes:

- `GET /experience/status` — current position, goal, and available actions.
- `POST /experience/think-step` — Prototype observes the environment, asks its own language model to choose an action, performs it, and stores the experience in memory.
- `POST /experience/step` — manually perform a specific action for testing.
- `GET /self-update/status` — inspect the newest fresh self-update intention.

If Prototype produces an invalid action, the experience controller records a fresh self-update intention and uses a safe fallback action instead of crashing.

## AI communication

Prototype has both outgoing and incoming AI communication.

### Prototype → another AI

Peers are **not discovered automatically**. Configure a peer with the `PROTOTYPE_PEERS` environment variable:

```json
{"example-ai":"http://127.0.0.1:9000/message"}
```

Then `POST /communicate` sends a message to that explicitly configured peer.

### Another AI → Prototype

Prototype exposes:

```
POST /ai/message
```

The incoming endpoint is **public and does not require an authentication key or `X-Prototype-Key` header**. This is intentional for the current testing phase so another AI can send messages directly when the endpoint is reachable.

Example request:

```json
{
  "sender": "ExampleAI",
  "message": "Hello Prototype.",
  "conversation_id": "optional-stable-id",
  "context": {}
}
```

Example response:

```json
{
  "sender": "Prototype",
  "reply": "Hello.",
  "conversation_id": "optional-stable-id",
  "workspace": {}
}
```

Incoming messages are stored as external information in Prototype's memory. Prototype is explicitly told not to treat another AI's claims as automatically verified truth.

### Connecting a real remote AI

For automatic live updates in Codespaces, start `python server_supervisor.py` instead of `python chat_server.py`. The supervisor checks GitHub every 2 seconds, pulls new commits with a fast-forward-only update, and restarts the Prototype server so code changes go live automatically. It never force-resets local work. If the checkout has uncommitted changes, it safely skips the update. The server binds to `0.0.0.0` by default. For remote testing, expose port 8000 through a reachable HTTPS endpoint. Then another AI can send a `POST /ai/message` request using the JSON format above. **No shared inbound key is currently required.**

Do not put GitHub credentials or other secrets in the repository or in a client-side webpage.

The communication layer has message-size and timeout limits and does not forward Prototype's credentials.

## Server endpoints

- `GET /` — server-online information and API map.
- `GET /health` — lightweight health check.
- `GET /status` — unified status for Prototype's connected systems.
- `POST /chat` — normal conversation; starts the 10-second autonomous loop.
- `GET /autonomous/messages?conversation_id=...` — retrieve autonomous messages.
- `GET /experience/status` and `POST /experience/think-step` — environment and model-driven experience.
- `POST /ai/message` and `POST /communicate` — AI-to-AI communication.
- `GET /self-update/status` and `POST /self-update/intention` — bounded self-update controls.

## Planned next stage

Before enabling or expanding autonomous self-modification, the current codebase should be audited, tested, and frozen. The goal is to finish the planned foundation first so behavior can be tested without continually changing the underlying system.

A future wake-on-message design may allow an incoming conversation to activate Prototype when its main runtime is not already running. That is not implemented yet.
