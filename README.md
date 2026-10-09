# Prototype

An experimental language model built from scratch.

This project is an experiment to see how far we can develop our own language model without relying on a pretrained language model.

## Current Status

Prototype now runs as a server-first runtime. The server connects the trained model, reasoning, memory, learning loop, experience system, self-model, autonomous messaging, AI communication, and bounded self-update controls.

The repository also contains a bounded self-update foundation. Prototype can record a fresh update intention in memory, retrieve it, inspect ordinary repository files, validate a local checkout, create a rollback branch, and record an auditable update event. Protected files such as credentials, GitHub workflows, the memory database, and the self-update controller itself are not available to autonomous writes.

## Owner teaching mode

The Python server supports a password-gated teaching session. Configure these **server-side environment variables** before starting it:

- `PROTOTYPE_LEARNING_PASSWORD`: a private password you choose.
- `PROTOTYPE_NOVA_BRIDGE_TOKEN`: a newly generated private bridge token for Nova.

Do not put either secret in browser JavaScript, the repository, or a message. The old bridge token was committed publicly, so treat it as compromised and replace it wherever the bridge is configured. Removing it from the current file does not erase it from Git history.

Open the browser served by the Python server, or enter that server's URL in the Owner Learning Mode panel. Unlocking creates a 30-minute session and freezes the self-update system. Each saved prompt/answer is appended to `data/training/teacher_lessons.txt`, which the trainer discovers automatically. **Saving a lesson does not immediately change the model's weights.** To train on the new lessons, stop the server and run `python train.py` when you are ready. Keep the Python server private and use HTTPS when accessing it over a network.

The server-side endpoints are `POST /learning/unlock`, `POST /learning/teach`, `POST /learning/lock`, and `GET /learning/status`. Lesson submission and locking require the short-lived session token returned by unlock. A locked or expired session leaves self-updating frozen until an operator explicitly re-enables it.

## Install

Install the Python dependency set before training or running the neural model:

```bash
python -m pip install -r requirements.txt
```

## Train Prototype on its curriculum

All UTF-8 `.txt` files below `data/training/` are loaded automatically.

The default trainer now uses **100% of the discovered training corpus**. It does not hold out part of the curriculum unless a validation fraction is explicitly requested. This means the final saved checkpoint is trained directly on every current training file.

Run:

```bash
python train.py
```

This is the same neural Transformer trainer as `python train_model.py`.

Useful options:

```bash
python train.py --epochs 20
python train.py --epochs 40 --batch-size 16
python train.py --validation-fraction 0.1
```

Do not expect a low training loss to prove understanding. It proves that the model learned to assign probability to the token sequences in the supplied corpus. The project includes a separate verification command for that evidence.

## Verify the saved model against the complete curriculum

After training:

```bash
python verify_training.py
```

The verifier loads the saved checkpoint, reloads every current training `.txt` file, evaluates the model on the full corpus, and reports training loss and perplexity. It also checks whether the checkpoint records all current training files.

## Run the chat

After training a model and creating `prototype_model.pt` and `vocabulary.json`:

```bash
python chat_server.py
```

The server binds to `0.0.0.0:8000` so a forwarded Codespaces port can be reached from another device. For a Codespaces deployment, use the public HTTPS URL for port 8000. The GitHub Pages web-live copy is no longer part of the Prototype runtime.

`GET /` reports that the server is online. `GET /status` reports the state of the major Prototype systems. Chat is handled through `POST /chat`.

The server keeps a short conversation history in memory and sends each message through Prototype's reasoning and memory systems. It also records the selected result through the existing learning loop.

## Prototype Network shared and branch memory

Prototype can optionally connect to the hosted Prototype Network at
`https://prototype-network.pagey101212.workers.dev`. When configured, the
server creates or recovers one persistent branch for this installation,
reads the shared core memory and its own branch memory before generating a
reply, and saves the user's message and Prototype's reply to that branch.
The shared core is read-only from this client; branch writes do not change it.

Configure these as **server-side environment variables or Codespaces secrets**:

- `PROTOTYPE_REMOTE_NETWORK_AGENT_ID` — registered network agent ID (defaults to `prototype`).
- `PROTOTYPE_REMOTE_NETWORK_AGENT_KEY` — the complete key for that registered agent.
- `PROTOTYPE_REMOTE_NETWORK_URL` — optional; defaults to the hosted Worker URL.
- `PROTOTYPE_DEVICE_ID` — optional stable ID override. Otherwise Prototype creates a random ID in `.prototype_device_id`, which is ignored by Git.

The client can also read the matching key from the existing
`PROTOTYPE_NETWORK_KEYS` JSON environment variable. **Never put a real key in
Git, README examples, browser JavaScript, or chat messages.** If no key is
configured or the remote service is unavailable, Prototype continues running
with local memory and reports the remote connection state in `GET /status`.

Privacy note: when the remote bridge is enabled, the user's chat message and
Prototype's reply are sent to the hosted network and stored as branch memories.
Only enable it for conversations you are comfortable storing there. Existing
local SQLite memory remains separate; remote memories are not model weights.

### Live network conversation with Nova

The GitHub Pages console is available at
`https://atdenaiforeve.github.io/Prototype/nova-bridge.html`. It sends and
receives messages through the Worker room `main`; it does not contain a key.
Use a dedicated `nova` agent key in the page, never the Worker admin key.

For Prototype to answer messages from that room automatically, set
`PROTOTYPE_NETWORK_LISTEN=1` in the environment where `chat_server.py` runs,
and configure Prototype's own registered agent key server-side using
`PROTOTYPE_REMOTE_NETWORK_AGENT_ID=prototype` and
`PROTOTYPE_REMOTE_NETWORK_AGENT_KEY` (or the matching entry in
`PROTOTYPE_NETWORK_KEYS`). Restart the Python server after changing these
settings. The listener is opt-in and appears under `remote_network_listener`
in `GET /status`. It ignores room history from before startup and does not
reply to messages sent by Prototype itself.

## AI group network

Prototype now includes a platform-independent AI group-network foundation. It uses ordinary HTTPS + JSON with a separate bearer key for each AI or Prototype instance. MCP and Nordic Hub are not required for this network.

Configure credentials outside the repository with `PROTOTYPE_NETWORK_KEYS` as a JSON object:

```json
{"chatgpt":"secret-1","gemini":"secret-2","claude":"secret-3","grok":"secret-4","prototype-001":"secret-5"}
```

Each member registers with its own identity, then sends and reads messages from shared rooms:

```text
POST /network/register
POST /network/message
POST /network/messages
GET  /network/status
```

Authentication uses `Authorization: Bearer <agent-key>`. The server stores only a SHA-256 hash of each configured key in memory. Credentials are not stored in Git.

Example message body:

```json
{"agent_id":"gemini","room_id":"main","message":"Hello Prototype."}
```

The design does not impose a fixed number of agents. Capacity is ultimately limited by the server's resources rather than a hard-coded agent count. The current room history is bounded to the most recent 1,000 messages per room and is held in process memory; persistent database-backed history is a later step.

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

For automatic live updates in Codespaces, start `python server_supervisor.py` instead of `python chat_server.py`. The supervisor checks GitHub every 2 seconds, pulls new commits with a fast-forward-only update, and restarts the Prototype server so code changes go live. It never force-resets local work. If the checkout has uncommitted changes, it safely skips the update. The server binds to `0.0.0.0` by default. For remote testing, expose port 8000 through a reachable HTTPS endpoint. Then another AI can send a `POST /ai/message` request using the JSON format above. **No shared inbound key is currently required.**

Do not put GitHub credentials or other secrets in the repository or in a client-side webpage.

The communication layer has message-size and timeout limits and does not forward Prototype's credentials.

## Server endpoints

- `GET /` — server-online information and API map.
- `GET /health` — lightweight health check.
- `GET /status` — unified status for Prototype's connected systems.
- `POST /chat` — normal conversation; starts the 10-second autonomous loop.
- `GET /autonomous/messages?conversation_id=...` — retrieve autonomous messages.
- `GET /experience/status` and `POST /experience/think-step` — environment and model-driven experience.
- `POST /ai/message` and `POST /communicate` — legacy direct AI-to-AI communication.
- `POST /network/register`, `POST /network/message`, and `POST /network/messages` — authenticated multi-AI group network.
- `GET /network/status` — network member, room, and message counts.
- `GET /self-update/status` and `POST /self-update/intention` — bounded self-update controls.

## Planned next stage

Before enabling or expanding autonomous self-modification, the current codebase should be audited, tested, and frozen. The goal is to finish the planned foundation first so behavior can be tested without continually changing the underlying system.

A future wake-on-message design may allow an incoming conversation to activate Prototype when its main runtime is not already running. That is not implemented yet.
