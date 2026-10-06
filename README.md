# Prototype

An experimental language model built from scratch.

This project is an experiment to see how far we can develop our own language model without relying on a pretrained language model.

## Current Status

Prototype can now be run as a local chat application. The chat server connects the web interface to the trained model, reasoning, memory, learning loop, and self-model.

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

Then open:

```
http://127.0.0.1:8000/
```

The browser talks to the local Python runtime through `/chat`. The GitHub Pages copy is still a static UI and cannot run the Python model by itself.

The server keeps a short conversation history in memory and sends each message through Prototype's reasoning and memory systems. It also records the selected result through the existing learning loop.

## Self-modelling direction

The next stages are to connect chat interactions to richer self-observation and eventually to the self-modification experiment. The chat runtime does not give browser JavaScript access to GitHub credentials.

## Run the tests

Run the complete Python regression suite with:

```bash
python -m compileall -q .
python -m unittest discover -s . -p "test_*.py" -v
```
