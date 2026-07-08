# API Documentation

This file describes the HTTP and WebSocket endpoints provided by `lion_server`. The implementation files are `lion_server/src/api.rs` and `lion_server/src/ws.rs`.

HTTP endpoints (examples)
-------------------------

- `GET /health` — Returns 200 OK with basic health info.
- `POST /ingest` — Accepts JSON or JSONL payloads to ingest training data. Example request body: `{ "source": "train.jsonl" }`.
- `GET /state` — Returns a snapshot of the current model state (summary only; full binary state may be unavailable via HTTP).
- `POST /episode` — Trigger a new training episode with parameters: `{ "episodes": 1, "seed": 42 }`.

Responses use JSON with a top-level `status` or `error` field and data in `result`.

WebSocket
---------

The server supports a WebSocket endpoint (e.g. `/ws`) for real-time interactions. Typical message types:

- `subscribe` — subscribe to a stream of episodic events
- `episode_event` — server->client events carrying step-level traces
- `command` — client->server command to run inference or control the model

Message format (JSON):

```json
{ "type": "command", "action": "run_inference", "payload": { "input": "Hello" } }
```

Authentication
--------------

The current server does not enforce authentication by default. For production deployments, add TLS and an auth layer (API keys, JWT) in front of endpoints.

Extending the API
------------------

To add endpoints, update `lion_server/src/api.rs`, add handler functions, and register routes in `lion_server/src/main.rs`. Keep handlers small and delegate to `lion_core` for logic.
