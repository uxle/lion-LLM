# Data Format

This document describes the dataset and session file formats used in the repository. The `data/` folder contains `train.jsonl`, `train_chat.jsonl`, and other artifacts.

JSONL
-----

We use JSON Lines (JSONL) for training and session data. Each line is a standalone JSON object.

Example `train.jsonl` entry:

```json
{ "id": "example-001", "input": "What is the capital of France?", "output": "Paris", "metadata": { "source": "wiki" } }
```

Example `train_chat.jsonl` entry:

```json
{ "session_id": "s-0001", "turns": [ { "role": "user", "text": "Hello" }, { "role": "assistant", "text": "Hi, how can I help?" } ] }
```

Guidelines
----------

- Keep each JSON object minimal and include a stable `id` or `session_id` when possible.
- Prefer UTF-8 text encoding.
- Avoid embedding large binary blobs; store them separately and reference by path or URL.

Preprocessing
-------------

Use `simulate_environment.py` or your own scripts to normalize text, strip trailing whitespace, and token-level sanitization before ingestion.

Versioning
----------

If you change the schema, bump a `data_version` field inside files (or maintain separate versioned file names) and implement migration helpers in `lion_core/persist.rs`.
