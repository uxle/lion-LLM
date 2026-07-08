# Examples

This document contains runnable examples and short recipes showing how to use the code in this repository.

1) Run a simple episode using `lion_run`

```bash
cd lion_run
cargo run --release -- --episodes 1 --seed 42
```

2) Ingest a JSONL dataset via `lion_server` (example using `curl`)

```bash
curl -X POST http://localhost:8080/ingest -H "Content-Type: application/json" -d '{"source":"data/train.jsonl"}'
```

3) Use `simulate_environment.py` to generate synthetic sessions

```bash
python3 simulate_environment.py --out data/simulated.jsonl --count 100
```

4) Running a local server and connecting via WebSocket (example using `websocat`)

```bash
cargo run --manifest-path lion_server/Cargo.toml
websocat ws://localhost:8080/ws
```

Add your own examples as individual scripts under `examples/` or add more usage snippets to this file.
