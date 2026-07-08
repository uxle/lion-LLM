# Tutorials

This file contains step-by-step tutorials for common tasks.

Tutorial 1 — From data to an episode summary

1. Prepare or select a JSONL dataset in `data/`.
2. Start `lion_server` locally:

```bash
cd lion_server
cargo run
```

3. Ingest the dataset:

```bash
curl -X POST http://localhost:8080/ingest -H "Content-Type: application/json" -d '{"source":"data/train.jsonl"}'
```

4. Trigger an episode and wait for completion:

```bash
curl -X POST http://localhost:8080/episode -H "Content-Type: application/json" -d '{"episodes":1, "seed":42}'
```

5. Retrieve a state snapshot:

```bash
curl http://localhost:8080/state
```

Tutorial 2 — Running local experiments with `lion_run`

1. Build and run with a fixed seed:

```bash
cd lion_run
cargo run -- --episodes 10 --seed 123
```

2. Collect output from `target/` or from configured persistence location for analysis.

Add more tutorials here for dataset curation, evaluation metrics, and debugging common failures.
