# Architecture

This document explains the architecture and responsibilities of the major components in the repository.

Overview
--------

The project is organized as a small workspace of Rust crates plus a Python helper. The design emphasizes modularity, testability, and small, well-scoped binaries.

Top-level components
--------------------

- `lion_core/` — Core library that implements the brain, synapse models, episode handling, persistence, random generators, and unit-tested algorithms. This crate is intended to be dependency-free from application-specific code and provides primitives for storage, learning, and simulation.

- `lion_run/` — CLI/binary that exercises `lion_core` in batch or interactive experiments. Useful for running simulations, benchmarks, or CLI-driven tasks.

- `lion_server/` — Web server exposing APIs and a small front-end for visualizing or interacting with the model. It contains an HTTP API (`api.rs`) and WebSocket handlers (`ws.rs`) for real-time interactions.

- `simulate_environment.py` — Python helper used for lightweight experiments, dataset preparation, or integration testing with external tooling.

Data flow
---------

1. Training / data preparation: JSONL files under `data/` are prepared (see `data_format.md`).
2. Ingestion: the `lion_run` binary or `lion_server` ingestion endpoints load data into `lion_core` structures.
3. Processing: `lion_core` runs episodes, propagation, and synaptic updates using deterministic RNG from `rng.rs` for reproducible experiments.
4. Persistence: `persist.rs` handles saving state to disk; the storage format is intentionally minimal and documented in `developer_guide.md`.
5. Serving: `lion_server` provides REST endpoints and a WebSocket stream for live interaction and monitoring.

Testing and isolation
---------------------

Each crate contains focused unit tests under `tests/` (see `lion_core/src/tests/`). The design encourages small, fast tests rather than end-to-end monolith tests. Integration tests live in the binaries and the `simulate_environment.py` harness.

Extensibility
-------------

The `lion_core` APIs are designed to be used both by native Rust binaries and via language bindings (if needed in future). Keep public-facing APIs minimal and stable; internal modules can evolve more quickly.
