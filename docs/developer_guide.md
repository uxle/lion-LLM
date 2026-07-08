# Developer Guide

This document explains the repository layout, coding conventions, where to find core modules, and how to add features or debug issues.

Repository layout
-----------------

- `lion_core/` — core library
  - `src/brain.rs` — high-level brain orchestration
  - `src/synapse.rs` — synaptic structures and updates
  - `src/propgation.rs` — propagation algorithms
  - `src/persist.rs` — persistence helpers
  - `src/tests/` — unit tests for core algorithms
- `lion_run/` — CLI runner
- `lion_server/` — web server with API and WebSocket handlers
- `data/` — sample and training datasets

Coding conventions
------------------

- Follow Rust idioms and `rustfmt` formatting. Enforce formatting locally via `rustfmt` or `cargo fmt`.
- Keep functions small and well-documented with doc comments `///`.
- Use unit tests for algorithm correctness; include small, deterministic examples in tests.

Building & format
------------------

Use the following commands:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo build --workspace
```

Debugging
---------

Run a single crate in debug mode:

```bash
cd lion_core
cargo test --lib -- --nocapture
```

Profiling and benchmarks
------------------------

Benchmarks are in `lion_core/benches/`. Use `cargo bench` (requires nightly for some benchmark harnesses) or run manual benchmarks in `lion_run`.

Adding a new feature
---------------------

1. Open an issue describing the feature and design decisions.
2. Implement on a feature branch.
3. Add unit tests and integration tests where appropriate.
4. Update `docs/` with API changes.

Persistence format
------------------

The persistence layer aims for a simple, robust on-disk representation. Before changing formats, add a migration strategy and preserve backwards compatibility when possible.
