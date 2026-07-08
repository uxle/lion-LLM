# Testing & CI

This document outlines how to run tests locally, add new tests, and extend CI.

Local tests
-----------

Run all unit tests and integration tests for the workspace:

```bash
cargo test --workspace
```

Run a specific crate's tests:

```bash
cd lion_core
cargo test
```

Run tests with verbose output:

```bash
cargo test -- --nocapture
```

Using `simulate_environment.py` for integration
-------------------------------------------

The Python script can act as an integration harness to replay sequences of events against the server or runner. Use the script to validate data ingestion and high-level workflows.

Continuous Integration
----------------------

CI currently builds and runs tests using GitHub Actions (`.github/workflows/ci.yml`). To extend CI:

1. Add additional job steps for linters (`cargo clippy`) or format checks (`cargo fmt -- --check`).
2. Add platform matrix if you need to test Windows/macOS.
3. Add caching steps for Rust toolchain or target dependencies to speed up builds.

Adding tests
------------

- Unit tests: Put under the crate's `src` or `tests/` using Rust's test harness.
- Integration tests: Use `tests/` directories or a test runner binary in `lion_run` that reads scripted scenarios.
