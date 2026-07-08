# Getting Started

This guide helps new contributors set up a development environment, build the workspace, run tests, and start the server or runner.

Prerequisites
-------------

- Rust toolchain (stable). Install via `rustup`.
- `cargo` (bundled with rustup).
- Python 3.8+ (optional, for `simulate_environment.py`).
- Node.js / npm (optional, for working on `lion_server/public`).

Clone and prepare
-----------------

```bash
git clone <repo-url>
cd lion-LLM
rustup toolchain install stable
rustup default stable
```

Build and test
--------------

Build entire workspace:

```bash
cargo build --workspace
```

Run all tests:

```bash
cargo test --workspace
```

Run server
----------

From the repo root:

```bash
cd lion_server
cargo run --release -- --help
```

Run runner
----------

```bash
cd lion_run
cargo run --release
```

Use the Python simulator
------------------------

```bash
python3 simulate_environment.py
```

Troubleshooting
---------------

- If build fails, check the Rust toolchain version with `rustc --version`.
- Missing system libraries: install common build tools (`build-essential`/`make`, `pkg-config`, `libssl-dev` on Linux for TLS features).
