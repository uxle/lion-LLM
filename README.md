# lion-LLM

Lightweight research workspace for the lion-LLM project.

## Overview

This repository contains a small collection of Rust crates and a Python helper for simulating environments. Key folders:

- `lion_core/` — core Rust library and unit tests
- `lion_run/` — Rust binary entrypoint
- `lion_server/` — Rust web server (public JS assets under `public/`)
- `simulate_environment.py` — small Python helper script

## Requirements

- Rust (stable) and `cargo`
- Python 3.8+ (optional, for `simulate_environment.py`)
- Node.js / npm (optional, for editing `lion_server/public` assets)

## Quick start

Build the entire workspace:

```
cargo build --workspace --release
```

Run tests for the workspace:

```
cargo test --workspace
```

Run the server (from repo root):

```
cd lion_server
cargo run --release
```

Run the runner:

```
cd lion_run
cargo run --release
```

Run the Python simulator:

```
python3 simulate_environment.py
```

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for contribution guidelines.

## Documentation

Comprehensive documentation lives in the `docs/` folder. Start with [docs/README.md](docs/README.md).

## License

This project is available under the MIT License. See [LICENSE](LICENSE) for details.
