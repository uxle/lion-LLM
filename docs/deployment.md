# Deployment

This guide covers deploying `lion_server` and running the system in production-like settings.

1) Build release artifacts

```bash
cargo build --workspace --release
```

2) Recommended production stack

- Containerize the server with Docker and run behind a reverse proxy (Nginx) for TLS and routing.
- Use systemd or container orchestrators (Docker Compose, Kubernetes) for process supervision.

3) Example Dockerfile (simple)

```
FROM rust:slim
WORKDIR /app
COPY . .
RUN cargo build --release --manifest-path lion_server/Cargo.toml
CMD ["/app/target/release/lion_server"]
```

4) Environment variables

- `LION_SERVER_PORT` — port to bind (default 8080)
- `RUST_LOG` — logging level (e.g., `info`, `debug`)

5) Backups and persistence

Regularly back up persisted model state (files written by `persist.rs`). Use incremental backups and checksum verification for important experiments.

6) Scaling

If serving many concurrent experiments or inference requests, shard tasks across multiple server instances and use shared storage for model checkpoints. Consider adding queueing (RabbitMQ, Redis) for long-running jobs.
