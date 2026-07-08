# Security

This file documents security considerations, dependency management, and responsible disclosure guidelines.

Dependency hygiene
------------------

- Regularly run `cargo audit` (install via `cargo install cargo-audit`) to detect vulnerabilities in Rust dependencies.
- Pin critical dependency versions in `Cargo.toml` and review transitive dependency updates.

Secrets and credentials
-----------------------

- Do not store secrets, access tokens, or private keys in the repository. Use environment variables or external secret stores.
- Add `.env` to `.gitignore` (already present) and avoid committing any files with credentials.

Responsible disclosure
----------------------

If you discover a security vulnerability, please contact the maintainers privately at security@example.com with details, reproduction steps, and suggested mitigation. Do not post vulnerabilities publicly until they are addressed.

Runtime hardening
-----------------

- Run the server behind a reverse proxy (Nginx) with TLS termination.
- Use process supervision (systemd) and resource limits for production.

Threat model
------------

Consider threats such as data exfiltration from persistent storage, malicious ingestion payloads, and denial-of-service via expensive training runs. Defensive measures include input validation, rate limiting, and careful resource accounting for training jobs.
