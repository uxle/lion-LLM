# Architecture Diagrams

Below are high-level architecture diagrams (Mermaid) to help visualize component interactions.

System overview

```mermaid
flowchart LR
  subgraph Server
    A[lion_server]
    A -->|calls| B[lion_core]
  end
  C[lion_run]
  C -->|uses| B
  D[simulate_environment.py] -->|generates| E[data/*.jsonl]
  E --> A
```

Sequence: ingestion -> episode

```mermaid
sequenceDiagram
  participant Client
  participant Server as lion_server
  participant Core as lion_core
  Client->>Server: POST /ingest {source}
  Server->>Core: load_data(source)
  Core->>Core: run_episode()
  Core-->>Server: summary
  Server-->>Client: 200 OK
```

These diagrams are intentionally small; expand them with additional nodes for storage, proxies, and monitoring when planning a deployment.
