# Lion AI & Footprint Architecture — Master Index

Ten architecture specifications covering memory, orchestration (V1 & V2), serving optimization, footprint execution substrate, determinism envelopes, canonical IR contracts, and full LFMF/ENFS specifications.

| File | Layer | Status |
|---|---|---|
| `01_MEMORY_AND_STORAGE.md` | Filesystem/model-container format (ENFS + LFMF) overview | Conceptual — memory hierarchy & container layout |
| `02_ORCHESTRATION_RUNTIME.md` | Agent loop V1: routing, planning, DAG execution, HITL auth | Retained for reference — superseded by V2 |
| `08_AGENT_RUNTIME_V2.md` | Agent loop V2: explicit HITL resume, tool contracts, DAG scheduler, 3-plane security | Active design spec |
| `03_SERVING_OPTIMIZATION.md` | Inference-time optimization (caching, batching, quantization, speculative decoding) | Established techniques + honest analysis of multiplier claims (v2) |
| `04_EXECUTION_SUBSTRATE.md` | Footprint: typed IR, contracts, sandboxing, replay ledger | Conceptual blueprint & algorithm reference |
| `05_FOOTPRINT_DETERMINISTIC_SUBSTRATE.md` | Footprint Core Philosophy & Paradigm Shift | Core Architectural Principles |
| `06_FOOTPRINT_ORCHESTRATION_SANDBOX_LEDGER.md` | DAG Orchestrator, Determinism Envelopes & Causal Replay | Hybrid Scheduler & Security Policy |
| `07_FOOTPRINT_CANONICAL_IR_CONTRACTS.md` | Canonical IR, Type Primitives & Verification Contracts | Type Contracts & AST Canonicalization |
| `09_LFMF_FULL_SPEC.md` | LFMF v1.0 — Lion Flexible Model Format full specification | All chunk types, tensor dtypes, compression, encryption, delta updates |
| `10_ENFS_FULL_SPEC.md` | ENFS v1.0 — Einstein Neurons Filesystem full specification | Cognitive memory tiers, domain stores, 73-doc index, hybrid packaging |

## Scope Note

Everything in this specification directory — filesystem layout, model container format, agent orchestration loop, execution substrate — defines infrastructure *around* an AI model.

Key distinctions:
- Infrastructure manages context, routing, caching, tool invocation, and memory storage.
- Reasoning capability, factual grounding, and generalization come from model architecture, training data, and alignment.
- Systems are designed for robust engineering, graceful fallback, and sandboxed execution rather than theoretical unconstrained guarantees.

## Algorithm Integration Guidelines

When evaluating new algorithms or specification components:
1. Check for functional overlap with existing modules (`lion_core`, `lion_brain`, `lion_agent`, `lion_senses`).
2. Verify correctness, memory limits, and security constraints (e.g. sandbox path validation, RBAC policy checks, determinism envelopes).
3. Record changes using Merge Notes (what was kept, replaced, or upgraded and why).
