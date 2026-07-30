# Footprint: Orchestration, Sandboxing & The Execution Ledger

**Version:** 1.0.0  
**Status:** Architectural Blueprint  

## 1. The DAG Orchestrator & Hybrid Scheduler

Once the Canonical IR is verified by the Compiler, it is handed to the DAG Orchestrator. The Orchestrator treats the IR as a Directed Acyclic Graph (DAG) of discrete computational nodes.

### Resource-Aware Scheduling
The scheduler does not blindly execute tasks. It analyzes the specific `Opcode`, the `ComputeClass` metadata, and resource availability (CPU pressure, RAM, GPU availability) to determine the optimal execution backend.

* **Thread-Local Sync Execution:** Tiny, fast expressions (e.g., `< 10` AST nodes) run in-memory with zero overhead.
* **Foreign Function Interface (FFI):** Heavy exact-math or CPU-bound tasks are passed via zero-copy memory to compiled C/Rust workers.
* **Isolated Sub-Processes:** Logical predicates or risky external I/O are routed to dedicated processes (e.g., Z3/SMT solvers) with strict PID lifecycle management. If a timeout triggers, the Orchestrator sends a `SIGKILL`, ensuring zero CPU leakage.

---

## 2. Capability-Based Sandboxing

Backends are given zero inherent trust. Footprint utilizes capability-based security: a worker process can only perform the exact actions explicitly permitted by its injected Sandbox Policy.

### The Determinism Envelope
Because true bit-for-bit determinism is impossible when touching OS-level I/O or network calls, Footprint enforces a strict "Determinism Envelope" contract for every node:
1. **EXACT:** Bit-for-bit memory and hash match (pure math, logic).
2. **LOGICALLY_EQUIVALENT:** Data matches, but underlying memory layouts or timestamps may drift.
3. **NUMERIC_TOLERANCE:** Floating-point bounds controlled by explicit epsilon values.
4. **AUDITABLE_ND:** Non-deterministic (RNG, Network), but strictly bounded by capability scopes and intercepted by Mocks.

---

## 3. The Execution Ledger (Causal Hash Chain)

Footprint abandons traditional linear logging. Because execution happens in parallel across a DAG, the ledger uses a **Causal DAG Hash Chain**. 

When a node completes successfully, the Orchestrator records an `ExecutionJournalEntry`. The identity of this entry is a cryptographic hash derived from:
1. The pre-canonicalized inputs and Opcode.
2. The specific environment fingerprint.
3. The **causal_parent_hashes** (the exact, lexically sorted hashes of the upstream nodes that produced the inputs).

This creates an unbreakable, branching cryptographic proof of execution topology.

---

## 4. The Replay Manifest (Forensic Fidelity)

If a Footprint node fails or produces anomalous output, the Orchestrator packages the state into a `ReplayManifest`. This is a standalone, legally auditable artifact that guarantees 100% forensic reproduction of the fault—completely bypassing the LLM planner.

### Contents of the Manifest:
* **The High-Fidelity Environment:** Pins the orchestrator version, backend binary digests (BLAKE3), `cgroup` memory limits, `io_uring` kernel flags, RNG seeds, and dependency lockfile hashes.
* **The Canonical Input Snapshot:** The exact, cryptographically verified memory state before execution.
* **Remote Mock States:** If the node made an external API call, the manifest includes the canonical response blob, latency, and request hash injected at the exact moment of execution.
* **The Divergence Check:** Contains both the `expected_backend_result` (what the contract demanded) and the `actual_backend_result` (the fault that occurred).

**The Result:** An engineer can download this single JSON/YAML manifest, feed it to a local Footprint Sandbox, and watch the exact failure recreate itself down to the millisecond, independent of the AI that initially proposed it.
