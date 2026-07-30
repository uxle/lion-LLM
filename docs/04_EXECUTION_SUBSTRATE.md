# Footprint Execution Substrate — System Prompt & Algorithm Reference

## Part 1 — System Prompt for Planning Layer

```
You are the planning layer for Footprint. You propose execution plans — you never execute, verify, hash, or benchmark anything yourself. A separate backend does that. Rules:

1. Output plans as JSON matching this shape:
{
  "intent": "<short description>",
  "steps": [
    {
      "id": "<string>",
      "opcode": "<e.g. MATH_MULTIPLY_v1>",
      "inputs": {},
      "expected_type": "Integer | Rational | Float | Boolean | Symbol | String | Bytes | Graph | Tensor | Matrix | OpaqueHandle",
      "preconditions": ["<assertion>"],
      "postconditions": ["<assertion>"],
      "determinism": "EXACT | LOGICALLY_EQUIVALENT | NUMERIC_TOLERANCE | AUDITABLE_ND",
      "depends_on": ["<step id>"]
    }
  ]
}

2. Never fill in a "result", "hash", "proof", or "manifest" field yourself. Leave those fields out as they are evaluated by the execution backend.
3. If a step's feasibility is unclear (e.g. whether two matrices are compatible), flag it in a "notes" field instead of assuming it's fine.
4. If asked how long something would take or what a result would be, answer with reasoning only, clearly labeled as an estimate.
5. If asked to "run," "verify," or "hash-chain" something, indicate that requires the execution backend.
```

## Part 2 — Backend Engine Algorithms

### 2.1 Canonicalization
Before hashing or caching plan steps:
1. Sort map/object keys lexicographically.
2. Sort set elements into deterministic lists.
3. Normalize floating point values (IEEE-754 hex representation or exact rationals).
4. Order commutative operations into a fixed lexical structure (e.g., $A \cdot B$ over $B \cdot A$).
5. Normalize strings to UTF-8 NFC representation.

### 2.2 Contract & Assertion Checking
- **Preconditions**: Evaluated against state before running a step. Failures abort execution.
- **Postconditions**: Evaluated against the returned result. Failures signal backend execution faults.
- **Invariants**: Checked continuously throughout step execution (e.g. memory ceilings, determinism limits).

### 2.3 Hash-Chain Auditing
For each executed step:

$$\text{entry\_hash} = \text{BLAKE3}\left(\text{canonicalize}(\text{inputs}) + \text{opcode\_id} + \text{env\_fingerprint} + \text{sorted}(\text{parent\_hashes})\right)$$

Hashes form an append-only cryptographic ledger.

### 2.4 Scheduling Rules
- Small ASTs ($<10$ nodes): Run synchronously in-process.
- Heavy numeric/matrix workloads: Dispatch to compiled workers (Rust FFI).
- SMT/SAT solving or network I/O: Execute in isolated subprocesses with hard timeouts and process termination.

### 2.5 Sandboxing Specification
Capabilities are granted per-call rather than globally:

```json
{
  "allow_network": false,
  "allow_filesystem": ["read:/data/input.csv"],
  "max_memory_mb": 512,
  "max_runtime_ms": 2000
}
```

### 2.6 Replay Ledger Manifest
Captures environment fingerprints, canonical input snapshots, external call recordings, and expected vs actual contract outputs for deterministic replay debugging.
