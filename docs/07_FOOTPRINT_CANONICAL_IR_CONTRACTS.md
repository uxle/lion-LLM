# Footprint: Canonical IR & Verification Contracts

**Version:** 1.0.0  
**Status:** Architectural Blueprint  

## 1. Overview of the Compilation Layer

The compilation layer is the gatekeeper of the Footprint architecture. Its sole purpose is to treat the AI planner as an untrusted source and convert its proposed logic into a **Canonical Intermediate Representation (IR)**. 

If the proposed plan violates typing rules, security policy, determinism constraints, or IR semantics, the compiler rejects it entirely. The AI cannot "talk its way out" of a semantic failure.

---

## 2. The Formal IR Specification

Footprint IR is not a generic JSON payload; it is a strictly typed language specification designed for machine execution. 

### Core Primitives
Variables are strictly typed at compile-time to prevent runtime coercion failures.
* `Integer` (BigInt arbitrary precision)
* `Rational` (Exact fractions, no floating-point drift)
* `Float` (IEEE 754, requires explicit epsilon rounding policies)
* `Boolean`, `Symbol`, `String`, `Bytes`
* `Graph`, `Tensor`, `Matrix`
* `OpaqueHandle` (Pointers to external states)

### Opcode Semantics
Every operation in the Footprint IR is defined by an Opcode. Opcodes are immutable and versioned.

**Example: `Opcode.MATH_MULTIPLY_v1`**
* **Inputs:** `Operand A (Numeric)`, `Operand B (Numeric)`
* **Output:** `Result (Numeric)`

---

## 3. Design by Contract (Semantic Verification)

Footprint moves error handling from "after execution" to "before execution." Every Opcode defines mathematical and logical bounds through **Verification Contracts**. 

The Semantic Analyzer evaluates these contracts statically before assigning the task to a backend worker.

### Contract Types:
1. **Preconditions:** Must be true before the backend receives the operation.
2. **Postconditions:** Must be true about the result returned by the backend.
3. **Invariants:** Must remain unviolated throughout the execution boundary.

### Example: Matrix Multiplication Contract

```yaml
Opcode: MATRIX_MULTIPLY
Version: 1.1.0

Preconditions:
  - assert type(A) == Matrix
  - assert type(B) == Matrix
  - assert A.columns == B.rows  # Fails statically if impossible

Postconditions:
  - assert shape(Result) == (A.rows, B.columns)

Invariants:
  - DeterminismLevel: EXACT
  - MemoryCeiling: 512MB
```

*Result: If the LLM proposes multiplying a 2x3 matrix with a 4x4 matrix, the Semantic Analyzer rejects the graph instantly, saving CPU cycles and preventing backend crashes.*

---

## 4. The Canonicalization Protocol

To ensure that $x + y$ and $y + x$ produce the exact same cryptographic hash (allowing aggressive caching and perfect deduplication), all IR must be **canonicalized** before hashing or execution.

The normalizer enforces the following strict rules on the IR AST:
1. **Dictionary Ordering:** All map/dictionary keys are strictly lexicographically sorted.
2. **Set Serialization:** Unordered sets are sorted into deterministic lists.
3. **Float Normalization:** Floats are either coerced to exact Rationals or serialized to a deterministic IEEE 754 hex string to prevent cross-architecture drift.
4. **Algebraic Commutativity:** Commutative operations are lexically ordered (e.g., $B \cdot A$ is always rewritten to $A \cdot B$).
5. **Unicode:** All strings are enforced to UTF-8 with NFC Unicode normalization.

### Why This Matters
When the Semantic Analyzer produces a hashed `StateSnapshot` from the canonicalized IR, the hash acts as the absolute identity of the logic. If two different AI planners propose logically identical steps written in different ways, they will yield the exact same hash, triggering an instant cache hit rather than redundant execution.
