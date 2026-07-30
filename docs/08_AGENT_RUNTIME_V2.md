# Agent Execution Runtime — V2 Specification

**Version:** 2.0.0
**Status:** Design Specification
**Supersedes:** `02_ORCHESTRATION_RUNTIME.md` (V1 — retained for reference)

Key additions over V1:
- Explicit HITL resume path that re-hydrates the **exact saved batch** instead of replanning.
- `ToolDefinition` contract with input/output JSON Schemas.
- `ExecutionPlan` as a machine-validatable struct (no prose output from planner).
- Bounded retry with exponential back-off per action.
- 3-plane security separation: Cognitive → Control → Tool.
- `INVARIANTS` constant as a non-negotiable checklist.

---

## 0. Core Types

```typescript
type Route =
  | "direct"
  | "tool"
  | "rag"
  | "mixed";

type RiskLevel =
  | "low"
  | "medium"
  | "high"
  | "critical";

type RuntimeStatus =
  | "running"
  | "pending_auth"
  | "completed"
  | "failed"
  | "cancelled";

type ToolResult =
  | {
      ok: true;
      toolId: string;
      data: unknown;
      latencyMs: number;
    }
  | {
      ok: false;
      toolId: string;
      error: {
        code: string;
        message: string;
        retryable: boolean;
      };
    };

interface AgentRequest {
  requestId: string;
  sessionId: string;
  input?: string;
  resume?: {
    authorizationId: string;
    decision: "approve" | "deny";
  };
}

interface AgentState {
  sessionId: string;
  executionId: string;
  status: RuntimeStatus;
  iteration: number;
  maxIterations: number;
  context: ContextState;
  plan?: ExecutionPlan;
  pendingAuthorization?: PendingAuthorization;
  results: ToolResult[];
  createdAt: number;
  updatedAt: number;
}

interface ContextState {
  userInput: string;
  messages: Message[];
  retrievedData: unknown[];
  critiques: Critique[];
  constraints: Constraint[];
}

interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

interface Constraint {
  type: string;
  value: unknown;
}

interface Critique {
  source: string;
  code: string;
  message: string;
  severity: "info" | "warning" | "error";
}
```

---

## 1. Tool Contract

Every tool must have an explicit contract.
The runtime never executes an arbitrary function — only a registered `ToolDefinition`.

```typescript
interface ToolDefinition<TInput = unknown, TOutput = unknown> {
  id: string;
  version: string;
  description: string;
  inputSchema: JSONSchema;
  outputSchema: JSONSchema;
  capabilities: Capability[];
  riskLevel: RiskLevel;
  execute(
    input: TInput,
    context: ToolExecutionContext
  ): Promise<TOutput>;
}

interface ToolExecutionContext {
  executionId: string;
  sessionId: string;
  authorizationToken?: string;
  signal: AbortSignal;
}

// Lookup only — no dynamic registration at runtime:
ToolRegistry.has(toolId)
ToolRegistry.get(toolId)
```

---

## 2. Structured Execution Plan

The planner does not return prose. It returns a machine-validatable plan.

```typescript
interface ExecutionPlan {
  planId: string;
  version: string;
  actions: PlanAction[];
}

interface PlanAction {
  id: string;
  toolId: string;
  arguments: Record<string, unknown>;
  dependsOn: string[];
  riskLevel: RiskLevel;
  requiresAuthorization: boolean;
  timeoutMs: number;
  retryPolicy: RetryPolicy;
}

interface RetryPolicy {
  maxAttempts: number;
  backoffMs: number;
  exponential: boolean;
}
```

**Example plan:**

```json
{
  "planId": "plan_001",
  "version": "2.0",
  "actions": [
    {
      "id": "search_1",
      "toolId": "web.search",
      "arguments": { "query": "latest AI runtime research" },
      "dependsOn": [],
      "riskLevel": "low",
      "requiresAuthorization": false,
      "timeoutMs": 5000,
      "retryPolicy": {
        "maxAttempts": 2,
        "backoffMs": 200,
        "exponential": true
      }
    }
  ]
}
```

---

## 3. Main Runtime Entry Point

```typescript
async function AgentRuntime(
  request: AgentRequest
): Promise<RuntimeResponse> {

  const state = await LoadOrCreateState(request);

  // RESUME PATH — hydrates exact saved batch, never replans
  if (request.resume) {
    return await ResumeExecution(state, request.resume);
  }

  // NORMAL EXECUTION
  state.context.userInput = request.input ?? "";
  await PersistState(state);
  return await RunAgentLoop(state);
}
```

---

## 4. Resume / HITL Hydration Path

This is the critical addition over V1: approval resumes the **exact pending batch**,
not a replanned or regenerated version.

```typescript
async function ResumeExecution(
  state: AgentState,
  resume: AgentRequest["resume"]
): Promise<RuntimeResponse> {

  if (!state.pendingAuthorization) {
    return {
      status: "failed",
      error: { code: "NO_PENDING_AUTH", message: "No authorization request is waiting." }
    };
  }

  const pending = state.pendingAuthorization;

  if (pending.authorizationId !== resume!.authorizationId) {
    return {
      status: "failed",
      error: { code: "AUTHORIZATION_MISMATCH", message: "Authorization does not match pending action." }
    };
  }

  if (resume!.decision === "deny") {
    state.status = "cancelled";
    state.pendingAuthorization = undefined;
    await PersistState(state);
    return { status: "cancelled" };
  }

  // AUTHORIZED: execute the exact saved batch
  //
  // CRITICAL — do NOT:
  //   • Re-run the planner
  //   • Regenerate arguments
  //   • Create a new plan
  //
  const batch = pending.batch;
  state.pendingAuthorization = undefined;
  await PersistState(state);

  const results = await ExecuteAuthorizedBatch(state, batch, pending.authorizationId);
  state.results.push(...results);
  await PersistState(state);

  return await ContinueAfterToolExecution(state);
}
```

**Suspend/Resume flow:**

```
PLAN
  ↓
AUTH REQUIRED
  ↓
SAVE EXACT STATE
  ↓
RETURN pending_auth
  ↓
USER APPROVES
  ↓
LOAD STATE
  ↓
VERIFY AUTHORIZATION
  ↓
EXECUTE SAVED BATCH
  ↓
CONTINUE
```

---

## 5. Agent Loop

```typescript
async function RunAgentLoop(
  state: AgentState
): Promise<RuntimeResponse> {

  while (state.iteration < state.maxIterations) {
    state.iteration++;
    await PersistState(state);

    // ── ROUTING ──────────────────────────────────────────────
    const route = await ClassifyRoute(state.context);

    // ── RISK ─────────────────────────────────────────────────
    const risk = await ScoreRisk(state.context);

    // ── POLICY ───────────────────────────────────────────────
    const policy = await EvaluatePolicy(state.context, risk);
    if (policy.denied) return await FailSafely(state, policy.reason);

    // ── DIRECT PATH ──────────────────────────────────────────
    if (route === "direct") {
      const answer = await GenerateDirectAnswer(state.context);
      const verdict = await VerifyDraft(answer, state.context);
      if (verdict.pass) return await Complete(state, answer);
      AppendCritique(state, verdict.critique);
      continue;
    }

    // ── PLANNING ─────────────────────────────────────────────
    const plan = await PlanStructuredActions(state.context, route, risk);
    const planValidation = ValidateExecutionPlan(plan);
    if (!planValidation.valid) {
      AppendCritique(state, planValidation.error);
      continue;
    }
    state.plan = plan;

    // ── DAG EXECUTION ────────────────────────────────────────
    const batches = TopologicallyBatch(plan.actions);

    for (const batch of batches) {

      // Input validation
      const inputs = GenerateToolInputs(batch, state.context);
      const inputValidation = ValidateToolInputs(inputs, batch);
      if (!inputValidation.valid) {
        state.results.push({
          ok: false,
          toolId: inputValidation.toolId,
          error: { code: "INVALID_TOOL_INPUT", message: inputValidation.error, retryable: true }
        });
        continue;
      }

      // Authorization check
      const authorization = await CheckAuthorization(batch, risk);
      if (authorization.required) return await SuspendForAuthorization(state, batch);

      // Parallel execution
      const results = await ExecuteBatchParallel(batch, inputs);
      state.results.push(...results);
      await PersistState(state);

      // Output validation — never silently discard
      const checked = ValidateToolOutputs(results);
      state.results.push(...checked.errors);
    }

    // ── RESULT CRITIC (before synthesis) ─────────────────────
    const resultCritique = await CriticReviewResults(state.context, state.plan, state.results);
    if (!resultCritique.pass) {
      AppendCritique(state, resultCritique.critique);
      continue;
    }

    // ── FINAL SYNTHESIS + VERIFY ─────────────────────────────
    const draft = await SynthesizeFinalAnswer(state.context, state.results);
    const finalCheck = await VerifyDraft(draft, state.context);
    if (finalCheck.pass) return await Complete(state, draft);
    AppendCritique(state, finalCheck.critique);
  }

  return await FailSafely(state, "MAX_ITERATIONS_REACHED");
}
```

---

## 6. DAG Scheduler

Build the DAG first. Never infer dependencies during execution.

```typescript
function TopologicallyBatch(
  actions: PlanAction[]
): PlanAction[][] {

  const remaining = new Set(actions);
  const completed = new Set<string>();
  const batches: PlanAction[][] = [];

  while (remaining.size > 0) {
    const ready = [...remaining].filter(action =>
      action.dependsOn.every(dep => completed.has(dep))
    );

    if (ready.length === 0) {
      throw new Error("CYCLIC_OR_INVALID_DEPENDENCY_GRAPH");
    }

    batches.push(ready);
    for (const action of ready) {
      remaining.delete(action);
      completed.add(action.id);
    }
  }

  return batches;
}
```

**Example:** Given the DAG `A → {B, C} → D`:

```
Batch 1: [A]
Batch 2: [B, C]   ← parallel
Batch 3: [D]
```

---

## 7. Parallel Tool Execution

```typescript
async function ExecuteBatchParallel(
  batch: PlanAction[],
  inputs: ToolInput[]
): Promise<ToolResult[]> {

  return await Promise.all(
    batch.map(async action => {
      const input = inputs.find(x => x.actionId === action.id);
      return await ExecuteWithRetry(action, input);
    })
  );
}
```

---

## 8. Bounded Retry with Back-off

```typescript
async function ExecuteWithRetry(
  action: PlanAction,
  input: ToolInput
): Promise<ToolResult> {

  const tool = ToolRegistry.get(action.toolId);
  let attempt = 0;

  while (attempt < action.retryPolicy.maxAttempts) {
    attempt++;
    try {
      const output = await ExecuteWithTimeout(tool, input, action.timeoutMs);
      return { ok: true, toolId: action.toolId, data: output, latencyMs: 0 };
    } catch (error) {
      if (!IsRetryable(error)) {
        return {
          ok: false, toolId: action.toolId,
          error: { code: "TOOL_EXECUTION_FAILED", message: String(error), retryable: false }
        };
      }
      if (attempt >= action.retryPolicy.maxAttempts) break;
      await Sleep(CalculateBackoff(action.retryPolicy, attempt));
    }
  }

  return {
    ok: false, toolId: action.toolId,
    error: { code: "RETRY_LIMIT_REACHED", message: "Tool failed after maximum attempts.", retryable: false }
  };
}
```

---

## 9. Authorization Suspension

```typescript
async function SuspendForAuthorization(
  state: AgentState,
  batch: PlanAction[]
): Promise<RuntimeResponse> {

  const authorizationId = GenerateSecureId();

  state.status = "pending_auth";
  state.pendingAuthorization = {
    authorizationId,
    batch,
    createdAt: Date.now(),
    expiresAt: Date.now() + AUTHORIZATION_TTL
  };

  await PersistState(state);

  return {
    status: "tool_call_pending_authorization",
    authorizationId,
    actions: DescribeAuthorization(batch),
    expiresAt: state.pendingAuthorization.expiresAt
  };
}
```

**Frontend response shape:**

```json
{
  "status": "tool_call_pending_authorization",
  "authorizationId": "auth_91x...",
  "actions": [{ "tool": "send_email", "description": "Send email to..." }]
}
```

The backend does not block while waiting.

---

## 10. Authorization Security Verification

Authorization must bind to the **exact saved action** — not a prose description of intent.

```typescript
async function VerifyAuthorization(
  state: AgentState,
  authorizationId: string
): Promise<boolean> {

  const pending = state.pendingAuthorization;
  if (!pending)                                return false;
  if (pending.authorizationId !== authorizationId) return false;
  if (Date.now() > pending.expiresAt)          return false;
  if (pending.sessionId && pending.sessionId !== state.sessionId) return false;
  return true;
}
```

The token must bind to:

```
authorization_id  +  execution_id  +  plan_id
+  action_ids  +  tool_versions  +  validated_arguments
```

Not just: `"User approved sending email."`

---

## 11. Input Validation

```typescript
function ValidateToolInputs(
  inputs: ToolInput[],
  actions: PlanAction[]
): ValidationResult {

  for (const input of inputs) {
    const action = actions.find(a => a.id === input.actionId);
    if (!action) return { valid: false, error: "UNKNOWN_ACTION", toolId: input.toolId };

    const tool = ToolRegistry.get(action.toolId);
    const valid = JSONSchemaValidate(tool.inputSchema, input.arguments);
    if (!valid) return { valid: false, error: "TOOL_ARGUMENT_SCHEMA_INVALID", toolId: tool.id };
  }

  return { valid: true };
}
```

---

## 12. Output Validation

Never silently discard bad results.

```typescript
function ValidateToolOutputs(results: ToolResult[]) {
  const valid: ToolResult[] = [];
  const errors: ToolResult[] = [];

  for (const result of results) {
    if (!result.ok) { errors.push(result); continue; }

    const tool = ToolRegistry.get(result.toolId);
    const validOutput = JSONSchemaValidate(tool.outputSchema, result.data);

    if (!validOutput) {
      errors.push({
        ok: false, toolId: result.toolId,
        error: {
          code: "INVALID_TOOL_OUTPUT",
          message: "Tool returned data that does not match its declared output schema.",
          retryable: true
        }
      });
      continue;
    }

    valid.push(result);
  }

  return { valid, errors };
}
```

---

## 13. Result Critic (runs before synthesis)

```typescript
async function CriticReviewResults(
  context: ContextState,
  plan: ExecutionPlan,
  results: ToolResult[]
): Promise<CriticResult> {

  const failures = results.filter(r => !r.ok);
  if (failures.length > 0) {
    return {
      pass: false,
      critique: {
        source: "tool_validation", code: "TOOL_FAILURE",
        message: "One or more required tool operations failed.", severity: "error"
      }
    };
  }

  const consistency = CheckCrossToolConsistency(results);
  if (!consistency.valid) {
    return {
      pass: false,
      critique: {
        source: "result_critic", code: "RESULT_CONFLICT",
        message: consistency.reason, severity: "error"
      }
    };
  }

  return { pass: true };
}
```

---

## 14. Self-Correction Loop

Failures feed back as structured critiques, not generic fallbacks:

```
Failure  →  Structured error  →  Context  →  Planner  →  New plan  →  New execution
```

**Example:**

```
Iteration 1:
  search_database → INVALID_TOOL_INPUT → Critic → "Missing required field: user_id"

Iteration 2:
  Planner sees: "Missing required field: user_id"
  → generate corrected arguments → validate → execute
```

---

## 15. Minimal Context Management

Do not send the entire history to the model on every iteration.

```typescript
interface ContextBudget {
  systemTokens: number;
  userTokens: number;
  relevantMemoryTokens: number;
  toolResultTokens: number;
  critiqueTokens: number;
}

function BuildMinimalContext(state: AgentState): ContextState {
  return {
    userInput:     state.context.userInput,
    messages:      SelectRelevantMessages(state.context.messages),
    retrievedData: SelectRelevantResults(state.results),
    critiques:     SelectRecentCritiques(state.context.critiques),
    constraints:   state.context.constraints
  };
}
```

This is one of the major latency optimizations in the loop.

---

## 16. Runtime State Machine

```
┌───────────────┐
│    CREATED    │
└───────┬───────┘
        ↓
┌───────────────┐
│    RUNNING    │
└───────┬───────┘
        │
   ┌────┼──────────┐
   ↓    ↓          ↓
COMPLETE PENDING_AUTH  FAILED
         │
         ↓
     APPROVED
         │
         ↓
     RUNNING
```

Execution state lives outside the process → runtime is horizontally scalable.

---

## 17. Fast-Path Routing

The fastest execution is the one you don't execute.

```typescript
if (cacheHit)             return cache;
if (highConfidenceDirect) return direct;
if (singleCheapTool)      executeSingleTool();
if (independentTools)     executeParallel();
if (complexDAG)           executeDAG();
```

**Full routing diagram:**

```
REQUEST
   ↓
Ultra-Fast Router
   ├── DIRECT
   ├── CACHE
   └── AGENT → PLAN → DAG → PARALLEL TOOLS
          └────────────────────────────────→ VERIFY → ANSWER
```

---

## 18. 3-Plane Security Architecture

The LLM must never be the security boundary.
The policy engine, validator, authorization system, and sandbox remain authoritative
even if the model is fully compromised by prompt injection.

```
┌──────────────────────┐
│    COGNITIVE PLANE   │  Router · Planner · Critic · Synthesizer
└──────────┬───────────┘
           │  POLICY BOUNDARY
┌──────────▼───────────┐
│     CONTROL PLANE    │  Auth · Permissions · Schema validation
│                      │  Rate limits · Sandboxing · Audit logs · Resource limits
└──────────┬───────────┘
           │  EXECUTION BOUNDARY
┌──────────▼───────────┐
│      TOOL PLANE      │  Web · Database · Filesystem · APIs · Code sandbox
└──────────────────────┘
```

---

## 19. Non-Negotiable Invariants

```typescript
const INVARIANTS = {
  INPUTS_VALIDATED:              true,
  OUTPUTS_VALIDATED:             true,
  PLAN_SCHEMA_VALIDATED:         true,
  TOOL_ALLOWLIST_ENFORCED:       true,
  AUTHORIZATION_STATEFUL:        true,
  DEPENDENCIES_DAG_VALIDATED:    true,
  PARALLEL_EXECUTION_BOUNDED:    true,
  RETRIES_BOUNDED:               true,
  ITERATIONS_BOUNDED:            true,
  TOOL_FAILURES_EXPLICIT:        true,
  RESULT_CRITIC_BEFORE_SYNTHESIS: true,
  FINAL_OUTPUT_VERIFIED:         true,
  MEMORY_PERMISSIONED:           true,
  MODEL_NOT_SECURITY_BOUNDARY:   true,
};
```

---

## 20. Full Execution Pipeline

```
INPUT
  ↓
FAST ROUTER
  ↓
RISK ENGINE
  ↓
POLICY GATE
  ↓
PLANNER
  ↓
PLAN VALIDATOR
  ↓
DAG SCHEDULER
  ↓
INPUT GENERATOR
  ↓
INPUT VALIDATOR
  ↓
AUTHORIZATION ──── DENIED → STOP
  ↓ APPROVED
PARALLEL EXECUTION
  ↓
OUTPUT VALIDATOR
  ↓
RESULT CRITIC
  ├── FAILURE → SELF-CORRECT → (back to PLANNER)
  └── PASS
       ↓
   SYNTHESIZE
       ↓
   FINAL VERIFY
       ↓
   RESPONSE
```
