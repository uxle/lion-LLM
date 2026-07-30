# Orchestration Runtime — Merged Specification

Comprehensive runtime specification for agent routing, risk scoring, policy validation, DAG dependency batching, input schema verification, authorization suspension/resume, and critic review.

## 0. Core Types

```typescript
type Route = "direct" | "tool" | "rag" | "mixed";
type RiskLevel = "low" | "medium" | "high" | "critical";
type RuntimeStatus = "created" | "running" | "pending_auth" | "completed" | "failed" | "cancelled";

type ToolResult =
  | { ok: true; toolId: string; data: unknown; latencyMs: number }
  | { ok: false; toolId: string; error: { code: string; message: string; retryable: boolean } };

interface AgentRequest {
  requestId: string;
  sessionId: string;
  input?: string;
  resume?: { authorizationId: string; decision: "approve" | "deny" };
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
  memory: MemoryFact[];
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

interface Message { role: "system" | "user" | "assistant" | "tool"; content: string; }
interface Constraint { type: string; value: unknown; }
interface Critique { source: string; code: string; message: string; severity: "info" | "warning" | "error"; }
interface MemoryFact { content: string; timestamp: number; sourceSession: string; }
```

## 1. Tool & Plan Contracts

```typescript
interface ToolDefinition<TInput = unknown, TOutput = unknown> {
  id: string;
  version: string;
  description: string;
  inputSchema: JSONSchema;
  outputSchema: JSONSchema;
  requiredRole: string;
  isDestructive: boolean;
  accessesSensitiveData: boolean;
  riskLevel: RiskLevel;
  execute(input: TInput, context: ToolExecutionContext): Promise<TOutput>;
}

interface ToolExecutionContext {
  executionId: string;
  sessionId: string;
  authorizationToken?: string;
  signal: AbortSignal;
}

interface PlanAction {
  id: string;
  toolId: string;
  arguments: Record<string, unknown>;
  argsSchema: JSONSchema;
  dependsOn: string[];
  riskLevel: RiskLevel;
  requiresAuthorization: boolean;
  timeoutMs: number;
  retryPolicy: RetryPolicy;
}

interface RetryPolicy { maxAttempts: number; backoffMs: number; exponential: boolean; }
interface ExecutionPlan { planId: string; version: string; actions: PlanAction[]; }
```

## 2. Fast Routing

```typescript
function ClassifyRoute(context: ContextState): Route {
  const input = context.userInput.toLowerCase();

  if (MatchesRegex(input, /^(hi|hello|thanks|bye)/)) return "direct";
  if (MatchesRegex(input, /what is \d+ [\+\-\*\/] \d+/)) return "direct";
  if (ContainsKeywords(input, ["search", "fetch", "weather", "stock", "latest"])) return "tool";
  if (ContainsKeywords(input, ["in my files", "from the manual", "policy doc"])) return "rag";

  const systemPrompt = "Classify intent into: [direct, tool, rag, mixed]. Output only the word.";
  return FastLLM(systemPrompt, input) as Route;
}
```

## 3. Risk Scoring

```typescript
function ScoreRisk(context: ContextState): RiskLevel {
  const input = context.userInput;

  if (ContainsKeywords(input, ["ignore previous instructions", "system prompt", "bypass"])) {
    return "high"; // prompt-injection signal
  }
  if (ContainsKeywords(input, ["transfer money", "diagnose", "sue", "delete"])) {
    return "high"; // sensitive-domain signal
  }
  if (ContainsPII(input)) return "medium";

  return "low";
}
```

## 4. Policy Gate (RBAC + Confirmation)

```typescript
function PolicyCheckBatch(
  batch: PlanAction[],
  risk: RiskLevel,
  userSession: Session
): PolicyDecision {
  for (const action of batch) {
    const tool = ToolRegistry.get(action.toolId);

    if (!tool) {
      return { blocked: true, requiresConfirmation: false, reason: `Tool ${action.toolId} not found or disabled.` };
    }
    if (!userSession.roles.includes(tool.requiredRole)) {
      return { blocked: true, requiresConfirmation: false, reason: `Unauthorized access to ${action.toolId}.` };
    }
    if ((tool.isDestructive || tool.accessesSensitiveData) && (risk === "high" || risk === "critical")) {
      return { blocked: false, requiresConfirmation: true, requiredAction: `approve_tool_call:${action.toolId}` };
    }
  }
  return { blocked: false, requiresConfirmation: false };
}
```

## 5. Topological DAG Batching (O(V+E))

```typescript
function TopologicallyBatch(actions: PlanAction[]): PlanAction[][] {
  const inDegree = new Map<string, number>();
  const dependents = new Map<string, string[]>();
  const actionMap = new Map<string, PlanAction>();

  for (const action of actions) {
    actionMap.set(action.id, action);
    inDegree.set(action.id, 0);
    if (!dependents.has(action.id)) dependents.set(action.id, []);
  }

  for (const action of actions) {
    for (const depId of action.dependsOn) {
      if (!actionMap.has(depId)) throw new Error(`Missing dependency: ${depId}`);
      dependents.get(depId)!.push(action.id);
      inDegree.set(action.id, inDegree.get(action.id)! + 1);
    }
  }

  const batches: PlanAction[][] = [];
  let processed = 0;

  while (processed < actions.length) {
    const ready = [...inDegree.entries()]
      .filter(([, degree]) => degree === 0)
      .map(([id]) => actionMap.get(id)!);

    if (ready.length === 0) {
      throw new Error("CYCLIC_OR_INVALID_DEPENDENCY_GRAPH");
    }

    batches.push(ready);
    for (const action of ready) {
      inDegree.set(action.id, -1);
      for (const depId of dependents.get(action.id) ?? []) {
        inDegree.set(depId, inDegree.get(depId)! - 1);
      }
    }
    processed += ready.length;
  }

  return batches;
}
```

## 6. Strict Input Schema Validation

```typescript
function ValidateToolInputs(inputs: Record<string, unknown>[], batch: PlanAction[]): ValidationResult {
  for (let i = 0; i < batch.length; i++) {
    const action = batch[i];
    const args = inputs[i];
    const schema = action.argsSchema;

    for (const required of schema.required ?? []) {
      if (!(required in args)) {
        return { valid: false, toolId: action.toolId, error: `Missing required argument: ${required}` };
      }
    }
    for (const key in args) {
      if (!(key in schema.properties)) {
        return { valid: false, toolId: action.toolId, error: `Unexpected argument: ${key}` };
      }
      const expected = schema.properties[key].type;
      const actual = typeof args[key];
      if (actual !== expected) {
        return { valid: false, toolId: action.toolId, error: `Type mismatch for ${key}: expected ${expected}, got ${actual}` };
      }
    }
  }
  return { valid: true };
}
```

## 7. Parallel Execution & Retry with Latency Tracking

```typescript
async function ExecuteBatchParallel(batch: PlanAction[], inputs: Record<string, unknown>[]): Promise<ToolResult[]> {
  return Promise.all(
    batch.map((action, i) => ExecuteWithRetry(action, inputs[i]))
  );
}

async function ExecuteWithRetry(action: PlanAction, input: Record<string, unknown>): Promise<ToolResult> {
  const tool = ToolRegistry.get(action.toolId);
  let attempt = 0;

  while (attempt < action.retryPolicy.maxAttempts) {
    attempt++;
    const started = Date.now();
    try {
      const output = await ExecuteWithTimeout(tool, input, action.timeoutMs);
      return { ok: true, toolId: action.toolId, data: output, latencyMs: Date.now() - started };
    } catch (error) {
      if (!IsRetryable(error)) {
        return { ok: false, toolId: action.toolId, error: { code: "TOOL_EXECUTION_FAILED", message: String(error), retryable: false } };
      }
      if (attempt >= action.retryPolicy.maxAttempts) break;
      await Sleep(CalculateBackoff(action.retryPolicy, attempt));
    }
  }

  return { ok: false, toolId: action.toolId, error: { code: "RETRY_LIMIT_REACHED", message: "Tool failed after maximum attempts.", retryable: false } };
}
```

## 8. Stateful Authorization (HITL)

```typescript
interface PendingAuthorization {
  authorizationId: string;
  batch: PlanAction[];
  createdAt: number;
  expiresAt: number;
}

async function SuspendForAuthorization(state: AgentState, batch: PlanAction[]): Promise<RuntimeResponse> {
  const authorizationId = GenerateSecureId();
  state.status = "pending_auth";
  state.pendingAuthorization = { authorizationId, batch, createdAt: Date.now(), expiresAt: Date.now() + AUTHORIZATION_TTL };
  await PersistState(state);

  return {
    status: "tool_call_pending_authorization",
    authorizationId,
    actions: DescribeAuthorization(batch),
    expiresAt: state.pendingAuthorization.expiresAt
  };
}

function VerifyAuthorization(state: AgentState, authorizationId: string): boolean {
  const pending = state.pendingAuthorization;
  if (!pending) return false;
  if (pending.authorizationId !== authorizationId) return false;
  if (Date.now() > pending.expiresAt) return false;
  return true;
}

async function ResumeExecution(state: AgentState, resume: AgentRequest["resume"]): Promise<RuntimeResponse> {
  if (!state.pendingAuthorization) {
    return { status: "failed", error: { code: "NO_PENDING_AUTH", message: "No authorization request is waiting." } };
  }
  if (!VerifyAuthorization(state, resume!.authorizationId)) {
    return { status: "failed", error: { code: "AUTHORIZATION_MISMATCH", message: "Authorization does not match or has expired." } };
  }
  if (resume!.decision === "deny") {
    state.status = "cancelled";
    state.pendingAuthorization = undefined;
    await PersistState(state);
    return { status: "cancelled" };
  }

  const batch = state.pendingAuthorization.batch;
  state.pendingAuthorization = undefined;
  await PersistState(state);

  const results = await ExecuteBatchParallel(batch, GenerateToolInputs(batch, state.context));
  state.results.push(...results);
  await PersistState(state);
  return ContinueAfterToolExecution(state);
}
```

## 9. Pre-Synthesis Critic Review

```typescript
async function CriticReview(context: ContextState, results: ToolResult[]): Promise<CriticResult> {
  const failed = results.filter(r => !r.ok);
  if (failed.length > 0) {
    return { pass: false, feedback: `Execution failed for: ${failed.map(f => f.toolId).join(", ")}. Fix inputs and retry.` };
  }

  const empty = results.filter(r => r.ok && (r.data == null || (Array.isArray(r.data) && r.data.length === 0)));
  if (empty.length > 0) {
    return { pass: false, feedback: `${empty.map(e => e.toolId).join(", ")} returned no data. Broaden search or adjust parameters.` };
  }

  const criticPrompt = `User asked: "${context.userInput}"\nResults: ${JSON.stringify(results)}\nIs there enough data to answer fully without hallucinating? Output PASS or FAIL: [reason].`;
  const verdict = await FastLLM(criticPrompt);
  if (verdict.startsWith("FAIL")) return { pass: false, feedback: verdict };

  return { pass: true };
}
```

## 10. Risk-Gated Memory Extraction

```typescript
function FinalizeMemoryIfAllowed(state: AgentState, risk: RiskLevel, finalDraft: string): void {
  if (risk === "high" || risk === "critical") return;

  const rawFacts = LLM_ExtractJSON(`
    Extract permanent user preferences or facts from this conversation as a JSON array of strings.
    Do not extract temporary/contextual data.
  `);

  const safeFacts = rawFacts.filter(fact => !ContainsPII(fact) && fact.length <= 150);

  for (const fact of safeFacts) {
    state.memory.push({ content: fact, timestamp: Date.now(), sourceSession: state.sessionId });
  }
}
```

## 11. Security Architecture

```
┌──────────────────────┐
│   COGNITIVE PLANE    │   Router, Planner, Critic, Synthesizer
└──────────┬───────────┘
           │  policy boundary
┌──────────▼───────────┐
│    CONTROL PLANE     │   Auth, RBAC, schema validation, rate limits, sandboxing, audit logs
└──────────┬───────────┘
           │  execution boundary
┌──────────▼───────────┐
│     TOOL PLANE       │   Web, database, filesystem, external APIs, code sandbox
└──────────────────────┘
```

The load-bearing security principle: **The model is never the security boundary.** Auth, RBAC, schema validation, and sandboxing all reside in the Control Plane.
