// lion_agent/src/runtime/types.rs — Agent Runtime V2 Core Types
//
// Rust translation of the TypeScript spec.
// All types are Serialize/Deserialize so AgentState can be persisted externally.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ── Enumerations ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    Direct,
    Tool,
    Rag,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Running,
    PendingAuth,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

// ── Messages & Context ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role:    MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub r#type: String,
    pub value:  serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Critique {
    pub source:   String,
    pub code:     String,
    pub message:  String,
    pub severity: CritiqueSeverity,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextState {
    pub user_input:     String,
    pub messages:       Vec<Message>,
    pub retrieved_data: Vec<serde_json::Value>,
    pub critiques:      Vec<Critique>,
    pub constraints:    Vec<Constraint>,
}

// ── Tool Results ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    pub code:      String,
    pub message:   String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub ok:         bool,
    pub tool_id:    String,
    pub data:       Option<serde_json::Value>,
    pub error:      Option<ToolError>,
    pub latency_ms: u64,
}

impl ToolResult {
    pub fn success(tool_id: impl Into<String>, data: serde_json::Value, latency_ms: u64) -> Self {
        Self { ok: true, tool_id: tool_id.into(), data: Some(data), error: None, latency_ms }
    }

    pub fn failure(tool_id: impl Into<String>, code: &str, message: &str, retryable: bool) -> Self {
        Self {
            ok: false,
            tool_id: tool_id.into(),
            data: None,
            latency_ms: 0,
            error: Some(ToolError { code: code.to_string(), message: message.to_string(), retryable }),
        }
    }
}

// ── Execution Plan ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms:   u64,
    pub exponential:  bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { max_attempts: 2, backoff_ms: 200, exponential: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanAction {
    pub id:                    String,
    pub tool_id:               String,
    pub arguments:             HashMap<String, serde_json::Value>,
    pub depends_on:            Vec<String>,
    pub risk_level:            RiskLevel,
    pub requires_authorization: bool,
    pub timeout_ms:            u64,
    pub retry_policy:          RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub plan_id: String,
    pub version: String,
    pub actions: Vec<PlanAction>,
}

// ── Pending Authorization ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingAuthorization {
    pub authorization_id: String,
    pub batch:            Vec<PlanAction>,
    pub created_at:       u64,
    pub expires_at:       u64,
    pub session_id:       String,
}

// ── Agent Request / State ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeDecision {
    pub authorization_id: String,
    pub decision:         AuthDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthDecision {
    Approve,
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    pub request_id: String,
    pub session_id: String,
    pub input:      Option<String>,
    pub resume:     Option<ResumeDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id:    String,
    pub execution_id:  String,
    pub status:        RuntimeStatus,
    pub iteration:     u32,
    pub max_iterations: u32,
    pub context:       ContextState,
    pub plan:          Option<ExecutionPlan>,
    pub pending_authorization: Option<PendingAuthorization>,
    pub results:       Vec<ToolResult>,
    pub created_at:    u64,
    pub updated_at:    u64,
}

impl AgentState {
    pub fn new(session_id: String, execution_id: String, max_iterations: u32) -> Self {
        let now = unix_now();
        Self {
            session_id,
            execution_id,
            status: RuntimeStatus::Running,
            iteration: 0,
            max_iterations,
            context: ContextState::default(),
            plan: None,
            pending_authorization: None,
            results: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = unix_now();
    }
}

// ── Runtime Response ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RuntimeResponse {
    Completed {
        answer: String,
        execution_id: String,
    },
    PendingAuthorization {
        authorization_id: String,
        actions:          Vec<AuthActionDescription>,
        expires_at:       u64,
    },
    Failed {
        code:    String,
        message: String,
    },
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthActionDescription {
    pub tool:        String,
    pub description: String,
    pub risk_level:  RiskLevel,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
