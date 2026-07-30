// lion_agent/src/runtime/error.rs — Runtime V2 Error Types

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("No pending authorization in state")]
    NoPendingAuth,

    #[error("Authorization ID mismatch: expected {expected}, got {got}")]
    AuthorizationMismatch { expected: String, got: String },

    #[error("Authorization expired at {expires_at}")]
    AuthorizationExpired { expires_at: u64 },

    #[error("Session mismatch in authorization")]
    SessionMismatch,

    #[error("Cyclic or invalid dependency graph in plan")]
    CyclicDependencyGraph,

    #[error("Tool '{tool_id}' not found in registry")]
    ToolNotFound { tool_id: String },

    #[error("Tool input validation failed for '{tool_id}': {reason}")]
    InputValidationFailed { tool_id: String, reason: String },

    #[error("Tool output validation failed for '{tool_id}': {reason}")]
    OutputValidationFailed { tool_id: String, reason: String },

    #[error("Tool '{tool_id}' timed out after {timeout_ms}ms")]
    ToolTimeout { tool_id: String, timeout_ms: u64 },

    #[error("Tool '{tool_id}' failed after {attempts} attempts")]
    RetryLimitReached { tool_id: String, attempts: u32 },

    #[error("Policy denied execution: {reason}")]
    PolicyDenied { reason: String },

    #[error("Max iterations reached ({max})")]
    MaxIterationsReached { max: u32 },

    #[error("Plan validation failed: {reason}")]
    PlanValidationFailed { reason: String },

    #[error("State persistence error: {0}")]
    StatePersistence(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;
