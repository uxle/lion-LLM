// lion_agent/src/runtime/mod.rs — Agent Runtime V2 Module Root

pub mod auth;
pub mod critic;
pub mod dag;
pub mod engine;
pub mod error;
pub mod executor;
pub mod tool_registry;
pub mod types;
pub mod validation;

pub use engine::{AgentRuntime, RuntimeConfig};
pub use error::{RuntimeError, RuntimeResult};
pub use types::{
    AgentRequest, AgentState, AuthDecision, Critique, CritiqueSeverity,
    ContextState, ExecutionPlan, PlanAction, RetryPolicy, ResumeDecision,
    RiskLevel, Route, RuntimeResponse, RuntimeStatus, ToolResult,
};
pub use tool_registry::{ToolDefinition, ToolRegistry, MathTool, EchoTool};
