// lion_agent/src/runtime/tool_registry.rs — Tool Contract & Registry
//
// ToolDefinition is the contract every tool must satisfy.
// ToolRegistry is the authoritative allowlist — only registered tools execute.
// No arbitrary function pointer is ever called outside of this registry.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;
use crate::runtime::types::{RiskLevel, ToolResult, unix_now_ms};
use crate::runtime::error::{RuntimeError, RuntimeResult};

// ── Tool Execution Context ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ToolExecutionContext {
    pub execution_id:        String,
    pub session_id:          String,
    pub authorization_token: Option<String>,
}

// ── Tool Definition Trait ─────────────────────────────────────────────────────

#[async_trait]
pub trait ToolDefinition: Send + Sync {
    fn id(&self) -> &str;
    fn version(&self) -> &str;
    fn description(&self) -> &str;
    fn risk_level(&self) -> RiskLevel;

    /// JSON Schema for the input arguments (simplified: just validate key presence).
    fn required_input_fields(&self) -> Vec<&'static str> { vec![] }

    /// Validate an input Value against the tool's schema.
    fn validate_input(&self, input: &Value) -> Result<(), String> {
        let obj = input.as_object()
            .ok_or_else(|| "Input must be a JSON object".to_string())?;
        for field in self.required_input_fields() {
            if !obj.contains_key(field) {
                return Err(format!("Missing required field: {}", field));
            }
        }
        Ok(())
    }

    /// Validate an output Value against the tool's schema.
    fn validate_output(&self, _output: &Value) -> Result<(), String> { Ok(()) }

    /// Execute the tool. Returns raw JSON output or an error string.
    async fn execute(
        &self,
        input: &Value,
        ctx: &ToolExecutionContext,
    ) -> Result<Value, String>;
}

// ── Tool Registry ─────────────────────────────────────────────────────────────

/// The authoritative tool allowlist. Immutable after construction.
/// Invariant: TOOL_ALLOWLIST_ENFORCED — only registered tools are ever called.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn ToolDefinition>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: Arc<dyn ToolDefinition>) {
        self.tools.insert(tool.id().to_string(), tool);
    }

    pub fn has(&self, tool_id: &str) -> bool {
        self.tools.contains_key(tool_id)
    }

    pub fn get(&self, tool_id: &str) -> RuntimeResult<&Arc<dyn ToolDefinition>> {
        self.tools.get(tool_id)
            .ok_or_else(|| RuntimeError::ToolNotFound { tool_id: tool_id.to_string() })
    }

    pub fn tool_ids(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}

// ── Execute With Timeout ──────────────────────────────────────────────────────

/// Executes a tool with a per-action timeout.
/// Returns a ToolResult — never panics.
pub async fn execute_with_timeout(
    tool: &Arc<dyn ToolDefinition>,
    input: &Value,
    ctx: &ToolExecutionContext,
    timeout_ms: u64,
) -> ToolResult {
    let start = unix_now_ms();
    let timeout = tokio::time::Duration::from_millis(timeout_ms);

    match tokio::time::timeout(timeout, tool.execute(input, ctx)).await {
        Ok(Ok(data)) => {
            let elapsed = unix_now_ms() - start;
            ToolResult::success(tool.id(), data, elapsed)
        }
        Ok(Err(msg)) => {
            ToolResult::failure(tool.id(), "TOOL_EXECUTION_FAILED", &msg, true)
        }
        Err(_elapsed) => {
            ToolResult::failure(
                tool.id(),
                "TOOL_TIMEOUT",
                &format!("Tool timed out after {}ms", timeout_ms),
                true,
            )
        }
    }
}

// ── Built-in: Math Tool ───────────────────────────────────────────────────────

/// Simple expression evaluator — kept as the default registered tool.
pub struct MathTool;

#[async_trait]
impl ToolDefinition for MathTool {
    fn id(&self) -> &str { "math.eval" }
    fn version(&self) -> &str { "1.0.0" }
    fn description(&self) -> &str { "Evaluate a mathematical expression" }
    fn risk_level(&self) -> RiskLevel { RiskLevel::Low }
    fn required_input_fields(&self) -> Vec<&'static str> { vec!["expression"] }

    async fn execute(&self, input: &Value, _ctx: &ToolExecutionContext) -> Result<Value, String> {
        let expr = input["expression"]
            .as_str()
            .ok_or("expression must be a string")?;
        let result = evalexpr::eval(expr)
            .map_err(|e| format!("eval error: {}", e))?;
        Ok(Value::String(result.to_string()))
    }
}

/// Echo tool for testing — low risk, reflects input as output.
pub struct EchoTool;

#[async_trait]
impl ToolDefinition for EchoTool {
    fn id(&self) -> &str { "echo" }
    fn version(&self) -> &str { "1.0.0" }
    fn description(&self) -> &str { "Echo input back as output" }
    fn risk_level(&self) -> RiskLevel { RiskLevel::Low }

    async fn execute(&self, input: &Value, _ctx: &ToolExecutionContext) -> Result<Value, String> {
        Ok(input.clone())
    }
}
