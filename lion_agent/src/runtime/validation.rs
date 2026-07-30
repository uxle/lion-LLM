// lion_agent/src/runtime/validation.rs — Input & Output Validation
//
// Invariants: INPUTS_VALIDATED, OUTPUTS_VALIDATED
// Bad results are never silently discarded.

use std::sync::Arc;
use serde_json::Value;
use crate::runtime::types::{PlanAction, ToolResult};
use crate::runtime::tool_registry::ToolRegistry;

// ── Input Validation ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct InputValidationResult {
    pub valid:   bool,
    pub tool_id: String,
    pub error:   Option<String>,
}

impl InputValidationResult {
    pub fn ok() -> Self {
        Self { valid: true, tool_id: String::new(), error: None }
    }

    pub fn fail(tool_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self { valid: false, tool_id: tool_id.into(), error: Some(error.into()) }
    }
}

/// Validate inputs for a batch of actions against their registered tool schemas.
/// Returns the first validation failure, or Ok if all pass.
///
/// Invariant: INPUTS_VALIDATED — this runs before any execution.
pub fn validate_tool_inputs(
    actions: &[PlanAction],
    registry: &ToolRegistry,
) -> InputValidationResult {
    for action in actions {
        let tool = match registry.get(&action.tool_id) {
            Ok(t) => t,
            Err(_) => return InputValidationResult::fail(
                &action.tool_id, format!("Tool '{}' not in registry", action.tool_id)
            ),
        };

        let args_value = Value::Object(
            action.arguments.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        );

        if let Err(reason) = tool.validate_input(&args_value) {
            return InputValidationResult::fail(&action.tool_id, reason);
        }
    }

    InputValidationResult::ok()
}

// ── Output Validation ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct OutputValidationResult {
    pub valid:  Vec<ToolResult>,
    pub errors: Vec<ToolResult>,
}

/// Validate outputs from a batch execution.
/// Failed results are moved to `errors`. Schema-invalid outputs become error results.
///
/// Invariant: OUTPUTS_VALIDATED — bad results are never silently discarded.
pub fn validate_tool_outputs(
    results: Vec<ToolResult>,
    registry: &ToolRegistry,
) -> OutputValidationResult {
    let mut valid  = Vec::new();
    let mut errors = Vec::new();

    for result in results {
        if !result.ok {
            errors.push(result);
            continue;
        }

        // Validate output schema if tool is registered
        if let Ok(tool) = registry.get(&result.tool_id) {
            if let Some(data) = &result.data {
                if let Err(reason) = tool.validate_output(data) {
                    errors.push(ToolResult::failure(
                        &result.tool_id,
                        "INVALID_TOOL_OUTPUT",
                        &reason,
                        true,
                    ));
                    continue;
                }
            }
        }

        valid.push(result);
    }

    OutputValidationResult { valid, errors }
}
