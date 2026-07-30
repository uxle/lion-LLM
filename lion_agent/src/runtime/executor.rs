// lion_agent/src/runtime/executor.rs — Parallel Tool Execution & Bounded Retry
//
// Invariants: PARALLEL_EXECUTION_BOUNDED, RETRIES_BOUNDED, TOOL_FAILURES_EXPLICIT

use std::sync::Arc;
use serde_json::Value;
use tokio::task::JoinSet;
use crate::runtime::types::{PlanAction, ToolResult, RetryPolicy, unix_now_ms};
use crate::runtime::tool_registry::{ToolRegistry, ToolExecutionContext, execute_with_timeout};

// ── Parallel Batch Execution ──────────────────────────────────────────────────

/// Execute a batch of independent actions in parallel.
/// Every action is wrapped in ExecuteWithRetry — no action can stall the batch forever.
///
/// Invariant: PARALLEL_EXECUTION_BOUNDED
pub async fn execute_batch_parallel(
    batch: &[PlanAction],
    registry: &Arc<ToolRegistry>,
    ctx: &ToolExecutionContext,
) -> Vec<ToolResult> {
    let mut join_set: JoinSet<ToolResult> = JoinSet::new();

    for action in batch {
        let action = action.clone();
        let registry = Arc::clone(registry);
        let ctx = ctx.clone();

        join_set.spawn(async move {
            execute_with_retry(&action, &registry, &ctx).await
        });
    }

    let mut results = Vec::new();
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(tool_result) => results.push(tool_result),
            Err(join_err) => {
                results.push(ToolResult::failure(
                    "unknown",
                    "TASK_JOIN_ERROR",
                    &join_err.to_string(),
                    false,
                ));
            }
        }
    }

    results
}

// ── Bounded Retry With Exponential Back-off ───────────────────────────────────

/// Execute a single action with bounded retries and exponential backoff.
///
/// Invariant: RETRIES_BOUNDED — maxAttempts is always enforced.
/// Invariant: TOOL_FAILURES_EXPLICIT — errors surface as ToolResult, never panic.
pub async fn execute_with_retry(
    action: &PlanAction,
    registry: &ToolRegistry,
    ctx: &ToolExecutionContext,
) -> ToolResult {
    let tool = match registry.get(&action.tool_id) {
        Ok(t) => t.clone(),
        Err(_) => {
            return ToolResult::failure(
                &action.tool_id,
                "TOOL_NOT_FOUND",
                &format!("Tool '{}' not in registry", action.tool_id),
                false,
            );
        }
    };

    let args_value = Value::Object(
        action.arguments.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    );

    let max = action.retry_policy.max_attempts.max(1);
    let mut attempt = 0u32;

    loop {
        attempt += 1;
        let result = execute_with_timeout(&tool, &args_value, ctx, action.timeout_ms).await;

        if result.ok {
            return result;
        }

        // Non-retryable or limit reached
        let retryable = result.error.as_ref().map(|e| e.retryable).unwrap_or(false);
        if !retryable || attempt >= max {
            if attempt >= max {
                return ToolResult::failure(
                    &action.tool_id,
                    "RETRY_LIMIT_REACHED",
                    &format!("Tool failed after {} attempts", attempt),
                    false,
                );
            }
            return result;
        }

        // Back-off before next attempt
        let delay = calculate_backoff(&action.retry_policy, attempt);
        tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
    }
}

// ── Back-off Calculator ───────────────────────────────────────────────────────

/// Calculate the delay before the next retry attempt.
/// Exponential: delay = backoff_ms * 2^(attempt - 1), capped at 30s.
pub fn calculate_backoff(policy: &RetryPolicy, attempt: u32) -> u64 {
    if !policy.exponential {
        return policy.backoff_ms;
    }
    let multiplier = 2u64.saturating_pow(attempt.saturating_sub(1));
    let delay = policy.backoff_ms.saturating_mul(multiplier);
    delay.min(30_000) // cap at 30 seconds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_linear() {
        let p = RetryPolicy { max_attempts: 3, backoff_ms: 100, exponential: false };
        assert_eq!(calculate_backoff(&p, 1), 100);
        assert_eq!(calculate_backoff(&p, 2), 100);
        assert_eq!(calculate_backoff(&p, 3), 100);
    }

    #[test]
    fn test_backoff_exponential() {
        let p = RetryPolicy { max_attempts: 5, backoff_ms: 100, exponential: true };
        assert_eq!(calculate_backoff(&p, 1), 100);   // 100 * 2^0
        assert_eq!(calculate_backoff(&p, 2), 200);   // 100 * 2^1
        assert_eq!(calculate_backoff(&p, 3), 400);   // 100 * 2^2
        assert_eq!(calculate_backoff(&p, 4), 800);   // 100 * 2^3
    }

    #[test]
    fn test_backoff_cap() {
        let p = RetryPolicy { max_attempts: 10, backoff_ms: 1000, exponential: true };
        // 1000 * 2^30 would overflow — must be capped at 30s
        assert_eq!(calculate_backoff(&p, 30), 30_000);
    }
}
