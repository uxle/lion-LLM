// lion_agent/src/runtime/critic.rs — Result Critic & Self-Correction
//
// Invariant: RESULT_CRITIC_BEFORE_SYNTHESIS
// The critic runs before expensive LLM synthesis.
// Failures produce Critique structs — never silent fallbacks.

use crate::runtime::types::{
    AgentState, Critique, CritiqueSeverity, ExecutionPlan, ToolResult,
};

// ── Critic Result ─────────────────────────────────────────────────────────────

pub struct CriticResult {
    pub pass:    bool,
    pub critique: Option<Critique>,
}

impl CriticResult {
    pub fn pass() -> Self { Self { pass: true, critique: None } }
    pub fn fail(source: &str, code: &str, message: &str) -> Self {
        Self {
            pass: false,
            critique: Some(Critique {
                source:   source.to_string(),
                code:     code.to_string(),
                message:  message.to_string(),
                severity: CritiqueSeverity::Error,
            }),
        }
    }
}

// ── Result Critic ─────────────────────────────────────────────────────────────

/// Review tool results before synthesis begins.
///
/// Invariant: RESULT_CRITIC_BEFORE_SYNTHESIS
/// Checks:
///   1. No failed tool results in the batch
///   2. Cross-tool consistency (basic heuristic — extend with domain logic)
pub fn critic_review_results(
    _plan: &ExecutionPlan,
    results: &[ToolResult],
) -> CriticResult {
    // Check 1: any failures?
    let failures: Vec<_> = results.iter().filter(|r| !r.ok).collect();
    if !failures.is_empty() {
        let codes: Vec<_> = failures.iter()
            .filter_map(|r| r.error.as_ref())
            .map(|e| e.code.as_str())
            .collect();
        return CriticResult::fail(
            "tool_validation",
            "TOOL_FAILURE",
            &format!(
                "{} tool(s) failed: [{}]",
                failures.len(),
                codes.join(", ")
            ),
        );
    }

    // Check 2: cross-tool consistency (placeholder — extend with real logic)
    // For now: pass if all tools succeeded
    CriticResult::pass()
}

// ── Append Critique to State ──────────────────────────────────────────────────

/// Append a critique to the agent's context so the next planner iteration
/// can see the structured error and generate a corrected plan.
///
/// This is the Self-Correction path:
///   Failure → Structured error → Context → Planner → New plan → New execution
pub fn append_critique(state: &mut AgentState, critique: Critique) {
    state.context.critiques.push(critique);
    state.touch();
}

// ── Context Budget / Minimal Context ─────────────────────────────────────────

/// Trim the context to avoid sending the full history on every iteration.
///
/// Invariant: ITERATIONS_BOUNDED (indirectly — avoids token explosion)
pub fn build_minimal_context_summary(state: &AgentState) -> String {
    let user_input = &state.context.user_input;

    // Last 3 critiques only
    let recent_critiques: Vec<String> = state.context.critiques
        .iter()
        .rev()
        .take(3)
        .map(|c| format!("[{}] {}: {}", c.code, c.source, c.message))
        .collect();

    // Last 5 successful results
    let recent_results: Vec<String> = state.results
        .iter()
        .filter(|r| r.ok)
        .rev()
        .take(5)
        .map(|r| format!("{} → {}", r.tool_id,
            r.data.as_ref().map(|d| d.to_string()).unwrap_or_default()))
        .collect();

    let mut summary = format!("Task: {}\n", user_input);
    if !recent_critiques.is_empty() {
        summary.push_str(&format!("Recent errors:\n  {}\n", recent_critiques.join("\n  ")));
    }
    if !recent_results.is_empty() {
        summary.push_str(&format!("Recent results:\n  {}\n", recent_results.join("\n  ")));
    }
    summary.push_str(&format!("Iteration: {}/{}\n", state.iteration, state.max_iterations));

    summary
}
