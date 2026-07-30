// lion_agent/src/runtime/engine.rs — Agent Runtime V2 State Machine
//
// This is the main entry point: AgentRuntime.
// It implements:
//   • Normal execution path (RunAgentLoop)
//   • Resume/HITL path (ResumeExecution)
//   • State machine: Running → PendingAuth → Running | Completed | Failed | Cancelled
//
// Invariants enforced here:
//   ITERATIONS_BOUNDED, PLAN_SCHEMA_VALIDATED, RESULT_CRITIC_BEFORE_SYNTHESIS,
//   FINAL_OUTPUT_VERIFIED, AUTHORIZATION_STATEFUL, MODEL_NOT_SECURITY_BOUNDARY

use std::sync::Arc;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::runtime::types::{
    AgentRequest, AgentState, AuthDecision, Critique, CritiqueSeverity, ExecutionPlan,
    PlanAction, RetryPolicy, RiskLevel, Route, RuntimeResponse, RuntimeStatus,
    ToolResult, unix_now,
};
use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::tool_registry::{ToolRegistry, ToolExecutionContext};
use crate::runtime::dag::{topologically_batch, validate_plan_dag};
use crate::runtime::auth::{
    suspend_for_authorization, verify_authorization,
    requires_authorization, batch_max_risk,
};
use crate::runtime::executor::execute_batch_parallel;
use crate::runtime::validation::{validate_tool_inputs, validate_tool_outputs};
use crate::runtime::critic::{critic_review_results, append_critique, build_minimal_context_summary};

// ── Runtime Config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_iterations: u32,
    /// Risk level at or above which authorization is always required.
    pub auth_threshold: RiskLevel,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { max_iterations: 10, auth_threshold: RiskLevel::High }
    }
}

// ── Agent Runtime ─────────────────────────────────────────────────────────────

pub struct AgentRuntime {
    registry: Arc<ToolRegistry>,
    config:   RuntimeConfig,
}

impl AgentRuntime {
    pub fn new(registry: Arc<ToolRegistry>, config: RuntimeConfig) -> Self {
        Self { registry, config }
    }

    pub fn with_defaults(registry: Arc<ToolRegistry>) -> Self {
        Self::new(registry, RuntimeConfig::default())
    }

    // ── Main Entry Point ───────────────────────────────────────────────────────

    pub async fn run(
        &self,
        request: AgentRequest,
        mut state: AgentState,
    ) -> (AgentState, RuntimeResponse) {
        // RESUME PATH — takes priority over normal execution
        if let Some(resume) = request.resume {
            let response = self.resume_execution(&mut state, resume).await;
            return (state, response);
        }

        // NORMAL EXECUTION
        state.context.user_input = request.input.unwrap_or_default();
        state.touch();

        let response = self.run_agent_loop(&mut state).await;
        (state, response)
    }

    // ── Resume / HITL Path ─────────────────────────────────────────────────────
    //
    // CRITICAL: This path MUST NOT re-run the planner.
    //           MUST NOT regenerate arguments.
    //           MUST NOT create a new plan.
    //           It MUST execute the exact saved batch.

    async fn resume_execution(
        &self,
        state: &mut AgentState,
        resume: crate::runtime::types::ResumeDecision,
    ) -> RuntimeResponse {
        info!("Resuming execution with authorization_id={}", resume.authorization_id);

        // 1. Verify the pending authorization exists and is valid
        if let Err(e) = verify_authorization(state, &resume.authorization_id) {
            warn!("Authorization verification failed: {}", e);
            return RuntimeResponse::Failed {
                code:    "AUTHORIZATION_FAILED".to_string(),
                message: e.to_string(),
            };
        }

        // 2. Denied? Cancel cleanly.
        if resume.decision == AuthDecision::Deny {
            state.status = RuntimeStatus::Cancelled;
            state.pending_authorization = None;
            state.touch();
            info!("Authorization denied — execution cancelled");
            return RuntimeResponse::Cancelled;
        }

        // 3. Approved — load the EXACT saved batch (never re-plan)
        let batch = state.pending_authorization
            .as_ref()
            .map(|p| p.batch.clone())
            .unwrap_or_default();

        state.pending_authorization = None;
        state.status = RuntimeStatus::Running;
        state.touch();

        info!("Authorization approved — executing saved batch ({} actions)", batch.len());

        // 4. Execute the exact saved batch
        let ctx = self.make_ctx(state);
        let results = execute_batch_parallel(&batch, &self.registry, &ctx).await;
        state.results.extend(results);
        state.touch();

        // 5. Continue the agent loop from where it left off
        self.continue_after_tool_execution(state).await
    }

    // ── Agent Loop ─────────────────────────────────────────────────────────────

    async fn run_agent_loop(&self, state: &mut AgentState) -> RuntimeResponse {
        // Invariant: ITERATIONS_BOUNDED
        while state.iteration < state.max_iterations {
            state.iteration += 1;
            state.touch();
            debug!("Iteration {}/{}", state.iteration, state.max_iterations);

            // ── ROUTING ───────────────────────────────────────────────────────
            let route = self.classify_route(state);
            debug!("Route: {:?}", route);

            // ── RISK ──────────────────────────────────────────────────────────
            let risk = self.score_risk(state);
            debug!("Risk: {:?}", risk);

            // ── POLICY GATE ───────────────────────────────────────────────────
            // Invariant: MODEL_NOT_SECURITY_BOUNDARY — policy is checked here,
            // outside the cognitive (LLM) layer.
            if risk == RiskLevel::Critical {
                warn!("Critical risk detected — policy gate denied");
                return self.fail(state, "POLICY_DENIED", "Critical risk level — execution blocked");
            }

            // ── DIRECT PATH ───────────────────────────────────────────────────
            if route == Route::Direct {
                let answer = self.generate_direct_answer(state);
                // Invariant: FINAL_OUTPUT_VERIFIED
                if self.verify_draft(&answer, state) {
                    return self.complete(state, answer);
                }
                append_critique(state, Critique {
                    source:   "verifier".to_string(),
                    code:     "VERIFICATION_FAILED".to_string(),
                    message:  "Direct answer failed verification".to_string(),
                    severity: CritiqueSeverity::Warning,
                });
                continue;
            }

            // ── PLANNING ──────────────────────────────────────────────────────
            let plan = match self.plan_structured_actions(state, &route, &risk) {
                Some(p) => p,
                None => {
                    append_critique(state, Critique {
                        source:   "planner".to_string(),
                        code:     "PLANNING_FAILED".to_string(),
                        message:  "Could not generate a valid execution plan".to_string(),
                        severity: CritiqueSeverity::Error,
                    });
                    continue;
                }
            };

            // ── PLAN VALIDATION ───────────────────────────────────────────────
            // Invariant: PLAN_SCHEMA_VALIDATED
            if let Err(reason) = validate_plan_dag(&plan.actions) {
                append_critique(state, Critique {
                    source:   "plan_validator".to_string(),
                    code:     "PLAN_INVALID".to_string(),
                    message:  reason,
                    severity: CritiqueSeverity::Error,
                });
                continue;
            }

            state.plan = Some(plan.clone());

            // ── DAG SCHEDULING ────────────────────────────────────────────────
            // Invariant: DEPENDENCIES_DAG_VALIDATED — DAG built before execution
            let batches = match topologically_batch(plan.actions.clone()) {
                Ok(b) => b,
                Err(_) => {
                    append_critique(state, Critique {
                        source:   "dag_scheduler".to_string(),
                        code:     "CYCLIC_DAG".to_string(),
                        message:  "Cyclic dependency detected in plan".to_string(),
                        severity: CritiqueSeverity::Error,
                    });
                    continue;
                }
            };

            let mut batch_failed = false;

            for batch in batches {
                // ── INPUT VALIDATION ─────────────────────────────────────────
                // Invariant: INPUTS_VALIDATED
                let iv = validate_tool_inputs(&batch, &self.registry);
                if !iv.valid {
                    let msg = iv.error.unwrap_or_default();
                    state.results.push(ToolResult::failure(
                        &iv.tool_id, "INVALID_TOOL_INPUT", &msg, true,
                    ));
                    append_critique(state, Critique {
                        source:   "input_validator".to_string(),
                        code:     "INVALID_TOOL_INPUT".to_string(),
                        message:  msg,
                        severity: CritiqueSeverity::Error,
                    });
                    batch_failed = true;
                    break;
                }

                // ── AUTHORIZATION ─────────────────────────────────────────────
                // Invariant: AUTHORIZATION_STATEFUL
                let max_risk = batch_max_risk(&batch);
                if requires_authorization(&batch, &max_risk) {
                    return suspend_for_authorization(state, batch);
                }

                // ── PARALLEL EXECUTION ────────────────────────────────────────
                // Invariant: PARALLEL_EXECUTION_BOUNDED
                let ctx = self.make_ctx(state);
                let results = execute_batch_parallel(&batch, &self.registry, &ctx).await;
                state.results.extend(results.clone());
                state.touch();

                // ── OUTPUT VALIDATION ─────────────────────────────────────────
                // Invariant: OUTPUTS_VALIDATED — errors surface, never silently dropped
                let validated = validate_tool_outputs(results, &self.registry);
                if !validated.errors.is_empty() {
                    let err_codes: Vec<_> = validated.errors.iter()
                        .filter_map(|r| r.error.as_ref())
                        .map(|e| e.code.as_str())
                        .collect();
                    append_critique(state, Critique {
                        source:   "output_validator".to_string(),
                        code:     "OUTPUT_VALIDATION_FAILED".to_string(),
                        message:  format!("Output errors: [{}]", err_codes.join(", ")),
                        severity: CritiqueSeverity::Error,
                    });
                    state.results.extend(validated.errors);
                    batch_failed = true;
                    break;
                }
            }

            if batch_failed { continue; }

            // ── RESULT CRITIC (before synthesis) ──────────────────────────────
            // Invariant: RESULT_CRITIC_BEFORE_SYNTHESIS
            let plan_ref = state.plan.as_ref().unwrap();
            let critic = critic_review_results(plan_ref, &state.results);
            if !critic.pass {
                if let Some(c) = critic.critique {
                    append_critique(state, c);
                }
                continue; // Self-correction: back to planning with structured error
            }

            // ── SYNTHESIS + FINAL VERIFICATION ───────────────────────────────
            let draft = self.synthesize_answer(state);
            // Invariant: FINAL_OUTPUT_VERIFIED
            if self.verify_draft(&draft, state) {
                return self.complete(state, draft);
            }

            append_critique(state, Critique {
                source:   "final_verifier".to_string(),
                code:     "FINAL_VERIFICATION_FAILED".to_string(),
                message:  "Synthesized answer failed final verification".to_string(),
                severity: CritiqueSeverity::Warning,
            });
        }

        // Invariant: ITERATIONS_BOUNDED
        self.fail(state, "MAX_ITERATIONS_REACHED",
            &format!("Stopped after {} iterations", state.max_iterations))
    }

    // ── Continue After Tool Execution (resume path) ────────────────────────────

    async fn continue_after_tool_execution(&self, state: &mut AgentState) -> RuntimeResponse {
        // Run the critic and synthesis pass on accumulated results
        if let Some(plan) = state.plan.clone() {
            let critic = critic_review_results(&plan, &state.results);
            if !critic.pass {
                if let Some(c) = critic.critique {
                    append_critique(state, c);
                }
                // Re-enter the loop for self-correction
                return self.run_agent_loop(state).await;
            }
        }

        let draft = self.synthesize_answer(state);
        if self.verify_draft(&draft, state) {
            self.complete(state, draft)
        } else {
            self.run_agent_loop(state).await
        }
    }

    // ── Cognitive Plane Stubs ─────────────────────────────────────────────────
    //
    // These are the Cognitive Plane components.
    // In production, they call the LLM (lion_brain::OllamaClient).
    // The LLM is NEVER the security boundary — that's the Control Plane above.

    fn classify_route(&self, state: &AgentState) -> Route {
        let input = &state.context.user_input;
        // Simple heuristic: use tools if the input mentions action words
        let action_words = ["calculate", "compute", "search", "find", "get", "fetch",
                            "send", "create", "delete", "update", "list"];
        if action_words.iter().any(|w| input.to_lowercase().contains(w)) {
            Route::Tool
        } else {
            Route::Direct
        }
    }

    fn score_risk(&self, state: &AgentState) -> RiskLevel {
        let input = &state.context.user_input;
        let danger_words = ["delete", "drop", "remove", "send email", "payment", "transfer"];
        if danger_words.iter().any(|w| input.to_lowercase().contains(w)) {
            RiskLevel::High
        } else {
            RiskLevel::Low
        }
    }

    fn plan_structured_actions(
        &self,
        state: &AgentState,
        _route: &Route,
        _risk: &RiskLevel,
    ) -> Option<ExecutionPlan> {
        // Minimal planner: look for math expressions in the input
        let input = &state.context.user_input;

        // Extract anything that looks like a math expression
        let expr = extract_math_expression(input)?;

        let plan_id = format!("plan_{}", state.iteration);
        Some(ExecutionPlan {
            plan_id: plan_id.clone(),
            version: "2.0".to_string(),
            actions: vec![
                PlanAction {
                    id:                    "step_1".to_string(),
                    tool_id:               "math.eval".to_string(),
                    arguments:             [("expression".to_string(),
                                            Value::String(expr))].into(),
                    depends_on:            vec![],
                    risk_level:            RiskLevel::Low,
                    requires_authorization: false,
                    timeout_ms:            5_000,
                    retry_policy:          RetryPolicy::default(),
                }
            ],
        })
    }

    fn generate_direct_answer(&self, state: &AgentState) -> String {
        let summary = build_minimal_context_summary(state);
        format!("Direct answer for: {}", state.context.user_input.trim())
    }

    fn synthesize_answer(&self, state: &AgentState) -> String {
        let results: Vec<String> = state.results.iter()
            .filter(|r| r.ok)
            .filter_map(|r| r.data.as_ref())
            .map(|d| d.to_string())
            .collect();

        if results.is_empty() {
            format!("Completed: {}", state.context.user_input)
        } else {
            results.join(", ")
        }
    }

    fn verify_draft(&self, draft: &str, _state: &AgentState) -> bool {
        // Invariant: FINAL_OUTPUT_VERIFIED
        !draft.is_empty()
    }

    // ── Terminal States ───────────────────────────────────────────────────────

    fn complete(&self, state: &mut AgentState, answer: String) -> RuntimeResponse {
        state.status = RuntimeStatus::Completed;
        state.touch();
        info!("Execution completed: execution_id={}", state.execution_id);
        RuntimeResponse::Completed {
            answer,
            execution_id: state.execution_id.clone(),
        }
    }

    fn fail(&self, state: &mut AgentState, code: &str, message: &str) -> RuntimeResponse {
        state.status = RuntimeStatus::Failed;
        state.touch();
        warn!("Execution failed: {} — {}", code, message);
        RuntimeResponse::Failed {
            code:    code.to_string(),
            message: message.to_string(),
        }
    }

    fn make_ctx(&self, state: &AgentState) -> ToolExecutionContext {
        ToolExecutionContext {
            execution_id:        state.execution_id.clone(),
            session_id:          state.session_id.clone(),
            authorization_token: None,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_math_expression(input: &str) -> Option<String> {
    // Look for patterns like "2 + 2", "sqrt(16)", "10 * 5", etc.
    // Simple heuristic: if input contains math operators, use the whole trimmed input.
    let math_chars = ['+', '-', '*', '/', '^', '(', ')'];
    let digits_present = input.chars().any(|c| c.is_ascii_digit());
    let ops_present = input.chars().any(|c| math_chars.contains(&c));

    // Also handle "calculate X" or "compute X" patterns
    let cleaned = input.trim()
        .trim_start_matches("calculate")
        .trim_start_matches("compute")
        .trim_start_matches("what is")
        .trim_start_matches("eval")
        .trim()
        .to_string();

    if !cleaned.is_empty() && (digits_present || ops_present) {
        Some(cleaned)
    } else {
        None
    }
}
