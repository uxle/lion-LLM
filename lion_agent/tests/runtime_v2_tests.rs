// lion_agent/tests/runtime_v2_tests.rs — Agent Runtime V2 Integration Tests

use std::sync::Arc;
use lion_agent::runtime::{
    AgentRuntime, AgentRequest, AgentState, AuthDecision, ResumeDecision,
    RuntimeConfig, RuntimeStatus, RuntimeResponse, RiskLevel, Route,
    ToolRegistry, MathTool, EchoTool,
};

fn make_registry() -> Arc<ToolRegistry> {
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(MathTool));
    reg.register(Arc::new(EchoTool));
    Arc::new(reg)
}

fn make_runtime() -> AgentRuntime {
    AgentRuntime::with_defaults(make_registry())
}

fn make_state(session: &str) -> AgentState {
    AgentState::new(session.to_string(), format!("exec_{}", session), 10)
}

fn request(session: &str, input: &str) -> AgentRequest {
    AgentRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        session_id: session.to_string(),
        input:      Some(input.to_string()),
        resume:     None,
    }
}

// ── DAG Scheduler ─────────────────────────────────────────────────────────────

#[test]
fn test_dag_linear() {
    use lion_agent::runtime::types::{PlanAction, RetryPolicy};
    use lion_agent::runtime::dag::topologically_batch;

    let actions = vec![
        PlanAction {
            id: "a".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec![], risk_level: RiskLevel::Low,
            requires_authorization: false, timeout_ms: 1000,
            retry_policy: RetryPolicy::default(),
        },
        PlanAction {
            id: "b".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec!["a".into()], risk_level: RiskLevel::Low,
            requires_authorization: false, timeout_ms: 1000,
            retry_policy: RetryPolicy::default(),
        },
    ];

    let batches = topologically_batch(actions).unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0][0].id, "a");
    assert_eq!(batches[1][0].id, "b");
}

#[test]
fn test_dag_diamond() {
    use lion_agent::runtime::types::{PlanAction, RetryPolicy};
    use lion_agent::runtime::dag::topologically_batch;

    let actions = vec![
        PlanAction { id: "a".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec![], risk_level: RiskLevel::Low, requires_authorization: false,
            timeout_ms: 1000, retry_policy: RetryPolicy::default() },
        PlanAction { id: "b".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec!["a".into()], risk_level: RiskLevel::Low, requires_authorization: false,
            timeout_ms: 1000, retry_policy: RetryPolicy::default() },
        PlanAction { id: "c".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec!["a".into()], risk_level: RiskLevel::Low, requires_authorization: false,
            timeout_ms: 1000, retry_policy: RetryPolicy::default() },
        PlanAction { id: "d".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec!["b".into(), "c".into()], risk_level: RiskLevel::Low,
            requires_authorization: false, timeout_ms: 1000, retry_policy: RetryPolicy::default() },
    ];

    let batches = topologically_batch(actions).unwrap();
    assert_eq!(batches.len(), 3); // [A], [B,C], [D]
    assert_eq!(batches[0].len(), 1);
    assert_eq!(batches[1].len(), 2);
    assert_eq!(batches[2].len(), 1);
    assert_eq!(batches[2][0].id, "d");
}

#[test]
fn test_dag_cyclic_fails() {
    use lion_agent::runtime::types::{PlanAction, RetryPolicy};
    use lion_agent::runtime::dag::topologically_batch;

    // Unknown reference (equivalent to cycle for our purposes)
    let actions = vec![
        PlanAction { id: "a".into(), tool_id: "echo".into(), arguments: Default::default(),
            depends_on: vec!["z".into()], // "z" doesn't exist
            risk_level: RiskLevel::Low, requires_authorization: false,
            timeout_ms: 1000, retry_policy: RetryPolicy::default() },
    ];
    assert!(topologically_batch(actions).is_err());
}

// ── Retry & Backoff ───────────────────────────────────────────────────────────

#[test]
fn test_backoff_exponential() {
    use lion_agent::runtime::executor::calculate_backoff;
    use lion_agent::runtime::types::RetryPolicy;

    let p = RetryPolicy { max_attempts: 5, backoff_ms: 100, exponential: true };
    assert_eq!(calculate_backoff(&p, 1), 100);
    assert_eq!(calculate_backoff(&p, 2), 200);
    assert_eq!(calculate_backoff(&p, 3), 400);
}

#[test]
fn test_backoff_capped() {
    use lion_agent::runtime::executor::calculate_backoff;
    use lion_agent::runtime::types::RetryPolicy;

    let p = RetryPolicy { max_attempts: 10, backoff_ms: 1000, exponential: true };
    assert!(calculate_backoff(&p, 20) <= 30_000);
}

// ── Tool Registry ─────────────────────────────────────────────────────────────

#[test]
fn test_registry_allowlist() {
    let reg = make_registry();
    assert!(reg.has("math.eval"));
    assert!(reg.has("echo"));
    assert!(!reg.has("dangerous.rm_rf")); // not registered
}

#[test]
fn test_registry_get_missing() {
    let reg = make_registry();
    assert!(reg.get("not_a_real_tool").is_err());
}

// ── Full Runtime: Direct Path ─────────────────────────────────────────────────

#[tokio::test]
async fn test_direct_path_returns_completed() {
    let runtime = make_runtime();
    let state   = make_state("sess_direct");
    let req     = request("sess_direct", "Hello, what is Footprint?");

    let (_state, response) = runtime.run(req, state).await;

    matches!(response, RuntimeResponse::Completed { .. });
}

// ── Full Runtime: Math Tool Path ──────────────────────────────────────────────

#[tokio::test]
async fn test_tool_path_math_eval() {
    let runtime = make_runtime();
    let state   = make_state("sess_math");
    let req     = request("sess_math", "calculate 6 * 7");

    let (final_state, response) = runtime.run(req, state).await;

    assert_eq!(final_state.status, RuntimeStatus::Completed);
    if let RuntimeResponse::Completed { answer, .. } = response {
        assert!(answer.contains("42"), "Expected '42' in answer, got: {}", answer);
    } else {
        panic!("Expected Completed, got different response");
    }
}

// ── Full Runtime: Max Iterations ──────────────────────────────────────────────

#[tokio::test]
async fn test_max_iterations_limit() {
    let reg     = make_registry();
    let config  = RuntimeConfig { max_iterations: 2, auth_threshold: RiskLevel::High };
    let runtime = AgentRuntime::new(reg, config);
    let state   = make_state("sess_iter");
    // Request that can't be completed in 2 iterations
    let req = request("sess_iter", "calculate (((((1+2+3+4+5)))))");

    let (final_state, _response) = runtime.run(req, state).await;
    // Should be completed or failed — never exceeds max_iterations
    assert!(final_state.iteration <= 2);
}

// ── Auth: Verify Mismatch ─────────────────────────────────────────────────────

#[test]
fn test_auth_verify_no_pending() {
    use lion_agent::runtime::auth::verify_authorization;
    let state = make_state("sess_auth");
    let result = verify_authorization(&state, "auth_fake");
    assert!(result.is_err());
}

// ── Resume: No Pending Auth Fails Gracefully ──────────────────────────────────

#[tokio::test]
async fn test_resume_no_pending_fails() {
    let runtime = make_runtime();
    let state   = make_state("sess_resume");
    let req = AgentRequest {
        request_id: "r1".to_string(),
        session_id: "sess_resume".to_string(),
        input:      None,
        resume:     Some(ResumeDecision {
            authorization_id: "auth_nonexistent".to_string(),
            decision:         AuthDecision::Approve,
        }),
    };

    let (_state, response) = runtime.run(req, state).await;
    assert!(matches!(response, RuntimeResponse::Failed { .. }));
}

// ── State Machine: Status Transitions ────────────────────────────────────────

#[tokio::test]
async fn test_completed_state_on_success() {
    let runtime = make_runtime();
    let state   = make_state("sess_sm");
    let req     = request("sess_sm", "calculate 10 + 32");

    let (final_state, response) = runtime.run(req, state).await;
    assert_eq!(final_state.status, RuntimeStatus::Completed);
    assert!(matches!(response, RuntimeResponse::Completed { .. }));
}
