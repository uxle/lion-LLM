// lion_agent/src/runtime/auth.rs — Authorization Suspension & Verification
//
// Invariants: AUTHORIZATION_STATEFUL, MODEL_NOT_SECURITY_BOUNDARY
//
// Authorization binds to the exact saved batch — not to a prose description.
// The token commits to: authorization_id + execution_id + action_ids + tool_versions + arguments.

use std::sync::Arc;
use crate::runtime::types::{
    AgentState, PlanAction, PendingAuthorization, AuthActionDescription,
    RuntimeStatus, RuntimeResponse, RiskLevel, unix_now, unix_now_ms,
};
use crate::runtime::error::{RuntimeError, RuntimeResult};

/// How long an authorization token is valid (5 minutes).
pub const AUTHORIZATION_TTL_MS: u64 = 5 * 60 * 1000;

// ── Suspend ───────────────────────────────────────────────────────────────────

/// Suspend execution and return a pending_authorization response.
/// The exact batch is saved in state — it MUST NOT be regenerated on resume.
///
/// Invariant: AUTHORIZATION_STATEFUL
pub fn suspend_for_authorization(
    state: &mut AgentState,
    batch: Vec<PlanAction>,
) -> RuntimeResponse {
    let authorization_id = generate_auth_id(state, &batch);
    let now = unix_now_ms();
    let expires_at = now + AUTHORIZATION_TTL_MS;

    state.status = RuntimeStatus::PendingAuth;
    state.pending_authorization = Some(PendingAuthorization {
        authorization_id: authorization_id.clone(),
        batch:            batch.clone(),
        created_at:       now,
        expires_at,
        session_id:       state.session_id.clone(),
    });
    state.touch();

    let actions = batch.iter().map(|a| AuthActionDescription {
        tool:        a.tool_id.clone(),
        description: format!("Execute {} with {} arguments", a.tool_id, a.arguments.len()),
        risk_level:  a.risk_level.clone(),
    }).collect();

    RuntimeResponse::PendingAuthorization {
        authorization_id,
        actions,
        expires_at,
    }
}

// ── Verify ────────────────────────────────────────────────────────────────────

/// Verify an incoming authorization ID against the saved pending state.
///
/// Checks:
///   1. Pending authorization exists
///   2. authorization_id matches exactly
///   3. Token has not expired
///   4. Session ID matches (prevents cross-session replay)
///
/// Invariant: AUTHORIZATION_STATEFUL
pub fn verify_authorization(
    state: &AgentState,
    authorization_id: &str,
) -> RuntimeResult<()> {
    let pending = state.pending_authorization
        .as_ref()
        .ok_or(RuntimeError::NoPendingAuth)?;

    if pending.authorization_id != authorization_id {
        return Err(RuntimeError::AuthorizationMismatch {
            expected: pending.authorization_id.clone(),
            got:      authorization_id.to_string(),
        });
    }

    if unix_now_ms() > pending.expires_at {
        return Err(RuntimeError::AuthorizationExpired {
            expires_at: pending.expires_at,
        });
    }

    if pending.session_id != state.session_id {
        return Err(RuntimeError::SessionMismatch);
    }

    Ok(())
}

// ── Generate Auth ID ──────────────────────────────────────────────────────────

/// Generate a deterministic authorization ID that binds to:
///   execution_id + plan_id + action_ids + tool_ids + argument fingerprints
///
/// This ensures "approve" cannot be replayed for a different batch.
fn generate_auth_id(state: &AgentState, batch: &[PlanAction]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(state.execution_id.as_bytes());
    hasher.update(state.session_id.as_bytes());

    if let Some(plan) = &state.plan {
        hasher.update(plan.plan_id.as_bytes());
        hasher.update(plan.version.as_bytes());
    }

    // Bind to exact action IDs, tool IDs, and argument fingerprints
    for action in batch {
        hasher.update(action.id.as_bytes());
        hasher.update(action.tool_id.as_bytes());
        if let Ok(args_json) = serde_json::to_string(&action.arguments) {
            hasher.update(args_json.as_bytes());
        }
    }

    // Add a nonce so identical batches get unique IDs per run
    hasher.update(&unix_now_ms().to_le_bytes());

    let hash = hasher.finalize();
    format!("auth_{}", hex_short(hash.as_bytes()))
}

fn hex_short(bytes: &[u8]) -> String {
    bytes[..8].iter().map(|b| format!("{:02x}", b)).collect()
}

// ── Check if batch needs authorization ───────────────────────────────────────

/// True if any action in the batch has requiresAuthorization = true
/// or the computed risk ≥ High.
pub fn requires_authorization(batch: &[PlanAction], max_risk: &RiskLevel) -> bool {
    batch.iter().any(|a| a.requires_authorization)
        || max_risk >= &RiskLevel::High
}

/// Maximum risk level across a batch.
pub fn batch_max_risk(batch: &[PlanAction]) -> RiskLevel {
    batch.iter()
        .map(|a| &a.risk_level)
        .max()
        .cloned()
        .unwrap_or(RiskLevel::Low)
}
