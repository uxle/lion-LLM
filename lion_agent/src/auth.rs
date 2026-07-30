// lion_agent/src/auth.rs — Stateful Human-In-The-Loop (HITL) Authorization
//
// Implements SuspendForAuthorization, VerifyAuthorization, and ResumeExecution from 02_ORCHESTRATION_RUNTIME.md.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingAuthorization {
    pub authorization_id: String,
    pub tool_name: String,
    pub tool_input: String,
    pub created_at: u64,
    pub expires_at: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthorizationManager {
    pub pending: Vec<PendingAuthorization>,
}

impl AuthorizationManager {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Suspend execution for authorization of high-risk or destructive actions.
    pub fn suspend(&mut self, tool_name: impl Into<String>, tool_input: impl Into<String>, ttl_secs: u64) -> PendingAuthorization {
        let auth_id = format!("auth_{:016x}", rand::Rng::gen::<u64>(&mut rand::thread_rng()));
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let pending = PendingAuthorization {
            authorization_id: auth_id,
            tool_name: tool_name.into(),
            tool_input: tool_input.into(),
            created_at: timestamp,
            expires_at: timestamp + ttl_secs,
        };

        self.pending.push(pending.clone());
        pending
    }

    /// Verify whether an authorization token exists and is valid (unexpired).
    pub fn verify(&self, authorization_id: &str) -> bool {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.pending
            .iter()
            .any(|p| p.authorization_id == authorization_id && timestamp <= p.expires_at)
    }

    /// Approve an authorization, returning the saved (tool_name, tool_input).
    pub fn approve(&mut self, authorization_id: &str) -> Option<(String, String)> {
        if !self.verify(authorization_id) {
            return None;
        }

        if let Some(pos) = self.pending.iter().position(|p| p.authorization_id == authorization_id) {
            let item = self.pending.remove(pos);
            return Some((item.tool_name, item.tool_input));
        }
        None
    }

    /// Deny an authorization request.
    pub fn deny(&mut self, authorization_id: &str) -> bool {
        if let Some(pos) = self.pending.iter().position(|p| p.authorization_id == authorization_id) {
            self.pending.remove(pos);
            return true;
        }
        false
    }
}
