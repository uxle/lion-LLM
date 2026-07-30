// lion_core/src/ledger.rs — Footprint Cryptographic Audit Ledger
//
// Implements append-only BLAKE3 hash-chaining for plan execution steps.
// Ensures tamper-evident auditing across execution runs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// =============================================================================
// CANONICALIZATION
// =============================================================================

/// Canonicalize JSON value by sorting object keys and normalizing formatting.
pub fn canonicalize_json(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map.iter().collect();
            let mut parts = Vec::new();
            for (k, v) in sorted {
                parts.push(format!("{}:{}", serde_json::to_string(k).unwrap_or_default(), canonicalize_json(v)));
            }
            format!("{{{}}}", parts.join(","))
        }
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(canonicalize_json).collect();
            format!("[{}]", parts.join(","))
        }
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                format!("{:.6}", f)
            } else {
                n.to_string()
            }
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

// =============================================================================
// LEDGER ENTRY
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LedgerEntry {
    pub step_id: String,
    pub opcode: String,
    pub canonical_inputs: String,
    pub env_fingerprint: String,
    pub parent_hash: String,
    pub entry_hash: String,
    pub timestamp: u64,
}

impl LedgerEntry {
    /// Compute entry hash using BLAKE3.
    pub fn compute_hash(
        canonical_inputs: &str,
        opcode: &str,
        env_fingerprint: &str,
        parent_hash: &str,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(canonical_inputs.as_bytes());
        hasher.update(b"|");
        hasher.update(opcode.as_bytes());
        hasher.update(b"|");
        hasher.update(env_fingerprint.as_bytes());
        hasher.update(b"|");
        hasher.update(parent_hash.as_bytes());
        let result = hasher.finalize();
        result.to_hex().to_string()
    }
}

// =============================================================================
// HASH LEDGER
// =============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HashLedger {
    pub entries: Vec<LedgerEntry>,
    pub env_fingerprint: String,
}

impl HashLedger {
    pub fn new(env_fingerprint: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            env_fingerprint: env_fingerprint.into(),
        }
    }

    /// Record a completed execution step into the append-only ledger.
    pub fn append(&mut self, step_id: &str, opcode: &str, inputs: &serde_json::Value) -> &LedgerEntry {
        let canonical = canonicalize_json(inputs);
        let parent_hash = self
            .entries
            .last()
            .map(|e| e.entry_hash.as_str())
            .unwrap_or("0000000000000000000000000000000000000000000000000000000000000000");

        let entry_hash = LedgerEntry::compute_hash(&canonical, opcode, &self.env_fingerprint, parent_hash);

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let entry = LedgerEntry {
            step_id: step_id.to_string(),
            opcode: opcode.to_string(),
            canonical_inputs: canonical,
            env_fingerprint: self.env_fingerprint.clone(),
            parent_hash: parent_hash.to_string(),
            entry_hash,
            timestamp,
        };

        self.entries.push(entry);
        self.entries.last().unwrap()
    }

    /// Verify full tamper-evidence of the ledger history.
    pub fn verify_chain(&self) -> bool {
        let mut expected_parent = "0000000000000000000000000000000000000000000000000000000000000000";

        for entry in &self.entries {
            if entry.parent_hash != expected_parent {
                return false;
            }
            let computed = LedgerEntry::compute_hash(
                &entry.canonical_inputs,
                &entry.opcode,
                &entry.env_fingerprint,
                &entry.parent_hash,
            );
            if computed != entry.entry_hash {
                return false;
            }
            expected_parent = &entry.entry_hash;
        }

        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
