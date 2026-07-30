// lion_core/src/determinism.rs — Footprint Determinism Envelope
//
// Implements the Determinism Envelope contracts from 06_FOOTPRINT_ORCHESTRATION_SANDBOX_LEDGER.md.

use serde::{Deserialize, Serialize};
use crate::ir::TypedPrimitive;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeterminismEnvelope {
    /// Exact bit-for-bit memory and hash match (pure math, logic)
    Exact,
    /// Data matches, but underlying memory layouts or timestamps may drift
    LogicallyEquivalent,
    /// Floating-point bounds controlled by explicit epsilon values
    NumericTolerance(f64),
    /// Non-deterministic (RNG, Network), strictly bounded by capability scopes
    AuditableND,
}

impl DeterminismEnvelope {
    pub fn validate_match(&self, a: &TypedPrimitive, b: &TypedPrimitive) -> bool {
        match (self, a, b) {
            (DeterminismEnvelope::Exact, x, y) => x == y,
            (DeterminismEnvelope::LogicallyEquivalent, TypedPrimitive::String(s1), TypedPrimitive::String(s2)) => {
                s1.trim() == s2.trim()
            }
            (DeterminismEnvelope::NumericTolerance(eps), TypedPrimitive::Float(f1), TypedPrimitive::Float(f2)) => {
                (f1 - f2).abs() <= *eps
            }
            (DeterminismEnvelope::AuditableND, _, _) => true,
            _ => a == b,
        }
    }
}
