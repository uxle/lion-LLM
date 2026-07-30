// lion_core/src/lib.rs — LionAI v1.0 Foundation
//
// Exports all core types: TernaryEncoder, SensoryInput, Role,
// persistent knowledge/memory/versioning/evaluation modules,
// LFMF container specs, Footprint IR, contracts, and cryptographic ledgers.

pub mod contracts;
pub mod determinism;
pub mod encoder;
pub mod evaluation;
pub mod ir;
pub mod knowledge;
pub mod ledger;
pub mod lfmf;
pub mod longmem;
pub mod versioning;

pub use contracts::{SemanticAnalyzer, VerificationContract};
pub use determinism::DeterminismEnvelope;
pub use encoder::{
    Activation, TernaryEncoder, TernaryEncoderConfig, TernaryLayer,
};
pub use ir::{CanonicalIR, IRNode, Opcode, TypedPrimitive};
pub use ledger::{canonicalize_json, HashLedger, LedgerEntry};
pub use lfmf::{LfmfHeader, MemoryTier, TieredMemoryManager, TieredMemoryRecord};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Size of every embedding vector produced by the TernaryEncoder.
pub const FEATURE_SIZE: usize = 32;

/// Alias for the fixed-size feature vector.
pub type Features = [f32; FEATURE_SIZE];

// ── Role ─────────────────────────────────────────────────────────────────────

/// Cognitive roles for sensory input channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Role {
    /// Visual perception input (images).
    Vision,
    /// Danger / alert signal (loud audio, threat detection).
    Danger,
    /// Episodic memory recall channel.
    Memory,
    /// Motor / output planning channel.
    Motor,
}

// ── SensoryInput ─────────────────────────────────────────────────────────────

/// A multi-modal sensory frame: maps each Role to a 32-dim feature vector.
/// Fed into the cognitive graph each tick.
#[derive(Debug, Clone, Default)]
pub struct SensoryInput {
    channels: std::collections::HashMap<Role, Features>,
}

impl SensoryInput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a channel's embedding.
    pub fn insert(&mut self, role: Role, features: Features) {
        self.channels.insert(role, features);
    }

    /// Get the feature vector for a given role.
    pub fn get(&self, role: Role) -> Option<&Features> {
        self.channels.get(&role)
    }

    /// Number of active channels.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Iterate over (role, features) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Role, &Features)> {
        self.channels.iter()
    }
}

// ── Quantization helpers ──────────────────────────────────────────────────────

/// Clamps and quantizes an f32 in [-1.0, +1.0] to i8 in [-127, +127].
#[inline]
pub fn f32_to_i8(x: f32) -> i8 {
    (x.clamp(-1.0, 1.0) * 127.0).round() as i8
}

/// Dequantizes an i8 back to f32 in [-1.0, +1.0].
#[inline]
pub fn i8_to_f32(x: i8) -> f32 {
    x as f32 / 127.0
}

// ── Seeded RNG alias ──────────────────────────────────────────────────────────

/// Convenience alias for the seeded RNG used throughout LionAI.
pub type BrainRng = rand::rngs::StdRng;

/// Create a seeded BrainRng.
pub fn seeded_rng(seed: u64) -> BrainRng {
    use rand::SeedableRng;
    BrainRng::seed_from_u64(seed)
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length slices.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32   = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a < 1e-9 || mag_b < 1e-9 { return 0.0; }
    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}
