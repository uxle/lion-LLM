// lion_core/src/types.rs

use crate::constants::FEATURE_SIZE;
use serde::{Deserialize, Serialize};

// =============================================================================
// GENERATIONAL INDEX
// =============================================================================

/// A safe, non-pointer reference to an arena slot.
///
/// `index`      — The position inside `Vec<Neuron>` or `Vec<Synapse>`.
/// `generation` — A counter that increments every time a slot is reused.
///
/// A `GenIndex` is only valid if:
///   arena.generations[index] == self.generation
///
/// This means stale references from dead neurons are automatically detected
/// without any pointer dereferencing or unsafe code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GenIndex {
    pub index:      usize,
    pub generation: u32,
}

impl GenIndex {
    pub fn new(index: usize, generation: u32) -> Self {
        Self { index, generation }
    }
}

// =============================================================================
// ROLE
// =============================================================================

/// The biological role of a neuron.
/// Matches Python: class Role(str, Enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Role {
    Vision = 0,
    Motor  = 1,
    Memory = 2,
    Danger = 3,
}

impl Role {
    /// Returns the string name used when matching sensory input modalities.
    /// Matches Python: n.role.value == modality.upper()
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Vision => "VISION",
            Role::Motor  => "MOTOR",
            Role::Memory => "MEMORY",
            Role::Danger => "DANGER",
        }
    }

    /// Total number of roles. Used when initializing the core brain.
    pub const COUNT: usize = 4;
}

// =============================================================================
// MEMORY TRACE
// =============================================================================

/// A single episodic memory fragment stored inside a neuron.
///
/// Matches Python:
///   @dataclass
///   class MemoryTrace:
///       trace_id: str
///       vector: np.ndarray
///       strength: float = 1.0
///       age: int = 0
///
/// CRITICAL DESIGN RULE:
/// MemoryTrace must be `Copy`. It contains ONLY fixed-size arrays and
/// primitive scalars — no `Vec`, no `Box`, no `String`.
/// This is what allows the entire `BrainMatrix` to be cloned in O(1)
/// during the evolutionary night cycle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryTrace {
    pub vector:   [f32; FEATURE_SIZE],
    pub strength: f32,
    pub age:      u32,
}

impl MemoryTrace {
    /// Creates a fresh trace with full strength from a raw feature vector.
    pub fn new(vector: [f32; FEATURE_SIZE]) -> Self {
        Self {
            vector,
            strength: 1.0,
            age:      0,
        }
    }
}

impl Default for MemoryTrace {
    /// A zero-initialized, zero-strength trace used to fill empty slots.
    fn default() -> Self {
        Self {
            vector:   [0.0_f32; FEATURE_SIZE],
            strength: 0.0,
            age:      0,
        }
    }
}
