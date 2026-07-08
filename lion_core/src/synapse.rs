// lion_core/src/synapse.rs

use crate::types::GenIndex;
use serde::{Deserialize, Serialize};

/// A directed weighted connection between two neurons.
///
/// Matches Python:
///   @dataclass
///   class Synapse:
///       pre_id: str
///       post_id: str
///       weight: float
///
/// Uses `GenIndex` instead of string IDs.
/// `alive` replaces deletion from the Python dict.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Synapse {
    /// The upstream neuron that sends the signal.
    pub pre_id:  GenIndex,

    /// The downstream neuron that receives the signal.
    pub post_id: GenIndex,

    /// Connection weight. Range clamped to [WEIGHT_MIN, WEIGHT_MAX] = [-2.0, 2.0].
    pub weight:  f32,

    /// Whether this synapse is alive.
    /// Dead synapses are skipped during propagation and available for reuse.
    pub alive: bool,
}

impl Synapse {
    /// Creates a new live synapse between two neurons.
    pub fn new(pre_id: GenIndex, post_id: GenIndex, weight: f32) -> Self {
        Self {
            pre_id,
            post_id,
            weight,
            alive: true,
        }
    }
}
