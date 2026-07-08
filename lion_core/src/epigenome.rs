// lion_core/src/epigenome.rs
use serde::{Deserialize, Serialize};

/// Dynamic phenotypic state that modulates mutation rate and exploration.
///
/// Matches Python:
///   @dataclass
///   class Epigenome:
///       plasticity: float = 0.05
///       exploration_drive: float = 0.1
///       accumulated_stress: float = 0.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Epigenome {
    /// Controls the rate of Hebbian weight updates during the sleep cycle.
    /// Range: [0.01, 0.5].
    pub plasticity: f32,

    /// Probability of taking a random action instead of the best known action.
    /// Range: [0.0, 1.0].
    pub exploration_drive: f32,

    /// Accumulated negative reward signal. Increases mutation rate at night.
    /// Range: [0.0, 1.0].
    pub accumulated_stress: f32,
}

impl Default for Epigenome {
    fn default() -> Self {
        Self {
            plasticity:         0.05,
            exploration_drive:  0.1,
            accumulated_stress: 0.0,
        }
    }
}

impl Epigenome {
    /// Increases stress in response to negative reward.
    ///
    /// Matches Python:
    ///   def adapt_live_stress(self, delta_stress: float):
    ///       self.accumulated_stress = max(0.0, min(1.0, ...))
    pub fn adapt_live_stress(&mut self, delta_stress: f32) {
        self.accumulated_stress =
            (self.accumulated_stress + delta_stress).clamp(0.0, 1.0);
    }

    /// Applies random drift to plasticity and exploration drive.
    ///
    /// Called once per child during the evolutionary night cycle.
    ///
    /// Matches Python:
    ///   def mutate(self):
    ///       self.plasticity = max(0.01, min(0.5, self.plasticity + random...))
    ///       self.exploration_drive = max(0.0, min(1.0, self.exploration_drive + random...))
    pub fn mutate(&mut self, rng_plasticity_delta: f32, rng_explore_delta: f32) {
        self.plasticity =
            (self.plasticity + rng_plasticity_delta).clamp(0.01, 0.5);
        self.exploration_drive =
            (self.exploration_drive + rng_explore_delta).clamp(0.0, 1.0);
    }

    /// Computes the effective mutation rate for the night cycle.
    ///
    /// Matches Python:
    ///   dyn_mutation_rate = base_mutation_rate * (1.0 + plasticity) + (stress * 0.2)
    pub fn effective_mutation_rate(&self, base_rate: f32) -> f32 {
        base_rate * (1.0 + self.plasticity) + (self.accumulated_stress * 0.2)
    }

    /// Decays stress after a failed night cycle (no child won).
    ///
    /// Matches Python: self.epi.accumulated_stress *= 0.8
    pub fn decay_stress(&mut self) {
        self.accumulated_stress *= 0.8;
    }

    /// Resets stress after a successful evolution event.
    ///
    /// Matches Python: self.epi.accumulated_stress = 0.0
    pub fn clear_stress(&mut self) {
        self.accumulated_stress = 0.0;
    }
}
