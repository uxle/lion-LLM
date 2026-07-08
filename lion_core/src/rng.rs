// lion_core/src/rng.rs

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::constants::{
    BASE_VECTOR_INIT_MAX, BASE_VECTOR_INIT_MIN,
    FEATURE_SIZE,
    INITIAL_WEIGHT_MAX, INITIAL_WEIGHT_MIN,
    MITOSIS_JITTER_MAX, MITOSIS_JITTER_MIN,
    MUTATION_DELTA_MAX, MUTATION_DELTA_MIN,
};

// =============================================================================
// BRAIN RNG
// =============================================================================

/// A deterministic, seeded random number generator for LionAI.
///
/// All random values produced by LionAI flow through this struct.
/// This centralises randomness so that:
///   1. Sandbox evaluations can be reproduced exactly.
///   2. Unit tests produce stable, predictable results.
///   3. Night cycle mutations are traceable to a known seed.
///
/// Wraps `rand::rngs::StdRng` which implements `SeedableRng`.
pub struct BrainRng {
    inner: StdRng,
}

impl BrainRng {
    // -------------------------------------------------------------------------
    // CONSTRUCTION
    // -------------------------------------------------------------------------

    /// Creates a BrainRng seeded from a specific u64 value.
    ///
    /// Two `BrainRng` instances constructed with the same seed will produce
    /// an identical stream of values in identical call order.
    ///
    /// Use this for: sandbox evaluation, unit tests, deterministic replay.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }

    /// Creates a BrainRng seeded from system entropy.
    ///
    /// Use this for: live execution where exploration is desired.
    /// Do NOT use inside the sandbox evaluator — results will not be reproducible.
    pub fn from_entropy() -> Self {
        Self {
            inner: StdRng::from_entropy(),
        }
    }

    // -------------------------------------------------------------------------
    // DOMAIN-SPECIFIC GENERATORS
    // -------------------------------------------------------------------------

    /// Generates a neuron base_vector with values uniformly distributed
    /// in [BASE_VECTOR_INIT_MIN, BASE_VECTOR_INIT_MAX] = [-0.1, +0.1].
    ///
    /// Matches Python:
    ///   self.base_vector = np.random.uniform(-0.1, 0.1, dna.feature_size)
    ///
    /// IMPORTANT: A zero base_vector causes division-by-zero in cosine alignment.
    /// The small random range guarantees no neuron starts with a zero vector.
    pub fn gen_base_vector(&mut self) -> [f32; FEATURE_SIZE] {
        let mut v = [0.0_f32; FEATURE_SIZE];
        for x in v.iter_mut() {
            *x = self
                .inner
                .gen_range(BASE_VECTOR_INIT_MIN..=BASE_VECTOR_INIT_MAX);
        }
        v
    }

    /// Generates a single initial synapse weight uniformly in [-0.5, +0.5].
    ///
    /// Matches Python:
    ///   random.uniform(-0.5, 0.5)
    pub fn gen_initial_weight(&mut self) -> f32 {
        self.inner
            .gen_range(INITIAL_WEIGHT_MIN..=INITIAL_WEIGHT_MAX)
    }

    /// Generates a weight mutation delta uniformly in [-0.3, +0.3].
    ///
    /// Matches Python (night cycle):
    ///   syn.weight += random.uniform(-0.3, 0.3)
    pub fn gen_mutation_delta(&mut self) -> f32 {
        self.inner
            .gen_range(MUTATION_DELTA_MIN..=MUTATION_DELTA_MAX)
    }

    /// Generates a mitosis base_vector jitter delta for a single f32 component.
    /// Uniformly in [-0.05, +0.05].
    ///
    /// Matches Python (mitosis):
    ///   child.base_vector = parent.base_vector + np.random.uniform(-0.05, 0.05, ...)
    pub fn gen_mitosis_jitter(&mut self) -> f32 {
        self.inner
            .gen_range(MITOSIS_JITTER_MIN..=MITOSIS_JITTER_MAX)
    }

    /// Generates a probability value uniformly in [0.0, 1.0).
    ///
    /// Matches Python: random.random()
    pub fn gen_prob(&mut self) -> f32 {
        self.inner.gen::<f32>()
    }

    /// Returns true with the given probability `p` ∈ [0.0, 1.0].
    ///
    /// Matches Python: random.random() < p
    pub fn gen_bool_with_prob(&mut self, p: f32) -> bool {
        self.inner.gen::<f32>() < p
    }

    /// Selects a random element from a non-empty slice and returns a reference.
    ///
    /// Matches Python: random.choice(actions)
    ///
    /// Panics if `items` is empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        assert!(!items.is_empty(), "BrainRng::choose called on empty slice");
        let idx = self.inner.gen_range(0..items.len());
        &items[idx]
    }

    /// Generates a random usize in [0, n).
    pub fn gen_index(&mut self, n: usize) -> usize {
        assert!(n > 0, "BrainRng::gen_index called with n=0");
        self.inner.gen_range(0..n)
    }

    /// Generates an epigenome plasticity mutation delta uniformly in [-0.02, +0.02].
    ///
    /// Matches Python:
    ///   self.plasticity += random.uniform(-0.02, 0.02)
    pub fn gen_plasticity_delta(&mut self) -> f32 {
        self.inner.gen_range(-0.02_f32..=0.02_f32)
    }

    /// Generates an epigenome exploration_drive mutation delta uniformly in [-0.05, +0.05].
    ///
    /// Matches Python:
    ///   self.exploration_drive += random.uniform(-0.05, 0.05)
    pub fn gen_exploration_delta(&mut self) -> f32 {
        self.inner.gen_range(-0.05_f32..=0.05_f32)
    }

    /// Generates a synapse weight jitter for mitosis inheritance.
    /// Uniformly in [MITOSIS_SYNAPSE_JITTER_MIN, MITOSIS_SYNAPSE_JITTER_MAX]
    /// = [-0.1, +0.1].
    ///
    /// Distinct from `gen_mitosis_jitter()` (which is for base_vector [-0.05, +0.05]).
    ///
    /// Matches Python (mitosis, synapse inheritance):
    ///   syn.weight + random.uniform(-0.1, 0.1)
    pub fn gen_synapse_mitosis_jitter(&mut self) -> f32 {
        self.inner
            .gen_range(crate::constants::MITOSIS_SYNAPSE_JITTER_MIN..=crate::constants::MITOSIS_SYNAPSE_JITTER_MAX)
    }
}
