// lion_core/src/hebbian.rs

use crate::brain::BrainMatrix;
use crate::constants::{ACTIVATION_DEAD_ZONE, FEATURE_SIZE, WEIGHT_MAX, WEIGHT_MIN};

// =============================================================================
// TRACE ALIGNMENT HELPER
// =============================================================================

/// Computes cosine similarity between two memory trace vectors.
///
/// Returns the clamped positive component A⁺ = max(0.0, cosine(v1, v2)).
///
/// ```text
///          v1 · v2
/// A = ─────────────────────
///      ‖v1‖ · ‖v2‖ + ε
/// ```
///
/// Returns 0.0 when either vector is zero (epsilon guard prevents NaN).
/// Returns only the positive component — negative alignment is ignored,
/// contributing neither boost nor penalty to the learning signal.
///
/// Matches Python:
///   mag = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
///   trace_align = float(np.dot(v1, v2) / mag)
///   (1.0 + max(0.0, trace_align))
#[inline]
pub fn trace_alignment(
    v1: &[f32; FEATURE_SIZE],
    v2: &[f32; FEATURE_SIZE],
) -> f32 {
    let mut dot  = 0.0_f32;
    let mut mag1 = 0.0_f32;
    let mut mag2 = 0.0_f32;

    for i in 0..FEATURE_SIZE {
        dot  += v1[i] * v2[i];
        mag1 += v1[i] * v1[i];
        mag2 += v2[i] * v2[i];
    }

    let cosine = dot / (mag1.sqrt() * mag2.sqrt() + 1e-9);

    // Only the positive component contributes, clamped to [0.0, 1.0] to handle float precision issues.
    cosine.clamp(0.0, 1.0)
}

// =============================================================================
// HEBBIAN PLASTICITY ON BRAIN MATRIX
// =============================================================================

impl BrainMatrix {
    /// Applies one round of Hebbian plasticity to all alive synapses.
    ///
    /// Must be called AFTER `propagate_steps()` so that neuron activations
    /// reflect the current episode state.
    ///
    /// For each alive synapse (pre → post):
    ///   1. Skip if pre.activation is in the dead zone (< ACTIVATION_DEAD_ZONE).
    ///   2. Compute trace alignment A⁺ between pre and post latest traces.
    ///      If either neuron has no traces, A⁺ = 0.0 (no semantic boost).
    ///   3. Compute learning signal S = pre.act × post.act × (1 + A⁺).
    ///   4. Update: w += S × hebbian_rate × plasticity.
    ///   5. Clamp: w = clamp(w, WEIGHT_MIN, WEIGHT_MAX).
    ///
    /// # Plasticity Parameter
    /// `plasticity` comes from `Epigenome::plasticity` (range 0.01..0.5).
    /// A value of 0.0 produces zero weight change (guard early-exit).
    ///
    /// # Borrow Strategy
    /// `neurons` is borrowed immutably before the mutable synapse loop.
    /// Rust field splitting allows simultaneous:
    ///   - `&self.neurons`   (immutable, reading activations and traces)
    ///   - `&mut self.synapses` (mutable, updating weights)
    ///
    /// Matches Python:
    ///   if apply_hebbian and plasticity > 0 and self.dna.hebbian_rate > 0:
    ///       for pre_id, connections in self.synapses.items():
    ///           pre_n = self.neurons[pre_id]
    ///           if abs(pre_n.activation) < 0.01: continue
    ///           for post_id, syn in connections.items():
    ///               post_n = self.neurons[post_id]
    ///               trace_align = ...
    ///               learning_signal = pre_n.activation * post_n.activation * (1.0 + max(0.0, trace_align))
    ///               syn.weight += learning_signal * hebbian_rate * plasticity
    ///               syn.weight = max(-2.0, min(2.0, syn.weight))
    pub fn apply_hebbian_plasticity(&mut self, plasticity: f32) {
        // Guard: no-op if plasticity is zero or hebbian_rate is zero.
        // This matches Python: if apply_hebbian and plasticity > 0 and hebbian_rate > 0
        if plasticity <= 0.0 || self.hebbian_rate <= 0.0 {
            return;
        }

        let rate = self.hebbian_rate * plasticity;

        // Borrow neurons immutably before the mutable synapse loop.
        // Rust field splitting: &self.neurons and &mut self.synapses are
        // different fields — both borrows coexist safely.
        let neurons = &self.neurons;

        for syn in self.synapses.iter_mut() {
            if !syn.alive {
                continue;
            }

            // Validate both endpoints are alive.
            let pre_idx  = syn.pre_id.index;
            let post_idx = syn.post_id.index;

            let pre_n  = &neurons[pre_idx];
            let post_n = &neurons[post_idx];

            if !pre_n.alive || !post_n.alive {
                continue;
            }

            // Dead zone: skip weakly activated pre-synaptic neurons.
            // Matches Python: if abs(pre_n.activation) < 0.01: continue
            if pre_n.activation.abs() < ACTIVATION_DEAD_ZONE {
                continue;
            }

            // Compute trace alignment.
            // If either neuron has no traces, alignment is 0.0 (no semantic boost).
            let align = if pre_n.trace_count > 0 && post_n.trace_count > 0 {
                // Use the latest (youngest) trace from each neuron.
                let t_pre  = pre_n.latest_trace().unwrap();
                let t_post = post_n.latest_trace().unwrap();
                trace_alignment(&t_pre.vector, &t_post.vector)
            } else {
                0.0
            };

            // Compute learning signal:
            // S = a_pre × a_post × (1 + A⁺)
            let learning_signal =
                pre_n.activation * post_n.activation * (1.0 + align);

            // Apply weight update and clamp.
            // Δw = S × hebbian_rate × plasticity
            syn.weight = (syn.weight + learning_signal * rate)
                .clamp(WEIGHT_MIN, WEIGHT_MAX);
        }
    }

    /// Returns a snapshot of all synapse weights for diagnostics and testing.
    ///
    /// Returns `(pre_id_index, post_id_index, weight)` tuples for all alive synapses.
    /// Used to assert weight changes during Hebbian tests.
    pub fn synapse_weight_snapshot(&self) -> Vec<(usize, usize, f32)> {
        self.synapses
            .iter()
            .filter(|s| s.alive)
            .map(|s| (s.pre_id.index, s.post_id.index, s.weight))
            .collect()
    }

    /// Returns the total absolute weight change between two snapshots.
    ///
    /// Used in tests to assert that Hebbian plasticity has modified weights.
    pub fn total_weight_delta(
        before: &[(usize, usize, f32)],
        after:  &[(usize, usize, f32)],
    ) -> f32 {
        before
            .iter()
            .zip(after.iter())
            .map(|((_, _, w_before), (_, _, w_after))| (w_after - w_before).abs())
            .sum()
    }
}
