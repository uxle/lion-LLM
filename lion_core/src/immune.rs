// lion_core/src/immune.rs

use crate::brain::BrainMatrix;
use crate::constants::{FEATURE_SIZE, WEIGHT_MAX, WEIGHT_MIN};

// =============================================================================
// PRIMITIVE SANITIZERS
// =============================================================================

/// Sanitizes a single f32 activation or feature-vector component.
///
/// Replacement table:
///   NaN  → 0.0   (no signal — safe neutral)
///   +Inf → 1.0   (clamp to tanh ceiling)
///   -Inf → -1.0  (clamp to tanh floor)
///   finite → unchanged
///
/// Matches Python:
///   np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
#[inline]
pub fn sanitize_scalar(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else if x == f32::INFINITY {
        1.0
    } else if x == f32::NEG_INFINITY {
        -1.0
    } else {
        x
    }
}

/// Sanitizes a fixed-size feature vector in-place.
///
/// Applies `sanitize_scalar` to every component.
/// Returns `true` if at least one value was corrected.
///
/// Used on: `neuron.base_vector`, `trace.vector`.
#[inline]
pub fn sanitize_vector(vec: &mut [f32; FEATURE_SIZE]) -> bool {
    let mut changed = false;
    for x in vec.iter_mut() {
        let sanitized = sanitize_scalar(*x);
        if sanitized.to_bits() != x.to_bits() {
            *x      = sanitized;
            changed = true;
        }
    }
    changed
}

/// Sanitizes a synapse weight.
///
/// Replacement table:
/// ```text
///   NaN         → 0.0        (dead synapse — no signal)
///   > WEIGHT_MAX → WEIGHT_MAX (clamp high)
///   < WEIGHT_MIN → WEIGHT_MIN (clamp low)
///   finite in range → unchanged
/// ```
///
/// Note: synapse weights are already clamped by Hebbian updates.
/// This is a redundant safety net for values produced by mitosis jitter
/// or direct arena manipulation.
#[inline]
pub fn sanitize_weight(w: f32) -> f32 {
    if w.is_nan() {
        0.0
    } else {
        w.clamp(WEIGHT_MIN, WEIGHT_MAX)
    }
}

/// Sanitizes a trace strength value.
///
/// Strength must be in [0.0, 1.0].
/// Replacement table:
/// ```text
///   NaN         → 0.0  (dead trace — evictable)
///   < 0.0       → 0.0
///   > 1.0       → 1.0
///   finite [0,1] → unchanged
/// ```
#[inline]
pub fn sanitize_strength(s: f32) -> f32 {
    if s.is_nan() {
        0.0
    } else {
        s.clamp(0.0, 1.0)
    }
}

// =============================================================================
// IMMUNE SCAN REPORT
// =============================================================================

/// A summary of what the immune system found and fixed in one scan pass.
///
/// Returned by `scan_and_heal()` so callers can log or react to the results.
#[derive(Debug, Clone, Default)]
pub struct ImmuneReport {
    /// Number of neuron activations corrected.
    pub activation_fixes: u32,

    /// Number of base_vector component values corrected.
    pub base_vector_fixes: u32,

    /// Number of trace vector component values corrected.
    pub trace_vector_fixes: u32,

    /// Number of trace strength values corrected.
    pub trace_strength_fixes: u32,

    /// Number of synapse weights corrected.
    pub weight_fixes: u32,
}

impl ImmuneReport {
    /// Total number of corrections across all categories.
    pub fn total(&self) -> u32 {
        self.activation_fixes
            + self.base_vector_fixes
            + self.trace_vector_fixes
            + self.trace_strength_fixes
            + self.weight_fixes
    }

    /// Returns true if the immune system found no invalid values.
    pub fn is_clean(&self) -> bool {
        self.total() == 0
    }
}

// =============================================================================
// IMMUNE SCAN ON BRAIN MATRIX
// =============================================================================

impl BrainMatrix {
    /// Scans the entire arena for invalid floating-point values
    /// and replaces them with safe defaults.
    ///
    /// Scans in this order:
    ///   1. Neuron activations
    ///   2. Neuron base_vectors
    ///   3. Memory trace vectors
    ///   4. Memory trace strengths
    ///   5. Synapse weights
    ///
    /// Increments `self.immune_interventions` by the total fix count.
    /// Returns a detailed `ImmuneReport` for logging.
    ///
    /// Called after every live propagation step.
    /// NOT called during sandbox evaluation.
    ///
    /// Matches Python:
    ///   def scan_and_heal(self, neurons):
    ///       for nid, n in neurons.items():
    ///           if not math.isfinite(n.activation):
    ///               n.activation = 0.0
    ///               self.interventions += 1
    ///           n.base_vector = self.sanitize_vector(n.base_vector)
    ///           for t in n.local_traces:
    ///               t.vector = self.sanitize_vector(t.vector)
    pub fn scan_and_heal(&mut self) -> ImmuneReport {
        let mut report = ImmuneReport::default();

        // ── Scan neurons ─────────────────────────────────────────────────────
        for n in self.neurons.iter_mut() {
            if !n.alive {
                continue;
            }

            // 1. Sanitize activation.
            let clean_act = sanitize_scalar(n.activation);
            if clean_act.to_bits() != n.activation.to_bits() {
                n.activation = clean_act;
                report.activation_fixes += 1;
            }

            // 2. Sanitize base_vector.
            for x in n.base_vector.iter_mut() {
                let clean = sanitize_scalar(*x);
                if clean.to_bits() != x.to_bits() {
                    *x = clean;
                    report.base_vector_fixes += 1;
                }
            }

            // 3. Sanitize all active memory trace vectors and strengths.
            for i in 0..n.trace_count {
                for x in n.traces[i].vector.iter_mut() {
                    let clean = sanitize_scalar(*x);
                    if clean.to_bits() != x.to_bits() {
                        *x = clean;
                        report.trace_vector_fixes += 1;
                    }
                }

                // 4. Sanitize trace strength.
                let clean_str = sanitize_strength(n.traces[i].strength);
                if clean_str.to_bits() != n.traces[i].strength.to_bits() {
                    n.traces[i].strength = clean_str;
                    report.trace_strength_fixes += 1;
                }
            }
        }

        // ── Scan synapses ─────────────────────────────────────────────────────
        for s in self.synapses.iter_mut() {
            if !s.alive {
                continue;
            }

            // 5. Sanitize synapse weight.
            let clean_w = sanitize_weight(s.weight);
            if clean_w.to_bits() != s.weight.to_bits() {
                s.weight = clean_w;
                report.weight_fixes += 1;
            }
        }

        // Accumulate into persistent counter.
        self.immune_interventions += report.total();

        report
    }

    /// Resets the immune intervention counter to zero.
    ///
    /// Called at the end of each sleep cycle, after printing the day's report.
    ///
    /// Matches Python:
    ///   self.immune.interventions = 0
    pub fn reset_immune_counter(&mut self) {
        self.immune_interventions = 0;
    }

    /// Returns the total immune interventions accumulated since last reset.
    pub fn immune_intervention_count(&self) -> u32 {
        self.immune_interventions
    }

    /// Returns `true` if the entire arena is currently free of invalid values.
    ///
    /// Performs a read-only scan — does not modify anything.
    /// Used in tests and diagnostics to assert brain health.
    pub fn is_numerically_healthy(&self) -> bool {
        // Check neurons.
        for n in self.neurons.iter() {
            if !n.alive {
                continue;
            }
            if !n.activation.is_finite() {
                return false;
            }
            if n.base_vector.iter().any(|x| !x.is_finite()) {
                return false;
            }
            for i in 0..n.trace_count {
                if n.traces[i].vector.iter().any(|x| !x.is_finite()) {
                    return false;
                }
                let s = n.traces[i].strength;
                if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                    return false;
                }
            }
        }

        // Check synapses.
        for s in self.synapses.iter() {
            if !s.alive {
                continue;
            }
            if !s.weight.is_finite()
                || s.weight > WEIGHT_MAX
                || s.weight < WEIGHT_MIN
            {
                return false;
            }
        }

        true
    }

    /// Injects a NaN into a specific neuron's activation for testing purposes.
    ///
    /// Only available in test builds — not compiled into release.
    #[cfg(test)]
    pub fn corrupt_neuron_activation(&mut self, id: crate::types::GenIndex) {
        if let Some(n) = self.get_neuron_mut(id) {
            n.activation = f32::NAN;
        }
    }

    /// Injects a NaN into a specific synapse weight for testing purposes.
    #[cfg(test)]
    pub fn corrupt_synapse_weight(&mut self, id: crate::types::GenIndex) {
        if self.is_valid_synapse(id) {
            self.synapses[id.index].weight = f32::NAN;
        }
    }

    /// Injects +Inf into a specific neuron's base_vector component.
    #[cfg(test)]
    pub fn corrupt_base_vector(&mut self, id: crate::types::GenIndex, component: usize) {
        if let Some(n) = self.get_neuron_mut(id) {
            if component < FEATURE_SIZE {
                n.base_vector[component] = f32::INFINITY;
            }
        }
    }
}
