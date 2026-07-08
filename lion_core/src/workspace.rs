// lion_core/src/workspace.rs

use crate::brain::BrainMatrix;
use crate::constants::{FEATURE_SIZE, PROCEDURAL_ACTIONS};
use crate::epigenome::Epigenome;
use crate::neuron::ActionLabel;
use crate::rng::BrainRng;
use crate::types::{GenIndex, Role};

// =============================================================================
// GLOBAL WORKSPACE
// =============================================================================

/// The number of top-k neurons admitted to the global workspace per tick.
///
/// Matches Python: top_k: int = 5
pub const WORKSPACE_TOP_K: usize = 5;

/// Gathers the brain's conscious state into a compressed gestalt vector.
///
/// Algorithm:
///   1. Find the top-k alive neurons ranked by |activation| (absolute saliency).
///   2. For each top-k neuron, accumulate:
///      ```text
///      gestalt += base_vector × activation
///      for each trace: gestalt += trace.vector × activation × trace.strength
///      ```
///   3. Normalize gestalt if its magnitude > 0.
///
/// The activation weight preserves polarity — inhibitory neurons (negative
/// activation) negate their contribution to the gestalt direction.
/// This is what allows DANGER neurons to actively suppress FORAGE direction.
///
/// Returns:
///   - The normalized gestalt vector.
///   - The GenIndex values of the top-k neurons (for diagnostics).
///
/// Matches Python:
///   def gather_consciousness(self, graph, top_k=5):
///       active_neurons = sorted(..., key=lambda n: abs(n.activation), reverse=True)[:top_k]
///       gestalt = np.zeros(FEATURE_SIZE)
///       for n in active_neurons:
///           weight = n.activation
///           gestalt += (n.base_vector * weight)
///           for trace in n.local_traces:
///               gestalt += (trace.vector * weight * trace.strength)
///       norm = np.linalg.norm(gestalt)
///       if norm > 0: gestalt /= norm
///       return active_neurons, gestalt
pub fn gather_consciousness(
    brain: &BrainMatrix,
    top_k: usize,
) -> ([f32; FEATURE_SIZE], Vec<GenIndex>) {

    // ── Step 1: Collect and rank alive neurons by absolute saliency ──────────
    let mut ranked: Vec<(GenIndex, f32)> = brain
        .alive_neurons()
        .map(|n| (n.id, n.activation.abs()))
        .collect();

    // Sort descending by absolute activation.
    ranked.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take the top-k most salient neurons.
    ranked.truncate(top_k);

    let conscious_ids: Vec<GenIndex> = ranked.iter().map(|(id, _)| *id).collect();

    // ── Step 2: Accumulate gestalt ────────────────────────────────────────────
    let mut gestalt = [0.0_f32; FEATURE_SIZE];

    for (id, _) in &ranked {
        if let Some(n) = brain.get_neuron(*id) {
            let weight = n.activation; // Preserves polarity.

            // Weighted base_vector contribution.
            for (g, &b) in gestalt.iter_mut().zip(n.base_vector.iter()) {
                *g += b * weight;
            }

            // Weighted trace contributions.
            for trace in n.active_traces() {
                let trace_weight = weight * trace.strength;
                for (g, &t) in gestalt.iter_mut().zip(trace.vector.iter()) {
                    *g += t * trace_weight;
                }
            }
        }
    }

    // ── Step 3: Normalize ─────────────────────────────────────────────────────
    let norm: f32 = gestalt.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in gestalt.iter_mut() {
            *x /= norm;
        }
    }

    (gestalt, conscious_ids)
}

// =============================================================================
// ACTION EXTRACTION
// =============================================================================

/// Extracts the agent's chosen action using an ε-greedy exploration policy.
///
/// During live execution (`force_exploit = false`):
///   With probability `epigenome.exploration_drive`, picks a random action.
///   Otherwise, picks the action of the most-activated Motor neuron.
///
/// During sandbox evaluation (`force_exploit = true`):
///   Always picks the most-activated Motor neuron. Deterministic.
///
/// Returns a `&'static str` from `PROCEDURAL_ACTIONS`.
/// Returns `"WANDER"` as fallback if no Motor neurons are alive.
///
/// Matches Python:
///   def extract_action(self, graph, epi, force_exploit=False):
///       motor_neurons = [n for n in graph.neurons.values() if n.role == Role.MOTOR]
///       if not motor_neurons: return "WANDER"
///       best_motor = max(motor_neurons, key=lambda n: n.activation)
///       if not force_exploit and random.random() < epi.exploration_drive:
///           return random.choice(self.actions)
///       return best_motor.action_label if best_motor.action_label else "WANDER"
pub fn extract_action(
    brain:         &BrainMatrix,
    epigenome:     &Epigenome,
    rng:           &mut BrainRng,
    force_exploit: bool,
) -> &'static str {

    // ε-greedy exploration during live execution.
    if !force_exploit && rng.gen_bool_with_prob(epigenome.exploration_drive) {
        return rng.choose(PROCEDURAL_ACTIONS);
    }

    // Exploitation: pick the most-activated Motor neuron.
    best_motor_action(brain)
}

/// Returns the `&'static str` action label of the most-activated Motor neuron.
///
/// Resolves the `ActionLabel` stored in the arena back to a `&'static str`
/// by matching against `PROCEDURAL_ACTIONS`. This avoids lifetime issues
/// from returning a reference into the arena.
///
/// Returns `"WANDER"` if no Motor neurons are alive or none have labels.
pub fn best_motor_action(brain: &BrainMatrix) -> &'static str {
    brain
        .neurons_by_role(Role::Motor)
        .filter(|n| n.action_label.is_some())
        .max_by(|a, b| {
            a.activation
                .partial_cmp(&b.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .and_then(|n| {
            n.action_label.as_ref().and_then(|label| {
                PROCEDURAL_ACTIONS
                    .iter()
                    .copied()
                    .find(|&s| s == label.as_str())
            })
        })
        .unwrap_or("WANDER")
}

/// Converts a `&'static str` action name to an `ActionLabel`.
///
/// Used when creating an `Episode` to store the chosen action compactly.
///
/// Panics if `action` exceeds 16 bytes (should never happen with PROCEDURAL_ACTIONS).
pub fn action_to_label(action: &str) -> ActionLabel {
    ActionLabel::new(action)
}
