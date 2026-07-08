// lion_core/src/propagation.rs

use std::collections::HashMap;

use crate::brain::BrainMatrix;
use crate::constants::{ACTIVATION_DEAD_ZONE, FEATURE_SIZE, MAX_NEURONS};
use crate::types::Role;
use crate::immune::ImmuneReport;

// =============================================================================
// SENSORY INPUT TYPE
// =============================================================================

/// A frame of sensory data delivered to the brain on a single tick.
///
/// Maps each active Role to a raw feature vector of length FEATURE_SIZE.
/// Not every role must be present — omitted roles receive no injection.
///
/// Matches Python:
///   sensory_inputs: Dict[str, np.ndarray]
///   e.g. {"vision": vec_forest, "danger": vec_danger}
///
/// Usage:
///   let mut frame = SensoryInput::new();
///   frame.insert(Role::Vision, vision_vec);
///   frame.insert(Role::Danger, danger_vec);
pub type SensoryInput = HashMap<Role, [f32; FEATURE_SIZE]>;

// =============================================================================
// VECTOR MATH PRIMITIVES
// =============================================================================

/// Computes the Euclidean magnitude (L2 norm) of a fixed-size f32 vector.
///
/// ‖v‖ = sqrt(Σ v[i]²)
///
/// Returns 0.0 for a zero vector.
/// Called once per neuron per sensory injection.
#[inline]
pub fn vec_magnitude(v: &[f32; FEATURE_SIZE]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Computes the dot product of two fixed-size f32 vectors.
///
/// dot(a, b) = Σ a[i] * b[i]
///
/// Both vectors must have length FEATURE_SIZE.
/// This is the inner product used for cosine alignment.
#[inline]
pub fn dot_product(a: &[f32; FEATURE_SIZE], b: &[f32; FEATURE_SIZE]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the cosine alignment between a neuron's base_vector and an input vector.
///
/// ```text
///         dot(base, input)
/// A = ─────────────────────────────
///      ‖base‖ · ‖input‖ + ε
/// ```
///
/// ε = 1e-9 prevents division-by-zero when either vector is zero.
///
/// Range: [-1.0, +1.0]
///   +1.0 → perfectly aligned    → strong excitation
///    0.0 → orthogonal           → no response
///   -1.0 → perfectly opposed    → strong inhibition
///
/// The epsilon guard is critical: a neuron freshly inserted with a zero
/// base_vector (which should not happen after Phase 2, but is defensively
/// handled here) will return alignment = 0.0 instead of NaN.
///
/// Matches Python:
///   alignment = float(np.dot(n.base_vector, vec)) / (vec_mag * base_mag + 1e-9)
#[inline]
pub fn cosine_alignment(
    base:  &[f32; FEATURE_SIZE],
    input: &[f32; FEATURE_SIZE],
    input_mag: f32,
) -> f32 {
    let base_mag = vec_magnitude(base);
    let denom = base_mag * input_mag + 1e-9;
    dot_product(base, input) / denom
}

// =============================================================================
// PROPAGATION IMPL ON BRAIN MATRIX
// =============================================================================

impl BrainMatrix {
    // -------------------------------------------------------------------------
    // SENSORY INJECTION
    // -------------------------------------------------------------------------

    /// Injects a single sensory modality into all neurons of the matching role.
    ///
    /// For each neuron whose role matches `modality`:
    ///   1. Compute cosine alignment between the neuron's base_vector and the input.
    ///   2. Set activation = tanh(alignment * input_magnitude).
    ///   3. Store the input as a new memory trace on the neuron.
    ///
    /// The trace storage call matches Python's `n.add_trace(vec)` inside update().
    /// Trace decay and eviction are handled inside `Neuron::add_trace()` (Phase 1).
    ///
    /// Matches Python (from LNNHyperNodeV16.update()):
    ///   for n in target_neurons:
    ///       base_mag = float(np.linalg.norm(n.base_vector))
    ///       alignment = float(np.dot(n.base_vector, vec)) / (vec_mag * base_mag + 1e-9)
    ///       n.activation = np.tanh(alignment * vec_mag)
    ///       n.add_trace(vec)
    ///
    /// # Sandbox variant
    /// When called from the fitness evaluator (no trace storage needed),
    /// use `inject_sensory_no_trace()` instead. This avoids polluting
    /// neuron memory during deterministic evaluation.
    pub fn inject_sensory(
        &mut self,
        modality: Role,
        input:    &[f32; FEATURE_SIZE],
    ) {
        let input_mag = vec_magnitude(input);

        // Iterate over the flat neuron arena.
        // We cannot call self.neurons_by_role() here because that returns
        // an immutable iterator and we need mutable access to set activation.
        for n in self.neurons.iter_mut() {
            if !n.alive || n.role != modality {
                continue;
            }

            let alignment = cosine_alignment(&n.base_vector, input, input_mag);
            n.activation  = (alignment * input_mag).tanh();

            // Store the raw input as a memory trace.
            n.add_trace(*input);
        }
    }

    /// Same as `inject_sensory()` but does NOT write memory traces.
    ///
    /// Use this inside the evolutionary sandbox fitness evaluator.
    /// The sandbox replays historical episodes to score child brains —
    /// writing traces during evaluation would corrupt the child's memory
    /// state and make fitness scores non-comparable across children.
    ///
    /// Matches Python (from EvolutionarySandbox.evaluate_fitness()):
    ///   for n in target_neurons:
    ///       alignment = float(np.dot(n.base_vector, vec)) / (vec_mag * base_mag + 1e-9)
    ///       n.activation = np.tanh(alignment * vec_mag)
    ///       # NOTE: No n.add_trace(vec) here
    pub fn inject_sensory_no_trace(
        &mut self,
        modality: Role,
        input:    &[f32; FEATURE_SIZE],
    ) {
        let input_mag = vec_magnitude(input);

        for n in self.neurons.iter_mut() {
            if !n.alive || n.role != modality {
                continue;
            }

            let alignment = cosine_alignment(&n.base_vector, input, input_mag);
            n.activation  = (alignment * input_mag).tanh();
        }
    }

    /// Injects all modalities in a SensoryInput frame in a single call.
    ///
    /// Resets all activations first, then injects each modality.
    /// This is the standard live-execution call path.
    ///
    /// Matches Python (from LNNHyperNodeV16.update()):
    ///   self.graph.reset_activations()
    ///   for modality, raw_vec in sensory_inputs.items():
    ///       ...inject...
    pub fn inject_frame(&mut self, frame: &SensoryInput) {
        self.reset_activations();
        for (modality, input) in frame {
            self.inject_sensory(*modality, input);
        }
    }

    /// Injects all modalities without trace storage. Used in the sandbox.
    ///
    /// Does NOT call reset_activations() — the sandbox calls that manually
    /// before each episode replay.
    pub fn inject_frame_no_trace(&mut self, frame: &SensoryInput) {
        for (modality, input) in frame {
            self.inject_sensory_no_trace(*modality, input);
        }
    }

    // -------------------------------------------------------------------------
    // SIGNAL PROPAGATION
    // -------------------------------------------------------------------------

    /// Propagates signals across the synapse graph for `steps` iterations.
    ///
    /// Each step is a two-pass algorithm:
    ///
    ///   Pass 1 (Accumulate):
    ///     For every alive synapse (pre → post):
    ///       If |pre.activation| > ACTIVATION_DEAD_ZONE:
    ///         buffer[post.index] += pre.activation * synapse.weight
    ///
    ///   Pass 2 (Squash):
    ///     For every alive neuron:
    ///       neuron.activation = tanh(buffer[neuron.index])
    ///
    /// The dead zone threshold (0.01) skips near-zero pre-synaptic neurons,
    /// which is both biologically motivated and a performance optimization —
    /// most neurons are inactive at any given tick.
    ///
    /// # The Two-Pass Necessity in Rust
    ///
    /// Python can read `self.neurons[pre_id].activation` and write
    /// `next_activations[post_id]` simultaneously because it uses a dict.
    /// Rust's borrow checker forbids simultaneous mutable and immutable
    /// borrows of the same `Vec<Neuron>`.
    ///
    /// Solution: `accumulator` is a flat `[f32; MAX_NEURONS]` array — a
    /// stack-allocated, zero-cost intermediate buffer. Pass 1 reads neurons
    /// immutably into the buffer. Pass 2 writes from the buffer into neurons
    /// mutably. No unsafe, no allocation.
    ///
    /// Matches Python:
    ///   def propagate(self, steps: int = 2, apply_hebbian: bool = False, ...):
    ///       for _ in range(steps):
    ///           next_activations = {nid: 0.0 for nid in self.neurons}
    ///           for pre_id, connections in self.synapses.items():
    ///               pre_act = self.neurons[pre_id].activation
    ///               if abs(pre_act) < 0.01: continue
    ///               for post_id, syn in connections.items():
    ///                   next_activations[post_id] += pre_act * syn.weight
    ///           for nid, n_sum in next_activations.items():
    ///               self.neurons[nid].process_activation(n_sum)
    pub fn propagate_steps(&mut self, steps: usize) {
        // Stack-allocated accumulation buffer. Indexed by neuron slot index.
        // Reused across all steps — zeroed at the start of each step.
        let mut accumulator = [0.0_f32; MAX_NEURONS];

        for _ in 0..steps {
            // Zero out the accumulator for this step.
            accumulator.iter_mut().for_each(|x| *x = 0.0);

            // ── Pass 1: Accumulate ──────────────────────────────────────────
            //
            // Read all synapse contributions into the accumulator.
            // This is a fully immutable pass over self.neurons and self.synapses.
            for synapse in self.synapses.iter() {
                if !synapse.alive {
                    continue;
                }

                // Validate that both endpoints are still alive.
                // A synapse might outlive one of its neurons if the neuron
                // was pruned without cleaning up its synapses (Phase 7 job).
                // Here we defend against that case silently.
                let pre_idx  = synapse.pre_id.index;
                let post_idx = synapse.post_id.index;

                if !self.is_valid_neuron(synapse.pre_id)
                    || !self.is_valid_neuron(synapse.post_id)
                {
                    continue;
                }

                let pre_activation = self.neurons[pre_idx].activation;

                // Dead zone: skip neurons with negligible activation.
                // Matches Python: if abs(pre_act) < 0.01: continue
                if pre_activation.abs() < ACTIVATION_DEAD_ZONE {
                    continue;
                }

                // Accumulate the weighted contribution to the post-synaptic neuron.
                accumulator[post_idx] += pre_activation * synapse.weight;
            }

            // ── Pass 2: Squash ──────────────────────────────────────────────
            //
            // Apply tanh to each neuron's accumulated input.
            // This is a fully mutable pass — safe because Pass 1 is complete.
            for neuron in self.neurons.iter_mut() {
                if !neuron.alive {
                    continue;
                }
                neuron.process_activation(accumulator[neuron.id.index]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // FULL TICK (COMPOSE INJECTION + PROPAGATION)
    // -------------------------------------------------------------------------

    /// Executes one complete live-execution forward tick:
    ///   1. Reset all activations to 0.0
    ///   2. Inject all sensory modalities (with trace storage)
    ///   3. Propagate for `steps` iterations
    ///   4. Scan and heal the arena (immune system)
    ///
    /// The immune scan in step 4 is the key addition over Phase 3.
    /// It ensures no NaN or Inf values leave this function.
    ///
    /// Matches Python (from LNNHyperNodeV16.update()):
    ///   self.graph.reset_activations()
    ///   ...inject...
    ///   self.graph.propagate(steps=2, apply_hebbian=False)
    ///   self.immune.scan_and_heal(self.graph.neurons)
    pub fn tick(&mut self, frame: &SensoryInput, steps: usize) -> ImmuneReport {
        self.inject_frame(frame);
        self.propagate_steps(steps);
        self.scan_and_heal() // Returns report for optional logging.
    }

    /// Executes one complete forward tick WITHOUT trace storage.
    ///
    /// Used by the evolutionary sandbox to replay historical episodes.
    /// Trace-free because sandbox evaluation must not modify neuron memory.
    ///
    /// Matches Python (from EvolutionarySandbox.evaluate_fitness()):
    ///   graph.reset_activations()
    ///   for modality, vec in episode.sensory_inputs.items():
    ///       ...inject (no trace)...
    ///   graph.propagate(steps=2, apply_hebbian=True, plasticity=epi.plasticity)
    ///
    /// Note: Hebbian plasticity (apply_hebbian=True) is implemented in Phase 4.
    /// For now this method only handles the propagation-only path.
    pub fn tick_sandbox(&mut self, frame: &SensoryInput, steps: usize) {
        self.reset_activations();
        self.inject_frame_no_trace(frame);
        self.propagate_steps(steps);
    }

    // -------------------------------------------------------------------------
    // MOTOR READOUT
    // -------------------------------------------------------------------------

    /// Returns the action label of the most activated Motor neuron.
    ///
    /// If multiple Motor neurons have equal activation, the one encountered
    /// first in the arena is returned (stable, deterministic).
    ///
    /// Returns `None` if no Motor neurons are alive or no action label is set.
    ///
    /// Matches Python (from EvolutionarySandbox.extract_action(), exploit path):
    ///   motor_neurons = [n for n in graph.neurons.values() if n.role == Role.MOTOR]
    ///   best_motor = max(motor_neurons, key=lambda n: n.activation)
    ///   return best_motor.action_label if best_motor.action_label else "WANDER"
    pub fn best_motor_action(&self) -> Option<&str> {
        self.neurons
            .iter()
            .filter(|n| n.alive && n.role == Role::Motor && n.action_label.is_some())
            .max_by(|a, b| {
                a.activation
                    .partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|n| n.action_label.as_ref().map(|l| l.as_str()))
    }

    // -------------------------------------------------------------------------
    // DIAGNOSTICS
    // -------------------------------------------------------------------------

    /// Returns the mean absolute activation across all alive neurons.
    ///
    /// Useful for detecting dead brains (mean ≈ 0.0) or saturated brains
    /// (mean ≈ 1.0 after too many propagation steps).
    pub fn mean_absolute_activation(&self) -> f32 {
        let alive: Vec<f32> = self
            .neurons
            .iter()
            .filter(|n| n.alive)
            .map(|n| n.activation.abs())
            .collect();

        if alive.is_empty() {
            return 0.0;
        }

        alive.iter().sum::<f32>() / alive.len() as f32
    }

    /// Returns the activation of a specific neuron, or 0.0 if invalid.
    ///
    /// Convenience method for tests and diagnostics.
    pub fn activation_of(&self, id: crate::types::GenIndex) -> f32 {
        self.get_neuron(id)
            .map(|n| n.activation)
            .unwrap_or(0.0)
    }

    /// Propagates signals AND applies Hebbian plasticity after the final step.
    ///
    /// This is the combined operation used by the evolutionary sandbox
    /// during fitness evaluation.
    ///
    /// Hebbian is applied ONCE after all propagation steps complete —
    /// not after each step. This matches Python's call sequence:
    ///   graph.propagate(steps=2, apply_hebbian=True, plasticity=epi.plasticity)
    ///
    /// Matches Python (from EvolutionarySandbox.evaluate_fitness()):
    ///   graph.propagate(steps=2, apply_hebbian=True, plasticity=epi.plasticity)
    pub fn propagate_with_hebbian(&mut self, steps: usize, plasticity: f32) {
        self.propagate_steps(steps);
        self.apply_hebbian_plasticity(plasticity);
    }

    /// Full sandbox tick: reset → inject (no trace) → propagate → Hebbian.
    ///
    /// This replaces the Phase 3 `tick_sandbox()` for use cases where
    /// the sandbox evaluator needs Hebbian weight updates during replay.
    ///
    /// The Phase 3 `tick_sandbox()` (no Hebbian) is still available for
    /// cases where plasticity is not needed (e.g., action-only readouts).
    ///
    /// Matches Python (from EvolutionarySandbox.evaluate_fitness()):
    ///   graph.reset_activations()
    ///   for modality, vec in episode.sensory_inputs.items():
    ///       ...inject (no trace)...
    ///   graph.propagate(steps=2, apply_hebbian=True, plasticity=epi.plasticity)
    pub fn tick_sandbox_hebbian(
        &mut self,
        frame:      &SensoryInput,
        steps:      usize,
        plasticity: f32,
    ) {
        self.reset_activations();
        self.inject_frame_no_trace(frame);
        self.propagate_with_hebbian(steps, plasticity);
    }
}
