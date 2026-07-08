// lion_core/src/constants.rs

use crate::types::Role;

/// Number of f32 values in every feature vector.
/// Matches Python: DNACore.feature_size = 32
pub const FEATURE_SIZE: usize = 32;

/// Maximum number of MemoryTrace entries stored per neuron.
/// Matches Python: DNACore.max_neuron_traces = 5
pub const MAX_TRACES: usize = 5;

/// Maximum number of neurons the arena pre-allocates.
/// Must be a power of 2 for cache alignment.
pub const MAX_NEURONS: usize = 1024;

/// Maximum number of synapses the arena pre-allocates.
/// Approximately 4x neurons to support dense connectivity.
pub const MAX_SYNAPSES: usize = 8192;

/// Base rate at which memory trace strength decays each tick.
/// Matches Python: DNACore.trace_decay_rate = 0.95
pub const TRACE_DECAY_RATE: f32 = 0.95;

/// Learning rate for Hebbian plasticity updates.
/// Matches Python: DNACore.hebbian_rate = 0.05
pub const HEBBIAN_RATE: f32 = 0.05;

/// Base probability that a weight is mutated during the night cycle.
/// Matches Python: DNACore.base_mutation_rate = 0.1
pub const BASE_MUTATION_RATE: f32 = 0.1;

/// Minimum trace saturation threshold for neuron mitosis eligibility.
/// Matches Python: NeuronNode.is_overloaded() threshold = 0.8
pub const OVERLOAD_THRESHOLD: f32 = 0.8;

/// Minimum absolute activation value to be considered "alive" during propagation.
/// Matches Python: if abs(pre_act) < 0.01: continue
pub const ACTIVATION_DEAD_ZONE: f32 = 0.01;

/// Hard clamp ceiling on synapse weights during Hebbian updates.
/// Matches Python: syn.weight = max(-2.0, min(2.0, syn.weight))
pub const WEIGHT_MAX: f32 = 2.0;
pub const WEIGHT_MIN: f32 = -2.0;

/// Fitness improvement margin required before replacing the sovereign brain.
/// Ensures the night cycle doesn't hot-swap for negligible gains.
pub const EVOLUTION_MARGIN: f64 = 0.001;

// ─── Appended to lion_core/src/constants.rs ──────────────────────────────────

/// Number of neurons spawned per non-motor role during core brain initialization.
///
/// Matches Python:
///   for _ in range(3):
///       self.add_neuron(NeuronNode(role, self.dna))
pub const NEURONS_PER_ROLE: usize = 3;

/// The full set of procedural actions bound to Motor neurons.
///
/// Matches Python:
///   self.procedural_actions = ["WANDER", "FORAGE", "FLEE", "ATTACK"]
///
/// Every string must be ≤ 16 bytes to fit inside `ActionLabel`.
pub const PROCEDURAL_ACTIONS: &[&str] = &["WANDER", "FORAGE", "FLEE", "ATTACK"];

/// Number of Motor neurons spawned — one per procedural action.
pub const MOTOR_NEURON_COUNT: usize = PROCEDURAL_ACTIONS.len();

/// Total neurons in the initial brain before any mitosis or pruning.
///
/// = (roles - 1) * NEURONS_PER_ROLE + MOTOR_NEURON_COUNT
/// = (4 - 1) * 3 + 4 = 13
///
/// Roles: Vision, Memory, Danger = 3 non-motor roles × 3 neurons each = 9
/// Motor: 4 neurons (one per action)
/// Total: 13
pub const INITIAL_NEURON_COUNT: usize =
    (Role::COUNT - 1) * NEURONS_PER_ROLE + MOTOR_NEURON_COUNT;

/// The probability threshold above which a random synapse is NOT created.
///
/// Matches Python:
///   if pre != post and random.random() > 0.5:
///       self.synapses[pre][post] = Synapse(...)
pub const SYNAPSE_CREATION_PROB: f32 = 0.5;

/// Uniform range for initial synapse weights.
///
/// Matches Python: random.uniform(-0.5, 0.5)
pub const INITIAL_WEIGHT_MIN: f32 = -0.5;
pub const INITIAL_WEIGHT_MAX: f32 =  0.5;

/// Uniform range for initial neuron base_vector values.
///
/// Matches Python:
///   self.base_vector = np.random.uniform(-0.1, 0.1, dna.feature_size)
pub const BASE_VECTOR_INIT_MIN: f32 = -0.1;
pub const BASE_VECTOR_INIT_MAX: f32 =  0.1;

/// Uniform range for weight mutation deltas during the night cycle.
///
/// Matches Python:
///   syn.weight += random.uniform(-0.3, 0.3)
pub const MUTATION_DELTA_MIN: f32 = -0.3;
pub const MUTATION_DELTA_MAX: f32 =  0.3;

/// Uniform range for base_vector jitter applied during neuron mitosis.
///
/// Matches Python:
///   child.base_vector = parent.base_vector.copy() + np.random.uniform(-0.05, 0.05, ...)
pub const MITOSIS_JITTER_MIN: f32 = -0.05;
pub const MITOSIS_JITTER_MAX: f32 =  0.05;

/// Synapses with |weight| below this threshold are pruned during mutation.
/// Matches Python: if abs(syn.weight) < 0.05
pub const SYNAPSE_PRUNE_THRESHOLD: f32 = 0.05;

/// Jitter range for inherited synapse weights during mitosis.
/// Matches Python: syn.weight + random.uniform(-0.1, 0.1)
pub const MITOSIS_SYNAPSE_JITTER_MIN: f32 = -0.1;
pub const MITOSIS_SYNAPSE_JITTER_MAX: f32 =  0.1;

/// Default population size for the evolutionary night cycle.
/// Matches Python: population_size=15 in trigger_sleep_cycle()
pub const NIGHT_CYCLE_POPULATION: usize = 15;

/// Complexity cost per alive synapse subtracted from fitness.
/// Matches Python: fitness -= total_synapses * 0.001
pub const FITNESS_SYNAPSE_COST: f64 = 0.001;

/// Complexity cost per alive neuron subtracted from fitness.
/// Matches Python: fitness -= total_neurons * 0.005
pub const FITNESS_NEURON_COST: f64 = 0.005;

/// Penalty multiplier applied when a child repeats a harmful action.
/// Matches Python: fitness -= abs(reward) * 2.0
pub const FITNESS_REPEAT_PENALTY: f64 = 2.0;

/// Reward multiplier for avoiding a previously punished action.
/// Matches Python: fitness += abs(reward) * 0.1
pub const FITNESS_EVASION_REWARD: f64 = 0.1;

// ── Appended to lion_core/src/constants.rs ───────────────────────────────────

/// LCG multiplier for per-child RNG seed derivation.
/// Knuth (1997), MMIX multiplier — maximally spread, period = 2^64.
pub const LCG_MULTIPLIER: u64 = 6364136223846793005;

/// Input size threshold above which the SIMD-friendly GEMV variant is used.
/// Below this threshold, the branchless variant (Phase 8) is used.
/// Tune this for your CPU's SIMD register width and cache parameters.
pub const GEMV_SIMD_THRESHOLD: usize = 32;

