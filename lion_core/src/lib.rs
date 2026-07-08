// lion_core/src/lib.rs

pub mod brain;
pub mod constants;
pub mod encoder;
pub mod epigenome;
pub mod episode;
pub mod hebbian;
pub mod immune;
pub mod init;
pub mod language;
pub mod neuron;
pub mod propagation;
pub mod rng;
pub mod sandbox;
pub mod sovereign;
pub mod synapse;
pub mod ternary;
pub mod types;
pub mod workspace;
pub mod persist;

pub use persist::{
    load_from, load_snapshot, save_snapshot, save_to,
    BrainSnapshot, SnapshotSummary, SNAPSHOT_VERSION,
};

// Re-exports at crate root.
pub use brain::BrainMatrix;
pub use constants::*;
pub use encoder::{TernaryEncoder, TernaryEncoderConfig};
pub use epigenome::Epigenome;
pub use episode::{Episode, EpisodicBuffer, RewardSummary};
pub use hebbian::trace_alignment;
pub use immune::{
    sanitize_scalar, sanitize_strength, sanitize_vector,
    sanitize_weight, ImmuneReport,
};
pub use neuron::{ActionLabel, Neuron};
pub use propagation::{
    cosine_alignment, dot_product, vec_magnitude, SensoryInput,
};
pub use rng::BrainRng;
pub use sandbox::{
    child_seed, evaluate_fitness, mutate_graph,
    run_night_cycle, run_night_cycle_parallel, NightCycleReport,
};
pub use sovereign::{Sovereign, TickResult};
pub use synapse::Synapse;
pub use ternary::{
    assume_len_multiple_of_8, f32_to_i8, gemv_row_simd_friendly,
    i8_to_f32, pack_weights, packed_byte_count, ternary_gemv,
    ternary_gemv_auto, ternary_gemv_dispatch, unpack_weight,
    unpack_weight_row, Activation, TernaryLayer,
    TERNARY_NEG, TERNARY_POS, TERNARY_ZERO, WEIGHTS_PER_BYTE,
};
pub use types::{GenIndex, MemoryTrace, Role};
pub use workspace::{
    action_to_label, best_motor_action, extract_action,
    gather_consciousness, WORKSPACE_TOP_K,
};
pub use language::{
    LanguageMotor, Tokenizer,
    target_speech_for_action, target_speech_for_input,
};

// Include tests module if testing
#[cfg(test)]
mod tests {
    mod arena_tests;
    mod init_tests;
    mod propagation_tests;
    mod hebbian_tests;
    mod immune_tests;
    mod episode_tests;
    mod workspace_tests;
    mod sovereign_tests;
    mod sandbox_tests;
    mod ternary_tests;
    mod parallel_tests;
}
