// lion_core/src/persist.rs

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};

use crate::brain::BrainMatrix;
use crate::encoder::TernaryEncoder;
use crate::episode::EpisodicBuffer;
use crate::language::LanguageMotor;

// =============================================================================
// FORMAT VERSION
// =============================================================================

/// Version stamp written at the start of every snapshot file.
/// Increment when the snapshot format changes to prevent silent corruption.
pub const SNAPSHOT_VERSION: u32 = 1;

// =============================================================================
// BRAIN SNAPSHOT
// =============================================================================

/// The complete serializable state of a LionAI sovereign agent.
///
/// `BrainRng` is excluded because `StdRng` is not `Serialize`.
/// On restore, a new RNG is created from `rng_seed`.
///
/// Includes the ternary encoder so the full perception-to-action pipeline
/// is preserved in a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    /// Format version. Checked on load.
    pub version: u32,

    /// The tick counter at the moment of saving.
    pub tick: u64,

    /// The evolutionary generation at the moment of saving.
    pub generation: u32,

    /// Seed used to recreate the RNG on load.
    /// The restored RNG will produce the same sequence as the original
    /// from this point, giving reproducible post-load behaviour.
    pub rng_seed: u64,

    /// Full cognitive graph: neurons, synapses, epigenome.
    pub brain: BrainMatrix,

    /// Full episode history for night-cycle replay.
    pub episodic_buffer: EpisodicBuffer,

    /// Ternary encoder for sensory preprocessing (optional).
    pub encoder: Option<TernaryEncoder>,

    /// Language Motor Cortex (the text generator). None for very old snapshots.
    pub language_motor: Option<LanguageMotor>,
}

impl BrainSnapshot {
    /// Creates a snapshot from the given components.
    pub fn new(
        tick:            u64,
        generation:      u32,
        rng_seed:        u64,
        brain:           BrainMatrix,
        episodic_buffer: EpisodicBuffer,
        encoder:         Option<TernaryEncoder>,
        language_motor:  Option<LanguageMotor>,
    ) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            tick,
            generation,
            rng_seed,
            brain,
            episodic_buffer,
            encoder,
            language_motor,
        }
    }

    /// Returns a human-readable summary of the snapshot.
    pub fn summary(&self) -> SnapshotSummary {
        SnapshotSummary {
            version:        self.version,
            tick:           self.tick,
            generation:     self.generation,
            neuron_count:   self.brain.alive_neuron_count(),
            synapse_count:  self.brain.alive_synapse_count(),
            episode_count:  self.episodic_buffer.len(),
            stress_level:   self.brain.epigenome.accumulated_stress,
            has_encoder:    self.encoder.is_some(),
        }
    }
}

/// A lightweight summary returned after save/load operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSummary {
    pub version:       u32,
    pub tick:          u64,
    pub generation:    u32,
    pub neuron_count:  usize,
    pub synapse_count: usize,
    pub episode_count: usize,
    pub stress_level:  f32,
    pub has_encoder:   bool,
}

// =============================================================================
// SAVE
// =============================================================================

/// Serializes a `BrainSnapshot` to a binary file at the given path.
///
/// Creates parent directories if they do not exist.
/// Overwrites the file if it already exists.
///
/// Format: `SNAPSHOT_VERSION (u32 LE)` + `bincode(BrainSnapshot)`
///
/// # Errors
/// Returns an `io::Error` if the file cannot be written.
pub fn save_snapshot(snapshot: &BrainSnapshot, path: &Path) -> io::Result<SnapshotSummary> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let bytes = serialize(snapshot)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut file = fs::File::create(path)?;

    // Write 4-byte version header.
    file.write_all(&SNAPSHOT_VERSION.to_le_bytes())?;

    // Write bincode payload.
    file.write_all(&bytes)?;

    Ok(snapshot.summary())
}

/// Convenience wrapper: save to a string path.
pub fn save_to(snapshot: &BrainSnapshot, path: &str) -> io::Result<SnapshotSummary> {
    save_snapshot(snapshot, Path::new(path))
}

// =============================================================================
// LOAD
// =============================================================================

/// Deserializes a `BrainSnapshot` from a binary file.
///
/// Validates the version header before deserialization.
///
/// # Errors
/// - `io::ErrorKind::InvalidData` if the version does not match `SNAPSHOT_VERSION`.
/// - `io::Error` if the file cannot be read.
pub fn load_snapshot(path: &Path) -> io::Result<BrainSnapshot> {
    let mut file  = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    if bytes.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Snapshot file too small (missing version header)",
        ));
    }

    // Check version header.
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != SNAPSHOT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Snapshot version mismatch: file has v{}, current is v{}",
                version, SNAPSHOT_VERSION
            ),
        ));
    }

    // Deserialize the payload (skip 4-byte header).
    let snapshot: BrainSnapshot = deserialize(&bytes[4..])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(snapshot)
}

/// Convenience wrapper: load from a string path.
pub fn load_from(path: &str) -> io::Result<BrainSnapshot> {
    load_snapshot(Path::new(path))
}

// =============================================================================
// SNAPSHOT SIZE ESTIMATION
// =============================================================================

/// Estimates the serialized size of a snapshot in bytes.
///
/// Useful for pre-checking available disk space.
pub fn estimated_snapshot_size(snapshot: &BrainSnapshot) -> usize {
    // bincode overhead is roughly:
    //   4 bytes version header
    //   + neuron_count  × ~220 bytes (Neuron with MAX_TRACES)
    //   + synapse_count × ~16 bytes
    //   + episode_count × ~200 bytes average
    //   + encoder size (if present)
    let neuron_bytes  = snapshot.brain.alive_neuron_count()  * 220;
    let synapse_bytes = snapshot.brain.alive_synapse_count() * 16;
    let episode_bytes = snapshot.episodic_buffer.len()       * 200;
    let encoder_bytes = snapshot.encoder.as_ref()
        .map(|e| e.total_memory_bytes())
        .unwrap_or(0);

    4 + neuron_bytes + synapse_bytes + episode_bytes + encoder_bytes
}
