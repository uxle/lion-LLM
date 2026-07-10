// lion_core/src/versioning.rs — Phase 14: Version Registry & Rollback
//
// Tracks every brain snapshot that was saved during evolution.
// Provides rollback to any previous version.
// Registry stored as JSON at ~/.lionai/versions/registry.json

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

// =============================================================================
// TYPES
// =============================================================================

/// Metadata for one saved brain snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainVersion {
    /// Sequential version ID (0, 1, 2, …).
    pub id: u32,

    /// Evolutionary generation at time of snapshot.
    pub generation: u32,

    /// Best fitness score seen when this version was promoted.
    pub fitness: f64,

    /// Unix-ish timestamp (tick counter used as proxy).
    pub timestamp: u64,

    /// Human description of why this version was saved.
    pub reason: String,

    /// Filename relative to the versions directory (e.g. "v3.bin").
    pub file: String,
}

/// The registry of all saved versions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VersionRegistry {
    pub versions:    Vec<BrainVersion>,
    pub current_id:  u32,
    pub next_id:     u32,
}

// =============================================================================
// IMPLEMENTATION
// =============================================================================

impl VersionRegistry {
    /// Register a new checkpoint and return its ID.
    ///
    /// The caller must separately write the actual `.bin` file using
    /// `snapshot_path(id, dir)`.
    pub fn checkpoint(
        &mut self,
        generation: u32,
        fitness:    f64,
        reason:     &str,
        timestamp:  u64,
    ) -> u32 {
        let id   = self.next_id;
        self.next_id += 1;
        let file = format!("v{}.bin", id);
        self.versions.push(BrainVersion {
            id,
            generation,
            fitness,
            timestamp,
            reason: reason.to_string(),
            file,
        });
        self.current_id = id;
        id
    }

    /// Path where the snapshot for `id` should be stored.
    pub fn snapshot_path(&self, id: u32, versions_dir: &Path) -> PathBuf {
        versions_dir.join(format!("v{}.bin", id))
    }

    /// Path for the current snapshot.
    pub fn current_path(&self, versions_dir: &Path) -> Option<PathBuf> {
        self.get(self.current_id).map(|_| self.snapshot_path(self.current_id, versions_dir))
    }

    /// Find a version by ID.
    pub fn get(&self, id: u32) -> Option<&BrainVersion> {
        self.versions.iter().find(|v| v.id == id)
    }

    /// The version with the highest fitness.
    pub fn best(&self) -> Option<&BrainVersion> {
        self.versions.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// The most recently added version.
    pub fn latest(&self) -> Option<&BrainVersion> {
        self.versions.last()
    }

    /// All versions, newest first.
    pub fn sorted_newest(&self) -> Vec<&BrainVersion> {
        let mut v: Vec<&BrainVersion> = self.versions.iter().collect();
        v.sort_by(|a, b| b.id.cmp(&a.id));
        v
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
}
