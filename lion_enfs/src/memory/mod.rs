// lion_enfs/src/memory/mod.rs — ENFS Cognitive Memory Tiers

pub mod sensory;
pub mod working;
pub mod domain;
pub mod archive;

pub use sensory::SensoryMemory;
pub use working::WorkingMemory;
pub use domain::{DomainMemory, DomainKey};
pub use archive::ArchiveMemory;

use crate::error::EnfsResult;

/// The full 4-tier memory hierarchy.
/// Each tier exposes a common `store` / `fetch` interface over binary payloads.
pub struct MemoryHierarchy {
    pub sensory:  SensoryMemory,
    pub working:  WorkingMemory,
    pub domain:   DomainMemory,
    pub archive:  ArchiveMemory,
}

impl MemoryHierarchy {
    pub fn new() -> Self {
        Self {
            sensory:  SensoryMemory::new(),
            working:  WorkingMemory::new(256),      // 256 slot ring buffer
            domain:   DomainMemory::new(),
            archive:  ArchiveMemory::new(),
        }
    }

    /// Promote a working memory entry to the appropriate domain store.
    pub fn promote_to_domain(&mut self, key: &str, domain_key: DomainKey) -> EnfsResult<()> {
        if let Some(payload) = self.working.evict(key) {
            self.domain.store(domain_key, key, payload)?;
        }
        Ok(())
    }

    /// Demote a cold domain entry to archive.
    pub fn demote_to_archive(&mut self, domain_key: DomainKey, key: &str) -> EnfsResult<()> {
        if let Some(payload) = self.domain.evict(domain_key, key) {
            self.archive.store(key, payload);
        }
        Ok(())
    }
}

impl Default for MemoryHierarchy {
    fn default() -> Self { Self::new() }
}
