// lion_core/src/lfmf.rs — LFMF Container & ENFS Memory Tier Hierarchy
//
// Implements the Lion Flexible Model Format (LFMF) header container parser/builder
// and the Einstein Neurons File System (ENFS) tiered memory manager.

use serde::{Deserialize, Serialize};

// =============================================================================
// LFMF MAGIC & CONSTANTS
// =============================================================================

pub const LFMF_MAGIC: [u8; 4] = [0x4C, 0x46, 0x4D, 0x46]; // "LFMF"
pub const LFMF_VERSION_MAJOR: u16 = 1;
pub const LFMF_VERSION_MINOR: u16 = 0;

// =============================================================================
// LFMF CONTAINER HEADER
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LfmfHeader {
    pub magic: [u8; 4],
    pub version_major: u16,
    pub version_minor: u16,
    pub alignment_bytes: u32,
    pub shard_count: u32,
    pub model_name: String,
    pub metadata_json: String,
}

impl LfmfHeader {
    pub fn new(model_name: impl Into<String>, shard_count: u32) -> Self {
        Self {
            magic: LFMF_MAGIC,
            version_major: LFMF_VERSION_MAJOR,
            version_minor: LFMF_VERSION_MINOR,
            alignment_bytes: 64,
            shard_count,
            model_name: model_name.into(),
            metadata_json: "{}".to_string(),
        }
    }

    /// Validate header magic bytes and format version.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != LFMF_MAGIC {
            return Err("Invalid LFMF magic number".to_string());
        }
        if self.version_major != LFMF_VERSION_MAJOR {
            return Err(format!("Unsupported LFMF major version: {}", self.version_major));
        }
        Ok(())
    }

    /// Serialize header to binary buffer.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Parse header from binary buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let header: Self = bincode::deserialize(bytes).map_err(|e| e.to_string())?;
        header.validate()?;
        Ok(header)
    }
}

// =============================================================================
// ENFS MEMORY TIERS
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    Sensory,
    Working,
    Domain(String),
    Archive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredMemoryRecord {
    pub id: u64,
    pub content: String,
    pub tier: MemoryTier,
    pub access_count: u32,
    pub last_access_tick: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TieredMemoryManager {
    pub records: Vec<TieredMemoryRecord>,
    pub promotion_threshold: u32,
    pub archive_tick_threshold: u64,
}

impl TieredMemoryManager {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            promotion_threshold: 5,
            archive_tick_threshold: 1000,
        }
    }

    pub fn add_record(&mut self, id: u64, content: String, initial_tier: MemoryTier, current_tick: u64) {
        self.records.push(TieredMemoryRecord {
            id,
            content,
            tier: initial_tier,
            access_count: 1,
            last_access_tick: current_tick,
        });
    }

    /// Access a memory record and trigger tier promotion if access threshold is met.
    pub fn touch_record(&mut self, id: u64, current_tick: u64) -> Option<&TieredMemoryRecord> {
        let promotion_threshold = self.promotion_threshold;
        if let Some(record) = self.records.iter_mut().find(|r| r.id == id) {
            record.access_count += 1;
            record.last_access_tick = current_tick;

            // Promotion policy: Demoted/Archive -> Domain/Working if accessed frequently
            if record.access_count >= promotion_threshold && record.tier == MemoryTier::Archive {
                record.tier = MemoryTier::Working;
            }
            return Some(record);
        }
        None
    }

    /// Demote stagnant records to Archive tier.
    pub fn consolidate_tiers(&mut self, current_tick: u64) -> usize {
        let threshold = self.archive_tick_threshold;
        let mut demoted = 0;
        for record in &mut self.records {
            if current_tick.saturating_sub(record.last_access_tick) > threshold && record.tier != MemoryTier::Archive {
                record.tier = MemoryTier::Archive;
                demoted += 1;
            }
        }
        demoted
    }
}
