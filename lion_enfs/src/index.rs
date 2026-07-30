// lion_enfs/src/index.rs — ENFS Tensor Index (O(1) name lookup)
//
// The index maps canonical tensor names to inode IDs in O(1).
// It is stored in binary in the index region of the volume (not human-readable).
// In RAM it lives as a HashMap for O(1) access.
// On disk it serializes as sorted (name_blake3_hash, inode_id) pairs for binary search.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::error::{EnfsError, EnfsResult};

/// One index entry: the BLAKE3 hash of the name → inode ID.
/// Stored in binary — the plaintext name is NEVER written to disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct IndexEntry {
    pub name_blake3: [u8; 32],   // BLAKE3 of canonical name — the only form stored on disk
    pub inode_id:    u64,
    pub domain:      u8,         // DomainTag numeric value
    pub _pad:        [u8; 7],
}

/// Domain tags for partitioned lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DomainTag {
    Root        = 0,
    Tensors     = 1,
    Language    = 2,
    Mathematics = 3,
    Physics     = 4,
    Science     = 5,
    Skills      = 6,
    Vision2d    = 7,
    Vision3d    = 8,
    Audio       = 9,
    Video       = 10,
    Safety      = 11,
    Intelligence= 12,
    Adapters    = 13,
    Cache       = 14,
    Checkpoints = 15,
    Custom      = 255,
}

impl DomainTag {
    pub fn from_path_prefix(prefix: &str) -> Self {
        match prefix {
            "tensors"       => DomainTag::Tensors,
            "language"      => DomainTag::Language,
            "mathematics"   => DomainTag::Mathematics,
            "physics"       => DomainTag::Physics,
            "science"       => DomainTag::Science,
            "skills"        => DomainTag::Skills,
            "2d"            => DomainTag::Vision2d,
            "3d"            => DomainTag::Vision3d,
            "audio"         => DomainTag::Audio,
            "video"         => DomainTag::Video,
            "safety"        => DomainTag::Safety,
            "intelligence"  => DomainTag::Intelligence,
            "adapters"      => DomainTag::Adapters,
            "cache"         => DomainTag::Cache,
            "checkpoints"   => DomainTag::Checkpoints,
            _               => DomainTag::Root,
        }
    }
}

/// In-memory O(1) index + serialisable on-disk sorted array.
#[derive(Debug, Default)]
pub struct TensorIndex {
    /// name_blake3 → inode_id (O(1) lookup)
    map:     HashMap<[u8; 32], u64>,
    /// domain → Vec<inode_id> for per-domain iteration
    domains: HashMap<u8, Vec<u64>>,
}

impl TensorIndex {
    pub fn new() -> Self {
        Self { map: HashMap::new(), domains: HashMap::new() }
    }

    /// Register a name (hashed, never stored in plaintext) → inode_id.
    pub fn insert(&mut self, name: &str, inode_id: u64, domain: DomainTag) {
        let hash = blake3_of(name.as_bytes());
        self.map.insert(hash, inode_id);
        self.domains
            .entry(domain as u8)
            .or_default()
            .push(inode_id);
    }

    /// O(1) lookup: hash the name, look up in the HashMap.
    pub fn lookup(&self, name: &str) -> EnfsResult<u64> {
        let hash = blake3_of(name.as_bytes());
        self.map
            .get(&hash)
            .copied()
            .ok_or_else(|| EnfsError::TensorNotFound { name: name.to_string() })
    }

    /// List all inode IDs in a domain.
    pub fn list_domain(&self, domain: DomainTag) -> Vec<u64> {
        self.domains
            .get(&(domain as u8))
            .cloned()
            .unwrap_or_default()
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let hash = blake3_of(name.as_bytes());
        self.map.remove(&hash).is_some()
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Serialize the index to a sorted binary buffer for writing to disk.
    /// Stored as: [u32 count][IndexEntry × count] — no plaintext names.
    pub fn to_bytes(&self) -> EnfsResult<Vec<u8>> {
        let mut entries: Vec<IndexEntry> = self.map
            .iter()
            .map(|(hash, &inode_id)| {
                IndexEntry {
                    name_blake3: *hash,
                    inode_id,
                    domain: self.domain_for(inode_id),
                    _pad: [0u8; 7],
                }
            })
            .collect();
        entries.sort(); // deterministic binary layout
        bincode::serialize(&entries)
            .map_err(|e| EnfsError::Serialization(e.to_string()))
    }

    /// Deserialize index from binary buffer (loaded on volume mount).
    pub fn from_bytes(buf: &[u8]) -> EnfsResult<Self> {
        let entries: Vec<IndexEntry> = bincode::deserialize(buf)
            .map_err(|e| EnfsError::Serialization(e.to_string()))?;
        let mut idx = Self::new();
        for e in entries {
            idx.map.insert(e.name_blake3, e.inode_id);
            idx.domains.entry(e.domain).or_default().push(e.inode_id);
        }
        Ok(idx)
    }

    fn domain_for(&self, inode_id: u64) -> u8 {
        for (&domain, ids) in &self.domains {
            if ids.contains(&inode_id) { return domain; }
        }
        0
    }
}

fn blake3_of(data: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(data);
    *h.finalize().as_bytes()
}
