// lion_enfs/src/memory/archive.rs — Archive Memory Tier (cold compressed storage)
//
// Historical checkpoints, cold embeddings, deprecated versions.
// Payloads are LZ4-compressed before storage. Access is infrequent.

use std::collections::HashMap;

pub struct ArchiveEntry {
    pub key_hash:          [u8; 32],
    pub compressed_payload: Vec<u8>,
    pub original_size:     usize,
    pub stored_at:         u64,
}

pub struct ArchiveMemory {
    entries: HashMap<[u8; 32], ArchiveEntry>,
}

impl ArchiveMemory {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Compress and store a payload.
    pub fn store(&mut self, key: &str, payload: Vec<u8>) {
        let hash = blake3_of(key.as_bytes());
        let original_size = payload.len();
        let compressed_payload = lz4_flex::compress_prepend_size(&payload);
        let stored_at = unix_now();
        self.entries.insert(hash, ArchiveEntry {
            key_hash: hash,
            compressed_payload,
            original_size,
            stored_at,
        });
    }

    /// Decompress and return a payload.
    pub fn fetch(&self, key: &str) -> Option<Vec<u8>> {
        let hash = blake3_of(key.as_bytes());
        let entry = self.entries.get(&hash)?;
        lz4_flex::decompress_size_prepended(&entry.compressed_payload).ok()
    }

    pub fn remove(&mut self, key: &str) -> bool {
        let hash = blake3_of(key.as_bytes());
        self.entries.remove(&hash).is_some()
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Total compressed bytes in archive.
    pub fn compressed_size_bytes(&self) -> usize {
        self.entries.values().map(|e| e.compressed_payload.len()).sum()
    }
}

impl Default for ArchiveMemory {
    fn default() -> Self { Self::new() }
}

fn blake3_of(data: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(data);
    *h.finalize().as_bytes()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
