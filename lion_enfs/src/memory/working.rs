// lion_enfs/src/memory/working.rs — Working Memory Tier (RAM-speed)
//
// Active session context, in-flight embeddings, current task state.
// LRU eviction policy. Keys are BLAKE3-hashed internally — no plaintext stored.

use std::collections::{HashMap, VecDeque};

pub struct WorkingEntry {
    pub key_hash: [u8; 32],
    pub payload:  Vec<u8>,
    pub access_count: u32,
}

pub struct WorkingMemory {
    capacity: usize,
    entries:  HashMap<[u8; 32], WorkingEntry>,
    lru:      VecDeque<[u8; 32]>,  // front = LRU, back = MRU
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            lru:     VecDeque::with_capacity(capacity),
        }
    }

    pub fn store(&mut self, key: &str, payload: Vec<u8>) {
        let hash = blake3_of(key.as_bytes());
        if self.entries.len() >= self.capacity {
            // Evict LRU
            if let Some(evict_hash) = self.lru.pop_front() {
                self.entries.remove(&evict_hash);
            }
        }
        self.lru.retain(|h| h != &hash);
        self.lru.push_back(hash);
        self.entries.insert(hash, WorkingEntry { key_hash: hash, payload, access_count: 1 });
    }

    pub fn fetch(&mut self, key: &str) -> Option<&[u8]> {
        let hash = blake3_of(key.as_bytes());
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.access_count += 1;
            // Promote to MRU
            self.lru.retain(|h| h != &hash);
            self.lru.push_back(hash);
            return Some(entry.payload.as_slice());
        }
        None
    }

    /// Remove and return the payload (used during promotion to domain).
    pub fn evict(&mut self, key: &str) -> Option<Vec<u8>> {
        let hash = blake3_of(key.as_bytes());
        self.lru.retain(|h| h != &hash);
        self.entries.remove(&hash).map(|e| e.payload)
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

fn blake3_of(data: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(data);
    *h.finalize().as_bytes()
}
