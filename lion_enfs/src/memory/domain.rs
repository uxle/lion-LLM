// lion_enfs/src/memory/domain.rs — Domain Memory Stores (NVMe-backed long-term)
//
// One independent store per cognitive domain. Each domain uses its own
// BLAKE3-keyed HashMap in RAM backed by the ENFS volume on NVMe/SSD.

use std::collections::HashMap;
use crate::error::EnfsResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainKey {
    Language,
    Mathematics,
    Physics,
    Science,
    Skills,
    Vision2d,
    Vision3d,
    Audio,
    Video,
    Safety,
    Intelligence,
    Custom(u8),
}

impl DomainKey {
    pub fn tag(&self) -> u8 {
        match self {
            DomainKey::Language     => 2,
            DomainKey::Mathematics  => 3,
            DomainKey::Physics      => 4,
            DomainKey::Science      => 5,
            DomainKey::Skills       => 6,
            DomainKey::Vision2d     => 7,
            DomainKey::Vision3d     => 8,
            DomainKey::Audio        => 9,
            DomainKey::Video        => 10,
            DomainKey::Safety       => 11,
            DomainKey::Intelligence => 12,
            DomainKey::Custom(t)    => *t,
        }
    }
}

/// One domain's in-memory store. Keys are BLAKE3 hashes of canonical names.
#[derive(Default)]
struct DomainStore {
    map: HashMap<[u8; 32], Vec<u8>>,
}

impl DomainStore {
    fn store(&mut self, key: &str, payload: Vec<u8>) {
        self.map.insert(blake3_of(key.as_bytes()), payload);
    }

    fn fetch(&self, key: &str) -> Option<&[u8]> {
        self.map.get(&blake3_of(key.as_bytes())).map(|v| v.as_slice())
    }

    fn evict(&mut self, key: &str) -> Option<Vec<u8>> {
        self.map.remove(&blake3_of(key.as_bytes()))
    }

    fn len(&self) -> usize { self.map.len() }
}

pub struct DomainMemory {
    stores: HashMap<u8, DomainStore>,
}

impl DomainMemory {
    pub fn new() -> Self {
        let mut stores = HashMap::new();
        for tag in [2u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
            stores.insert(tag, DomainStore::default());
        }
        Self { stores }
    }

    pub fn store(&mut self, domain: DomainKey, key: &str, payload: Vec<u8>) -> EnfsResult<()> {
        self.stores
            .entry(domain.tag())
            .or_default()
            .store(key, payload);
        Ok(())
    }

    pub fn fetch(&self, domain: DomainKey, key: &str) -> Option<&[u8]> {
        self.stores.get(&domain.tag())?.fetch(key)
    }

    pub fn evict(&mut self, domain: DomainKey, key: &str) -> Option<Vec<u8>> {
        self.stores.get_mut(&domain.tag())?.evict(key)
    }

    pub fn domain_size(&self, domain: DomainKey) -> usize {
        self.stores.get(&domain.tag()).map(|s| s.len()).unwrap_or(0)
    }
}

impl Default for DomainMemory {
    fn default() -> Self { Self::new() }
}

fn blake3_of(data: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(data);
    *h.finalize().as_bytes()
}
