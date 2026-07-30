// lion_enfs/src/memory/sensory.rs — Sensory Memory Tier (CPU/GPU cache-speed)
//
// Extremely short-lived. Holds raw input frames (token embeddings, audio samples,
// image patches) for at most one inference tick. Binary only — no strings.

use std::collections::VecDeque;

const MAX_SLOTS: usize = 64;

pub struct SensorySlot {
    pub id:      u64,
    pub payload: Vec<u8>,    // raw binary — no human-readable encoding
    pub tick:    u64,
}

pub struct SensoryMemory {
    slots:    VecDeque<SensorySlot>,
    tick:     u64,
    next_id:  u64,
}

impl SensoryMemory {
    pub fn new() -> Self {
        Self { slots: VecDeque::with_capacity(MAX_SLOTS), tick: 0, next_id: 1 }
    }

    /// Store raw bytes. Oldest slot evicted when full.
    pub fn push(&mut self, payload: Vec<u8>) -> u64 {
        if self.slots.len() >= MAX_SLOTS {
            self.slots.pop_front();
        }
        let id = self.next_id;
        self.next_id += 1;
        self.slots.push_back(SensorySlot { id, payload, tick: self.tick });
        id
    }

    /// Fetch by slot ID. Returns None if already evicted.
    pub fn fetch(&self, id: u64) -> Option<&[u8]> {
        self.slots.iter().find(|s| s.id == id).map(|s| s.payload.as_slice())
    }

    /// Advance time — slots older than 2 ticks are evicted.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.slots.retain(|s| self.tick - s.tick <= 2);
    }

    pub fn len(&self) -> usize { self.slots.len() }
}

impl Default for SensoryMemory {
    fn default() -> Self { Self::new() }
}
