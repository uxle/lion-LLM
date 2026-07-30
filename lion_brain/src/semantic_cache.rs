// lion_brain/src/semantic_cache.rs — O(1) Semantic Response Caching
//
// Implements semantic cache routing from 03_SERVING_OPTIMIZATION.md.
// Bypasses LLM inference on cache hit (cosine similarity >= threshold).

use serde::{Deserialize, Serialize};
use lion_core::cosine_sim;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub query_text: String,
    pub embedding: Vec<f32>,
    pub response_text: String,
    pub hit_count: u32,
    pub created_at: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticCache {
    pub entries: Vec<CacheEntry>,
    pub threshold: f32, // Default 0.95
}

impl SemanticCache {
    pub fn new(threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            threshold,
        }
    }

    /// Query the semantic cache.
    /// If cosine similarity meets or exceeds threshold (default 0.95), returns the cached answer.
    pub fn lookup(&mut self, query_embedding: &[f32]) -> Option<&CacheEntry> {
        let threshold = self.threshold;
        let mut best_idx = None;
        let mut best_sim = 0.0_f32;

        for (i, entry) in self.entries.iter().enumerate() {
            let sim = cosine_sim(query_embedding, &entry.embedding);
            if sim >= threshold && sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            self.entries[idx].hit_count += 1;
            return Some(&self.entries[idx]);
        }

        None
    }

    /// Insert new query and response into cache.
    pub fn insert(&mut self, query_text: String, embedding: Vec<f32>, response_text: String) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.entries.push(CacheEntry {
            query_text,
            embedding,
            response_text,
            hit_count: 0,
            created_at: timestamp,
        });

        if self.entries.len() > 500 {
            self.entries.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
