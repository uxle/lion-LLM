// lion_brain/src/memory.rs — Semantic Memory System
//
// Stores text entries with their embedding vectors.
// Retrieval uses cosine similarity search.
// Persisted to disk via bincode.

use std::path::Path;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use lion_core::cosine_sim;

// =============================================================================
// TYPES
// =============================================================================

/// Where a memory entry came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemorySource {
    UserInput,
    LlmAnswer,
    AgentTask,
    External,
    Knowledge,
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::UserInput  => write!(f, "user"),
            Self::LlmAnswer  => write!(f, "llm"),
            Self::AgentTask  => write!(f, "agent"),
            Self::External   => write!(f, "external"),
            Self::Knowledge  => write!(f, "knowledge"),
        }
    }
}

/// A single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id:         u64,
    pub content:    String,
    pub embedding:  Vec<f32>,
    pub source:     MemorySource,
    pub confidence: f32,
    pub tick:       u64,
    pub access_count: u32,
}

/// A retrieved memory with similarity score.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    pub entry:      MemoryEntry,
    pub similarity: f32,
}

/// Memory statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub embed_dim:     usize,
    pub oldest_tick:   u64,
    pub newest_tick:   u64,
}

// =============================================================================
// MEMORY SYSTEM
// =============================================================================

/// Semantic memory with cosine-similarity retrieval and disk persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySystem {
    pub entries:   Vec<MemoryEntry>,
    pub embed_dim: usize,
    next_id:       u64,
    tick:          u64,
}

impl MemorySystem {
    // ── Construction ──────────────────────────────────────────────────────────

    pub fn new(embed_dim: usize) -> Self {
        Self { entries: Vec::new(), embed_dim, next_id: 0, tick: 0 }
    }

    /// Loads from disk or creates a fresh instance if the file doesn't exist.
    pub fn load_or_new(path: &Path, embed_dim: usize) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => {
                match bincode::deserialize::<MemorySystem>(&bytes) {
                    Ok(mem) => {
                        info!("Memory loaded: {} entries from '{}'", mem.entries.len(), path.display());
                        mem
                    }
                    Err(e) => {
                        warn!("Memory deserialization failed: {}. Starting fresh.", e);
                        Self::new(embed_dim)
                    }
                }
            }
            Err(_) => {
                debug!("No memory file at '{}'. Starting fresh.", path.display());
                Self::new(embed_dim)
            }
        }
    }

    // ── Storage ───────────────────────────────────────────────────────────────

    /// Stores a new memory entry.
    pub fn store(
        &mut self,
        content:    String,
        embedding:  Vec<f32>,
        source:     MemorySource,
        confidence: f32,
    ) {
        self.tick += 1;
        let id = self.next_id;
        self.next_id += 1;

        self.entries.push(MemoryEntry {
            id,
            content,
            embedding,
            source,
            confidence,
            tick: self.tick,
            access_count: 0,
        });

        // Evict oldest low-confidence entries when over capacity.
        if self.entries.len() > 10_000 {
            self.evict(500);
        }
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    /// Returns the top-k most similar entries above `min_similarity`.
    pub fn search(
        &mut self,
        query_embedding: &[f32],
        top_k:           usize,
        min_similarity:  f32,
    ) -> Vec<MemoryResult> {
        let mut scored: Vec<(f32, usize)> = self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (cosine_sim(query_embedding, &e.embedding), i))
            .filter(|(sim, _)| *sim >= min_similarity)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored.into_iter().map(|(sim, i)| {
            self.entries[i].access_count += 1;
            MemoryResult {
                entry:      self.entries[i].clone(),
                similarity: sim,
            }
        }).collect()
    }

    /// Keyword-based search fallback (when no embedding is available).
    pub fn keyword_search(&self, query: &str, top_k: usize) -> Vec<&MemoryEntry> {
        let q     = query.to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();

        let mut scored: Vec<(usize, usize)> = self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let lower = e.content.to_lowercase();
                let score = words.iter().filter(|&&w| lower.contains(w)).count();
                if score > 0 { Some((score, i)) } else { None }
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(top_k);
        scored.into_iter().map(|(_, i)| &self.entries[i]).collect()
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, &bytes)?;
        debug!("Memory saved: {} entries ({:.1} KB)", self.entries.len(), bytes.len() as f32 / 1024.0);
        Ok(())
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    pub fn stats(&self) -> MemoryStats {
        let oldest = self.entries.iter().map(|e| e.tick).min().unwrap_or(0);
        let newest = self.entries.iter().map(|e| e.tick).max().unwrap_or(0);
        MemoryStats {
            total_entries: self.entries.len(),
            embed_dim:     self.embed_dim,
            oldest_tick:   oldest,
            newest_tick:   newest,
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Evict the `n` lowest-confidence entries.
    fn evict(&mut self, n: usize) {
        self.entries.sort_by(|a, b| {
            a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        let new_len = self.entries.len().saturating_sub(n);
        self.entries.truncate(new_len);
        info!("Memory evicted {} entries. Remaining: {}", n, self.entries.len());
    }
}
