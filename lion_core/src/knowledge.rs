// lion_core/src/knowledge.rs — Phase 14: Knowledge Graph
//
// A persistent graph of Concepts connected by typed edges.
// Saved as human-readable JSON to ~/.lionai/knowledge.json

use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

// =============================================================================
// TYPES
// =============================================================================

/// A single concept node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id:           String,
    pub name:         String,
    pub description:  String,
    pub confidence:   f32,
    pub properties:   Vec<String>,
    pub tags:         Vec<String>,
    pub created_tick: u64,
    pub updated_tick: u64,
    pub access_count: u32,
}

/// A directed, typed edge between two concept nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub from:     String,
    pub relation: String,
    pub to:       String,
}

/// The full knowledge graph: concepts + edges.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeGraph {
    pub concepts: HashMap<String, Concept>,
    pub edges:    Vec<Edge>,
}

// =============================================================================
// IMPLEMENTATION
// =============================================================================

impl KnowledgeGraph {
    /// Add or update a concept. Confidence grows on repeated mentions.
    pub fn learn(&mut self, name: &str, description: &str, tags: Vec<String>, tick: u64) {
        let id = slugify(name);
        if let Some(c) = self.concepts.get_mut(&id) {
            c.confidence   = (c.confidence + 0.08).min(1.0);
            c.updated_tick = tick;
            c.access_count += 1;
            if !description.is_empty() {
                c.description = description.to_string();
            }
            for tag in &tags {
                if !c.tags.contains(tag) { c.tags.push(tag.clone()); }
            }
        } else {
            self.concepts.insert(id.clone(), Concept {
                id,
                name:         name.to_string(),
                description:  description.to_string(),
                confidence:   0.6,
                properties:   vec![],
                tags,
                created_tick: tick,
                updated_tick: tick,
                access_count: 0,
            });
        }
    }

    /// Add a typed edge between two concept names.
    pub fn relate(&mut self, from: &str, relation: &str, to: &str) {
        let edge = Edge { from: slugify(from), relation: relation.to_string(), to: slugify(to) };
        if !self.edges.contains(&edge) { self.edges.push(edge); }
    }

    /// Return up to 5 concepts most relevant to a text query.
    pub fn search(&mut self, query: &str) -> Vec<Concept> {
        let q     = query.to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();
        let mut ranked: Vec<(usize, String)> = self.concepts.iter()
            .filter_map(|(id, c)| {
                let text = format!("{} {} {}", c.name.to_lowercase(),
                    c.description.to_lowercase(), c.tags.join(" ").to_lowercase());
                let score = words.iter().filter(|&&w| text.contains(w)).count();
                if score > 0 { Some((score, id.clone())) } else { None }
            })
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
        ranked.into_iter().take(5).filter_map(|(_, id)| {
            let c = self.concepts.get_mut(&id)?;
            c.access_count += 1;
            Some(c.clone())
        }).collect()
    }

    /// Return concepts related to `name` by any edge.
    pub fn related_to(&self, name: &str) -> Vec<&Concept> {
        let id = slugify(name);
        self.edges.iter()
            .filter(|e| e.from == id || e.to == id)
            .filter_map(|e| {
                let target = if e.from == id { &e.to } else { &e.from };
                self.concepts.get(target)
            })
            .collect()
    }

    pub fn concept_count(&self) -> usize { self.concepts.len() }
    pub fn edge_count(&self)    -> usize { self.edges.len() }

    pub fn most_accessed(&self, n: usize) -> Vec<&Concept> {
        let mut v: Vec<&Concept> = self.concepts.values().collect();
        v.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        v.into_iter().take(n).collect()
    }

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

/// Convert a human name into a stable slug: "Rust Ownership" → "rust-ownership".
pub fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}
