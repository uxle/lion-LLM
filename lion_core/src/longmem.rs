// lion_core/src/longmem.rs — Phase 14: Long-Term Memory
//
// Persistent store for facts, mistakes, and per-skill performance.
// Saved as human-readable JSON to ~/.lionai/memory.json

use std::path::Path;
use serde::{Deserialize, Serialize};

// =============================================================================
// TYPES
// =============================================================================

/// A single learned fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id:           u64,
    pub topic:        String,
    pub content:      String,
    pub confidence:   f32,     // 0.0..=1.0
    pub source_tick:  u64,
    pub access_count: u32,
}

/// A recorded mistake: wrong action + what would have been correct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mistake {
    pub id:             u64,
    pub context:        String,   // What the user said
    pub wrong_action:   String,
    pub correct_action: String,
    pub reason:         String,
    pub resolved:       bool,
    pub tick:           u64,
}

/// Per-skill performance tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name:         String,
    pub correct:      u32,
    pub total:        u32,
    pub last_updated: u64,
}

impl Skill {
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 { return 0.5; }
        self.correct as f32 / self.total as f32
    }

    /// A skill is "weak" once it has 5+ uses and accuracy < 75 %.
    pub fn is_weak(&self) -> bool {
        self.total >= 5 && self.accuracy() < 0.75
    }

    pub fn grade(&self) -> &'static str {
        match (self.accuracy() * 100.0) as u32 {
            0..=49  => "F",
            50..=64 => "D",
            65..=74 => "C",
            75..=84 => "B",
            85..=94 => "A",
            _       => "A+",
        }
    }
}

// =============================================================================
// LONG-TERM MEMORY
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LongTermMemory {
    pub facts:    Vec<Fact>,
    pub mistakes: Vec<Mistake>,
    pub skills:   Vec<Skill>,
    pub sessions: u32,
    next_id:      u64,
}

impl LongTermMemory {
    // ── Facts ─────────────────────────────────────────────────────────────────

    /// Store a fact. If an identical topic already exists, update it instead.
    pub fn store_fact(&mut self, topic: &str, content: &str, tick: u64) {
        let topic_lower = topic.to_lowercase();
        if let Some(f) = self.facts.iter_mut()
            .find(|f| f.topic.to_lowercase() == topic_lower)
        {
            f.content      = content.to_string();
            f.confidence   = (f.confidence + 0.1).min(1.0);
            f.access_count += 1;
            return;
        }
        let id = self.alloc_id();
        self.facts.push(Fact {
            id,
            topic:        topic.to_string(),
            content:      content.to_string(),
            confidence:   0.7,
            source_tick:  tick,
            access_count: 0,
        });
    }

    /// Keyword search. Returns up to 3 best-matching facts.
    pub fn search_facts(&mut self, query: &str) -> Vec<Fact> {
        let q     = query.to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();
        let mut ranked: Vec<(usize, usize)> = self.facts.iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let text = format!("{} {}", f.topic.to_lowercase(), f.content.to_lowercase());
                let score = words.iter().filter(|&&w| text.contains(w)).count();
                if score > 0 { Some((score, i)) } else { None }
            })
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0));

        ranked.into_iter().take(3).map(|(_, i)| {
            self.facts[i].access_count += 1;
            self.facts[i].clone()
        }).collect()
    }

    /// Remove all facts whose topic contains `topic`. Returns removed count.
    pub fn forget_topic(&mut self, topic: &str) -> usize {
        let q      = topic.to_lowercase();
        let before = self.facts.len();
        self.facts.retain(|f| !f.topic.to_lowercase().contains(&q));
        before - self.facts.len()
    }

    // ── Skills ────────────────────────────────────────────────────────────────

    /// Record a skill outcome (success = true → correct count goes up).
    pub fn record_outcome(&mut self, skill_name: &str, success: bool, tick: u64) {
        if let Some(s) = self.skills.iter_mut().find(|s| s.name == skill_name) {
            s.total += 1;
            if success { s.correct += 1; }
            s.last_updated = tick;
        } else {
            self.skills.push(Skill {
                name:         skill_name.to_string(),
                correct:      if success { 1 } else { 0 },
                total:        1,
                last_updated: tick,
            });
        }
    }

    /// All skills whose accuracy is below the weak threshold.
    pub fn weak_skills(&self) -> Vec<&Skill> {
        self.skills.iter().filter(|s| s.is_weak()).collect()
    }

    // ── Mistakes ──────────────────────────────────────────────────────────────

    pub fn record_mistake(
        &mut self,
        context:  &str,
        wrong:    &str,
        correct:  &str,
        reason:   &str,
        tick:     u64,
    ) {
        let id = self.alloc_id();
        self.mistakes.push(Mistake {
            id,
            context:        context.to_string(),
            wrong_action:   wrong.to_string(),
            correct_action: correct.to_string(),
            reason:         reason.to_string(),
            resolved:       false,
            tick,
        });
    }

    pub fn resolve_mistake(&mut self, id: u64) {
        if let Some(m) = self.mistakes.iter_mut().find(|m| m.id == id) {
            m.resolved = true;
        }
    }

    pub fn unresolved_mistakes(&self) -> Vec<&Mistake> {
        self.mistakes.iter().filter(|m| !m.resolved).collect()
    }

    // ── Persistence ───────────────────────────────────────────────────────────

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

    // ── Private ───────────────────────────────────────────────────────────────

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}
