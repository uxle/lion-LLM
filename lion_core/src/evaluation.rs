// lion_core/src/evaluation.rs — Phase 14: Self-Evaluation Engine
//
// Computes a composite ResponseScore for every tick and tracks running
// averages per skill. Flags weak skills for targeted night cycles.

// =============================================================================
// RESPONSE SCORE
// =============================================================================

/// Multi-dimensional quality score for a single response tick.
#[derive(Debug, Clone, Default)]
pub struct ResponseScore {
    /// Consistency with recent positive episodes (0.0..=1.0).
    pub accuracy: f32,

    /// Normalised user feedback this tick (-1.0..=+1.0 → 0.0..=1.0).
    pub user_satisfaction: f32,

    /// Speech readability: fraction of alphabetic+space chars (0.0..=1.0).
    pub speech_quality: f32,

    /// Did long-term memory return a relevant hit? (0.0 or 1.0)
    pub memory_usefulness: f32,

    /// Weighted composite: the headline score shown to the user.
    pub composite: f32,
}

impl ResponseScore {
    /// Build a score from its components.
    ///
    /// Formula:
    ///   composite = accuracy × 0.50
    ///             + user_sat × 0.30
    ///             + speech   × 0.10
    ///             + memory   × 0.10
    pub fn compute(
        accuracy:    f32,
        user_reward: f32,   // raw queued_reward (may be negative)
        speech_q:    f32,
        memory_hit:  bool,
    ) -> Self {
        // Normalise user reward from [-5, +5] to [0, 1]
        let user_sat = (user_reward.clamp(-5.0, 5.0) + 5.0) / 10.0;
        let mem_util = if memory_hit { 1.0 } else { 0.0 };
        let composite = accuracy    * 0.50
                      + user_sat    * 0.30
                      + speech_q    * 0.10
                      + mem_util    * 0.10;
        Self {
            accuracy,
            user_satisfaction: user_sat,
            speech_quality:    speech_q,
            memory_usefulness: mem_util,
            composite: composite.clamp(0.0, 1.0),
        }
    }

    /// Letter grade for display.
    pub fn grade(&self) -> &'static str {
        match (self.composite * 10.0) as u32 {
            0..=1 => "F",
            2..=3 => "D",
            4..=5 => "C",
            6..=7 => "B",
            8..=9 => "A",
            _     => "A+",
        }
    }

    /// Colour-coded grade bar for terminal display.
    pub fn bar(&self) -> String {
        let filled = (self.composite * 10.0).round() as usize;
        let empty  = 10_usize.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }
}

// =============================================================================
// EVALUATOR
// =============================================================================

/// Rolling evaluator that tracks response history and per-skill averages.
#[derive(Debug, Default)]
pub struct ResponseEvaluator {
    pub history:           Vec<ResponseScore>,
    pub running_composite: f32,
    pub count:             u32,
}

impl ResponseEvaluator {
    /// Record a new score and update the running average.
    pub fn record(&mut self, score: ResponseScore) {
        let n = self.count as f32;
        self.running_composite = (self.running_composite * n + score.composite) / (n + 1.0);
        self.count += 1;
        self.history.push(score);
        if self.history.len() > 200 { self.history.remove(0); }
    }

    /// Recent trend: positive = improving, negative = declining.
    pub fn trend(&self) -> f32 {
        let n = self.history.len();
        if n < 4 { return 0.0; }
        let half  = n / 2;
        let newer: f32 = self.history[half..].iter().map(|s| s.composite).sum::<f32>()
                       / (n - half) as f32;
        let older: f32 = self.history[..half].iter().map(|s| s.composite).sum::<f32>()
                       / half as f32;
        newer - older
    }

    /// Estimate accuracy from episode reward history.
    /// Simple heuristic: fraction of last N ticks with positive user satisfaction.
    pub fn recent_accuracy(&self, window: usize) -> f32 {
        let slice: Vec<&ResponseScore> = self.history.iter().rev().take(window).collect();
        if slice.is_empty() { return 0.5; }
        let pos = slice.iter().filter(|s| s.user_satisfaction > 0.5).count();
        pos as f32 / slice.len() as f32
    }
}
