// lion_core/src/sovereign.rs

use crate::brain::BrainMatrix;
use crate::episode::{Episode, EpisodicBuffer};
use crate::immune::ImmuneReport;
use crate::language::LanguageMotor;
use crate::propagation::SensoryInput;
use crate::rng::BrainRng;
use crate::workspace::{action_to_label, best_motor_action, extract_action, gather_consciousness};
use crate::sandbox::run_night_cycle_parallel;
// =============================================================================
// TICK RESULT
// =============================================================================

/// The output of one complete waking-day tick.
///
/// Contains the chosen action, the immune scan report, and diagnostic info.
#[derive(Debug)]
pub struct TickResult {
    /// The action chosen by the agent this tick.
    pub action: &'static str,

    /// The compressed conscious state computed during this tick.
    pub gestalt: [f32; crate::constants::FEATURE_SIZE],

    /// The immune scan report from this tick's propagation.
    pub immune_report: ImmuneReport,

    /// The current tick number.
    pub tick: u64,
}

// =============================================================================
// SOVEREIGN AGENT
// =============================================================================

/// The sovereign agent — the outermost orchestrator of LionAI's waking day.
///
/// Owns:
///   - `brain`           — the cognitive graph (neurons, synapses, epigenome)
///   - `episodic_buffer` — the ring buffer of recorded experiences
///   - `rng`             — the deterministic random number generator
///   - `tick`            — the current tick counter
///   - `generation`      — the current evolutionary generation
///   - `pending_episode` — the episode awaiting its delayed reward
///
/// Translates Python's `LNNHyperNodeV16`.
///
/// The night cycle / evolutionary sandbox (Phase 7) is called externally
/// on the Sovereign via `trigger_sleep_cycle()`.
pub struct Sovereign {
    /// The cognitive brain — neurons, synapses, hebbian weights, epigenome.
    pub brain: BrainMatrix,

    /// The language motor cortex (Transformer) that generates speech from gestalt.
    pub language_motor: LanguageMotor,

    /// Ring buffer of recorded episodes. Training data for Phase 7 sandbox.
    pub episodic_buffer: EpisodicBuffer,

    /// Seeded random number generator for exploration and mutation.
    pub rng: BrainRng,

    /// Monotonic tick counter. Incremented once per `update()` call.
    pub tick: u64,

    /// Evolutionary generation counter. Incremented on successful night-cycle evolution.
    pub generation: u32,

    /// The episode from the previous tick, waiting for its reward.
    /// None at initialization and after each flush.
    pending_episode: Option<Episode>,
}

impl Sovereign {
    // =========================================================================
    // CONSTRUCTION
    // =========================================================================

    /// Creates a new Sovereign with a freshly initialized brain.
    ///
    /// Uses the given seed for the RNG — identical seeds produce identical runs.
    ///
    /// Matches Python:
    ///   class LNNHyperNodeV16:
    ///       def __init__(self):
    ///           self.graph = NeuralGraph(self.dna, self.procedural_actions)
    ///           self.episodic_buffer = EpisodicBuffer()
    ///           ...
    pub fn new(seed: u64) -> Self {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(seed);
        brain.initialize_core_brain(&mut rng);
        let language_motor = LanguageMotor::random(&mut rng);

        Self {
            brain,
            language_motor,
            episodic_buffer: EpisodicBuffer::default_capacity(),
            rng,
            tick:            0,
            generation:      1,
            pending_episode: None,
        }
    }

    // =========================================================================
    // WAKING-DAY TICK
    // =========================================================================

    /// Executes one complete waking-day tick and returns the chosen action.
    ///
    /// Full pipeline per tick:
    ///   1. Increment tick counter.
    ///   2. Adapt epigenetic stress from prev_reward (if negative).
    ///   3. Flush pending episode — assign prev_reward and record to buffer.
    ///   4. Run forward pass: inject senses → propagate → immune scan.
    ///   5. Compute global workspace gestalt.
    ///   6. Extract action (ε-greedy explore/exploit).
    ///   7. Create new pending episode (reward = 0.0, filled next tick).
    ///   8. Return TickResult.
    ///
    /// Matches Python:
    ///   def update(self, sensory_inputs, prev_reward=0.0):
    ///       self.tick += 1
    ///       if prev_reward < 0: self.epi.adapt_live_stress(abs(prev_reward) * 0.5)
    ///       if self.pending_episode:
    ///           self.pending_episode.reward_received = prev_reward
    ///           self.episodic_buffer.record(self.pending_episode)
    ///           self.pending_episode = None
    ///       ...inject, propagate, heal...
    ///       conscious_nodes, gestalt = self.workspace.gather_consciousness(self.graph)
    ///       action = self.sandbox.extract_action(self.graph, self.epi, force_exploit=False)
    ///       self.pending_episode = Episode(sensory_inputs, gestalt, action, 0.0, stress)
    ///       return action
    pub fn update(
        &mut self,
        frame:       &SensoryInput,
        prev_reward: f32,
    ) -> TickResult {
        // ── 1. Increment tick ────────────────────────────────────────────────
        self.tick += 1;

        // ── 2. Stress adaptation ─────────────────────────────────────────────
        // Negative rewards increase accumulated stress.
        // Matches Python: if prev_reward < 0: self.epi.adapt_live_stress(...)
        if prev_reward < 0.0 {
            self.brain
                .epigenome
                .adapt_live_stress(prev_reward.abs() * 0.5);
        }

        // ── 3. Flush pending episode (1-tick delayed reward) ─────────────────
        // The episode from the PREVIOUS tick now receives its reward.
        if let Some(mut ep) = self.pending_episode.take() {
            ep.set_reward(prev_reward);
            self.episodic_buffer.record(ep);
        }

        // ── 4. Forward pass: inject → propagate → heal ───────────────────────
        let immune_report = self.brain.tick(frame, 2);

        // ── 5. Global workspace: compute gestalt ─────────────────────────────
        let (gestalt, _conscious_ids) =
            gather_consciousness(&self.brain, crate::workspace::WORKSPACE_TOP_K);

        // ── 6. Extract action (ε-greedy) ─────────────────────────────────────
        let action = extract_action(
            &self.brain,
            &self.brain.epigenome,
            &mut self.rng,
            false, // live execution: exploration enabled
        );

        // ── 7. Create pending episode (reward arrives next tick) ──────────────
        self.pending_episode = Some(Episode::new(
            frame.clone(),
            gestalt,
            action_to_label(action),
            self.brain.epigenome.accumulated_stress,
        ));

        TickResult {
            action,
            gestalt,
            immune_report,
            tick: self.tick,
        }
    }

    // =========================================================================
    // SLEEP CYCLE PREPARATION
    // =========================================================================

    /// Flushes the pending episode before the sleep cycle begins.
    ///
    /// The last action of the day receives `final_reward`.
    /// After this call, `pending_episode` is `None` and the buffer is complete.
    ///
    /// Matches Python:
    ///   def trigger_sleep_cycle(self, final_reward=0.0):
    ///       if self.pending_episode:
    ///           self.pending_episode.reward_received = final_reward
    ///           self.episodic_buffer.record(self.pending_episode)
    ///           self.pending_episode = None
    pub fn flush_pending_episode(&mut self, final_reward: f32) {
        if let Some(mut ep) = self.pending_episode.take() {
            ep.set_reward(final_reward);
            self.episodic_buffer.record(ep);
        }
    }

    // =========================================================================
    // POST-SLEEP CYCLE UPDATES (called by Phase 7)
    // =========================================================================

    /// Replaces the brain with an evolved child from the night cycle.
    ///
    /// Called by the evolutionary sandbox (Phase 7) after a successful evolution.
    ///
    /// Matches Python:
    ///   self.graph = new_graph
    ///   self.epi = new_epi
    ///   self.generation += 1
    ///   self.epi.accumulated_stress = 0.0
    pub fn hot_swap_brain(
        &mut self,
        new_brain: BrainMatrix,
        new_language: Option<LanguageMotor>,
    ) {
        self.brain = new_brain;
        if let Some(lang) = new_language {
            self.language_motor = lang;
        }
        self.brain.epigenome.clear_stress();
        self.generation += 1;
        self.brain.reset_immune_counter();
    }

    /// Called after a failed night cycle (no child outperformed sovereign).
    ///
    /// Decays stress naturally.
    ///
    /// Matches Python:
    ///   self.epi.accumulated_stress *= 0.8
    pub fn on_failed_evolution(&mut self) {
        self.brain.epigenome.decay_stress();
        self.brain.reset_immune_counter();
    }

    // =========================================================================
    // DIAGNOSTICS
    // =========================================================================

    /// Returns the current accumulated stress level.
    pub fn stress(&self) -> f32 {
        self.brain.epigenome.accumulated_stress
    }

    /// Returns true if there is a pending episode awaiting its reward.
    pub fn has_pending_episode(&self) -> bool {
        self.pending_episode.is_some()
    }

    /// Returns the number of episodes recorded so far.
    pub fn episode_count(&self) -> usize {
        self.episodic_buffer.len()
    }

    /// Returns the best motor action under current activations (no exploration).
    pub fn exploit_action(&self) -> &'static str {
        best_motor_action(&self.brain)
    }

    /// Runs the full night cycle and evolves the sovereign brain if possible.
    ///
    /// Pipeline:
    ///   1. Flush pending episode with `final_reward`.
    ///   2. Evaluate sovereign fitness as baseline.
    ///   3. Spawn N mutated children and evaluate each against episode history.
    ///   4. If a child outperforms the sovereign, hot-swap the brain.
    ///   5. Reset immune counter. Return the night cycle report.
    ///
    /// Uses a separate sandbox RNG seeded from the current tick number
    /// so mutations are deterministic per day but vary across days.
    ///
    /// Matches Python:
    ///   def trigger_sleep_cycle(self, final_reward=0.0):
    ///       if self.pending_episode: ...flush...
    ///       new_graph, new_epi, best_fitness = self.sandbox.run_night_cycle(
    ///           self.graph, self.epi, self.episodic_buffer.history, population_size=15
    ///       )
    ///       if new_graph is not self.graph:
    ///           self.graph = new_graph
    ///           self.generation += 1
    ///           self.epi.accumulated_stress = 0.0
    ///       else:
    ///           self.epi.accumulated_stress *= 0.8
    pub fn trigger_sleep_cycle(&mut self, final_reward: f32) -> crate::sandbox::NightCycleReport {
        // Step 1: Flush the pending episode with the final day's reward.
        self.flush_pending_episode(final_reward);

        // Step 2–4: Run night cycle with a deterministic per-day RNG.
        // Seeding from tick ensures different mutations each day while
        // remaining reproducible for the same tick count.
        let (winner_brain, winner_lang, report) = run_night_cycle_parallel(
            &self.brain,
            &self.language_motor,
            &self.episodic_buffer,
            self.tick,
            crate::constants::NIGHT_CYCLE_POPULATION,
        );

        // Step 5: Hot-swap or decay stress.
        match winner_brain {
            Some(new_brain) => {
                self.hot_swap_brain(new_brain, winner_lang);
            }
            None => {
                self.on_failed_evolution();
            }
        }

        report
    }
}
