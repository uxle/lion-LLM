// lion_core/src/episode.rs

use std::collections::VecDeque;

use crate::constants::{FEATURE_SIZE, PROCEDURAL_ACTIONS};
use crate::neuron::ActionLabel;
use crate::propagation::SensoryInput;
use serde::{Deserialize, Serialize};

// =============================================================================
// EPISODE
// =============================================================================

/// A single recorded experience from one tick of the waking day.
///
/// Episodes are the training corpus for the evolutionary night cycle.
/// The sandbox replays them to score each child brain's fitness.
///
/// Matches Python:
///   @dataclass
///   class Episode:
///       sensory_inputs:   Dict[str, np.ndarray]
///       gestalt_context:  np.ndarray
///       action_taken:     str
///       reward_received:  float
///       stress_level:     float
///
/// # Reward timing note
/// `reward_received` is set to `0.0` when the episode is first created.
/// It is filled in on the NEXT tick when the environment's response arrives.
/// This is the 1-tick delayed reward mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// The sensory frame the agent observed during this tick.
    /// Cloned from the live frame so the episode owns its data.
    pub sensory_inputs: SensoryInput,

    /// The compressed conscious state at the moment of decision.
    /// A normalized [f32; FEATURE_SIZE] gestalt vector from the GlobalWorkspace.
    pub gestalt_context: [f32; FEATURE_SIZE],

    /// The action the agent chose during this tick.
    /// Stored as a fixed-size label matching one of PROCEDURAL_ACTIONS.
    pub action_taken: ActionLabel,

    /// The reward received FROM THE ENVIRONMENT for the action taken.
    /// Always 0.0 at creation — filled in on the next tick.
    pub reward_received: f32,

    /// The epigenetic stress level at the moment of decision.
    /// Used by the fitness function to weight high-stress episodes more heavily.
    pub stress_level: f32,
}

impl Episode {
    /// Creates a new episode with reward = 0.0.
    ///
    /// Reward is filled in externally on the next tick via `set_reward()`.
    pub fn new(
        sensory_inputs:  SensoryInput,
        gestalt_context: [f32; FEATURE_SIZE],
        action_taken:    ActionLabel,
        stress_level:    f32,
    ) -> Self {
        Self {
            sensory_inputs,
            gestalt_context,
            action_taken,
            reward_received: 0.0,
            stress_level,
        }
    }

    /// Fills in the delayed reward. Called at the start of the next tick.
    ///
    /// Matches Python:
    ///   self.pending_episode.reward_received = prev_reward
    pub fn set_reward(&mut self, reward: f32) {
        self.reward_received = reward;
    }

    /// Returns true if this episode resulted in a positive reward outcome.
    pub fn was_positive(&self) -> bool {
        self.reward_received > 0.0
    }

    /// Returns true if this episode resulted in a negative reward outcome.
    pub fn was_negative(&self) -> bool {
        self.reward_received < 0.0
    }

    /// Returns the action taken as a &'static str by matching against PROCEDURAL_ACTIONS.
    /// Returns "WANDER" if the label doesn't match any known action.
    pub fn action_str(&self) -> &'static str {
        let label = self.action_taken.as_str();
        PROCEDURAL_ACTIONS
            .iter()
            .copied()
            .find(|&s| s == label)
            .unwrap_or("WANDER")
    }
}

// =============================================================================
// EPISODIC BUFFER
// =============================================================================

/// A fixed-capacity ring buffer of Episodes.
///
/// Episodes are appended to the back. When the buffer is full,
/// the oldest episode (front) is evicted — FIFO ordering.
///
/// The `VecDeque` gives O(1) push_back and pop_front,
/// making recording and eviction both constant time regardless of buffer size.
///
/// Matches Python:
///   class EpisodicBuffer:
///       def __init__(self, max_size=1000):
///           self.history: List[Episode] = []
///           self.max_size = max_size
///       def record(self, episode):
///           self.history.append(episode)
///           if len(self.history) > self.max_size:
///               self.history.pop(0)
///
/// The Python implementation uses `list.pop(0)` which is O(n).
/// The Rust `VecDeque::pop_front()` is O(1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicBuffer {
    /// The ring buffer of recorded episodes.
    pub history: VecDeque<Episode>,

    /// Maximum number of episodes to retain.
    /// When exceeded, the oldest episode is evicted.
    pub max_size: usize,
}

impl EpisodicBuffer {
    /// Creates a new empty buffer with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            history:  VecDeque::with_capacity(max_size.min(1024)),
            max_size,
        }
    }

    /// Creates a buffer with the default capacity (1000 episodes).
    ///
    /// Matches Python: EpisodicBuffer(max_size=1000)
    pub fn default_capacity() -> Self {
        Self::new(1000)
    }

    /// Records a new episode, evicting the oldest if the buffer is full.
    ///
    /// Matches Python:
    ///   self.history.append(episode)
    ///   if len(self.history) > self.max_size:
    ///       self.history.pop(0)
    pub fn record(&mut self, episode: Episode) {
        if self.history.len() >= self.max_size {
            self.history.pop_front(); // Evict oldest.
        }
        self.history.push_back(episode);
    }

    /// Returns the number of episodes currently stored.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns true if the buffer contains no episodes.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Returns a slice-like view over all episodes for sandbox replay.
    pub fn as_slice(&self) -> impl Iterator<Item = &Episode> {
        self.history.iter()
    }

    /// Returns all positive-reward episodes (for selective replay).
    pub fn positive_episodes(&self) -> impl Iterator<Item = &Episode> {
        self.history.iter().filter(|e| e.was_positive())
    }

    /// Returns all negative-reward episodes (for failure analysis).
    pub fn negative_episodes(&self) -> impl Iterator<Item = &Episode> {
        self.history.iter().filter(|e| e.was_negative())
    }

    /// Returns the most recent episode, or None if empty.
    pub fn last(&self) -> Option<&Episode> {
        self.history.back()
    }

    /// Clears all episodes. Called when resetting between experiments.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Returns a summary of episode reward distribution.
    pub fn reward_summary(&self) -> RewardSummary {
        let mut positive = 0u32;
        let mut negative = 0u32;
        let mut neutral  = 0u32;
        let mut total    = 0.0_f32;

        for ep in &self.history {
            match ep.reward_received {
                r if r > 0.0 => positive += 1,
                r if r < 0.0 => negative += 1,
                _             => neutral  += 1,
            }
            total += ep.reward_received;
        }

        RewardSummary {
            positive_count: positive,
            negative_count: negative,
            neutral_count:  neutral,
            total_reward:   total,
            episode_count:  self.history.len() as u32,
        }
    }
}

/// Summary statistics over an episode buffer's reward history.
#[derive(Debug, Clone, Default)]
pub struct RewardSummary {
    pub positive_count: u32,
    pub negative_count: u32,
    pub neutral_count:  u32,
    pub total_reward:   f32,
    pub episode_count:  u32,
}

impl RewardSummary {
    /// Mean reward per episode. Returns 0.0 if no episodes.
    pub fn mean_reward(&self) -> f32 {
        if self.episode_count == 0 {
            0.0
        } else {
            self.total_reward / self.episode_count as f32
        }
    }
}
