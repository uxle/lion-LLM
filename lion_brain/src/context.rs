// lion_brain/src/context.rs — Context Window Manager
//
// Tracks the conversation history within Gemma's 8192-token limit.
// Trims oldest turns when over budget.

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::llm::ChatMessage;

/// Estimate tokens from character count (1 token ≈ 4 chars).
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(|m| estimate_tokens(&m.content) + 4).sum()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Max total tokens for the whole context (Gemma 3 1B: 8192).
    pub max_tokens:       usize,
    pub system_reserve:   usize,
    pub memory_reserve:   usize,
    pub tool_reserve:     usize,
    pub input_reserve:    usize,
    /// Minimum number of recent turn pairs to always keep.
    pub min_recent_turns: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens:       6000,
            system_reserve:   500,
            memory_reserve:   800,
            tool_reserve:     300,
            input_reserve:    400,
            min_recent_turns: 4,
        }
    }
}

impl ContextConfig {
    pub fn history_budget(&self) -> usize {
        self.max_tokens
            .saturating_sub(self.system_reserve)
            .saturating_sub(self.memory_reserve)
            .saturating_sub(self.tool_reserve)
            .saturating_sub(self.input_reserve)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub history_tokens: usize,
    pub budget:         usize,
    pub total_turns:    usize,
}

impl TokenUsage {
    pub fn utilization(&self) -> f32 {
        if self.budget == 0 { return 0.0; }
        self.history_tokens as f32 / self.budget as f32
    }
}

#[derive(Debug, Clone)]
pub struct ContextManager {
    pub config:       ContextConfig,
    pub history:      Vec<ChatMessage>,
    pub total_turns:  usize,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self { config, history: Vec::new(), total_turns: 0 }
    }

    /// Records a completed turn (user + assistant pair).
    pub fn push_turn(&mut self, user: &str, assistant: &str) {
        self.history.push(ChatMessage::user(user));
        self.history.push(ChatMessage::assistant(assistant));
        self.total_turns += 1;
        self.trim_to_budget();
    }

    fn trim_to_budget(&mut self) {
        let budget   = self.config.history_budget();
        let min_msgs = self.config.min_recent_turns * 2;

        while estimate_messages_tokens(&self.history) > budget
            && self.history.len() > min_msgs
        {
            if self.history.len() >= 2 {
                self.history.drain(0..2);
            } else {
                break;
            }
        }

        debug!(
            "Context: {} msgs, ~{} tokens (budget: {})",
            self.history.len(),
            estimate_messages_tokens(&self.history),
            budget
        );
    }

    /// Build the full message list for an LLM call.
    pub fn build_messages(
        &self,
        system:         &str,
        memory_context: Option<&str>,
        current_input:  &str,
    ) -> Vec<ChatMessage> {
        let mut msgs = vec![ChatMessage::system(system)];

        if let Some(ctx) = memory_context {
            if !ctx.is_empty() {
                msgs.push(ChatMessage::system(ctx));
            }
        }

        msgs.extend(self.history.iter().cloned());
        msgs.push(ChatMessage::user(current_input));
        msgs
    }

    pub fn token_usage(&self) -> TokenUsage {
        TokenUsage {
            history_tokens: estimate_messages_tokens(&self.history),
            budget:         self.config.history_budget(),
            total_turns:    self.total_turns,
        }
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}
