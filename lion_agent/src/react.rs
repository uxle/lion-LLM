// lion_agent/src/react.rs — ReAct (Reason + Act + Observe) Loop

use std::sync::Arc;
use tracing::{debug, info};

use lion_brain::llm::{ChatMessage, OllamaClient};
use crate::registry::ToolRegistry;

// =============================================================================
// CONFIG
// =============================================================================

#[derive(Debug, Clone)]
pub struct ReActConfig {
    pub max_steps:   usize,
    pub temperature: f32,
    pub max_tokens:  i32,
}

impl Default for ReActConfig {
    fn default() -> Self {
        Self { max_steps: 10, temperature: 0.3, max_tokens: 600 }
    }
}

// =============================================================================
// RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct ReActStep {
    pub thought:     String,
    pub action:      String,
    pub action_input: String,
    pub observation: String,
    pub success:     bool,
}

#[derive(Debug, Clone)]
pub struct ReActResult {
    pub answer:     String,
    pub steps:      Vec<ReActStep>,
    pub total_steps: usize,
    pub reached_answer: bool,
}

// =============================================================================
// REACT RUNNER
// =============================================================================

pub struct ReActRunner {
    llm:      Arc<OllamaClient>,
    registry: Arc<ToolRegistry>,
    config:   ReActConfig,
}

impl ReActRunner {
    pub fn new(llm: Arc<OllamaClient>, registry: Arc<ToolRegistry>, config: ReActConfig) -> Self {
        Self { llm, registry, config }
    }

    pub async fn run(&self, task: &str) -> ReActResult {
        let system = self.build_system_prompt(task);
        let mut messages    = vec![ChatMessage::system(&system)];
        let mut steps       = Vec::new();
        let mut reached     = false;

        for step_num in 0..self.config.max_steps {
            debug!("ReAct step {}", step_num + 1);

            let response = match self.llm.chat(
                &messages, self.config.temperature, self.config.max_tokens
            ).await {
                Ok(r)  => r,
                Err(e) => {
                    info!("ReAct LLM error: {}", e);
                    return ReActResult {
                        answer:         format!("LLM error: {}", e),
                        steps,
                        total_steps:    step_num,
                        reached_answer: false,
                    };
                }
            };

            debug!("ReAct response:\n{}", response);

            // Check for final answer.
            if let Some(answer) = extract_final_answer(&response) {
                reached = true;
                return ReActResult {
                    answer,
                    steps,
                    total_steps: step_num + 1,
                    reached_answer: reached,
                };
            }

            // Parse Thought / Action / Action Input.
            let thought      = extract_field(&response, "Thought");
            let action       = extract_field(&response, "Action");
            let action_input = extract_field(&response, "Action Input");

            if action.is_empty() {
                // No action — treat the whole response as the answer.
                return ReActResult {
                    answer:         response.trim().to_string(),
                    steps,
                    total_steps:    step_num + 1,
                    reached_answer: true,
                };
            }

            // Execute the tool.
            let tool_result = self.registry.execute(&action, &action_input).await;
            let observation  = tool_result.observation.clone();

            info!("Tool '{}' → {}", action, if tool_result.success { "OK" } else { "ERR" });

            steps.push(ReActStep {
                thought: thought.clone(),
                action:  action.clone(),
                action_input: action_input.clone(),
                observation:  observation.clone(),
                success:      tool_result.success,
            });

            // Add the round to the message history.
            messages.push(ChatMessage::assistant(&response));
            messages.push(ChatMessage::user(&format!("Observation: {}", observation)));
        }

        ReActResult {
            answer:         "Max steps reached without a final answer.".to_string(),
            steps,
            total_steps:    self.config.max_steps,
            reached_answer: false,
        }
    }

    fn build_system_prompt(&self, task: &str) -> String {
        let tool_list = self.registry.tool_prompt();
        format!(
            "You are LionAI, an AI agent that solves tasks using tools.\n\n\
             {tool_list}\n\
             ## Output format (follow EXACTLY):\n\
             Thought: [your reasoning]\n\
             Action: [tool_name]\n\
             Action Input: [input to the tool]\n\n\
             After receiving 'Observation:', continue reasoning until you have the answer.\n\
             When done, write:\n\
             Final Answer: [your answer]\n\n\
             ## Task:\n{task}"
        )
    }
}

// =============================================================================
// PARSING HELPERS
// =============================================================================

fn extract_field(text: &str, field: &str) -> String {
    let prefix = format!("{}:", field);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            return trimmed[prefix.len()..].trim().to_string();
        }
    }
    String::new()
}

fn extract_final_answer(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Final Answer:") {
            return Some(trimmed["Final Answer:".len()..].trim().to_string());
        }
    }
    None
}
