// lion_brain/src/critic.rs — Pre-Synthesis Critic Review
//
// Implements the pre-synthesis Critic Review step from 02_ORCHESTRATION_RUNTIME.md.
// Evaluates tool results for failures, empty data, or hallucination risks prior to final response generation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticResult {
    pub pass: bool,
    pub feedback: String,
}

pub struct CriticReviewer;

impl CriticReviewer {
    /// Review tool outputs and context before final synthesis.
    pub fn review(user_input: &str, tool_outputs: &[(&str, bool, &str)]) -> CriticResult {
        // 1. Check for failed tools
        let failed_tools: Vec<&str> = tool_outputs
            .iter()
            .filter(|(_, ok, _)| !*ok)
            .map(|(id, _, _)| *id)
            .collect();

        if !failed_tools.is_empty() {
            return CriticResult {
                pass: false,
                feedback: format!("Execution failed for tools: {}. Adjust parameters and retry.", failed_tools.join(", ")),
            };
        }

        // 2. Check for empty observations
        let empty_tools: Vec<&str> = tool_outputs
            .iter()
            .filter(|(_, ok, obs)| *ok && obs.trim().is_empty())
            .map(|(id, _, _)| *id)
            .collect();

        if !empty_tools.is_empty() {
            return CriticResult {
                pass: false,
                feedback: format!("Tools returned no data: {}. Broaden search query.", empty_tools.join(", ")),
            };
        }

        // 3. User input validation
        if user_input.trim().is_empty() {
            return CriticResult {
                pass: false,
                feedback: "Input request was empty.".to_string(),
            };
        }

        CriticResult {
            pass: true,
            feedback: "PASS".to_string(),
        }
    }
}
