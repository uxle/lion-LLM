// lion_brain/src/router.rs — Per-turn routing

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Route {
    Direct,
    ThinkingPipeline,
    Agent,
}

impl Route {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Direct           => "⚡ direct",
            Self::ThinkingPipeline => "🧠 thinking",
            Self::Agent            => "🤖 agent",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub route:             Route,
    pub memory_similarity: f32,
    pub intent:            String,
}

#[derive(Debug, Clone)]
pub struct Router {
    pub direct_threshold: f32,
}

impl Default for Router {
    fn default() -> Self { Self { direct_threshold: 0.90 } }
}

impl Router {
    pub fn route(
        &self,
        input:           &str,
        intent:          &str,
        best_memory_sim: f32,
    ) -> RoutingDecision {
        let lower = input.to_lowercase();

        // Agent route — task keywords.
        let agent_keywords = [
            "create a file", "write a file", "run ", "execute ",
            "calculate", "fetch ", "web search", "download",
            "list files", "read the file", "write to file",
        ];
        if intent == "Creation" || agent_keywords.iter().any(|&kw| lower.contains(kw)) {
            return RoutingDecision {
                route: Route::Agent,
                memory_similarity: best_memory_sim,
                intent: intent.to_string(),
            };
        }

        // Direct route — high-confidence memory hit.
        if best_memory_sim >= self.direct_threshold {
            return RoutingDecision {
                route: Route::Direct,
                memory_similarity: best_memory_sim,
                intent: intent.to_string(),
            };
        }

        RoutingDecision {
            route: Route::ThinkingPipeline,
            memory_similarity: best_memory_sim,
            intent: intent.to_string(),
        }
    }
}
