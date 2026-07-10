// lion_agent/src/agent.rs — High-level Agent facade

use std::sync::Arc;
use tracing::info;

use lion_brain::llm::OllamaClient;
use crate::react::{ReActConfig, ReActResult, ReActRunner};
use crate::registry::ToolRegistry;
use crate::tools;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub ollama_base: String,
    pub model:       String,
    pub react:       ReActConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            ollama_base: "http://localhost:11434".to_string(),
            model:       "gemma3:1b".to_string(),
            react:       ReActConfig::default(),
        }
    }
}

pub struct Agent {
    pub config:   AgentConfig,
    pub registry: Arc<ToolRegistry>,
    llm:          Arc<OllamaClient>,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        let llm = Arc::new(OllamaClient::new(&config.ollama_base, &config.model));

        let mut registry = ToolRegistry::new();
        tools::register_all(&mut registry);
        let registry = Arc::new(registry);

        Self { config, registry, llm }
    }

    /// Returns true if Ollama is available.
    pub async fn is_available(&self) -> bool {
        self.llm.is_available().await
    }

    /// Runs the ReAct loop for a given task.
    pub async fn run_task(&self, task: &str) -> ReActResult {
        info!("Agent task: {}", &task[..task.len().min(80)]);
        let runner = ReActRunner::new(
            Arc::clone(&self.llm),
            Arc::clone(&self.registry),
            self.config.react.clone(),
        );
        runner.run(task).await
    }

    /// Executes a single named tool directly (bypasses LLM).
    pub async fn use_tool_directly(&self, tool: &str, input: &str) -> String {
        let result = self.registry.execute(tool, input).await;
        result.observation
    }

    /// Returns names of all registered tools.
    pub fn tool_names(&self) -> Vec<&str> {
        self.registry.names()
    }
}
