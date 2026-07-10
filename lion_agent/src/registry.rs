// lion_agent/src/registry.rs — Tool registry

use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::{Tool, ToolResult};

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self { Self { tools: HashMap::new() } }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub async fn execute(&self, name: &str, input: &str) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.execute(input).await,
            None       => ToolResult::err(format!("Unknown tool '{}'. Available: {}", name, self.names().join(", "))),
        }
    }

    /// Produces the tool description block injected into LLM prompts.
    pub fn tool_prompt(&self) -> String {
        let mut s = String::from("Available tools (use EXACTLY this format):\n\n");
        let mut names = self.names();
        names.sort();
        for name in names {
            if let Some(t) = self.get(name) {
                s.push_str(&format!(
                    "Tool: {}\nDescription: {}\nInput format: {}\n\n",
                    t.name(), t.description(), t.input_format()
                ));
            }
        }
        s
    }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}
