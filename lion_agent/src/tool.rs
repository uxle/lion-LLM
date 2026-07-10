// lion_agent/src/tool.rs — Tool trait and result type

use std::future::Future;
use std::pin::Pin;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub observation:    String,
    pub success:        bool,
    pub data:           Option<serde_json::Value>,
    pub token_estimate: usize,
}

impl ToolResult {
    pub fn ok(observation: impl Into<String>) -> Self {
        let s = observation.into();
        let tokens = s.len() / 4;
        Self { observation: s, success: true, data: None, token_estimate: tokens }
    }
    pub fn ok_data(observation: impl Into<String>, data: serde_json::Value) -> Self {
        let s = observation.into();
        let tokens = s.len() / 4;
        Self { observation: s, success: true, data: Some(data), token_estimate: tokens }
    }
    pub fn err(message: impl Into<String>) -> Self {
        let s = format!("Error: {}", message.into());
        Self { observation: s, success: false, data: None, token_estimate: 20 }
    }
}

pub trait Tool: Send + Sync {
    fn name(&self)         -> &'static str;
    fn description(&self)  -> &'static str;
    fn input_format(&self) -> &'static str;
    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>>;
}
