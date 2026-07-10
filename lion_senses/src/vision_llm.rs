// lion_senses/src/vision_llm.rs — Ollama Vision Model Client

use std::path::Path;
use tracing::{debug, info};

pub struct VisionLLM {
    client:   reqwest::Client,
    base_url: String,
    model:    String,
}

impl VisionLLM {
    pub fn new(base_url: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("HTTP client build failed");
        Self { client, base_url: base_url.trim_end_matches('/').to_string(), model: model.to_string() }
    }

    pub fn moondream(base_url: &str) -> Self { Self::new(base_url, "moondream") }
    pub fn llava(base_url: &str)     -> Self { Self::new(base_url, "llava") }

    pub async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        matches!(self.client.get(&url).send().await, Ok(r) if r.status().is_success())
    }

    /// Describes an image from a file path.
    pub async fn describe_file(&self, path: &Path, prompt: &str) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        use base64::Engine;
        let b64   = base64::engine::general_purpose::STANDARD.encode(&bytes);
        self.describe_base64(&b64, prompt).await
    }

    /// Describes an image from base64 data.
    pub async fn describe_base64(&self, b64: &str, prompt: &str) -> Result<String, String> {
        debug!("Vision request to {}/{}", self.base_url, self.model);

        let body = serde_json::json!({
            "model":  self.model,
            "stream": false,
            "messages": [{ "role": "user", "content": prompt, "images": [b64] }]
        });

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let text = json["message"]["content"].as_str().unwrap_or("").trim().to_string();

        if text.is_empty() {
            return Err("Empty vision response".to_string());
        }

        info!("Vision description ({} chars)", text.len());
        Ok(text)
    }

    /// Full image analysis with default prompt.
    pub async fn analyze(&self, path: &Path) -> Result<String, String> {
        self.describe_file(
            path,
            "Describe this image in detail. Include: main subject, colors, \
             setting, any visible text, and whether it appears dangerous, calm, or neutral.",
        ).await
    }
}
