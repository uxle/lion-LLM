// lion_brain/src/llm.rs — Ollama HTTP client
//
// Handles: chat completion, token streaming, text embedding, availability check.
// Gracefully handles the case where Ollama is not running.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

// =============================================================================
// CHAT MESSAGE
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: &str)    -> Self { Self { role: "system".into(),    content: content.into() } }
    pub fn user(content: &str)      -> Self { Self { role: "user".into(),      content: content.into() } }
    pub fn assistant(content: &str) -> Self { Self { role: "assistant".into(), content: content.into() } }
}

// =============================================================================
// ERROR
// =============================================================================

#[derive(Debug)]
pub enum OllamaError {
    Http(reqwest::Error),
    Api(String),
    NotAvailable,
    EmptyResponse,
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Http(e)       => write!(f, "HTTP error: {}", e),
            Self::Api(msg)      => write!(f, "Ollama API error: {}", msg),
            Self::NotAvailable  => write!(f, "Ollama is not running. Start it with: ollama serve"),
            Self::EmptyResponse => write!(f, "Ollama returned an empty response"),
        }
    }
}

impl From<reqwest::Error> for OllamaError {
    fn from(e: reqwest::Error) -> Self { Self::Http(e) }
}

// =============================================================================
// OLLAMA CLIENT
// =============================================================================

/// HTTP client for the Ollama local LLM server.
///
/// Works with any model pulled via `ollama pull <model>`.
/// Recommended: `ollama pull gemma3:1b`
#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub base_url: String,
    pub model:    String,
    client:       Client,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model:    model.to_string(),
            client,
        }
    }

    // ── Availability check ────────────────────────────────────────────────────

    /// Returns true if Ollama is running and the model is available.
    pub async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    /// Returns the list of available model names.
    pub async fn available_models(&self) -> Vec<String> {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(r) if r.status().is_success() => {
                match r.json::<serde_json::Value>().await {
                    Ok(v) => v["models"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m["name"].as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    Err(_) => vec![],
                }
            }
            _ => vec![],
        }
    }

    // ── Chat completion ───────────────────────────────────────────────────────

    /// Sends a chat request and returns the complete response.
    pub async fn chat(
        &self,
        messages:    &[ChatMessage],
        temperature: f32,
        max_tokens:  i32,
    ) -> Result<String, OllamaError> {
        let body = serde_json::json!({
            "model":   self.model,
            "stream":  false,
            "messages": messages,
            "options": {
                "temperature":    temperature,
                "num_predict":    max_tokens,
                "repeat_penalty": 1.1,
            }
        });

        debug!("Chat request to {}", self.base_url);

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() { OllamaError::NotAvailable }
                else { OllamaError::Http(e) }
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text   = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(format!("HTTP {}: {}", status, text)));
        }

        let json: serde_json::Value = resp.json().await.map_err(OllamaError::Http)?;
        let content = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        if content.is_empty() {
            return Err(OllamaError::EmptyResponse);
        }

        Ok(content)
    }

    // ── Streaming chat ────────────────────────────────────────────────────────

    /// Sends a chat request with streaming. Calls `on_token` for each token.
    /// Returns the complete assembled response.
    pub async fn chat_stream<F>(
        &self,
        messages:    &[ChatMessage],
        temperature: f32,
        max_tokens:  i32,
        mut on_token: F,
    ) -> Result<String, OllamaError>
    where
        F: FnMut(&str),
    {
        use futures_util::StreamExt;

        let body = serde_json::json!({
            "model":   self.model,
            "stream":  true,
            "messages": messages,
            "options": {
                "temperature":    temperature,
                "num_predict":    max_tokens,
                "repeat_penalty": 1.1,
            }
        });

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() { OllamaError::NotAvailable }
                else { OllamaError::Http(e) }
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text   = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(format!("HTTP {}: {}", status, text)));
        }

        let mut stream      = resp.bytes_stream();
        let mut full_text   = String::new();
        let mut buffer      = Vec::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(OllamaError::Http)?;
            buffer.extend_from_slice(&bytes);

            // Process all complete JSON lines.
            while let Some(nl) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = buffer.drain(..=nl).collect::<Vec<u8>>();
                let line       = String::from_utf8_lossy(&line_bytes);
                let line       = line.trim();
                if line.is_empty() { continue; }

                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(content) = obj["message"]["content"].as_str() {
                        if !content.is_empty() {
                            on_token(content);
                            full_text.push_str(content);
                        }
                    }
                    if obj["done"].as_bool().unwrap_or(false) {
                        break;
                    }
                }
            }
        }

        if full_text.is_empty() {
            return Err(OllamaError::EmptyResponse);
        }

        Ok(full_text)
    }

    // ── Embeddings ────────────────────────────────────────────────────────────

    /// Generates a text embedding vector via Ollama's /api/embed endpoint.
    ///
    /// Returns a Vec<f32> of length determined by the model (gemma3:1b = 1152).
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, OllamaError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        let resp = self.client
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() { OllamaError::NotAvailable }
                else { OllamaError::Http(e) }
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text   = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(format!("HTTP {}: {}", status, text)));
        }

        let json: serde_json::Value = resp.json().await.map_err(OllamaError::Http)?;

        // Ollama embed response: { "embeddings": [[f32, ...]] }
        let embedding = json["embeddings"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();

        if embedding.is_empty() {
            return Err(OllamaError::EmptyResponse);
        }

        Ok(embedding)
    }

    // ── Vision (multimodal) ───────────────────────────────────────────────────

    /// Describes an image using a vision-capable model (e.g. moondream, llava).
    /// `image_b64` is the base64-encoded image data.
    pub async fn describe_image(
        &self,
        image_b64: &str,
        prompt:    &str,
    ) -> Result<String, OllamaError> {
        let body = serde_json::json!({
            "model":  self.model,
            "stream": false,
            "messages": [{
                "role":    "user",
                "content": prompt,
                "images":  [image_b64]
            }]
        });

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() { OllamaError::NotAvailable }
                else { OllamaError::Http(e) }
            })?;

        if !resp.status().is_success() {
            return Err(OllamaError::Api(
                resp.text().await.unwrap_or_default()
            ));
        }

        let json: serde_json::Value = resp.json().await.map_err(OllamaError::Http)?;
        Ok(json["message"]["content"].as_str().unwrap_or("").trim().to_string())
    }
}
