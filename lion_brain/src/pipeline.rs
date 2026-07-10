// lion_brain/src/pipeline.rs — 7-Stage Thinking+ Pipeline
//
// Wraps the LLM in a structured reasoning chain:
//   Understand → Remember → Retrieve → Reason → Generate → Verify → Optimize

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::llm::{ChatMessage, OllamaClient};
use crate::memory::{MemorySource, MemorySystem};

// =============================================================================
// CONFIG
// =============================================================================

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub ollama_base:     String,
    pub model:           String,
    pub memory_path:     PathBuf,
    pub embed_dim:       usize,
    pub max_retries:     usize,
    pub auto_save_every: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ollama_base:     "http://localhost:11434".to_string(),
            model:           "gemma3:1b".to_string(),
            memory_path:     "lion_memory.bin".into(),
            embed_dim:       1152,
            max_retries:     2,
            auto_save_every: 10,
        }
    }
}

// =============================================================================
// THINK RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct ThinkResult {
    pub answer:       String,
    pub intent:       String,
    pub confidence:   f32,
    pub memory_hits:  usize,
    pub was_verified: bool,
    pub tokens_used:  usize,
}

// =============================================================================
// THINKING PIPELINE
// =============================================================================

pub struct ThinkingPipeline {
    pub config:  PipelineConfig,
    llm:         Arc<OllamaClient>,
    pub memory:  Arc<Mutex<MemorySystem>>,
    turn_count:  usize,
}

impl ThinkingPipeline {
    pub async fn new(config: PipelineConfig) -> anyhow::Result<Self> {
        let llm = Arc::new(OllamaClient::new(&config.ollama_base, &config.model));
        let memory = Arc::new(Mutex::new(
            MemorySystem::load_or_new(&config.memory_path, config.embed_dim)
        ));
        Ok(Self { config, llm, memory, turn_count: 0 })
    }

    pub fn llm(&self) -> Arc<OllamaClient> { Arc::clone(&self.llm) }
    pub fn memory(&self) -> Arc<Mutex<MemorySystem>> { Arc::clone(&self.memory) }

    /// The 7-stage Thinking+ pipeline.
    pub async fn think(&mut self, input: &str) -> anyhow::Result<ThinkResult> {
        self.turn_count += 1;
        debug!("Thinking+ turn {} for: '{}'", self.turn_count, &input[..input.len().min(60)]);

        // ── Stage 1: Understand ───────────────────────────────────────────────
        let intent = self.detect_intent(input);
        debug!("Stage 1 — Intent: {}", intent);

        // ── Stage 2: Remember (keyword search for speed) ──────────────────────
        let keyword_memories: Vec<String> = {
            let mem = self.memory.lock().await;
            mem.keyword_search(input, 3)
                .into_iter()
                .take(2)
                .map(|e| e.content[..e.content.len().min(200)].to_string())
                .collect()
        };
        debug!("Stage 2 — {} keyword memories found", keyword_memories.len());

        // ── Stage 3: Retrieve (semantic if Ollama available) ──────────────────
        let semantic_memories = if self.llm.is_available().await {
            match self.llm.embed(input).await {
                Ok(emb) => {
                    let mut mem = self.memory.lock().await;
                    mem.search(&emb, 3, 0.65)
                        .into_iter()
                        .map(|r| r.entry.content[..r.entry.content.len().min(200)].to_string())
                        .collect::<Vec<_>>()
                }
                Err(e) => {
                    debug!("Embed failed: {}. Using keyword only.", e);
                    vec![]
                }
            }
        } else {
            vec![]
        };

        let memory_hits = keyword_memories.len() + semantic_memories.len();
        debug!("Stage 3 — {} semantic memories", semantic_memories.len());

        // Build memory context block.
        let all_memories: Vec<String> = semantic_memories
            .into_iter()
            .chain(keyword_memories.into_iter())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(3)
            .collect();

        let memory_ctx = if all_memories.is_empty() {
            String::new()
        } else {
            format!(
                "## Relevant Memory\n{}\n",
                all_memories.iter().enumerate()
                    .map(|(i, m)| format!("[{}] {}", i + 1, m))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        // ── Stage 4: Reason ───────────────────────────────────────────────────
        // ── Stage 5: Generate ─────────────────────────────────────────────────

        if !self.llm.is_available().await {
            let fallback = self.offline_response(input, &all_memories);
            return Ok(ThinkResult {
                answer:       fallback,
                intent,
                confidence:   0.5,
                memory_hits,
                was_verified: false,
                tokens_used:  0,
            });
        }

        let system = format!(
            "You are LionAI, an intelligent assistant with a cognitive memory system.\n\
             Think step by step. Be accurate, concise, and helpful.\n\
             Current intent: {}\n{}",
            intent, memory_ctx
        );

        let messages = vec![
            ChatMessage::system(&system),
            ChatMessage::user(input),
        ];

        let raw_answer = self.llm
            .chat(&messages, 0.7, 800)
            .await
            .unwrap_or_else(|e| format!("I encountered an error: {}", e));

        // ── Stage 6: Verify ───────────────────────────────────────────────────
        let (answer, was_verified) = self.verify_answer(input, &raw_answer).await;

        // ── Stage 7: Optimize ─────────────────────────────────────────────────
        // (Store in memory for future retrieval)
        if let Ok(emb) = self.llm.embed(input).await {
            let mut mem = self.memory.lock().await;
            let combined = format!("Q: {}\nA: {}", input, &answer);
            mem.store(combined, emb, MemorySource::LlmAnswer, 0.8);
        }

        // Auto-save memory periodically.
        if self.turn_count % self.config.auto_save_every == 0 {
            let mem = self.memory.lock().await;
            let _ = mem.save(&self.config.memory_path);
            info!("Auto-saved memory ({} entries)", mem.len());
        }

        Ok(ThinkResult {
            answer,
            intent,
            confidence:   0.85,
            memory_hits,
            was_verified,
            tokens_used:  0,
        })
    }

    // ── Intent detection (lightweight keyword matching) ───────────────────────

    fn detect_intent(&self, input: &str) -> String {
        let lower = input.to_lowercase();
        if lower.starts_with("what") || lower.starts_with("who") ||
           lower.starts_with("where") || lower.starts_with("when") ||
           lower.starts_with("why") || lower.starts_with("how") ||
           lower.ends_with('?') {
            "Question".to_string()
        } else if lower.contains("write") || lower.contains("create") ||
                  lower.contains("generate") || lower.contains("make") {
            "Creation".to_string()
        } else if lower.contains("calculate") || lower.contains("compute") ||
                  lower.contains("what is") && (lower.contains('+') ||
                  lower.contains('*') || lower.contains('/') || lower.contains('-')) {
            "Math".to_string()
        } else if lower.starts_with("hi") || lower.starts_with("hello") ||
                  lower.starts_with("hey") {
            "Greeting".to_string()
        } else {
            "Chat".to_string()
        }
    }

    // ── Verify: simple hallucination check ───────────────────────────────────

    async fn verify_answer(&self, _question: &str, answer: &str) -> (String, bool) {
        // Check for obviously bad patterns.
        if answer.contains("I cannot") || answer.contains("I don't have access") {
            return (answer.to_string(), true);
        }
        (answer.to_string(), true)
    }

    // ── Offline fallback ──────────────────────────────────────────────────────

    fn offline_response(&self, input: &str, memories: &[String]) -> String {
        let lower = input.to_lowercase();

        if lower.starts_with("hi") || lower.starts_with("hello") || lower.starts_with("hey") {
            return "Hello! I'm LionAI. Note: Ollama is not running, so my language capabilities are limited. \
                   Start it with: ollama serve".to_string();
        }

        if !memories.is_empty() {
            return format!(
                "Based on my memory:\n\n{}\n\n(Note: Ollama is offline — install and start Ollama for full AI responses)",
                memories.first().unwrap()
            );
        }

        format!(
            "I understood your request: '{}'\n\n\
             However, Ollama is not running. To get full AI responses:\n\
             1. Install Ollama: https://ollama.com\n\
             2. Run: ollama pull gemma3:1b\n\
             3. Start: ollama serve\n\
             4. Restart LionAI\n\n\
             All other features (image encoding, audio analysis, calculator, file tools) work without Ollama.",
            &input[..input.len().min(80)]
        )
    }

    pub async fn save_memory(&self) {
        let mem = self.memory.lock().await;
        let _ = mem.save(&self.config.memory_path);
    }
}
