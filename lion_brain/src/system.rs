// lion_brain/src/system.rs — LionSystem: Unified Orchestrator

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::warn;

use crate::{
    context::{ContextConfig, ContextManager, TokenUsage},
    llm::OllamaClient,
    memory::{MemorySource, MemorySystem},
    pipeline::{PipelineConfig, ThinkingPipeline},
    router::{Route, Router},
};

// =============================================================================
// CONFIG
// =============================================================================

#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub ollama_base:  String,
    pub model:        String,
    pub memory_path:  String,
    pub streaming:    bool,
    pub show_routing: bool,
    pub show_tokens:  bool,
    pub enable_agent: bool,
    pub context:      ContextConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            ollama_base:  "http://localhost:11434".to_string(),
            model:        "gemma3:1b".to_string(),
            memory_path:  "lion_memory.bin".to_string(),
            streaming:    true,
            show_routing: false,
            show_tokens:  false,
            enable_agent: true,
            context:      ContextConfig::default(),
        }
    }
}

// =============================================================================
// TURN RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct TurnResult {
    pub answer:      String,
    pub route:       Route,
    pub intent:      String,
    pub memory_hits: usize,
    pub tokens:      TokenUsage,
    pub turn_number: usize,
}

// =============================================================================
// LION SYSTEM
// =============================================================================

pub struct LionSystem {
    pub config:      SystemConfig,
    pipeline:        ThinkingPipeline,
    context:         ContextManager,
    router:          Router,
    llm:             Arc<OllamaClient>,
    memory:          Arc<Mutex<MemorySystem>>,
    turn_number:     usize,
}

impl LionSystem {
    pub async fn new(config: SystemConfig) -> anyhow::Result<Self> {
        let pipeline_config = PipelineConfig {
            ollama_base:     config.ollama_base.clone(),
            model:           config.model.clone(),
            memory_path:     config.memory_path.clone().into(),
            embed_dim:       1152,
            max_retries:     2,
            auto_save_every: 20,
        };

        let pipeline = ThinkingPipeline::new(pipeline_config).await?;
        let llm      = pipeline.llm();
        let memory   = pipeline.memory();
        let context  = ContextManager::new(config.context.clone());

        Ok(Self {
            pipeline,
            context,
            router: Router::default(),
            llm,
            memory,
            turn_number: 0,
            config,
        })
    }

    /// Process one user turn. Returns the complete TurnResult.
    /// If `streaming = true` and Ollama is available, tokens are printed as they arrive.
    pub async fn process(
        &mut self,
        input:     &str,
        on_token:  Option<&dyn Fn(&str)>,
    ) -> TurnResult {
        self.turn_number += 1;
        let turn = self.turn_number;

        // Quick keyword memory for routing.
        let keyword_hits = {
            let mem = self.memory.lock().await;
            mem.keyword_search(input, 1)
                .into_iter()
                .count()
        };
        let best_sim = if keyword_hits > 0 { 0.7 } else { 0.0 };

        // Simple intent detection.
        let intent = detect_intent(input);

        // Route.
        let routing = self.router.route(input, &intent, best_sim);

        if self.config.show_routing {
            eprintln!("  [{}  intent={} mem_sim={:.2}]",
                routing.route.label(), routing.intent, routing.memory_similarity);
        }

        // Execute.
        let (answer, memory_hits) = match routing.route.clone() {
            Route::Agent | Route::ThinkingPipeline => {
                // Try streaming first if enabled and Ollama is available.
                if self.config.streaming && self.llm.is_available().await {
                    let system = self.build_system_prompt(&intent).await;
                    let msgs   = self.context.build_messages(&system, None, input);

                    let mut collected = String::new();
                    let result = self.llm.chat_stream(&msgs, 0.7, 800, |token| {
                        collected.push_str(token);
                        if let Some(cb) = on_token { cb(token); }
                    }).await;

                    match result {
                        Ok(_) => {
                            // Store in memory.
                            if let Ok(emb) = self.llm.embed(input).await {
                                let mut mem = self.memory.lock().await;
                                mem.store(
                                    format!("Q: {}\nA: {}", input, &collected),
                                    emb,
                                    MemorySource::LlmAnswer,
                                    0.8,
                                );
                            }
                            (collected, 0)
                        }
                        Err(e) => {
                            warn!("Streaming failed: {}. Falling back to pipeline.", e);
                            self.run_pipeline(input).await
                        }
                    }
                } else {
                    self.run_pipeline(input).await
                }
            }
            Route::Direct => {
                let mem    = self.memory.lock().await;
                let hits   = mem.keyword_search(input, 1);
                let answer = hits.first()
                    .map(|e| {
                        // Extract "A: ..." part if present.
                        if let Some(pos) = e.content.find("\nA: ") {
                            e.content[pos + 4..].trim().to_string()
                        } else {
                            e.content.clone()
                        }
                    })
                    .unwrap_or_else(|| "No direct answer found.".to_string());
                (answer, hits.len())
            }
        };

        // Update context.
        self.context.push_turn(input, &answer);

        let tokens = self.context.token_usage();

        if self.config.show_tokens {
            eprintln!("  [tokens: {}/{} ({:.0}%)]",
                tokens.history_tokens, tokens.budget,
                tokens.utilization() * 100.0);
        }

        TurnResult {
            answer,
            route:       routing.route,
            intent,
            memory_hits,
            tokens,
            turn_number: turn,
        }
    }

    async fn run_pipeline(&mut self, input: &str) -> (String, usize) {
        match self.pipeline.think(input).await {
            Ok(r)  => (r.answer, r.memory_hits),
            Err(e) => (format!("Error: {}", e), 0),
        }
    }

    async fn build_system_prompt(&self, intent: &str) -> String {
        format!(
            "You are LionAI, an intelligent AI assistant built in Rust.\n\
             You have semantic memory, multimodal perception, and tool-use capabilities.\n\
             Be concise, accurate, and helpful. Current intent: {}.",
            intent
        )
    }

    pub async fn save_memory(&self) {
        self.pipeline.save_memory().await;
    }

    pub async fn memory_entry_count(&self) -> usize {
        self.memory.lock().await.len()
    }

    pub fn clear_context(&mut self) {
        self.context.clear();
    }

    pub fn turn_number(&self) -> usize { self.turn_number }
}

fn detect_intent(input: &str) -> String {
    let lower = input.to_lowercase();
    if lower.ends_with('?') || lower.starts_with("what") || lower.starts_with("how") ||
       lower.starts_with("why") || lower.starts_with("who") {
        "Question"
    } else if lower.starts_with("hi") || lower.starts_with("hello") {
        "Greeting"
    } else if lower.contains("write") || lower.contains("create") || lower.contains("make") {
        "Creation"
    } else {
        "Chat"
    }.to_string()
}
