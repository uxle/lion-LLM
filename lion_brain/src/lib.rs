// lion_brain/src/lib.rs — LionAI v1.0 Brain

pub mod context;
pub mod llm;
pub mod memory;
pub mod pipeline;
pub mod router;
pub mod system;

pub use context::{ContextConfig, ContextManager, TokenUsage};
pub use llm::{ChatMessage, OllamaClient, OllamaError};
pub use memory::{MemoryEntry, MemoryResult, MemorySource, MemoryStats, MemorySystem};
pub use pipeline::{PipelineConfig, ThinkingPipeline, ThinkResult};
pub use router::{Route, Router, RoutingDecision};
pub use system::{LionSystem, SystemConfig, TurnResult};
