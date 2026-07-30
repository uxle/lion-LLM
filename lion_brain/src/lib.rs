// lion_brain/src/lib.rs — LionAI v1.0 Brain

pub mod context;
pub mod critic;
pub mod llm;
pub mod mars_solver;
pub mod memory;
pub mod pipeline;
pub mod risk;
pub mod router;
pub mod semantic_cache;
pub mod system;

pub use context::{ContextConfig, ContextManager, TokenUsage};
pub use critic::{CriticResult, CriticReviewer};
pub use llm::{ChatMessage, OllamaClient, OllamaError};
pub use mars_solver::{ByzantineFilter, MarsColonyStatus, MarsRecoveryPlan, MarsRecoverySolver, SensorReading};
pub use memory::{MemoryEntry, MemoryResult, MemorySource, MemoryStats, MemorySystem};
pub use pipeline::{PipelineConfig, ThinkingPipeline, ThinkResult};
pub use risk::{RiskAssessment, RiskAssessor, RiskLevel};
pub use router::{Route, Router, RoutingDecision};
pub use semantic_cache::{CacheEntry, SemanticCache};
pub use system::{LionSystem, SystemConfig, TurnResult};
