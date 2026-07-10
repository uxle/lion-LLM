// lion_agent/src/tools/mod.rs

pub mod calculator;
pub mod code_runner;
pub mod file_ops;
pub mod web_fetch;

pub use calculator::Calculator;
pub use code_runner::CodeRunner;
pub use file_ops::{FileRead, FileWrite};
pub use web_fetch::WebFetch;

use crate::registry::ToolRegistry;

/// Registers all built-in tools into the registry.
pub fn register_all(registry: &mut ToolRegistry) {
    registry.register(Calculator);
    registry.register(FileRead);
    registry.register(FileWrite);
    registry.register(CodeRunner);
    registry.register(WebFetch::new());
}
