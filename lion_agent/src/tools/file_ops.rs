// lion_agent/src/tools/file_ops.rs — File read/write tools

use std::pin::Pin;
use std::future::Future;
use std::path::Path;
use crate::tool::{Tool, ToolResult};

// =============================================================================
// PATH SAFETY VALIDATOR
// =============================================================================

/// Checks if a file path is safe for read/write.
/// Restricts access to files under the current working directory or /tmp folder.
fn is_safe_path(p: &str) -> bool {
    let p_trimmed = p.trim();
    if p_trimmed.is_empty() { return false; }
    
    let path = Path::new(p_trimmed);
    
    // Canonicalize existing path or search parents
    let mut current = path.to_path_buf();
    let mut resolved = None;
    for _ in 0..12 {
        if let Ok(c) = std::fs::canonicalize(&current) {
            resolved = Some(c);
            break;
        }
        if !current.pop() {
            break;
        }
    }
    
    let resolved_path = match resolved {
        Some(r) => r,
        None => {
            // Check for path traversal string tricks
            let s = p_trimmed.to_lowercase();
            return !s.contains("..") && !s.starts_with('/') && !s.contains(":");
        }
    };
    
    let cwd = std::fs::canonicalize(std::env::current_dir().unwrap_or_default())
        .unwrap_or_default();
        
    resolved_path.starts_with(&cwd) || resolved_path.starts_with("/tmp")
}

// =============================================================================
// FILE READ
// =============================================================================

pub struct FileRead;

impl Tool for FileRead {
    fn name(&self)         -> &'static str { "file_read" }
    fn description(&self)  -> &'static str { "Reads a text file and returns its contents safely" }
    fn input_format(&self) -> &'static str { "A file path relative to working directory or under /tmp" }

    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        Box::pin(async move {
            let path = input.trim();
            if !is_safe_path(path) {
                return ToolResult::err("Access denied: path is outside the sandbox environment.");
            }
            
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let preview = if content.len() > 4000 {
                        format!("{}… [truncated, {} bytes total]", &content[..4000], content.len())
                    } else { content };
                    ToolResult::ok(preview)
                }
                Err(e) => ToolResult::err(format!("Cannot read '{}': {}", path, e)),
            }
        })
    }
}

// =============================================================================
// FILE WRITE
// =============================================================================

pub struct FileWrite;

impl Tool for FileWrite {
    fn name(&self)         -> &'static str { "file_write" }
    fn description(&self)  -> &'static str { "Writes content to a file safely" }
    fn input_format(&self) -> &'static str {
        r#"JSON: {"path": "relative/file.txt", "content": "text to write"}"#
    }

    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        Box::pin(async move {
            let parsed: serde_json::Value = match serde_json::from_str(input.trim()) {
                Ok(v)  => v,
                Err(e) => return ToolResult::err(format!("Invalid JSON: {}", e)),
            };
            let path    = match parsed["path"].as_str()    { Some(p) => p, None => return ToolResult::err("Missing 'path'") };
            let content = match parsed["content"].as_str() { Some(c) => c, None => return ToolResult::err("Missing 'content'") };

            if !is_safe_path(path) {
                return ToolResult::err("Access denied: path is outside the sandbox environment.");
            }

            // Create parent directories safely.
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }

            match std::fs::write(path, content) {
                Ok(_)  => ToolResult::ok(format!("Written {} bytes to '{}'", content.len(), path)),
                Err(e) => ToolResult::err(format!("Cannot write '{}': {}", path, e)),
            }
        })
    }
}

// =============================================================================
// DIRECTORY LIST
// =============================================================================

pub struct DirList;

impl Tool for DirList {
    fn name(&self)         -> &'static str { "dir_list" }
    fn description(&self)  -> &'static str { "Lists the contents of a directory safely" }
    fn input_format(&self) -> &'static str { "A directory path" }

    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        Box::pin(async move {
            let path = input.trim();
            if !is_safe_path(path) {
                return ToolResult::err("Access denied: path is outside the sandbox environment.");
            }

            match std::fs::read_dir(path) {
                Ok(entries) => {
                    let mut lines = Vec::new();
                    for entry in entries.flatten() {
                        let meta = entry.metadata();
                        let kind = match meta.as_ref().map(|m| m.is_dir()) {
                            Ok(true)  => "DIR ",
                            Ok(false) => "FILE",
                            _         => "???",
                        };
                        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                        lines.push(format!("[{}] {:>10} B  {}", kind, size, entry.file_name().to_string_lossy()));
                    }
                    lines.sort();
                    ToolResult::ok(format!("Contents of '{}':\n{}", path, lines.join("\n")))
                }
                Err(e) => ToolResult::err(format!("Cannot list '{}': {}", path, e)),
            }
        })
    }
}
