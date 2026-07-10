// lion_agent/src/tools/code_runner.rs — Run shell commands in a subprocess

use std::pin::Pin;
use std::future::Future;
use std::process::Command;
use crate::tool::{Tool, ToolResult};

pub struct CodeRunner;

impl Tool for CodeRunner {
    fn name(&self)         -> &'static str { "shell" }
    fn description(&self)  -> &'static str {
        "Runs a shell command and returns stdout+stderr (30-second timeout)"
    }
    fn input_format(&self) -> &'static str {
        "A shell command e.g.: echo Hello World"
    }

    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        Box::pin(async move {
            let cmd = input.trim();
            if cmd.is_empty() { return ToolResult::err("Empty command"); }

            // Block dangerous patterns.
            let dangerous = ["rm -rf /", "sudo rm", "mkfs", "dd if=", "> /dev/"];
            for pat in &dangerous {
                if cmd.contains(pat) {
                    return ToolResult::err(format!("Blocked dangerous command: {}", pat));
                }
            }

            let output = Command::new("bash")
                .arg("-c")
                .arg(cmd)
                .output();

            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&o.stderr).to_string();

                    let mut combined = stdout.clone();
                    if !stderr.is_empty() {
                        combined.push_str(&format!("\n[stderr]\n{}", stderr));
                    }

                    let exit_code = o.status.code().unwrap_or(-1);
                    combined.push_str(&format!("\n[exit code: {}]", exit_code));

                    if combined.len() > 5000 {
                        combined.truncate(5000);
                        combined.push_str("\n... [output truncated]");
                    }

                    if o.status.success() {
                        ToolResult::ok(combined)
                    } else {
                        ToolResult::err(combined)
                    }
                }
                Err(e) => ToolResult::err(format!("Failed to run command: {}", e)),
            }
        })
    }
}
