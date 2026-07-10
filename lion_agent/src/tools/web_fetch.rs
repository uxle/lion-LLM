// lion_agent/src/tools/web_fetch.rs — Fetch a web page as plain text

use std::pin::Pin;
use std::future::Future;
use reqwest::Client;
use std::time::Duration;
use crate::tool::{Tool, ToolResult};

pub struct WebFetch {
    client: Client,
}

impl WebFetch {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("LionAI/1.0")
            .build()
            .expect("HTTP client build failed");
        Self { client }
    }
}

impl Tool for WebFetch {
    fn name(&self)         -> &'static str { "web_fetch" }
    fn description(&self)  -> &'static str { "Fetches text content from a URL" }
    fn input_format(&self) -> &'static str { "A full URL e.g.: https://example.com" }

    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        let url = input.trim().to_string();
        Box::pin(async move {
            if !url.starts_with("http") {
                return ToolResult::err("URL must start with http:// or https://");
            }

            match self.client.get(&url).send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    match resp.text().await {
                        Ok(body) => {
                            // Strip HTML tags crudely.
                            let text = strip_html(&body);
                            let truncated = if text.len() > 4000 {
                                format!("{}… [truncated, {} chars]", &text[..4000], text.len())
                            } else { text };
                            ToolResult::ok(format!("[HTTP {}] {}", status, truncated))
                        }
                        Err(e) => ToolResult::err(format!("Failed to read body: {}", e)),
                    }
                }
                Err(e) => ToolResult::err(format!("Fetch '{}' failed: {}", url, e)),
            }
        })
    }
}

fn strip_html(html: &str) -> String {
    let mut out     = String::with_capacity(html.len() / 2);
    let mut in_tag  = false;
    for ch in html.chars() {
        match ch {
            '<'  => { in_tag = true; }
            '>'  => { in_tag = false; out.push(' '); }
            '\n' | '\r' => { if !in_tag { out.push('\n'); } }
            c if !in_tag => { out.push(c); }
            _    => {}
        }
    }
    // Collapse whitespace.
    let lines: Vec<&str> = out.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n")
}
