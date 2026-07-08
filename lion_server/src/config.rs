// lion_server/src/config.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level server configuration, loaded from a TOML file.
///
/// Default config path: `lion_config.toml`
///
/// Example config:
/// ```toml
/// [server]
/// host = "127.0.0.1"
/// port = 8080
/// log_level = "info"
///
/// [agent]
/// seed = 42
/// snapshot_path = "snapshots/lion_brain.bin"
/// auto_save_every_n_cycles = 5
///
/// [brain]
/// encoder_input_size = 64
/// encoder_hidden_sizes = [64]
/// night_cycle_population = 15
/// episodic_buffer_size = 1000
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server:  ServerConfig,
    pub agent:   AgentConfig,
    pub brain:   BrainConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind host (e.g. "127.0.0.1" or "0.0.0.0")
    pub host: String,

    /// Bind port (default: 8080)
    pub port: u16,

    /// Log level filter string (e.g. "info", "debug", "warn")
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Seed for the sovereign's RNG.
    pub seed: u64,

    /// Path to save/load brain snapshots.
    pub snapshot_path: PathBuf,

    /// Auto-save after every N successful evolutions.
    /// 0 = disabled.
    pub auto_save_every_n_cycles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    /// TernaryEncoder input feature size.
    pub encoder_input_size: usize,

    /// TernaryEncoder hidden layer sizes.
    pub encoder_hidden_sizes: Vec<usize>,

    /// Night cycle child population size.
    pub night_cycle_population: usize,

    /// Maximum episodes retained in the episodic buffer.
    pub episodic_buffer_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host:      "127.0.0.1".to_string(),
                port:      8080,
                log_level: "info".to_string(),
            },
            agent: AgentConfig {
                seed:                      42,
                snapshot_path:             PathBuf::from("snapshots/lion_brain.bin"),
                auto_save_every_n_cycles:  5,
            },
            brain: BrainConfig {
                encoder_input_size:     64,
                encoder_hidden_sizes:   vec![64],
                night_cycle_population: 15,
                episodic_buffer_size:   1000,
            },
        }
    }
}

impl Config {
    /// Loads config from a TOML file, falling back to defaults.
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_else(|e| {
                tracing::warn!("Config parse error: {}. Using defaults.", e);
                Self::default()
            }),
            Err(_) => {
                tracing::info!("Config file not found at '{}'. Using defaults.", path);
                Self::default()
            }
        }
    }

    /// Returns the bind address as "host:port".
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }
}
