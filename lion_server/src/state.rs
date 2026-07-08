// lion_server/src/state.rs

use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

use lion_core::{
    Sovereign, TernaryEncoder,
};
use serde::{Deserialize, Serialize};

use crate::config::Config;

// =============================================================================
// BROADCAST EVENTS (WebSocket)
// =============================================================================

/// An event broadcast to all connected WebSocket clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsEvent {
    /// Emitted after every live tick.
    Tick {
        tick:              u64,
        action:            String,
        gestalt_norm:      f32,
        immune_fixes:      u32,
        stress:            f32,
        episode_count:     usize,
    },
    /// Emitted after a sleep cycle completes.
    SleepCycle {
        generation:        u32,
        evolution_occurred: bool,
        sovereign_fitness: f64,
        best_child_fitness: f64,
        mutation_rate:     f32,
    },
    /// Emitted after a snapshot is saved.
    Saved {
        path:              String,
        neuron_count:      usize,
        synapse_count:     usize,
    },
    /// Emitted after a snapshot is loaded.
    Loaded {
        path:              String,
        generation:        u32,
        tick:              u64,
    },
    /// Emitted when the immune system fires.
    ImmuneAlert {
        tick:              u64,
        total_fixes:       u32,
    },
}

// =============================================================================
// SHARED SERVER STATE
// =============================================================================

/// Shared state held by every axum handler.
///
/// The `Sovereign` is protected by a `tokio::sync::Mutex` so concurrent
/// REST calls and WebSocket ticks are serialized safely.
///
/// `broadcast::Sender<WsEvent>` fans out events to all connected WS clients.
pub struct AppState {
    pub sovereign: Mutex<Sovereign>,
    pub encoder:   Mutex<Option<TernaryEncoder>>,
    pub config:    Config,
    pub event_tx:  broadcast::Sender<WsEvent>,
    pub cycles_since_last_save: Mutex<u32>,
}

impl AppState {
    /// Creates a new AppState from config.
    /// If a snapshot exists at `config.agent.snapshot_path`, loads it.
    /// Otherwise starts fresh.
    pub fn new(config: Config) -> Arc<Self> {
        let (event_tx, _) = broadcast::channel(256);

        let snapshot_path = config.agent.snapshot_path.to_str().unwrap_or("").to_string();
        let (sovereign, encoder) = if std::path::Path::new(&snapshot_path).exists() {
            tracing::info!("Loading snapshot from '{}'", snapshot_path);
            match lion_core::persist::load_from(&snapshot_path) {
                Ok(snap) => {
                    let mut s = Sovereign::new(snap.rng_seed);
                    s.brain          = snap.brain;
                    s.episodic_buffer = snap.episodic_buffer;
                    s.tick           = snap.tick;
                    s.generation     = snap.generation;
                    tracing::info!(
                        "Loaded: gen={}, tick={}, neurons={}",
                        s.generation, s.tick,
                        s.brain.alive_neuron_count()
                    );
                    (s, snap.encoder)
                }
                Err(e) => {
                    tracing::warn!("Snapshot load failed: {}. Starting fresh.", e);
                    (Sovereign::new(config.agent.seed), None)
                }
            }
        } else {
            tracing::info!("No snapshot found. Starting fresh.");
            (Sovereign::new(config.agent.seed), None)
        };

        Arc::new(Self {
            sovereign:              Mutex::new(sovereign),
            encoder:                Mutex::new(encoder),
            config,
            event_tx,
            cycles_since_last_save: Mutex::new(0),
        })
    }

    /// Subscribes to the broadcast event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<WsEvent> {
        self.event_tx.subscribe()
    }

    /// Broadcasts an event to all connected WebSocket clients.
    /// Ignores errors from zero subscribers.
    pub fn broadcast(&self, event: WsEvent) {
        let _ = self.event_tx.send(event);
    }
}
