// lion_server/src/api.rs

use std::sync::Arc;
use axum::{
    extract::{Json, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use lion_core::{
    persist::{save_snapshot, load_from, BrainSnapshot},
    SensoryInput,
    constants::FEATURE_SIZE,
};

use crate::state::{AppState, WsEvent};

type AppResult<T> = Result<Json<T>, (StatusCode, String)>;

fn err(msg: impl ToString) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
}

// =============================================================================
// REQUEST / RESPONSE TYPES
// =============================================================================

/// A sensory input for one tick, sent as JSON.
///
/// ```json
/// {
///   "inputs": [
///     { "modality": "VISION", "values": [0.5, 0.3, ...] },
///     { "modality": "DANGER", "values": [1.0, 1.0, ...] }
///   ],
///   "prev_reward": -1.0
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct TickRequest {
    pub inputs:      Vec<ModalityInput>,
    pub prev_reward: f32,
}

#[derive(Debug, Deserialize)]
pub struct ModalityInput {
    pub modality: String,
    pub values:   Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct TickResponse {
    pub tick:            u64,
    pub action:          String,
    pub stress:          f32,
    pub episode_count:   usize,
    pub immune_fixes:    u32,
    pub gestalt_sample:  Vec<f32>, // First 8 components of gestalt
}

#[derive(Debug, Deserialize)]
pub struct SleepRequest {
    pub final_reward: f32,
}

#[derive(Debug, Serialize)]
pub struct SleepResponse {
    pub generation:         u32,
    pub evolution_occurred: bool,
    pub sovereign_fitness:  f64,
    pub best_child_fitness: f64,
    pub mutation_rate:      f32,
}

#[derive(Debug, Serialize)]
pub struct AgentState {
    pub tick:              u64,
    pub generation:        u32,
    pub stress:            f32,
    pub plasticity:        f32,
    pub exploration_drive: f32,
    pub neuron_count:      usize,
    pub synapse_count:     usize,
    pub episode_count:     usize,
    pub immune_total:      u32,
    pub is_healthy:        bool,
}

#[derive(Debug, Serialize)]
pub struct NeuronView {
    pub index:      usize,
    pub generation: u32,
    pub role:       String,
    pub activation: f32,
    pub trace_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct SaveRequest {
    pub path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SaveResponse {
    pub path:          String,
    pub neuron_count:  usize,
    pub synapse_count: usize,
    pub episode_count: usize,
    pub bytes_written: u64,
}

#[derive(Debug, Deserialize)]
pub struct LoadRequest {
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct LoadResponse {
    pub path:       String,
    pub generation: u32,
    pub tick:       u64,
    pub success:    bool,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status:          &'static str,
    pub version:         &'static str,
    pub uptime_ticks:    u64,
    pub is_healthy:      bool,
    pub neuron_count:    usize,
}

// =============================================================================
// HANDLERS
// =============================================================================

/// POST /api/tick
///
/// Runs one live tick. The JSON body specifies modality inputs and the
/// reward received from the environment for the PREVIOUS tick.
pub async fn tick(
    State(state): State<Arc<AppState>>,
    Json(req):    Json<TickRequest>,
) -> AppResult<TickResponse> {
    let mut frame = SensoryInput::new();

    {
        let encoder_guard = state.encoder.lock().await;

        for input in &req.inputs {
            let role = parse_role(&input.modality)
                .map_err(err)?;

            if let Some(enc) = encoder_guard.as_ref() {
                // Route through ternary encoder if available.
                let embedding = enc.encode_f32(&input.values);
                frame.insert(role, embedding);
            } else {
                // Direct injection: pad or truncate to FEATURE_SIZE.
                let mut arr = [0.0_f32; FEATURE_SIZE];
                let n = input.values.len().min(FEATURE_SIZE);
                arr[..n].copy_from_slice(&input.values[..n]);
                frame.insert(role, arr);
            }
        }
    }

    let mut s = state.sovereign.lock().await;
    let result = s.update(&frame, req.prev_reward);

    let immune_fixes = result.immune_report.total();
    let gestalt_sample: Vec<f32> = result.gestalt[..8.min(FEATURE_SIZE)].to_vec();
    let stress = s.stress();
    let episode_count = s.episode_count();
    let tick = result.tick;
    let action = result.action.to_string();

    // Broadcast tick event to WebSocket clients.
    let gestalt_norm: f32 = result.gestalt.iter().map(|x| x * x).sum::<f32>().sqrt();
    state.broadcast(WsEvent::Tick {
        tick,
        action:        action.clone(),
        gestalt_norm,
        immune_fixes,
        stress,
        episode_count,
    });

    if immune_fixes > 0 {
        state.broadcast(WsEvent::ImmuneAlert { tick, total_fixes: immune_fixes });
    }

    Ok(Json(TickResponse {
        tick,
        action,
        stress,
        episode_count,
        immune_fixes,
        gestalt_sample,
    }))
}

/// POST /api/sleep
///
/// Triggers the evolutionary night cycle.
/// Optionally saves a snapshot if auto-save is due.
pub async fn sleep(
    State(state): State<Arc<AppState>>,
    Json(req):    Json<SleepRequest>,
) -> AppResult<SleepResponse> {
    let report = {
        let mut s = state.sovereign.lock().await;
        s.trigger_sleep_cycle(req.final_reward)
    };

    let (generation, evolution_occurred) = {
        let s = state.sovereign.lock().await;
        (s.generation, report.evolution_occurred)
    };

    state.broadcast(WsEvent::SleepCycle {
        generation,
        evolution_occurred,
        sovereign_fitness:  report.sovereign_fitness,
        best_child_fitness: report.best_child_fitness,
        mutation_rate:      report.mutation_rate,
    });

    // Auto-save logic.
    if evolution_occurred && state.config.agent.auto_save_every_n_cycles > 0 {
        let mut counter = state.cycles_since_last_save.lock().await;
        *counter += 1;
        if *counter >= state.config.agent.auto_save_every_n_cycles {
            *counter = 0;
            drop(counter);
            // Trigger save in background.
            let state_clone = Arc::clone(&state);
            tokio::spawn(async move {
                let _ = save_handler_inner(&state_clone, None).await;
            });
        }
    }

    Ok(Json(SleepResponse {
        generation,
        evolution_occurred,
        sovereign_fitness:  report.sovereign_fitness,
        best_child_fitness: report.best_child_fitness,
        mutation_rate:      report.mutation_rate,
    }))
}

/// GET /api/state
pub async fn agent_state(State(state): State<Arc<AppState>>) -> AppResult<AgentState> {
    let s = state.sovereign.lock().await;
    Ok(Json(AgentState {
        tick:              s.tick,
        generation:        s.generation,
        stress:            s.brain.epigenome.accumulated_stress,
        plasticity:        s.brain.epigenome.plasticity,
        exploration_drive: s.brain.epigenome.exploration_drive,
        neuron_count:      s.brain.alive_neuron_count(),
        synapse_count:     s.brain.alive_synapse_count(),
        episode_count:     s.episode_count(),
        immune_total:      s.brain.immune_intervention_count(),
        is_healthy:        s.brain.is_numerically_healthy(),
    }))
}

/// GET /api/neurons
pub async fn neurons(State(state): State<Arc<AppState>>) -> AppResult<Vec<NeuronView>> {
    let s = state.sovereign.lock().await;
    let views: Vec<NeuronView> = s.brain
        .alive_neurons()
        .map(|n| NeuronView {
            index:      n.id.index,
            generation: n.generation,
            role:       format!("{:?}", n.role),
            activation: n.activation,
            trace_count: n.trace_count,
        })
        .collect();
    Ok(Json(views))
}

/// POST /api/save
pub async fn save(
    State(state): State<Arc<AppState>>,
    Json(req):    Json<SaveRequest>,
) -> AppResult<SaveResponse> {
    let response = save_handler_inner(&state, req.path).await
        .map_err(err)?;
    Ok(Json(response))
}

async fn save_handler_inner(
    state: &Arc<AppState>,
    path_override: Option<String>,
) -> Result<SaveResponse, String> {
    let path = path_override
        .unwrap_or_else(|| state.config.agent.snapshot_path.to_string_lossy().to_string());

    let (snapshot, neuron_count, synapse_count, episode_count) = {
        let s   = state.sovereign.lock().await;
        let enc = state.encoder.lock().await;
        let nc  = s.brain.alive_neuron_count();
        let sc  = s.brain.alive_synapse_count();
        let ec  = s.episode_count();
        let snap = BrainSnapshot::new(
            s.tick,
            s.generation,
            s.tick, // rng_seed = tick for reproducibility
            s.brain.clone(),
            s.episodic_buffer.clone(),
            enc.clone(),
            Some(s.language_motor.clone()),
        );
        (snap, nc, sc, ec)
    };

    let _summary = save_snapshot(&snapshot, std::path::Path::new(&path))
        .map_err(|e| e.to_string())?;

    let bytes_written = std::fs::metadata(&path)
        .map(|m| m.len())
        .unwrap_or(0);

    state.broadcast(WsEvent::Saved {
        path:         path.clone(),
        neuron_count,
        synapse_count,
    });

    tracing::info!("Saved snapshot to '{}' ({} bytes)", path, bytes_written);

    Ok(SaveResponse {
        path,
        neuron_count,
        synapse_count,
        episode_count,
        bytes_written: bytes_written as u64,
    })
}

/// POST /api/load
pub async fn load(
    State(state): State<Arc<AppState>>,
    Json(req):    Json<LoadRequest>,
) -> AppResult<LoadResponse> {
    let snap = load_from(&req.path)
        .map_err(|e| err(format!("Load failed: {}", e)))?;

    let generation = snap.generation;
    let tick       = snap.tick;
    let encoder    = snap.encoder.clone();

    {
        let mut s = state.sovereign.lock().await;
        s.brain           = snap.brain;
        s.episodic_buffer = snap.episodic_buffer;
        s.tick            = snap.tick;
        s.generation      = snap.generation;
    }

    if let Some(enc) = encoder {
        *state.encoder.lock().await = Some(enc);
    }

    state.broadcast(WsEvent::Loaded {
        path:       req.path.clone(),
        generation,
        tick,
    });

    tracing::info!("Loaded snapshot from '{}' (gen={}, tick={})", req.path, generation, tick);

    Ok(Json(LoadResponse {
        path:       req.path,
        generation,
        tick,
        success:    true,
    }))
}

/// GET /api/health
pub async fn health(State(state): State<Arc<AppState>>) -> AppResult<HealthResponse> {
    let (tick, is_healthy, neuron_count) = {
        let s = state.sovereign.lock().await;
        (s.tick, s.brain.is_numerically_healthy(), s.brain.alive_neuron_count())
    };

    Ok(Json(HealthResponse {
        status:       "ok",
        version:      env!("CARGO_PKG_VERSION"),
        uptime_ticks: tick,
        is_healthy,
        neuron_count,
    }))
}

// =============================================================================
// HELPERS
// =============================================================================

fn parse_role(s: &str) -> Result<lion_core::types::Role, String> {
    match s.to_uppercase().as_str() {
        "VISION" => Ok(lion_core::types::Role::Vision),
        "MOTOR"  => Ok(lion_core::types::Role::Motor),
        "MEMORY" => Ok(lion_core::types::Role::Memory),
        "DANGER" => Ok(lion_core::types::Role::Danger),
        other    => Err(format!("Unknown modality: '{}'", other)),
    }
}
