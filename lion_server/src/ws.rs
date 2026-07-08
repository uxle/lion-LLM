// lion_server/src/ws.rs

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use tokio::select;

use crate::state::AppState;

/// WebSocket upgrade handler at `GET /ws`.
///
/// Each connected client receives a broadcast of every WsEvent:
///   - Tick events (every live update)
///   - SleepCycle events (after every night cycle)
///   - Saved / Loaded events
///   - ImmuneAlert events (when the immune system fires)
///
/// Clients are read-only subscribers — they cannot send commands
/// via the WebSocket (use the REST API for that).
pub async fn ws_handler(
    ws:             WebSocketUpgrade,
    State(state):   State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    // Subscribe to the broadcast channel.
    let mut rx = state.subscribe();

    tracing::info!("WebSocket client connected");

    // Send a welcome message with current state.
    let welcome = {
        let s = state.sovereign.lock().await;
        serde_json::json!({
            "type": "Welcome",
            "data": {
                "tick":       s.tick,
                "generation": s.generation,
                "neurons":    s.brain.alive_neuron_count(),
                "synapses":   s.brain.alive_synapse_count(),
            }
        })
        .to_string()
    };

    if socket.send(Message::Text(welcome)).await.is_err() {
        return; // Client disconnected immediately.
    }

    // Event loop: relay broadcast events to this client until disconnect.
    loop {
        select! {
            // New broadcast event from the server.
            event = rx.recv() => {
                match event {
                    Ok(ev) => {
                        let json = match serde_json::to_string(&ev) {
                            Ok(j)  => j,
                            Err(e) => {
                                tracing::error!("WS serialize error: {}", e);
                                continue;
                            }
                        };
                        if socket.send(Message::Text(json)).await.is_err() {
                            break; // Client disconnected.
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("WebSocket client lagged by {} events", n);
                    }
                    Err(_) => break, // Sender dropped.
                }
            }

            // Client sent a message (ping/close).
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        let _ = socket.send(Message::Pong(data)).await;
                    }
                    _ => {} // Ignore other client messages.
                }
            }
        }
    }

    tracing::info!("WebSocket client disconnected");
}
