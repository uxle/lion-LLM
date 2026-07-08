// lion_server/src/main.rs

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

mod api;
mod config;
mod state;
mod ws;

use config::Config;
use state::AppState;

// =============================================================================
// CLI
// =============================================================================

/// LionAI cognitive agent server.
#[derive(Parser, Debug)]
#[command(name = "lion_server", version, about)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(short, long, default_value = "lion_config.toml")]
    config: String,

    /// Override the bind port.
    #[arg(short, long)]
    port: Option<u16>,

    /// Override the log level (trace, debug, info, warn, error).
    #[arg(short, long)]
    log_level: Option<String>,

    /// Path to a snapshot to load on startup (overrides config).
    #[arg(long)]
    load: Option<String>,
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // ── Load config ───────────────────────────────────────────────────────────
    let mut config = Config::load(&cli.config);

    if let Some(port) = cli.port {
        config.server.port = port;
    }
    if let Some(level) = cli.log_level {
        config.server.log_level = level;
    }
    if let Some(path) = cli.load {
        config.agent.snapshot_path = path.into();
    }

    // ── Initialize tracing ────────────────────────────────────────────────────
    let filter = EnvFilter::try_new(&config.server.log_level)
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .init();

    tracing::info!("LionAI server v{}", env!("CARGO_PKG_VERSION"));

    // ── Build shared state ────────────────────────────────────────────────────
    let addr = config.bind_addr();
    let state = AppState::new(config);

    // ── Build router ──────────────────────────────────────────────────────────
    let app = Router::new()
        // REST API
        .route("/api/tick",    post(api::tick))
        .route("/api/sleep",   post(api::sleep))
        .route("/api/state",   get(api::agent_state))
        .route("/api/neurons", get(api::neurons))
        .route("/api/save",    post(api::save))
        .route("/api/load",    post(api::load))
        .route("/api/health",  get(api::health))
        // WebSocket telemetry
        .route("/ws",          get(ws::ws_handler))
        // Middleware
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
        // Serve static files from "public" fallback
        .fallback_service(
            tower_http::services::ServeDir::new("lion_server/public")
                .fallback(tower_http::services::ServeDir::new("public"))
        );

    // ── Start server ──────────────────────────────────────────────────────────
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind address");

    tracing::info!("Listening on http://{}", addr);
    tracing::info!("WebSocket telemetry at ws://{}/ws", addr);

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
