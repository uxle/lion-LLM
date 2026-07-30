// lion_cli/src/main.rs — LionAI v1.0 — Unified CLI
//
// Usage: just type anything.
// Slash commands:
//   /help               — show all commands
//   /status             — system status
//   /memory             — memory stats
//   /ir                 — demonstrate Footprint Canonical IR AST
//   /contract           — run Design-by-Contract semantic verification
//   /envelope           — test Footprint Determinism Envelope matching
//   /mars               — solve Mars Colony Autonomous Recovery scenario
//   /risk <text>        — prompt injection & risk analysis
//   /ledger             — view BLAKE3 cryptographic audit ledger
//   /cache              — view O(1) semantic cache stats
//   /clear              — clear conversation context
//   /tools              — list available tools
//   /use <tool> <input> — call a tool directly
//   /calc <expr>        — fast calculator (no Ollama needed)
//   /image <path>       — encode and describe an image
//   /audio <path>       — analyze a WAV file
//   /agent <task>       — run the ReAct agent for a multi-step task
//   /save               — save memory to disk now
//   /exit               — quit

use std::io::{self, Write};
use std::path::PathBuf;

use crossterm::style::Color;

use lion_agent::{Agent, AgentConfig};
use lion_brain::{LionSystem, SystemConfig, RiskAssessor, MarsColonyStatus, MarsRecoverySolver, SensorReading};
use lion_core::{CanonicalIR, IRNode, Opcode, TypedPrimitive, DeterminismEnvelope, SemanticAnalyzer};
use lion_senses::{AudioEncoder, ImageEncoder, VisionLLM};

// =============================================================================
// ENTRY POINT
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging — only show WARN and above unless RUST_LOG is set.
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "warn".to_string()),
        )
        .with_target(false)
        .compact()
        .init();

    // Data directory.
    let data_dir = dirs_home().join(".lionai");
    std::fs::create_dir_all(&data_dir)?;

    let memory_path = data_dir.join("memory.bin").to_string_lossy().into_owned();

    // Config.
    let config = SystemConfig {
        ollama_base:  "http://localhost:11434".to_string(),
        model:        "gemma3:1b".to_string(),
        memory_path:  memory_path.clone(),
        streaming:    true,
        show_routing: false,
        show_tokens:  false,
        enable_agent: true,
        ..Default::default()
    };

    // Print banner.
    print_banner();

    // Boot LionSystem.
    eprint!("  Initializing LionAI system...");
    let mut system = LionSystem::new(config.clone()).await?;
    eprintln!(" done");

    // Boot Agent.
    let agent = Agent::new(AgentConfig {
        ollama_base: config.ollama_base.clone(),
        model:       config.model.clone(),
        ..Default::default()
    });

    // Check Ollama.
    let ollama_live = {
        let llm = lion_brain::llm::OllamaClient::new(&config.ollama_base, &config.model);
        llm.is_available().await
    };

    if ollama_live {
        print_color("  ✓ Ollama is running", Color::Green);
        let llm  = lion_brain::llm::OllamaClient::new(&config.ollama_base, &config.model);
        let models = llm.available_models().await;
        if !models.is_empty() {
            eprintln!("    Models: {}", models.join(", "));
        }
    } else {
        print_color(
            "  ⚠ Ollama not running — LLM features disabled\n  \
             To enable: ollama serve && ollama pull gemma3:1b",
            Color::Yellow,
        );
    }

    let memory_count = system.memory_entry_count().await;
    if memory_count > 0 {
        print_color(&format!("  ● Memory: {} entries loaded", memory_count), Color::Cyan);
    }

    eprintln!();
    print_color("  Type anything to chat. /help for commands.", Color::DarkGrey);
    eprintln!();

    // Main REPL.
    let stdin = io::stdin();
    loop {
        // Prompt.
        print_prefix("🦁 › ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() || line.is_empty() {
            break;
        }

        let input = line.trim();
        if input.is_empty() { continue; }

        // Slash commands.
        if input.starts_with('/') {
            let handled = handle_command(input, &mut system, &agent, &config).await;
            if !handled { break; }
            continue;
        }

        // Regular chat → LionSystem.
        eprintln!();
        print_color("🤖 LionAI:", Color::Magenta);

        // Streaming: print tokens as they arrive.
        let is_streaming = config.streaming && ollama_live;
        if is_streaming {
            print!("   ");
            io::stdout().flush().ok();
        }

        let print_tok = |tok: &str| {
            print!("{}", tok);
            io::stdout().flush().ok();
        };
        let cb: Option<&dyn Fn(&str)> = if is_streaming { Some(&print_tok) } else { None };
        let result = system.process(input, cb).await;

        if result.cache_hit {
            print_color("   ⚡ (Served from O(1) Semantic Cache)", Color::Cyan);
        }

        if matches!(result.risk_assessment.level, lion_brain::RiskLevel::High | lion_brain::RiskLevel::Critical) {
            print_color(&format!("   ⚠ Risk Warning: {:?}", result.risk_assessment.level), Color::Red);
        }

        if is_streaming { eprintln!(); }
        else { println!("   {}", result.answer); }

        eprintln!();
    }

    // Save memory on exit.
    system.save_memory().await;
    eprintln!("  Memory saved. Goodbye! 🦁");
    Ok(())
}

// =============================================================================
// COMMAND HANDLER
// =============================================================================

/// Returns `false` to quit.
async fn handle_command(
    input:  &str,
    system: &mut LionSystem,
    agent:  &Agent,
    config: &SystemConfig,
) -> bool {
    let parts: Vec<&str> = input.splitn(3, ' ').collect();
    let cmd = parts[0];

    match cmd {
        // ── /help ─────────────────────────────────────────────────────────────
        "/help" => {
            println!("{}", HELP_TEXT);
        }

        // ── /status ───────────────────────────────────────────────────────────
        "/status" => {
            let llm   = lion_brain::llm::OllamaClient::new(&config.ollama_base, &config.model);
            let alive = llm.is_available().await;
            let mem   = system.memory_entry_count().await;
            let turn  = system.turn_number();
            println!();
            println!("  ╔═══════════════════ LionAI v1.0 ════════════════════╗");
            println!("  ║  Ollama  : {:<41}║", if alive { "✓ running" } else { "✗ offline" });
            println!("  ║  Model   : {:<41}║", config.model);
            println!("  ║  Memory  : {:<41}║", format!("{} entries", mem));
            println!("  ║  Context : turn {:<37}║", turn);
            println!("  ║  Ledger  : {:<41}║", format!("{} audit blocks", system.ledger.len()));
            println!("  ║  Cache   : {:<41}║", format!("{} entries", system.semantic_cache.len()));
            println!("  ║  Tools   : {:<41}║", agent.tool_names().join(", "));
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /ir ───────────────────────────────────────────────────────────────
        "/ir" => {
            let mut ir = CanonicalIR::new();
            ir.add_node(IRNode {
                id: "n1".to_string(),
                opcode: Opcode::MathMultiply,
                inputs: vec![TypedPrimitive::Integer(42), TypedPrimitive::Float(3.14159)],
                expected_type: "Float".to_string(),
                depends_on: vec!["n0".to_string()],
            });
            ir.canonicalize();

            println!();
            println!("  ╔════════════════ Footprint Canonical IR ══════════════╗");
            println!("  ║  Nodes     : {:<40}║", ir.nodes.len());
            println!("  ║  Opcode    : {:<40}║", format!("{:?}", ir.nodes[0].opcode));
            println!("  ║  Type 0    : {:<40}║", ir.nodes[0].inputs[0].type_name());
            println!("  ║  Type 1    : {:<40}║", ir.nodes[0].inputs[1].type_name());
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /contract ─────────────────────────────────────────────────────────
        "/contract" => {
            let valid_node = IRNode {
                id: "mat_mult_1".to_string(),
                opcode: Opcode::MatrixMultiply,
                inputs: vec![
                    TypedPrimitive::Matrix { rows: 2, cols: 3, data: vec![1.0; 6] },
                    TypedPrimitive::Matrix { rows: 3, cols: 4, data: vec![1.0; 12] },
                ],
                expected_type: "Matrix".to_string(),
                depends_on: vec![],
            };

            let res = SemanticAnalyzer::verify_node(&valid_node);
            println!();
            println!("  ╔══════════════ Design-by-Contract Check ═════════════╗");
            println!("  ║  Opcode    : {:<40}║", format!("{:?}", valid_node.opcode));
            println!("  ║  Dimensions: {:<40}║", "2x3 * 3x4 -> Valid Contract");
            println!("  ║  Result    : {:<40}║", if res.is_ok() { "PASSED" } else { "FAILED" });
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /envelope ─────────────────────────────────────────────────────────
        "/envelope" => {
            let env = DeterminismEnvelope::NumericTolerance(1e-4);
            let val1 = TypedPrimitive::Float(42.00001);
            let val2 = TypedPrimitive::Float(42.00002);
            let matched = env.validate_match(&val1, &val2);

            println!();
            println!("  ╔══════════════ Determinism Envelope ════════════════╗");
            println!("  ║  Envelope  : {:<40}║", "NumericTolerance(0.0001)");
            println!("  ║  Val 1     : {:<40}║", "42.00001");
            println!("  ║  Val 2     : {:<40}║", "42.00002");
            println!("  ║  Match     : {:<40}║", if matched { "EXACT MATCH (Within Epsilon)" } else { "MISMATCH" });
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /mars ─────────────────────────────────────────────────────────────
        "/mars" => {
            let status = MarsColonyStatus::default();
            let mut raw_sensors = vec![
                SensorReading { sensor_id: "s1".to_string(), subsystem: "power".to_string(), value: 0.60 },
                SensorReading { sensor_id: "s2".to_string(), subsystem: "water".to_string(), value: 0.75 },
                SensorReading { sensor_id: "s3".to_string(), subsystem: "comm".to_string(), value: 0.82 },
            ];

            // Add 10% Byzantine corrupt sensor readings
            raw_sensors.push(SensorReading { sensor_id: "s_adv".to_string(), subsystem: "power".to_string(), value: 999.0 });

            let plan = MarsRecoverySolver::solve(status, &raw_sensors);

            println!();
            println!("  ╔════════════ Mars Colony Autonomous Recovery ═══════════╗");
            println!("  ║  Population: {:<40}║", format!("{} inhabitants", plan.status.population));
            println!("  ║  Storm Delay: {:<40}║", format!("{} hours (Duration {}d)", plan.status.hours_to_storm, plan.status.storm_duration_days));
            println!("  ║  Byzantine : {:<40}║", format!("{} rejected (10% Tol)", plan.rejected_sensors));
            println!("  ║  Casualties: {:<40}║", format!("{} (Verified)", plan.expected_casualties));
            println!("  ║  Proof Hash: {:<40}║", &plan.verification_proof_hash[..32]);
            println!("  ╚════════════════════════════════════════════════════╝");
            println!("  Recovery Schedule:");
            for step in &plan.repair_sequence {
                println!("    {}", step);
            }
            println!();
        }

        // ── /memory ───────────────────────────────────────────────────────────
        "/memory" => {
            let count = system.memory_entry_count().await;
            println!("  Memory entries: {}", count);
        }

        // ── /clear ────────────────────────────────────────────────────────────
        "/clear" => {
            system.clear_context();
            println!("  Context cleared.");
        }

        // ── /risk <text> ──────────────────────────────────────────────────────
        "/risk" => {
            if parts.len() < 2 {
                println!("  Usage: /risk <input_text>");
                return true;
            }
            let text = &input["/risk".len()..].trim();
            let assessment = RiskAssessor::assess(text);
            let allow_mem = RiskAssessor::allow_memory_extraction(assessment.level);
            println!();
            println!("  ╔══════════════════ Risk Assessment ═════════════════╗");
            println!("  ║  Risk Level: {:<40}║", format!("{:?}", assessment.level));
            println!("  ║  Injection : {:<40}║", if assessment.prompt_injection_detected { "DETECTED" } else { "Clean" });
            println!("  ║  Sensitive : {:<40}║", if assessment.sensitive_domain_detected { "DETECTED" } else { "Clean" });
            println!("  ║  PII Signal: {:<40}║", if assessment.pii_detected { "DETECTED" } else { "Clean" });
            println!("  ║  Allow Mem : {:<40}║", if allow_mem { "Yes (Allowed)" } else { "No (Gated)" });
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /ledger ───────────────────────────────────────────────────────────
        "/ledger" => {
            let valid = system.ledger.verify_chain();
            println!();
            println!("  ╔═══════════════ Cryptographic Ledger ═══════════════╗");
            println!("  ║  Blocks    : {:<40}║", system.ledger.len());
            println!("  ║  Integrity : {:<40}║", if valid { "VALID (Unbroken Chain)" } else { "INVALID (Tampered)" });
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /cache ────────────────────────────────────────────────────────────
        "/cache" => {
            println!();
            println!("  ╔══════════════════ Semantic Cache ══════════════════╗");
            println!("  ║  Entries   : {:<40}║", system.semantic_cache.len());
            println!("  ║  Threshold : {:<40}║", format!("{:.2}", system.semantic_cache.threshold));
            println!("  ╚════════════════════════════════════════════════════╝");
            println!();
        }

        // ── /tools ────────────────────────────────────────────────────────────
        "/tools" => {
            println!("  Available tools:");
            for name in agent.tool_names() {
                println!("    • {}", name);
            }
        }

        // ── /use <tool> <input> ───────────────────────────────────────────────
        "/use" => {
            if parts.len() < 3 {
                println!("  Usage: /use <tool_name> <input>");
                return true;
            }
            let tool  = parts[1];
            let tool_input = parts[2];
            println!("  Running tool '{}'...", tool);
            let out = agent.use_tool_directly(tool, tool_input).await;
            println!("  → {}", out);
        }

        // ── /calc <expr> ──────────────────────────────────────────────────────
        "/calc" => {
            if parts.len() < 2 {
                println!("  Usage: /calc <expression>");
                return true;
            }
            let expr_parts = input["calc".len() + 1..].trim();
            let out = agent.use_tool_directly("calculator", expr_parts).await;
            println!("  {}", out);
        }

        // ── /image <path> ─────────────────────────────────────────────────────
        "/image" => {
            if parts.len() < 2 {
                println!("  Usage: /image <path>");
                return true;
            }
            let path = std::path::Path::new(parts[1]);
            let enc  = ImageEncoder::default();

            match enc.encode_file(path) {
                Ok(feat) => {
                    println!();
                    println!("  ╔═══════════════════ Image Analysis ═════════════════╗");
                    println!("  ║  File     : {:<40}║", path.file_name().unwrap_or_default().to_string_lossy());
                    println!("  ║  Size     : {}×{}{:<35}║", feat.original_width, feat.original_height, "");
                    println!("  ║  Mode     : {:<40}║", feat.color_mode);
                    println!("  ║  Edge     : {:<40}║", format!("{:.3} ({})", feat.edge_energy, if feat.is_high_contrast { "high contrast" } else { "low contrast" }));
                    println!("  ║  Features : {} dimensions{:<26}║", feat.features.len(), "");
                    println!("  ╚════════════════════════════════════════════════════╝");

                    // Try vision LLM if available.
                    let vision = VisionLLM::moondream(&config.ollama_base);
                    if vision.is_available().await {
                        println!("  Sending to vision model...");
                        match vision.analyze(path).await {
                            Ok(desc) => {
                                println!();
                                print_color("🤖 Vision:", Color::Magenta);
                                println!("   {}", desc);
                            }
                            Err(e) => println!("  Vision error: {}", e),
                        }
                    } else {
                        println!("  (Vision LLM unavailable — pixel features only)");
                    }
                    println!();
                }
                Err(e) => println!("  Error: {}", e),
            }
        }

        // ── /audio <path> ─────────────────────────────────────────────────────
        "/audio" => {
            if parts.len() < 2 {
                println!("  Usage: /audio <path.wav>");
                return true;
            }
            let path = std::path::Path::new(parts[1]);
            let enc  = AudioEncoder::default();

            match enc.encode_file(path) {
                Ok(feat) => {
                    println!();
                    println!("  ╔═══════════════════ Audio Analysis ═════════════════╗");
                    println!("  ║  File     : {:<40}║", path.file_name().unwrap_or_default().to_string_lossy());
                    println!("  ║  Rate     : {}Hz{:<38}║", feat.sample_rate, "");
                    println!("  ║  Duration : {:<40}║", format!("{:.2}s", feat.duration_secs));
                    println!("  ║  Samples  : {:<40}║", feat.total_samples);
                    println!("  ║  RMS      : {:<40}║", format!("{:.4} ({})", feat.rms_energy, if feat.is_loud { "LOUD" } else { "quiet" }));
                    println!("  ║  ZCR      : {:<40}║", format!("{:.4}", feat.zero_cross_rate));
                    println!("  ║  Features : {} dimensions{:<26}║", feat.features.len(), "");
                    println!("  ╚════════════════════════════════════════════════════╝");
                    println!();
                }
                Err(e) => println!("  Error: {}", e),
            }
        }

        // ── /agent <task> ─────────────────────────────────────────────────────
        "/agent" => {
            if parts.len() < 2 {
                println!("  Usage: /agent <task description>");
                return true;
            }
            let task_start = cmd.len() + 1;
            let task = if input.len() > task_start { &input[task_start..] } else { "" };

            if task.is_empty() {
                println!("  Usage: /agent <task description>");
                return true;
            }

            if !agent.is_available().await {
                println!("  Ollama is not running — agent requires Ollama for reasoning.");
                println!("  You can still use tools directly with /use <tool> <input>");
                return true;
            }

            println!();
            print_color("🤖 Agent starting...", Color::Cyan);
            println!();

            let result = agent.run_task(task).await;

            if !result.steps.is_empty() {
                println!("  ── ReAct Trace ──────────────────────────────────────");
                for (i, step) in result.steps.iter().enumerate() {
                    println!("  Step {}:", i + 1);
                    if !step.thought.is_empty()      { println!("    Thought: {}", step.thought); }
                    if !step.action.is_empty()        { println!("    Action:  {}", step.action); }
                    if !step.action_input.is_empty()  { println!("    Input:   {}", step.action_input); }
                    println!("    Obs:     {}", &step.observation[..step.observation.len().min(200)]);
                }
                println!("  ─────────────────────────────────────────────────────");
            }

            println!();
            print_color("🤖 Answer:", Color::Magenta);
            println!("   {}", result.answer);
            println!();
        }

        // ── /save ─────────────────────────────────────────────────────────────
        "/save" => {
            system.save_memory().await;
            println!("  Memory saved.");
        }

        // ── /exit / /quit ─────────────────────────────────────────────────────
        "/exit" | "/quit" | "/q" => {
            return false;
        }

        _ => {
            println!("  Unknown command '{}'. Type /help for help.", cmd);
        }
    }

    true
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

fn print_banner() {
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════════════════════╗");
    eprintln!("  ║                                                           ║");
    eprintln!("  ║   🦁  L I O N A I   ─   Version 1.0                    ║");
    eprintln!("  ║                                                           ║");
    eprintln!("  ║   Ternary Cortex  ·  Semantic Memory  ·  ReAct Agent    ║");
    eprintln!("  ║   Multimodal Senses  ·  Streaming  ·  Knowledge Graph   ║");
    eprintln!("  ║                                                           ║");
    eprintln!("  ╚═══════════════════════════════════════════════════════════╝");
    eprintln!();
}

fn print_prefix(s: &str) {
    print!("\x1b[1m{}\x1b[0m", s);
}


fn print_color(s: &str, color: Color) {
    let code = match color {
        Color::Red      => "31",
        Color::Green    => "32",
        Color::Yellow   => "33",
        Color::Blue     => "34",
        Color::Magenta  => "35",
        Color::Cyan     => "36",
        Color::White    => "37",
        Color::DarkGrey => "90",
        _               => "37",
    };
    println!("\x1b[{}m{}\x1b[0m", code, s);
}


const HELP_TEXT: &str = "
  ╔═══════════════════ LionAI v1.0 Commands ═══════════════════╗
  ║                                                            ║
  ║  <anything>          Chat with LionAI                      ║
  ║                                                            ║
  ║  /help               Show this help                        ║
  ║  /status             System + Ollama status                ║
  ║  /memory             Show memory entry count               ║
  ║  /ir                 Show Footprint Canonical IR AST       ║
  ║  /contract           Run Design-by-Contract semantic check ║
  ║  /envelope           Test Determinism Envelope matching    ║
  ║  /mars               Solve Mars Colony Autonomous Recovery ║
  ║  /risk <text>        Run prompt injection & risk analysis  ║
  ║  /ledger             Check cryptographic audit ledger      ║
  ║  /cache              Show O(1) semantic cache stats        ║
  ║  /clear              Clear conversation context            ║
  ║  /save               Save memory to disk now               ║
  ║                                                            ║
  ║  /tools              List available tools                   ║
  ║  /use <tool> <input> Run a tool directly                   ║
  ║  /calc <expr>        Calculator (no Ollama needed)         ║
  ║                                                            ║
  ║  /image <path>       Encode + describe an image file       ║
  ║  /audio <path>       Analyze a WAV audio file              ║
  ║                                                            ║
  ║  /agent <task>       Run ReAct agent for multi-step tasks  ║
  ║                                                            ║
  ║  /exit               Quit (memory saved automatically)     ║
  ║                                                            ║
  ╚════════════════════════════════════════════════════════════╝
";

// =============================================================================
// HELPERS
// =============================================================================

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
