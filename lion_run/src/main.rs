// lion_run/src/main.rs — LionAI Phase 13: Maximum Enhancement Edition

use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

use lion_core::{
    BrainRng, Role, Sovereign, TernaryEncoder, TernaryEncoderConfig,
    target_speech_for_input, FEATURE_SIZE,
};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Encoder input size (128 rich n-gram features).
const ENCODER_INPUT: usize = 128;

/// Auto night-cycle every N ticks (0 = manual only).
const AUTO_SLEEP_TICKS: u64 = 20;

/// Conversation history to show (last N turns).
const HISTORY_LEN: usize = 5;

/// Temperature for live speech display (0.7 = mild sampling).
const DISPLAY_TEMP: f32 = 0.7;

// =============================================================================
// TEXT → FEATURE VECTOR  (Phase 13: Rich 128-dim encoding)
// =============================================================================

/// Converts text into a rich 128-dimensional feature vector.
///
/// Layout (all normalized to [-1, +1]):
///   [  0.. 63] — byte values of first 64 chars (zero-padded)
///   [ 64.. 79] — character-class histogram (letter, digit, space, punct, upper)
///               + word-count, sentence-count, avg-word-len, question/exclamation
///               + 4 character-n-gram hash buckets (bigrams, trigrams)
///   [ 80..127] — hash of the whole string projected into 48 random buckets
fn text_to_features(text: &str) -> Vec<f32> {
    let mut features = vec![0.0_f32; ENCODER_INPUT];
    let bytes = text.as_bytes();

    // ── Slot 0..63: raw byte values ──────────────────────────────────────────
    for (i, &b) in bytes.iter().take(64).enumerate() {
        features[i] = (b as f32) / 127.5 - 1.0;
    }

    // ── Slot 64..79: character statistics ────────────────────────────────────
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1);
    let letters  = chars.iter().filter(|c| c.is_alphabetic()).count();
    let digits   = chars.iter().filter(|c| c.is_numeric()).count();
    let spaces   = chars.iter().filter(|c| **c == ' ').count();
    let uppers   = chars.iter().filter(|c| c.is_uppercase()).count();
    let puncts   = chars.iter().filter(|c| c.is_ascii_punctuation()).count();
    let has_q    = text.contains('?') as u8 as f32;
    let has_ex   = text.contains('!') as u8 as f32;
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count  = words.len();
    let avg_word_len = if word_count > 0 {
        words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32 / 15.0
    } else { 0.0 };

    features[64] = letters  as f32 / n as f32;
    features[65] = digits   as f32 / n as f32;
    features[66] = spaces   as f32 / n as f32;
    features[67] = uppers   as f32 / n as f32;
    features[68] = puncts   as f32 / n as f32;
    features[69] = (word_count  as f32 / 20.0).min(1.0);
    features[70] = avg_word_len.min(1.0);
    features[71] = has_q;
    features[72] = has_ex;
    features[73] = (n as f32 / 80.0).min(1.0); // length normalised

    // ── Slot 74..79: bigram/trigram hash buckets ──────────────────────────────
    for window in chars.windows(2) {
        let h = (window[0] as u32 * 31 + window[1] as u32) as usize % 3;
        features[74 + h] = (features[74 + h] + 0.1).min(1.0);
    }
    for window in chars.windows(3) {
        let h = (window[0] as u32 * 961 + window[1] as u32 * 31 + window[2] as u32) as usize % 3;
        features[77 + h] = (features[77 + h] + 0.1).min(1.0);
    }

    // ── Slot 80..127: full-string hash projection ─────────────────────────────
    // FNV-1a hash of each 3-char window, spread into 48 buckets
    for window in chars.windows(3) {
        let mut h: u64 = 14695981039346656037;
        for c in window {
            h ^= *c as u64;
            h = h.wrapping_mul(1099511628211);
        }
        let bucket = (h % 48) as usize;
        features[80 + bucket] = (features[80 + bucket] + 0.1).min(1.0) * 2.0 - 1.0;
    }

    features
}

/// Detects the most relevant sensory role for a text input.
fn detect_role(text: &str) -> Role {
    let lower = text.to_lowercase();
    let danger  = ["danger","attack","threat","fear","flee","enemy","wolf","fire","hurt","kill","run","help"];
    let memory  = ["remember","recall","history","before","last time","forget","memory","learned","past"];
    let ds: usize = danger.iter().filter(|&&w| lower.contains(w)).count();
    let ms: usize = memory.iter().filter(|&&w| lower.contains(w)).count();
    if ds > 0 && ds >= ms { Role::Danger }
    else if ms > 0        { Role::Memory }
    else                  { Role::Vision }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

fn banner() {
    println!();
    println!("  ╔═════════════════════════════════════════════════════════════╗");
    println!("  ║    🦁  L I O N A I  ─  Phase 13  ·  Maximum Enhancement   ║");
    println!("  ║   1.58-bit Ternary Cortex · Hebbian Memory · Evolution     ║");
    println!("  ║   Parallel Night Cycle · Language Motor · Neural Immune    ║");
    println!("  ╚═════════════════════════════════════════════════════════════╝");
    println!();
}

fn help() {
    println!("  ┌───────────────────────────────────────────────────────────────┐");
    println!("  │  COMMANDS                                                    │");
    println!("  │                                                              │");
    println!("  │  Just type anything  →  brain thinks, speaks, and evolves   │");
    println!("  │                                                              │");
    println!("  │  REWARD (applied on your NEXT message):                      │");
    println!("  │    /good           reward +1.0  (reinforce current action)   │");
    println!("  │    /bad            reward -1.0  (punish current action)      │");
    println!("  │    /reward <N>     custom reward  (-5.0 … +5.0)              │");
    println!("  │    <number>        e.g. \"0.5\" or \"-1\" (shorthand)           │");
    println!("  │                                                              │");
    println!("  │  /learn <text>  teach target speech for next action          │");
    println!("  │  /sleep   manual night-cycle evolution                       │");
    println!("  │  /stats   brain diagnostics & language samples               │");
    println!("  │  /help    this menu                                          │");
    println!("  │  /quit    exit                                               │");
    println!("  └───────────────────────────────────────────────────────────────┘");
    println!();
}

/// Format an 8-element activation bar from gestalt values.
fn gestalt_bar(gestalt: &[f32; FEATURE_SIZE]) -> String {
    let step  = FEATURE_SIZE / 8;
    let chars = ["▁","▂","▃","▄","▅","▆","▇","█"];
    (0..8).map(|i| {
        let slice = &gestalt[i * step..(i + 1) * step];
        let avg   = slice.iter().sum::<f32>() / step as f32;
        let idx   = ((avg + 1.0) / 2.0 * 7.0).clamp(0.0, 7.0) as usize;
        chars[idx]
    }).collect()
}

fn role_icon(role: Role) -> &'static str {
    match role {
        Role::Vision => "👁  Vision",
        Role::Danger => "⚠️  Danger",
        Role::Memory => "🧠 Memory",
        _            => "📡 Sensor",
    }
}

fn action_icon(action: &str) -> &'static str {
    match action {
        "FLEE"   => "💨 FLEE",
        "ATTACK" => "⚔️  ATTACK",
        "FORAGE" => "🌿 FORAGE",
        "HIDE"   => "🫥 HIDE",
        _        => "🚶 WANDER",
    }
}

/// Speech quality score (simple: what fraction of chars are letters/spaces).
fn speech_quality(speech: &str) -> f32 {
    if speech.is_empty() { return 0.0; }
    let good = speech.chars().filter(|c| c.is_alphabetic() || *c == ' ').count();
    good as f32 / speech.len() as f32
}

/// Quality bar: 0..5 stars based on speech quality.
fn quality_bar(q: f32) -> &'static str {
    match (q * 5.0) as u32 {
        0          => "·····",
        1          => "★····",
        2          => "★★···",
        3          => "★★★··",
        4          => "★★★★·",
        _          => "★★★★★",
    }
}

fn print_tick(
    tick:    u64,
    gen:     u32,
    stress:  f32,
    input:   &str,
    role:    Role,
    action:  &str,
    speech:  &str,
    gestalt: &[f32; FEATURE_SIZE],
    target:  &str,
) {
    let bar      = gestalt_bar(gestalt);
    let q        = speech_quality(speech);
    let qbar     = quality_bar(q);
    let inp_disp = if input.len() > 42 { format!("{}…", &input[..41]) } else { input.to_string() };

    println!();
    println!("  ┌─ Tick {:>5}  Gen {:>3}  Stress {:.2} ───────────────────────────┐", tick, gen, stress);
    println!("  │  Input   : {:<48}│", inp_disp);
    println!("  │  Modality: {:<48}│", role_icon(role));
    println!("  │  Action  : {:<48}│", action_icon(action));
    println!("  │  Speech  : {:<48}│", speech);
    println!("  │  Quality : {}  ({:.0}%)                              │", qbar, q * 100.0);
    println!("  │  Target  : {:<48}│", target);
    println!("  │  Gestalt : {:<48}│", bar);
    println!("  └──────────────────────────────────────────────────────────────┘");
}

fn print_sleep_report(sov_fit: f64, child_fit: f64, evolved: bool, mut_rate: f32, count: usize, gen: u32) {
    println!();
    println!("  ╔═══ 🌙 Night Cycle Complete ══════════════════════════════════╗");
    println!("  ║  Sovereign fitness  :  {:>+10.4}                          ║", sov_fit);
    println!("  ║  Best child fitness :  {:>+10.4}                          ║", child_fit);
    println!("  ║  Children evaluated :  {:<4}                               ║", count);
    println!("  ║  Mutation rate      :  {:.4}                              ║", mut_rate);
    if evolved {
        println!("  ║  Result : ✅ EVOLVED  → Generation {}                      ║", gen);
    } else {
        println!("  ║  Result : ─  No improvement found                         ║");
    }
    println!("  ╚══════════════════════════════════════════════════════════════╝");
}

fn print_stats(sovereign: &Sovereign, encoder: &TernaryEncoder) {
    let nc  = sovereign.brain.alive_neuron_count();
    let sc  = sovereign.brain.alive_synapse_count();
    let ec  = sovereign.episode_count();
    let enc = encoder.total_weight_bytes();
    println!();
    println!("  ╔═══ 🔬 Brain Diagnostics ══════════════════════════════════╗");
    println!("  ║  Tick          : {}", sovereign.tick);
    println!("  ║  Generation    : {}", sovereign.generation);
    println!("  ║  Stress        : {:.3}", sovereign.stress());
    println!("  ║  Neurons       : {} alive", nc);
    println!("  ║  Synapses      : {} alive", sc);
    println!("  ║  Episodes      : {}", ec);
    println!("  ║  Encoder size  : {} bytes", enc);
    println!("  ║  Plasticity    : {:.3}", sovereign.brain.epigenome.plasticity);
    println!("  ║  Exploration   : {:.3}", sovereign.brain.epigenome.exploration_drive);
    println!("  ╠═══ 🗣  Language Motor Samples ══════════════════════════════╣");
    let gestalts = [
        (&[0.5f32; FEATURE_SIZE], "FORAGE-gestalt"),
        (&[-0.5f32; FEATURE_SIZE], "FLEE-gestalt"),
        (&[0.0f32; FEATURE_SIZE], "Neutral-gestalt"),
    ];
    for (g, label) in &gestalts {
        let speech = sovereign.language_motor.generate(g, 20);
        let q      = speech_quality(&speech);
        println!("  ║  [{}] {:>15} → {:?}", quality_bar(q), label, speech);
    }
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();
}

// =============================================================================
// MAIN LOOP
// =============================================================================

fn main() {
    banner();
    help();

    print!("  🔧 Initialising brain (seed=42) … ");
    io::stdout().flush().unwrap();

    let mut rng       = BrainRng::from_seed(42);
    let mut sovereign = Sovereign::new(42);

    let encoder_cfg = TernaryEncoderConfig {
        input_size:   ENCODER_INPUT,
        hidden_sizes: vec![96, 64],     // Deeper: 128→96→64→32
        output_size:  FEATURE_SIZE,
    };
    let encoder = TernaryEncoder::random(encoder_cfg, &mut rng);

    println!("done.");
    println!("  🦁 LionAI is alive. Type anything to begin.\n");

    // ── State ─────────────────────────────────────────────────────────────────
    let mut queued_reward: f32 = 0.0;
    let mut awaiting_feedback  = false;
    let mut history: VecDeque<(String, String, String)> = VecDeque::new();
    let mut best_speech_quality = 0.0f32;
    let mut learned_target: Option<String> = None; // /learn override
    let mut display_rng = BrainRng::from_seed(777);

    let stdin  = io::stdin();
    let stdout = io::stdout();

    loop {
        // ── Prompt ───────────────────────────────────────────────────────────
        {
            let mut out = stdout.lock();
            write!(out, "  [you] ▶ ").unwrap();
            out.flush().unwrap();
        }

        // ── Read input ───────────────────────────────────────────────────────
        let mut raw = String::new();
        if stdin.lock().read_line(&mut raw).is_err() { break; }
        let input = raw.trim().to_string();
        if input.is_empty() {
            // Empty input: if awaiting feedback, just do no-op
            if awaiting_feedback { continue; }
            // Otherwise, repeat last input (or ignore)
            continue;
        }

        // ── Command dispatch ─────────────────────────────────────────────────
        match input.as_str() {
            "/quit" => {
                println!("  💾 Saving … (snapshot persistence not configured in lion_run)");
                println!("  🦁 Goodbye. Brain reached generation {}.", sovereign.generation);
                break;
            }

            "/good" => {
                queued_reward     = 1.0;
                awaiting_feedback = false;
                println!("  ✅ Reward +1.0 set — will apply on your next message.");
                continue;
            }

            "/bad" => {
                queued_reward     = -1.0;
                awaiting_feedback = false;
                println!("  ❌ Reward -1.0 set — will apply on your next message.");
                continue;
            }

            "/sleep" => {
                println!("  💤 Triggering night cycle …");
                let report = sovereign.trigger_sleep_cycle(queued_reward);
                queued_reward     = 0.0;
                awaiting_feedback = false;
                print_sleep_report(
                    report.sovereign_fitness,
                    report.best_child_fitness,
                    report.evolution_occurred,
                    report.mutation_rate,
                    report.children_evaluated,
                    sovereign.generation,
                );
                continue;
            }

            "/stats" => {
                print_stats(&sovereign, &encoder);
                if !history.is_empty() {
                    println!("  📜 Recent conversation:");
                    for (inp, act, sp) in history.iter() {
                        println!("     you: {:>24}  →  [{}]  \"{}\"", inp, act, sp);
                    }
                    println!();
                }
                continue;
            }

            "/help" => { help(); continue; }

            _ if input.starts_with("/reward ") => {
                let val_str = input.trim_start_matches("/reward ").trim();
                match val_str.parse::<f32>() {
                    Ok(v) => {
                        let v = v.clamp(-5.0, 5.0);
                        queued_reward     = v;
                        awaiting_feedback = false;
                        println!("  🎯 Reward {:.2} set — will apply on your next message.", v);
                    }
                    Err(_) => println!("  ⚠  Usage: /reward <number>   e.g. /reward 0.5"),
                }
                continue;
            }

            _ if input.starts_with("/learn ") => {
                let target = input.trim_start_matches("/learn ").trim().to_string();
                if target.is_empty() {
                    println!("  ⚠  Usage: /learn <text>   e.g. /learn hello friend");
                } else {
                    println!("  📚 Learned target: \"{}\"", target);
                    println!("     This will be the speech goal for the next night cycle.");
                    learned_target = Some(target);
                }
                continue;
            }

            _ if awaiting_feedback => {
                if let Ok(v) = input.parse::<f32>() {
                    let v = v.clamp(-5.0, 5.0);
                    queued_reward     = v;
                    awaiting_feedback = false;
                    println!("  🎯 Reward {:.2} set — will apply on your next message.", v);
                    continue;
                }
                // Not a number → fall through to brain tick
            }

            _ => {}
        }

        // ── Encode text → brain tick ────────────────────────────────────────────────────
        let features = text_to_features(&input);
        let role     = detect_role(&input);
        let frame    = encoder.encode_frame(&[(role, &features)]);

        let result = sovereign.update(&frame, queued_reward);

        // Use temperature sampling for a richer display output
        let speech = sovereign.language_motor.generate_with_temp(
            &result.gestalt, 24, DISPLAY_TEMP, &mut display_rng,
        );

        // Choose target: /learn override, then input-aware, then action-aware
        let target_speech = learned_target.as_deref()
            .unwrap_or_else(|| target_speech_for_input(&input));

        let applied_reward = queued_reward;
        queued_reward      = 0.0;
        awaiting_feedback  = true;

        let q = speech_quality(&speech);
        if q > best_speech_quality + 0.05 {
            best_speech_quality = q;
            println!();
            println!("  🎉 Speech quality milestone! Now {:.0}% recognisable!", q * 100.0);
        }

        // ── Display ───────────────────────────────────────────────────────────
        print_tick(
            result.tick,
            sovereign.generation,
            sovereign.stress(),
            &input,
            role,
            result.action,
            &speech,
            &result.gestalt,
            target_speech,
        );

        if applied_reward != 0.0 {
            println!("  ↳ reward {:.2} applied this tick", applied_reward);
        }
        println!("  ↳ /good · /bad · /reward N · /sleep · /stats · or keep typing");

        // Update conversation history
        history.push_back((
            if input.len() > 20 { format!("{}…", &input[..19]) } else { input.clone() },
            result.action.to_string(),
            if speech.len() > 20 { format!("{}…", &speech[..19]) } else { speech.clone() },
        ));
        if history.len() > HISTORY_LEN { history.pop_front(); }

        // ── Auto night cycle ─────────────────────────────────────────────────
        if AUTO_SLEEP_TICKS > 0
            && result.tick > 0
            && result.tick % AUTO_SLEEP_TICKS == 0
        {
            println!("\n  💤 Auto night cycle at tick {} …", result.tick);
            let report = sovereign.trigger_sleep_cycle(0.0);
            awaiting_feedback = false;
            print_sleep_report(
                report.sovereign_fitness,
                report.best_child_fitness,
                report.evolution_occurred,
                report.mutation_rate,
                report.children_evaluated,
                sovereign.generation,
            );
        }
    }
}
