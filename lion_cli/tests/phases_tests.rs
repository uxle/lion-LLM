// lion_cli/tests/phases_tests.rs — Integrated test suite for Phases 1 to 16
//
// Run with: cargo test -p lion_cli

use rand::Rng;

use lion_core::{
    cosine_sim, f32_to_i8, i8_to_f32, seeded_rng, Activation as CoreActivation, Role,
    SensoryInput, TernaryEncoder, TernaryEncoderConfig, TernaryLayer,
};
use lion_core::knowledge::KnowledgeGraph;
use lion_core::longmem::LongTermMemory;
use lion_core::evaluation::{ResponseEvaluator, ResponseScore};

use lion_brain::{ContextConfig, ContextManager};
use lion_brain::context::estimate_tokens;
use lion_senses::{AudioEncoder, ImageEncoder};
use lion_agent::{Agent, AgentConfig};

// =============================================================================
// PHASES 1 - 3 & 7 & 10: COGNITIVE GRAPH & PERSISTENCE
// =============================================================================

#[test]
fn test_phase_1_sensory_input() {
    let mut input = SensoryInput::new();
    let features = [0.5_f32; 32];
    input.insert(Role::Vision, features);
    assert_eq!(input.len(), 1);
    assert_eq!(input.get(Role::Vision), Some(&features));
    assert!(input.get(Role::Danger).is_none());
}

#[test]
fn test_phase_2_seeded_rng() {
    let mut rng1 = seeded_rng(42);
    let mut rng2 = seeded_rng(42);
    let r1: f32 = rng1.gen();
    let r2: f32 = rng2.gen();
    assert_eq!(r1, r2);
}

#[test]
fn test_phase_3_longmem_facts() {
    let mut mem = LongTermMemory::default();
    mem.store_fact("test_topic", "test_content", 1);
    let hits = mem.search_facts("test_topic");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content, "test_content");
}

#[test]
fn test_phase_6_persistence() {
    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join("test_memory.json");

    let mut mem = LongTermMemory::default();
    mem.store_fact("rust", "rust is memory safe", 1);
    mem.save(&test_path).expect("Failed to save memory");

    let loaded = LongTermMemory::load(&test_path);
    let mut loaded_mut = loaded.clone();
    let hits = loaded_mut.search_facts("rust");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content, "rust is memory safe");

    let _ = std::fs::remove_file(test_path);
}

#[test]
fn test_phase_7_immune_evaluation() {
    let score1 = ResponseScore::compute(0.8, 0.9, 0.5, false);
    let score2 = ResponseScore::compute(0.9, 0.95, 0.8, true);
    
    let mut eval = ResponseEvaluator::default();
    eval.record(score1);
    eval.record(score2);
    
    assert_eq!(eval.count, 2);
    assert!(eval.running_composite > 0.0);
}

// =============================================================================
// PHASES 4 - 5 & 8 - 9: TERNARY QUANTIZATION & CORE UTILS
// =============================================================================

#[test]
fn test_phase_4_token_estimator() {
    let text = "Hello, world!";
    let tokens = estimate_tokens(text);
    assert!(tokens > 0);
}

#[test]
fn test_phase_8_ternary_encoder() {
    let mut rng = seeded_rng(12345);
    let layer = TernaryLayer::random(4, 2, CoreActivation::TanhF32, &mut rng);
    assert_eq!(layer.weights.len(), 8);
    for &w in &layer.weights {
        assert!(w == -1 || w == 0 || w == 1);
    }

    let input = [1.0_f32, -1.0_f32, 0.5_f32, 0.0_f32];
    let output = layer.forward_f32(&input);
    assert_eq!(output.len(), 2);
}

#[test]
fn test_phase_8_quantization_helpers() {
    let x = 0.5_f32;
    let q = f32_to_i8(x);
    let d = i8_to_f32(q);
    assert!((x - d).abs() < 0.02);
}

// =============================================================================
// PHASES 11 - 13: THINKING+, REACT AGENT, KNOWLEDGE GRAPH
// =============================================================================

#[test]
fn test_phase_11_context_manager() {
    let config = ContextConfig {
        max_tokens: 1000,
        system_reserve: 100,
        memory_reserve: 100,
        tool_reserve: 100,
        input_reserve: 100,
        min_recent_turns: 1,
    };
    let mut manager = ContextManager::new(config);
    manager.push_turn("Who are you?", "I am LionAI.");
    let usage = manager.token_usage();
    assert_eq!(usage.total_turns, 1);
    assert!(usage.history_tokens > 0);
}

#[test]
fn test_phase_12_calculator_tool() {
    let agent = Agent::new(AgentConfig::default());
    let names = agent.tool_names();
    assert!(names.contains(&"calculator"));

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let out = agent.use_tool_directly("calculator", "12 * 12").await;
        assert!(out.contains("144"));

        let out2 = agent.use_tool_directly("calculator", "sqrt(1764) * 5").await;
        assert!(out2.contains("210"));
    });
}

#[test]
fn test_phase_13_knowledge_graph() {
    let mut graph = KnowledgeGraph::default();
    graph.learn("Rust", "System language", vec!["programming".to_string()], 1);
    graph.learn("C++", "Legacy language", vec!["programming".to_string()], 1);
    graph.relate("Rust", "safer_than", "C++");

    let results = graph.search("Rust");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "Rust");
}

// =============================================================================
// PHASES 14 - 16: MULTIMODAL, ALIGNMENT, ROUTER
// =============================================================================

#[test]
fn test_phase_14_image_encoder_dims() {
    let enc = ImageEncoder::default();
    assert_eq!(enc.feature_count(), 68); // 8*8 + 3 (RGB) + 1 (edge)
}

#[test]
fn test_phase_14_audio_features() {
    let enc = AudioEncoder::default();
    // 40 raw samples + rms + zcr + 16 bands = 58 features, padded to 64.
    assert_eq!(enc.feature_size, 64);
}

#[test]
fn test_phase_15_cross_modal_alignment_f32() {
    let a = [1.0_f32; 32];
    let b = [1.0_f32; 32];
    let sim = cosine_sim(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);

    let c = [-1.0_f32; 32];
    let sim_opposite = cosine_sim(&a, &c);
    assert!((sim_opposite - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_phase_16_router_decisions() {
    let router = lion_brain::Router::default();
    // High-confidence memory match → Direct route
    let dec1 = router.route("Hello", "Chat", 0.95);
    assert_eq!(dec1.route, lion_brain::Route::Direct);

    // Math query or intent → Agent route
    let dec2 = router.route("calculate 2+2", "Math", 0.0);
    assert_eq!(dec2.route, lion_brain::Route::Agent);

    // Default query → Thinking pipeline route
    let dec3 = router.route("What is a star?", "Question", 0.0);
    assert_eq!(dec3.route, lion_brain::Route::ThinkingPipeline);
}

#[test]
fn test_phase_12_path_traversal_denial() {
    use lion_agent::tool::Tool;
    let tool = lion_agent::tools::FileRead;
    
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let result1 = tool.execute("/etc/passwd").await;
        assert!(!result1.success);
        assert!(result1.observation.contains("Access denied"));

        let result2 = tool.execute("../../../etc/passwd").await;
        assert!(!result2.success);
        assert!(result2.observation.contains("Access denied"));
    });
}
