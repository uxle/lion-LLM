// lion_core/src/language.rs
//
// Language Motor Cortex — Phase 13b: Fixed English Bias
//
// Key fixes over Phase 13a:
//   - Proper `lm_bias` field that survives layernorm (additive to logits directly)
//   - EOS suppression for first MIN_LEN tokens
//   - Repetition penalty to avoid `aaaaaaa` collapse
//   - Temperature 0.7 for generation (mild sampling prevents stuck loops)
//   - Stronger English frequencies: top-20 chars get +4.0 bias
//   - Top-3 most-English chars (space, e, t) get +6.0 bias

use crate::constants::FEATURE_SIZE;
use crate::rng::BrainRng;
use serde::{Deserialize, Serialize};

pub const VOCAB_SIZE:  usize = 96;   // Printable ASCII (space=1..~=95, EOS=0)
pub const MAX_SEQ_LEN: usize = 48;
const MIN_GEN_LEN:     usize = 8;    // Never emit EOS before this many tokens

// ==============================================================================
// TOKENIZER
// ==============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer;

impl Tokenizer {
    #[inline]
    pub fn encode_char(c: char) -> usize {
        let b = c as u8;
        if b == b'\n'               { 0 }
        else if b == b' '           { 1 }
        else if b >= 33 && b < 127  { (b - 31) as usize }
        else                        { 0 }
    }

    pub fn encode(text: &str) -> Vec<usize> {
        text.chars().map(Self::encode_char).collect()
    }

    #[inline]
    pub fn decode_token(t: usize) -> Option<char> {
        if t == 0             { None }
        else if t == 1        { Some(' ') }
        else if t < VOCAB_SIZE { Some((t as u8 + 31) as char) }
        else                  { None }
    }

    pub fn decode(tokens: &[usize]) -> String {
        tokens.iter().filter_map(|&t| Self::decode_token(t)).collect()
    }
}

// ==============================================================================
// ENGLISH CHARACTER FREQUENCY TABLE
// ==============================================================================

/// Returns the (token_id, bias_weight) pairs for English character priors.
/// Based on English letter frequency corpus data (Norvig, 2009).
fn english_priors() -> &'static [(usize, f32)] {
    // Token 1 = space, token N = ASCII byte N+31
    // space=1, e=38, t=57, a=34, o=48, i=40, n=47, s=52, r=51, h=39,
    // l=45, d=37, c=36, u=58, m=46, f=39→wait let me recalculate
    // ASCII 'a'=97 → token = 97-31 = 66 ... wait that's wrong
    // My tokenizer: space → 1, char b (33≤b<127) → b-31
    // 'a' = 97 → 97-31 = 66
    // 'e' = 101 → 101-31 = 70
    // 't' = 116 → 116-31 = 85
    // 'o' = 111 → 111-31 = 80
    // 'i' = 105 → 105-31 = 74
    // 'n' = 110 → 110-31 = 79
    // 's' = 115 → 115-31 = 84
    // 'r' = 114 → 114-31 = 83
    // 'h' = 104 → 104-31 = 73
    // 'l' = 108 → 108-31 = 77
    // 'd' = 100 → 100-31 = 69
    // 'c' = 99  → 99-31  = 68
    // 'u' = 117 → 117-31 = 86
    // 'm' = 109 → 109-31 = 78
    // 'w' = 119 → 119-31 = 88
    // 'f' = 102 → 102-31 = 71
    // 'g' = 103 → 103-31 = 72
    // 'y' = 121 → 121-31 = 90
    // 'p' = 112 → 112-31 = 81
    // 'b' = 98  → 98-31  = 67
    &[
        (1,  6.0),   // space — MOST important
        (70, 6.0),   // e
        (85, 5.5),   // t
        (66, 5.0),   // a
        (80, 5.0),   // o
        (74, 5.0),   // i
        (79, 5.0),   // n
        (84, 5.0),   // s
        (83, 5.0),   // r
        (73, 4.5),   // h
        (77, 4.5),   // l
        (69, 4.0),   // d
        (68, 4.0),   // c
        (86, 4.0),   // u
        (78, 4.0),   // m
        (88, 3.5),   // w
        (71, 3.5),   // f
        (72, 3.5),   // g
        (90, 3.5),   // y
        (81, 3.0),   // p
        (67, 3.0),   // b
    ]
}

// ==============================================================================
// LANGUAGE MOTOR
// ==============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageMotor {
    // Embeddings
    pub token_embedding: Vec<f32>,  // [VOCAB_SIZE × FEATURE_SIZE]
    pub pos_embedding:   Vec<f32>,  // [MAX_SEQ_LEN × FEATURE_SIZE]

    // Attention
    pub w_q: Vec<f32>,  // [FEATURE_SIZE × FEATURE_SIZE]
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,

    // Feed-forward MLP
    pub w_up:   Vec<f32>,  // [FEATURE_SIZE × (FEATURE_SIZE*4)]
    pub w_down: Vec<f32>,  // [(FEATURE_SIZE*4) × FEATURE_SIZE]

    // Output
    pub w_lm_head: Vec<f32>,  // [VOCAB_SIZE × FEATURE_SIZE]

    /// CRITICAL: Direct additive bias applied to logits AFTER GEMV.
    /// This is NOT zeroed out by layernorm. English common chars get large positive bias.
    pub lm_bias: Vec<f32>,  // [VOCAB_SIZE]
}

impl LanguageMotor {
    // ── Construction ──────────────────────────────────────────────────────────

    pub fn random(rng: &mut BrainRng) -> Self {
        let scale = (2.0 / FEATURE_SIZE as f32).sqrt();
        let mut gen = || rng.gen_prob() * scale * 2.0 - scale;

        // Build lm_bias: English char frequency priors
        let mut lm_bias = vec![0.0f32; VOCAB_SIZE];
        lm_bias[0] = -12.0; // Strong EOS suppression in bias too
        for &(tok, weight) in english_priors() {
            if tok < VOCAB_SIZE {
                lm_bias[tok] = weight;
            }
        }
        // Penalize control-like chars (!, @, #, etc. — low-freq punctuation)
        for tok in 2..10 {
            lm_bias[tok] -= 2.0; // !, ", #, $, %, &, ', (, )
        }

        Self {
            token_embedding: (0..VOCAB_SIZE * FEATURE_SIZE).map(|_| gen()).collect(),
            pos_embedding:   (0..MAX_SEQ_LEN * FEATURE_SIZE).map(|_| gen()).collect(),
            w_q:    (0..FEATURE_SIZE * FEATURE_SIZE).map(|_| gen()).collect(),
            w_k:    (0..FEATURE_SIZE * FEATURE_SIZE).map(|_| gen()).collect(),
            w_v:    (0..FEATURE_SIZE * FEATURE_SIZE).map(|_| gen()).collect(),
            w_o:    (0..FEATURE_SIZE * FEATURE_SIZE).map(|_| gen()).collect(),
            w_up:   (0..FEATURE_SIZE * (FEATURE_SIZE * 4)).map(|_| gen()).collect(),
            w_down: (0..(FEATURE_SIZE * 4) * FEATURE_SIZE).map(|_| gen()).collect(),
            w_lm_head: (0..VOCAB_SIZE * FEATURE_SIZE).map(|_| gen()).collect(),
            lm_bias,
        }
    }

    // ── Math utilities ─────────────────────────────────────────────────────────

    #[inline]
    fn gemv(mat: &[f32], vec: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        for r in 0..rows {
            let base = r * cols;
            out[r] = mat[base..base + cols]
                .iter()
                .zip(vec.iter())
                .map(|(w, x)| w * x)
                .sum();
        }
    }

    fn layernorm(v: &mut [f32]) {
        let n    = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var  = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std  = (var + 1e-5).sqrt();
        for x in v.iter_mut() { *x = (*x - mean) / std; }
    }

    fn gelu(v: &mut [f32]) {
        for x in v.iter_mut() {
            let u = *x;
            *x = 0.5 * u * (1.0 + (0.7978845608 * (u + 0.044715 * u.powi(3))).tanh());
        }
    }

    /// Softmax with numerical stability.
    fn softmax(logits: &mut Vec<f32>) {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in logits.iter_mut() { *x = (*x - max).exp(); sum += *x; }
        if sum > 0.0 { for x in logits.iter_mut() { *x /= sum; } }
    }

    /// Temperature sampling.
    fn sample(logits: &[f32], temp: f32, rng: &mut BrainRng) -> usize {
        if temp <= 0.0 {
            return logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(1);
        }
        let mut probs: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
        Self::softmax(&mut probs);
        let r = rng.gen_prob();
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum { return i; }
        }
        1 // fallback: space
    }

    // ── Forward pass (one step) ───────────────────────────────────────────────

    fn step(&self, h: &[f32; FEATURE_SIZE], pos: usize) -> Vec<f32> {
        let mut hn: Vec<f32> = h.to_vec();
        Self::layernorm(&mut hn);

        // Attention (single-token, so softmax(qkᵀ/√d) = 1 → out = V·Wₒ)
        let mut v_proj = vec![0.0f32; FEATURE_SIZE];
        Self::gemv(&self.w_v, &hn, &mut v_proj, FEATURE_SIZE, FEATURE_SIZE);
        let mut attn = vec![0.0f32; FEATURE_SIZE];
        Self::gemv(&self.w_o, &v_proj, &mut attn, FEATURE_SIZE, FEATURE_SIZE);

        // Residual 1
        let mut h2: Vec<f32> = h.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();
        let mut h2n = h2.clone();
        Self::layernorm(&mut h2n);

        // MLP: up → gelu → down
        let mut up = vec![0.0f32; FEATURE_SIZE * 4];
        Self::gemv(&self.w_up, &h2n, &mut up, FEATURE_SIZE * 4, FEATURE_SIZE);
        Self::gelu(&mut up);
        let mut down = vec![0.0f32; FEATURE_SIZE];
        Self::gemv(&self.w_down, &up, &mut down, FEATURE_SIZE, FEATURE_SIZE * 4);

        // Residual 2
        for i in 0..FEATURE_SIZE { h2[i] += down[i]; }
        Self::layernorm(&mut h2);

        // Logits = lm_head × h_final + lm_bias
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        Self::gemv(&self.w_lm_head, &h2, &mut logits, VOCAB_SIZE, FEATURE_SIZE);

        // ADD bias (this survives layernorm because it's applied AFTER)
        for (l, &b) in logits.iter_mut().zip(self.lm_bias.iter()) {
            *l += b;
        }

        // Word boundary nudge: prefer space every ~5 chars
        if pos > 3 && pos % 5 == 0 {
            logits[1] += 2.5; // space
        }

        logits
    }

    // ── Public generation ─────────────────────────────────────────────────────

    /// Deterministic greedy generation (for scoring/comparison).
    pub fn generate(&self, gestalt: &[f32; FEATURE_SIZE], max_tokens: usize) -> String {
        let mut rng = BrainRng::from_seed(42);
        self.generate_with_temp(gestalt, max_tokens, 0.0, &mut rng)
    }

    /// Generate with temperature sampling and repetition penalty.
    pub fn generate_with_temp(
        &self,
        gestalt:    &[f32; FEATURE_SIZE],
        max_tokens: usize,
        temp:       f32,
        rng:        &mut BrainRng,
    ) -> String {
        let mut state = *gestalt;
        let mut tokens: Vec<usize> = Vec::with_capacity(max_tokens);
        // Recent-token frequency for repetition penalty
        let mut recent = [0u32; VOCAB_SIZE];

        for pos in 0..max_tokens {
            let mut logits = self.step(&state, pos);

            // Suppress EOS until minimum length
            if pos < MIN_GEN_LEN {
                logits[0] = -20.0;
            }

            // Repetition penalty: reduce logits for recently generated tokens
            for (tok, &cnt) in recent.iter().enumerate() {
                if cnt > 0 {
                    logits[tok] -= (cnt as f32) * 1.5;
                }
            }

            let tok = Self::sample(&logits, temp, rng);
            if tok == 0 && pos >= MIN_GEN_LEN { break; }
            if tok == 0 { continue; } // skip EOS if suppressed

            tokens.push(tok);

            // Track recent tokens (decay older counts)
            for c in recent.iter_mut() { *c = c.saturating_sub(1); }
            if tok < VOCAB_SIZE { recent[tok] += 3; }

            // Update state with token + position embedding
            for i in 0..FEATURE_SIZE {
                let te = self.token_embedding[tok * FEATURE_SIZE + i];
                let pe = self.pos_embedding[(pos % MAX_SEQ_LEN) * FEATURE_SIZE + i];
                state[i] = state[i] * 0.65 + (te + pe) * 0.35;
            }
        }

        Tokenizer::decode(&tokens)
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    pub fn mutate(&mut self, rng: &mut BrainRng, mut_rate: f32) {
        let clamp = 4.0f32;
        for vec in [
            &mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o,
            &mut self.w_up, &mut self.w_down, &mut self.w_lm_head,
            &mut self.token_embedding, &mut self.pos_embedding,
        ] {
            for x in vec.iter_mut() {
                if rng.gen_bool_with_prob(mut_rate) {
                    *x = (*x + rng.gen_mutation_delta()).clamp(-clamp, clamp);
                }
            }
        }
        // Mutate lm_bias too (but gently — the English priors are precious)
        let bias_rate = mut_rate * 0.3;
        for x in self.lm_bias.iter_mut() {
            if rng.gen_bool_with_prob(bias_rate) {
                *x = (*x + rng.gen_mutation_delta() * 0.5).clamp(-15.0, 10.0);
            }
        }
        // Re-enforce EOS suppression after mutation
        self.lm_bias[0] = self.lm_bias[0].min(-8.0);
    }

    // ── Fitness scoring ───────────────────────────────────────────────────────

    /// Bigram + character-frequency similarity (0.0..=1.0).
    pub fn compute_language_score(
        &self,
        gestalt:     &[f32; FEATURE_SIZE],
        target_text: &str,
    ) -> f64 {
        if target_text.is_empty() { return 0.0; }
        let generated = self.generate(gestalt, target_text.len() + 8);
        bigram_similarity(&generated, target_text)
    }
}

// ==============================================================================
// SIMILARITY UTILITIES
// ==============================================================================

fn bigram_similarity(generated: &str, target: &str) -> f64 {
    if target.is_empty() { return 0.0; }
    let gen: Vec<char> = generated.chars().collect();
    let tgt: Vec<char> = target.chars().collect();

    // Character frequency overlap
    let mut tgt_freq = std::collections::HashMap::new();
    for c in &tgt { *tgt_freq.entry(c).or_insert(0u32) += 1; }
    let mut gen_freq = std::collections::HashMap::new();
    for c in &gen { *gen_freq.entry(c).or_insert(0u32) += 1; }
    let char_match: u32 = tgt_freq.iter()
        .map(|(c, &cnt)| cnt.min(*gen_freq.get(c).unwrap_or(&0)))
        .sum();
    let char_score = char_match as f64 / tgt.len().max(1) as f64;

    // Bigram overlap
    if tgt.len() < 2 { return char_score; }
    let mut tgt_bi = std::collections::HashMap::new();
    for w in tgt.windows(2) { *tgt_bi.entry((w[0], w[1])).or_insert(0u32) += 1; }
    let mut gen_bi = std::collections::HashMap::new();
    for w in gen.windows(2) { *gen_bi.entry((w[0], w[1])).or_insert(0u32) += 1; }
    let bi_match: u32 = tgt_bi.iter()
        .map(|(b, &cnt)| cnt.min(*gen_bi.get(b).unwrap_or(&0)))
        .sum();
    let bi_score = bi_match as f64 / tgt_bi.values().sum::<u32>().max(1) as f64;

    bi_score * 0.6 + char_score * 0.4
}

// ==============================================================================
// SPEECH TARGETS
// ==============================================================================

pub fn target_speech_for_action(action: &str) -> &'static str {
    match action {
        "FLEE"   => "run away from danger",
        "ATTACK" => "i will fight back",
        "FORAGE" => "looking for food now",
        "HIDE"   => "stay quiet and hide",
        _        => "i am here with you",
    }
}

pub fn target_speech_for_input(input: &str) -> &'static str {
    let l = input.to_lowercase();
    if l.contains("hi") || l.contains("hello") || l.contains("hey") {
        "hello there friend"
    } else if l.contains("how") && l.contains("you") {
        "i am doing well"
    } else if (l.contains("what") || l.contains("who")) && l.contains("name") {
        "i am lion ai"
    } else if l.contains("help") {
        "i will help you"
    } else if l.contains("food") || l.contains("eat") || l.contains("hungry") {
        "i found some food"
    } else if l.contains("danger") || l.contains("attack") || l.contains("threat") {
        "we must flee now"
    } else if l.contains("good") || l.contains("great") || l.contains("nice") {
        "that is good news"
    } else if l.contains("bad") || l.contains("no") || l.contains("wrong") {
        "i understand that"
    } else if l.contains("what") || l.contains("doing") {
        "i am thinking now"
    } else if l.contains("where") || l.contains("here") {
        "i am right here now"
    } else {
        "i hear you now"
    }
}
