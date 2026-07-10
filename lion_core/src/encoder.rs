// lion_core/src/encoder.rs — Ternary Encoder (1.58-bit weights)
//
// A multi-layer perceptron whose weights are stored as i8 ∈ {-1, 0, +1}.
// Input: Vec<f32> of arbitrary length (padded/truncated to input_size).
// Output: [f32; FEATURE_SIZE] — a 32-dimensional embedding.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{i8_to_f32, Features, FEATURE_SIZE};

// =============================================================================
// ACTIVATION
// =============================================================================

/// Activation function for each TernaryLayer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// ReLU followed by re-quantization to i8.
    ReluQuantize,
    /// Tanh in f32 (used for the output layer).
    TanhF32,
}

// =============================================================================
// TERNARY LAYER
// =============================================================================

/// A single fully-connected layer with ternary weights.
///
/// Forward pass (vectorized):
///   h_j = Σ_i (w_ij × scale_j × x_i) + bias_j
///   out_j = activation(h_j)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryLayer {
    /// Weights stored as i8 ∈ {-1, 0, +1}.  Shape: [out_size × in_size].
    pub weights: Vec<i8>,

    /// Per-output-neuron scale factors (dequantization).
    pub scales: Vec<f32>,

    /// Per-output-neuron bias.
    pub biases: Vec<f32>,

    pub in_size:    usize,
    pub out_size:   usize,
    pub activation: Activation,
}

impl TernaryLayer {
    /// Constructs a layer from raw ternary weights.
    pub fn from_raw_weights(
        weights:    &[i8],
        in_size:    usize,
        out_size:   usize,
        scales:     Vec<f32>,
        biases:     Vec<f32>,
        activation: Activation,
    ) -> Self {
        assert_eq!(weights.len(), in_size * out_size);
        assert_eq!(scales.len(), out_size);
        assert_eq!(biases.len(), out_size);
        Self {
            weights: weights.to_vec(),
            scales,
            biases,
            in_size,
            out_size,
            activation,
        }
    }

    /// Constructs a random ternary layer.
    pub fn random<R: Rng>(
        in_size:    usize,
        out_size:   usize,
        activation: Activation,
        rng:        &mut R,
    ) -> Self {
        let n = in_size * out_size;
        let weights: Vec<i8> = (0..n)
            .map(|_| {
                let r: f32 = rng.gen();
                if r < 0.33 { -1 } else if r < 0.66 { 0 } else { 1 }
            })
            .collect();

        let scale = 1.0 / (in_size as f32).sqrt();
        let scales = vec![scale; out_size];
        let biases = vec![0.0_f32; out_size];

        Self { weights, scales, biases, in_size, out_size, activation }
    }

    /// Forward pass: input [in_size] → output [out_size].
    pub fn forward_f32(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0_f32; self.out_size];

        for (j, out_j) in output.iter_mut().enumerate() {
            let row_start = j * self.in_size;
            let mut sum = 0.0_f32;
            for i in 0..self.in_size {
                let w = self.weights[row_start + i];
                if w != 0 {
                    sum += w as f32 * input[i];
                }
            }
            let pre_act = sum * self.scales[j] + self.biases[j];
            *out_j = match self.activation {
                Activation::ReluQuantize => pre_act.max(0.0),
                Activation::TanhF32      => pre_act.tanh(),
            };
        }

        output
    }

    /// Forward pass using quantized i8 inputs.
    pub fn forward_i8(&self, input: &[i8]) -> Vec<f32> {
        let f32_input: Vec<f32> = input.iter().map(|&x| i8_to_f32(x)).collect();
        self.forward_f32(&f32_input)
    }
}

// =============================================================================
// TERNARY ENCODER CONFIG
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryEncoderConfig {
    pub input_size:   usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size:  usize,
}

impl Default for TernaryEncoderConfig {
    fn default() -> Self {
        Self {
            input_size:   64,
            hidden_sizes: vec![64],
            output_size:  FEATURE_SIZE,
        }
    }
}

impl TernaryEncoderConfig {
    /// All layer sizes in order: [input, hidden1, ..., output]
    pub fn all_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.input_size];
        sizes.extend(&self.hidden_sizes);
        sizes.push(self.output_size);
        sizes
    }
}

// =============================================================================
// TERNARY ENCODER
// =============================================================================

/// Multi-layer ternary network that encodes arbitrary inputs into
/// a [f32; FEATURE_SIZE] embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryEncoder {
    pub layers: Vec<TernaryLayer>,
    pub config: TernaryEncoderConfig,
}

impl TernaryEncoder {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Builds a random TernaryEncoder from config.
    pub fn random<R: Rng>(config: TernaryEncoderConfig, rng: &mut R) -> Self {
        let sizes = config.all_sizes();
        let n_layers = sizes.len() - 1;
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let is_last  = i == n_layers - 1;
            let act = if is_last { Activation::TanhF32 } else { Activation::ReluQuantize };
            layers.push(TernaryLayer::random(sizes[i], sizes[i + 1], act, rng));
        }
        Self { layers, config }
    }

    /// Builds a TernaryEncoder from pre-constructed layers.
    pub fn from_layers(layers: Vec<TernaryLayer>, config: TernaryEncoderConfig) -> Self {
        Self { layers, config }
    }

    // ── Encoding ──────────────────────────────────────────────────────────────

    /// Encodes a float input slice → [f32; FEATURE_SIZE].
    ///
    /// Input is padded with zeros or truncated to `config.input_size`.
    pub fn encode_f32(&self, input: &[f32]) -> Features {
        let mut padded = vec![0.0_f32; self.config.input_size];
        let copy_len = input.len().min(self.config.input_size);
        padded[..copy_len].copy_from_slice(&input[..copy_len]);

        let mut current: Vec<f32> = padded;
        for layer in &self.layers {
            current = layer.forward_f32(&current);
        }

        // L2-normalize the final embedding.
        let norm: f32 = current.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let normalized: Vec<f32> = current.iter().map(|x| x / norm).collect();

        let mut out = [0.0_f32; FEATURE_SIZE];
        let copy_len = normalized.len().min(FEATURE_SIZE);
        out[..copy_len].copy_from_slice(&normalized[..copy_len]);
        out
    }

    /// Encodes an i8-quantized input slice → [f32; FEATURE_SIZE].
    pub fn encode_i8(&self, input: &[i8]) -> Features {
        let f32_input: Vec<f32> = input.iter().map(|&x| i8_to_f32(x)).collect();
        self.encode_f32(&f32_input)
    }

    /// Encodes text by mapping each character to a float in [-1, +1].
    pub fn encode_text(&self, text: &str) -> Features {
        let input_size = self.config.input_size;
        let mut padded = vec![0.0_f32; input_size];
        for (i, ch) in text.chars().enumerate().take(input_size) {
            let scalar = ch as u32;
            padded[i] = if scalar <= 255 {
                (scalar as f32 / 127.5) - 1.0
            } else {
                ((scalar % 256) as f32 / 127.5) - 1.0
            };
        }
        self.encode_f32(&padded)
    }
}
