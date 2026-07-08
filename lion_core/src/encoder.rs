// lion_core/src/encoder.rs

use crate::constants::FEATURE_SIZE;
use crate::propagation::SensoryInput;
use crate::rng::BrainRng;
use crate::ternary::{f32_to_i8, Activation, TernaryLayer};
use crate::types::Role;
use serde::{Deserialize, Serialize};

// =============================================================================
// ENCODER CONFIGURATION
// =============================================================================

/// Configuration for building a TernaryEncoder.
///
/// # Typical small encoder (default):
///   input_size:   64   (e.g., a compressed tokenized input)
///   hidden_sizes: [64]  (one hidden layer)
///   output_size:  32   (= FEATURE_SIZE)
///
/// The final layer always uses TanhF32 activation.
/// All intermediate layers use ReluQuantize activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryEncoderConfig {
    /// Number of input features (length of the raw quantized input vector).
    pub input_size: usize,

    /// Sizes of intermediate hidden layers.
    /// Empty = no hidden layers (input → output directly).
    pub hidden_sizes: Vec<usize>,

    /// Output size. Must equal `FEATURE_SIZE` to integrate with the cognitive graph.
    pub output_size: usize,
}

impl TernaryEncoderConfig {
    /// The default small encoder configuration for LionAI.
    ///
    /// Architecture: 64 → 64 → 32
    ///   Input:  64 i8 features (quantized sensory input)
    ///   Hidden: 64 features (ReLU, quantized)
    ///   Output: 32 f32 features (tanh, = FEATURE_SIZE)
    pub fn default_small() -> Self {
        Self {
            input_size:   64,
            hidden_sizes: vec![64],
            output_size:  FEATURE_SIZE,
        }
    }

    /// Minimal encoder with no hidden layers (input → output directly).
    ///
    /// Architecture: input_size → FEATURE_SIZE
    pub fn minimal(input_size: usize) -> Self {
        Self {
            input_size,
            hidden_sizes: vec![],
            output_size:  FEATURE_SIZE,
        }
    }

    /// Returns the complete list of layer input/output sizes in order.
    pub fn layer_sizes(&self) -> Vec<(usize, usize)> {
        let mut sizes = Vec::new();
        let mut prev  = self.input_size;

        for &hidden in &self.hidden_sizes {
            sizes.push((prev, hidden));
            prev = hidden;
        }

        sizes.push((prev, self.output_size));
        sizes
    }
}

// =============================================================================
// TERNARY ENCODER
// =============================================================================

/// A multi-layer ternary neural encoder.
///
/// Converts a raw `[f32]` sensory input of arbitrary length into a fixed
/// `[f32; FEATURE_SIZE]` embedding vector that feeds directly into
/// `BrainMatrix::inject_sensory()`.
///
/// All intermediate layers use ternary weights with ReLU + i8 quantization.
/// The final layer uses ternary weights with tanh + f32 output.
///
/// # Zero floating-point multiplication
/// Every GEMV operation uses only integer addition and subtraction.
/// Scale and bias (f32) are applied only once per neuron AFTER the GEMV,
/// not during the inner loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryEncoder {
    pub layers: Vec<TernaryLayer>,

    /// The encoder config this was built from (for diagnostics).
    pub config: TernaryEncoderConfig,
}

impl TernaryEncoder {
    // =========================================================================
    // CONSTRUCTION
    // =========================================================================

    /// Builds a TernaryEncoder from a list of pre-built TernaryLayers.
    ///
    /// Validates that:
    ///   - At least one layer exists.
    ///   - Layer dimensions are consistent (output of layer i = input of layer i+1).
    ///   - The final layer uses TanhF32 activation.
    ///   - The final layer's output size equals FEATURE_SIZE.
    ///
    /// # Panics
    /// Panics if any of the above validation checks fail.
    pub fn from_layers(
        layers: Vec<TernaryLayer>,
        config: TernaryEncoderConfig,
    ) -> Self {
        assert!(!layers.is_empty(), "TernaryEncoder requires at least one layer");

        // Validate dimension chain.
        for i in 1..layers.len() {
            assert_eq!(
                layers[i - 1].out_features,
                layers[i].in_features,
                "Layer {} output ({}) must equal layer {} input ({})",
                i - 1, layers[i - 1].out_features,
                i,     layers[i].in_features
            );
        }

        // Validate final layer.
        let last = layers.last().unwrap();
        assert_eq!(
            last.activation,
            Activation::TanhF32,
            "Final encoder layer must use TanhF32 activation"
        );
        assert_eq!(
            last.out_features,
            FEATURE_SIZE,
            "Final encoder layer output must equal FEATURE_SIZE={}, got {}",
            FEATURE_SIZE,
            last.out_features
        );

        Self { layers, config }
    }

    /// Creates a randomly initialized encoder from a config.
    ///
    /// All weights drawn uniformly from {-1, 0, +1}.
    /// Scales = 1.0 / in_features per layer.
    /// Biases = 0.0.
    ///
    /// This encoder is suitable for testing and as a starting point for training.
    /// In production, weights would be loaded from a trained model file.
    pub fn random(config: TernaryEncoderConfig, rng: &mut BrainRng) -> Self {
        let layer_sizes = config.layer_sizes();
        let n_layers    = layer_sizes.len();

        let layers: Vec<TernaryLayer> = layer_sizes
            .into_iter()
            .enumerate()
            .map(|(idx, (in_size, out_size))| {
                let activation = if idx == n_layers - 1 {
                    Activation::TanhF32
                } else {
                    Activation::ReluQuantize
                };
                TernaryLayer::random(in_size, out_size, activation, rng)
            })
            .collect();

        Self::from_layers(layers, config)
    }

    // =========================================================================
    // FORWARD PASS
    // =========================================================================

    /// Encodes a raw `&[f32]` input into a `[f32; FEATURE_SIZE]` embedding.
    ///
    /// Pipeline:
    ///   1. Quantize f32 input to i8: each value clamped to [-1, +1] × 127.
    ///   2. Run each hidden layer: ternary_gemv → scale → relu → i8.
    ///   3. Run the final layer:   ternary_gemv → scale → tanh → f32.
    ///   4. Return `[f32; FEATURE_SIZE]`.
    ///
    /// # Input length
    /// The input slice must have exactly `config.input_size` elements.
    /// If it is shorter, it is zero-padded.
    /// If it is longer, it is truncated.
    ///
    /// # Panics
    /// Panics in debug builds if the output vec is not FEATURE_SIZE.
    pub fn encode_f32(&self, input: &[f32]) -> [f32; FEATURE_SIZE] {
        // Step 1: Quantize and pad/truncate to input_size.
        let in_size = self.config.input_size;
        let mut i8_input = vec![0i8; in_size];
        for (i, chunk) in i8_input.iter_mut().enumerate() {
            *chunk = if i < input.len() { f32_to_i8(input[i]) } else { 0 };
        }

        // Steps 2–3: Forward through all layers.
        self.forward_from_i8(&i8_input)
    }

    /// Encodes a pre-quantized `&[i8]` input into a `[f32; FEATURE_SIZE]` embedding.
    ///
    /// Use this when the input is already in INT8 format to skip re-quantization.
    pub fn encode_i8(&self, input: &[i8]) -> [f32; FEATURE_SIZE] {
        debug_assert_eq!(
            input.len(),
            self.config.input_size,
            "encode_i8: input length {} != config.input_size {}",
            input.len(),
            self.config.input_size
        );
        self.forward_from_i8(input)
    }

    /// Runs the multi-layer forward pass from an i8 input.
    fn forward_from_i8(&self, input: &[i8]) -> [f32; FEATURE_SIZE] {
        let n_layers = self.layers.len();

        let mut current = input.to_vec();

        // Hidden layers (all but the last).
        for layer in &self.layers[..n_layers - 1] {
            current = layer.forward_hidden(&current);
        }

        // Final layer → f32.
        let final_output = self.layers[n_layers - 1].forward_final(&current);

        debug_assert_eq!(final_output.len(), FEATURE_SIZE);

        let mut result = [0.0_f32; FEATURE_SIZE];
        result.copy_from_slice(&final_output);
        result
    }

    // =========================================================================
    // SENSORY FRAME HELPERS
    // =========================================================================

    /// Encodes a single raw input and inserts it into a SensoryInput frame.
    ///
    /// This is the main integration point with the cognitive graph.
    ///
    /// # Usage
    /// ```text
    /// let mut frame = SensoryInput::new();
    /// encoder.encode_into_frame(Role::Vision, &raw_vision_data, &mut frame);
    /// sovereign.update(&frame, prev_reward);
    /// ```
    pub fn encode_into_frame(
        &self,
        modality: Role,
        raw:      &[f32],
        frame:    &mut SensoryInput,
    ) {
        let embedding = self.encode_f32(raw);
        frame.insert(modality, embedding);
    }

    /// Encodes multiple modalities into a complete SensoryInput frame.
    ///
    /// # Parameters
    /// `inputs` — slice of (role, raw_f32_data) pairs.
    pub fn encode_frame(&self, inputs: &[(Role, &[f32])]) -> SensoryInput {
        let mut frame = SensoryInput::new();
        for &(role, raw) in inputs {
            self.encode_into_frame(role, raw, &mut frame);
        }
        frame
    }

    // =========================================================================
    // DIAGNOSTICS
    // =========================================================================

    /// Returns the total number of packed weight bytes across all layers.
    pub fn total_weight_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.weight_bytes()).sum()
    }

    /// Returns the total memory footprint in bytes (weights + scales + biases).
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Returns the number of layers.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Returns the sparsity of the weight matrix (fraction of zero weights).
    ///
    /// Higher sparsity = fewer additions during GEMV = faster inference.
    pub fn weight_sparsity(&self) -> f32 {
        let mut total   = 0usize;
        let mut nonzero = 0usize;

        for layer in &self.layers {
            let n = layer.in_features * layer.out_features;
            total += n;
            for i in 0..n {
                if crate::ternary::unpack_weight(&layer.weights, i) != 0 {
                    nonzero += 1;
                }
            }
        }

        if total == 0 { return 0.0; }
        1.0 - (nonzero as f32 / total as f32)
    }
}
