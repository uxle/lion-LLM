// lion_core/src/ternary.rs

use serde::{Deserialize, Serialize};

/// Packed ternary weight value for zero.
pub const TERNARY_ZERO: u8 = 0b00;

/// Packed ternary weight value for +1.
pub const TERNARY_POS: u8 = 0b01;

/// Packed ternary weight value for -1.
pub const TERNARY_NEG: u8 = 0b10;

/// Number of ternary weights packed per byte (2 bits each → 4 per byte).
pub const WEIGHTS_PER_BYTE: usize = 4;

// =============================================================================
// WEIGHT PACKING
// =============================================================================

/// Converts a slice of ternary values `{-1, 0, +1}` into a packed `Vec<u8>`.
///
/// Input must contain only values from `{-1i8, 0i8, 1i8}`.
/// Values outside this range are clamped to the nearest ternary value.
///
/// Packing layout (little-endian, 4 weights per byte):
///   byte[k] = w[4k] | w[4k+1]<<2 | w[4k+2]<<4 | w[4k+3]<<6
///
/// If `weights.len()` is not divisible by 4, the final byte is zero-padded.
///
/// # Example
/// ```text
/// let ternary = vec![1i8, -1, 0, 1];
/// // w0=+1→0b01, w1=-1→0b10, w2=0→0b00, w3=+1→0b01
/// // byte = 0b01 | 0b10<<2 | 0b00<<4 | 0b01<<6
/// //      = 0b01001001 = 0x49
/// let packed = pack_weights(&ternary);
/// assert_eq!(packed[0], 0x49);
/// ```
pub fn pack_weights(weights: &[i8]) -> Vec<u8> {
    let num_bytes = weights.len().div_ceil(WEIGHTS_PER_BYTE);
    let mut packed = vec![0u8; num_bytes];

    for (i, &w) in weights.iter().enumerate() {
        let code: u8 = match w {
            1          => TERNARY_POS,
            w if w < 0 => TERNARY_NEG,
            _          => TERNARY_ZERO,
        };
        let byte_idx  = i / WEIGHTS_PER_BYTE;
        let bit_shift = (i % WEIGHTS_PER_BYTE) * 2;
        packed[byte_idx] |= code << bit_shift;
    }

    packed
}

/// Extracts a single ternary weight from a packed byte array.
///
/// Returns the weight as `i8` in `{-1, 0, +1}`.
///
/// # Panics
/// Panics if `flat_index` is out of range for the packed array.
#[inline]
pub fn unpack_weight(packed: &[u8], flat_index: usize) -> i8 {
    let byte_idx  = flat_index / WEIGHTS_PER_BYTE;
    let bit_shift = (flat_index % WEIGHTS_PER_BYTE) * 2;
    let code = (packed[byte_idx] >> bit_shift) & 0x03;

    match code {
        TERNARY_POS => 1,
        TERNARY_NEG => -1,
        _           => 0,
    }
}

/// Returns the number of packed bytes needed for `n` ternary weights.
#[inline]
pub fn packed_byte_count(n: usize) -> usize {
    n.div_ceil(WEIGHTS_PER_BYTE)
}

// =============================================================================
// BRANCHLESS TERNARY GEMV
// =============================================================================

/// Computes a General Matrix-Vector Multiply (GEMV) with packed ternary weights.
///
/// Mathematical operation:
///   output[i] = Σⱼ W[i,j] × input[j]
///
/// where W[i,j] ∈ {-1, 0, +1} — no multiplications, only add/subtract.
///
/// Branchless implementation using arithmetic bitmasks:
///   mask_pos = -(w_packed == 1)  →  0xFFFFFFFF if +1, else 0x00000000
///   mask_neg = -(w_packed == 2)  →  0xFFFFFFFF if -1, else 0x00000000
///   accumulator += input[j] & mask_pos
///   accumulator -= input[j] & mask_neg
///
/// # Parameters
/// - `input`       — i8 activations of length `in_features`
/// - `weights`     — packed ternary weights of size ceil(in*out/4)
/// - `output`      — i32 accumulation buffer of length `out_features` (caller zeroes)
/// - `in_features` — number of input neurons
/// - `out_features`— number of output neurons
///
/// # Weight layout
/// Weights are stored in row-major order: W[out_idx * in_features + in_idx].
/// Row i contains all weights from input to output neuron i.
///
/// Matches C pseudocode from the architecture design session:
///   for i in 0..out_features:
///       for j in 0..in_features:
///           flat_index = i * in_features + j
///           extract 2-bit w from packed
///           mask_pos = -(w == 1); mask_neg = -(w == 2)
///           acc += input[j] & mask_pos
///           acc -= input[j] & mask_neg
#[allow(clippy::needless_range_loop)]
pub fn ternary_gemv(
    input:        &[i8],
    weights:      &[u8],
    output:       &mut [i32],
    in_features:  usize,
    out_features: usize,
) {
    debug_assert_eq!(input.len(),  in_features);
    debug_assert_eq!(output.len(), out_features);
    debug_assert_eq!(weights.len(), packed_byte_count(in_features * out_features));

    for i in 0..out_features {
        let mut accumulator = 0i32;
        let row_offset = i * in_features;

        for j in 0..in_features {
            let flat_index = row_offset + j;
            let byte_idx   = flat_index / WEIGHTS_PER_BYTE;
            let bit_shift  = (flat_index % WEIGHTS_PER_BYTE) * 2;

            let w_packed = (weights[byte_idx] >> bit_shift) & 0x03;

            // Arithmetic bitmask — zero branches.
            let mask_pos = -((w_packed == TERNARY_POS) as i32);
            let mask_neg = -((w_packed == TERNARY_NEG) as i32);

            accumulator += (input[j] as i32) & mask_pos;
            accumulator -= (input[j] as i32) & mask_neg;
        }

        output[i] = accumulator;
    }
}

// =============================================================================
// ACTIVATION & QUANTIZATION
// =============================================================================

/// Quantizes a single f32 activation to i8 range [-127, 127].
///
/// Clamps to [-1.0, +1.0] then scales.
/// Used when converting f32 inputs to i8 before GEMV.
#[inline]
pub fn f32_to_i8(x: f32) -> i8 {
    (x.clamp(-1.0, 1.0) * 127.0).round() as i8
}

/// Converts an i8 activation back to f32 in [-1.0, +1.0].
#[inline]
pub fn i8_to_f32(x: i8) -> f32 {
    x as f32 / 127.0
}

/// Applies ReLU to an i8 value: max(0, x).
#[inline]
pub fn relu_i8(x: i8) -> i8 {
    x.max(0)
}

/// Applies tanh to an i32 accumulator scaled by `scale`, plus `bias`.
/// Returns the result as f32.
///
/// Used in the final encoder layer where output is f32.
#[inline]
pub fn apply_scale_tanh(acc: i32, scale: f32, bias: f32) -> f32 {
    ((acc as f32) * scale + bias).tanh()
}

/// Applies scale + bias to an i32 accumulator, then ReLU-quantizes to i8.
///
/// Used in intermediate encoder layers where output feeds back into GEMV.
#[inline]
pub fn apply_scale_relu_quantize(acc: i32, scale: f32, bias: f32) -> i8 {
    let activated = ((acc as f32) * scale + bias).max(0.0);
    f32_to_i8(activated)
}

// =============================================================================
// TERNARY LAYER
// =============================================================================

/// Activation function for a TernaryLayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// ReLU then quantize to i8 — used in hidden layers.
    ReluQuantize,
    /// tanh then output as f32 — used in the final encoder layer.
    TanhF32,
}

/// A single layer in the ternary encoder.
///
/// Stores packed ternary weights, per-output scale factors, and biases.
/// Scales and biases remain full-precision f32 — there are only `out_features`
/// of them, so their memory cost is negligible.
///
/// Weight layout (row-major):
///   W[i, j] = weight from input j to output i
///   Flat index: i * in_features + j
///   Packed: ceil(in * out / 4) bytes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryLayer {
    /// Packed ternary weights. ceil(in_features × out_features / 4) bytes.
    pub weights: Vec<u8>,

    pub in_features:  usize,
    pub out_features: usize,

    /// Per-output scale factors applied after GEMV.
    /// Learned during training; default to 1.0 / in_features for random init.
    pub scales: Vec<f32>,

    /// Per-output biases applied after scaling.
    pub biases: Vec<f32>,

    /// The activation function applied to this layer's output.
    pub activation: Activation,
}

impl TernaryLayer {
    /// Creates a TernaryLayer from raw ternary weights (values in {-1, 0, 1}).
    ///
    /// `raw_weights` must have length `in_features × out_features`.
    pub fn from_raw_weights(
        raw_weights:  &[i8],
        in_features:  usize,
        out_features: usize,
        scales:       Vec<f32>,
        biases:       Vec<f32>,
        activation:   Activation,
    ) -> Self {
        assert_eq!(raw_weights.len(), in_features * out_features);
        assert_eq!(scales.len(), out_features);
        assert_eq!(biases.len(), out_features);

        Self {
            weights:     pack_weights(raw_weights),
            in_features,
            out_features,
            scales,
            biases,
            activation,
        }
    }

    /// Creates a randomly initialized TernaryLayer for testing.
    ///
    /// Ternary weights drawn uniformly from {-1, 0, +1} with equal probability.
    /// Scales set to 1.0 / in_features (standard initialization).
    /// Biases set to 0.0.
    pub fn random(
        in_features:  usize,
        out_features: usize,
        activation:   Activation,
        rng:          &mut crate::rng::BrainRng,
    ) -> Self {
        let n = in_features * out_features;
        let raw_weights: Vec<i8> = (0..n)
            .map(|_| {
                // Uniform distribution over {-1, 0, +1}
                match rng.gen_index(3) {
                    0 => -1i8,
                    1 =>  0i8,
                    _ =>  1i8,
                }
            })
            .collect();

        let scale_val = 1.0 / in_features as f32;
        let scales = vec![scale_val; out_features];
        let biases = vec![0.0_f32;   out_features];

        Self::from_raw_weights(&raw_weights, in_features, out_features, scales, biases, activation)
    }

    /// Runs the forward pass for a hidden layer (ReLU → i8 output).
    ///
    /// Input:  `&[i8]` of length `in_features`
    /// Output: `Vec<i8>` of length `out_features`
    ///
    /// Steps:
    ///   1. ternary_gemv → i32 accumulators
    ///   2. apply_scale_relu_quantize for each output
    pub fn forward_hidden(&self, input: &[i8]) -> Vec<i8> {
        debug_assert_eq!(self.activation, Activation::ReluQuantize);
        debug_assert_eq!(input.len(), self.in_features);

        let mut accumulators = vec![0i32; self.out_features];
        ternary_gemv(input, &self.weights, &mut accumulators, self.in_features, self.out_features);

        accumulators
            .iter()
            .enumerate()
            .map(|(i, &acc)| apply_scale_relu_quantize(acc, self.scales[i], self.biases[i]))
            .collect()
    }

    /// Runs the forward pass for the final encoder layer (tanh → f32 output).
    ///
    /// Input:  `&[i8]` of length `in_features`
    /// Output: `Vec<f32>` of length `out_features`
    pub fn forward_final(&self, input: &[i8]) -> Vec<f32> {
        debug_assert_eq!(self.activation, Activation::TanhF32);
        debug_assert_eq!(input.len(), self.in_features);

        let mut accumulators = vec![0i32; self.out_features];
        ternary_gemv(input, &self.weights, &mut accumulators, self.in_features, self.out_features);

        accumulators
            .iter()
            .enumerate()
            .map(|(i, &acc)| apply_scale_tanh(acc, self.scales[i], self.biases[i]))
            .collect()
    }

    /// Returns the number of packed weight bytes.
    pub fn weight_bytes(&self) -> usize {
        self.weights.len()
    }

    /// Returns the approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.weight_bytes()
            + self.scales.len() * 4
            + self.biases.len() * 4
    }
}

// ── Appended to lion_core/src/ternary.rs ─────────────────────────────────────

use crate::constants::GEMV_SIMD_THRESHOLD;

// =============================================================================
// SIMD-FRIENDLY TERNARY GEMV
// =============================================================================

/// Pre-unpacks a single weight row into a scratch buffer of i8 values.
///
/// The scratch buffer is reused across output neurons to avoid re-allocation.
///
/// # Parameters
/// - `weights`      — packed ternary weight array
/// - `row`          — which output neuron's row to unpack (0-indexed)
/// - `in_features`  — number of input features
/// - `scratch`      — output buffer of length `in_features`, overwritten
#[inline]
pub fn unpack_weight_row(
    weights:     &[u8],
    row:         usize,
    in_features: usize,
    scratch:     &mut [i8],
) {
    debug_assert_eq!(scratch.len(), in_features);
    let row_offset = row * in_features;
    for (j, item) in scratch.iter_mut().enumerate().take(in_features) {
        *item = unpack_weight(weights, row_offset + j);
    }
}

/// SIMD-friendly ternary GEMV using a pre-unpacked weight row.
///
/// Inner loop: `acc += input[j] * scratch[j]`
///
/// This pattern is auto-vectorized by LLVM to AVX2 / NEON / SSE4 when
/// compiled with `opt-level=3` and `target-cpu=native`.
///
/// The multiply is effectively free for ternary weights:
///   x * 1  = x
///   x * -1 = -x
///   x * 0  = 0
/// LLVM replaces integer multiply by conditional add/subtract in the vectorized
/// path, matching the manual branchless variant but in SIMD width.
///
/// # Parameters
/// - `input`   — i8 activations
/// - `scratch` — pre-unpacked i8 weights for THIS output neuron's row
///
/// Returns the i32 accumulator before scale/bias.
#[inline]
pub fn gemv_row_simd_friendly(input: &[i8], scratch: &[i8]) -> i32 {
    debug_assert_eq!(input.len(), scratch.len());
    let mut acc = 0i32;
    for j in 0..input.len() {
        acc += (input[j] as i32) * (scratch[j] as i32);
    }
    acc
}

/// Dispatching ternary GEMV.
///
/// Routes to the SIMD-friendly variant when `in_features >= GEMV_SIMD_THRESHOLD`,
/// and to the branchless variant (Phase 8) for small inputs.
///
/// Callers should always use this function rather than calling either
/// variant directly, to benefit from future threshold tuning.
///
/// # Parameters
/// Same as `ternary_gemv()`.
pub fn ternary_gemv_dispatch(
    input:        &[i8],
    weights:      &[u8],
    output:       &mut [i32],
    in_features:  usize,
    out_features: usize,
) {
    if in_features >= GEMV_SIMD_THRESHOLD {
        ternary_gemv_auto(input, weights, output, in_features, out_features);
    } else {
        ternary_gemv(input, weights, output, in_features, out_features);
    }
}

/// Auto-vectorizable ternary GEMV (SIMD-friendly variant).
///
/// Uses a pre-allocated scratch buffer allocated once per call,
/// then reuses it for all output neurons.
///
/// Compile with `opt-level=3` and `target-cpu=native` for full vectorization.
pub fn ternary_gemv_auto(
    input:        &[i8],
    weights:      &[u8],
    output:       &mut [i32],
    in_features:  usize,
    out_features: usize,
) {
    debug_assert_eq!(input.len(),  in_features);
    debug_assert_eq!(output.len(), out_features);

    // One scratch allocation per GEMV call, reused for all output neurons.
    let mut scratch = vec![0i8; in_features];

    for (i, out_val) in output.iter_mut().enumerate().take(out_features) {
        // Unpack this output neuron's weight row.
        unpack_weight_row(weights, i, in_features, &mut scratch);

        // LLVM auto-vectorizes this loop to AVX2 / NEON.
        *out_val = gemv_row_simd_friendly(input, &scratch);
    }
}

/// Hint to the compiler that a slice length is a compile-time-known multiple.
///
/// Wraps the inner loop body to allow LLVM to assume aligned access,
/// enabling better vectorization for layer sizes that are multiples of 8.
///
/// Only effective with `opt-level >= 2`.
#[inline(always)]
pub fn assume_len_multiple_of_8(slice: &[i8]) -> &[i8] {
    let len = slice.len();
    let rounded = len - (len % 8);
    &slice[..rounded]
}
