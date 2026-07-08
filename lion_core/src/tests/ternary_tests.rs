// lion_core/src/tests/ternary_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_rng() -> BrainRng {
        BrainRng::from_seed(42)
    }

    // =========================================================================
    // PACKING / UNPACKING
    // =========================================================================

    #[test]
    fn test_pack_and_unpack_roundtrip_all_values() {
        let raw = vec![-1i8, 0, 1, -1, 1, 0, 1, -1];
        let packed = pack_weights(&raw);

        for (i, &expected) in raw.iter().enumerate() {
            let got = unpack_weight(&packed, i);
            assert_eq!(
                got, expected,
                "Round-trip failed at index {}: expected {}, got {}",
                i, expected, got
            );
        }
    }

    #[test]
    fn test_pack_zero_weights() {
        let raw = vec![0i8; 8];
        let packed = pack_weights(&raw);
        // All zeros → all bytes should be 0x00
        assert!(packed.iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_pack_all_positive() {
        let raw = vec![1i8; 4]; // 4 weights of +1
        let packed = pack_weights(&raw);
        // Each weight = 0b01, so byte = 0b01010101 = 0x55
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x55);
    }

    #[test]
    fn test_pack_all_negative() {
        let raw = vec![-1i8; 4]; // 4 weights of -1
        let packed = pack_weights(&raw);
        // Each weight = 0b10, so byte = 0b10101010 = 0xAA
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0xAA);
    }

    #[test]
    fn test_packed_byte_count_alignment() {
        assert_eq!(packed_byte_count(4),  1); // Exactly 1 byte
        assert_eq!(packed_byte_count(5),  2); // Needs 2 bytes
        assert_eq!(packed_byte_count(8),  2); // Exactly 2 bytes
        assert_eq!(packed_byte_count(1),  1);
        assert_eq!(packed_byte_count(16), 4);
    }

    #[test]
    fn test_pack_known_pattern() {
        // w0=+1(0b01), w1=-1(0b10), w2=0(0b00), w3=+1(0b01)
        // byte = 01 | 10<<2 | 00<<4 | 01<<6
        //      = 0b01 | 0b1000 | 0b000000 | 0b01000000
        //      = 0b01001001 = 0x49
        let raw = vec![1i8, -1, 0, 1];
        let packed = pack_weights(&raw);
        assert_eq!(packed[0], 0x49);
    }

    // =========================================================================
    // TERNARY GEMV — CORRECTNESS
    // =========================================================================

    #[test]
    fn test_gemv_all_zero_weights_gives_zero_output() {
        let input   = vec![100i8; 8];
        let weights = vec![0u8; packed_byte_count(8 * 4)]; // 4 outputs
        let mut output = vec![0i32; 4];

        ternary_gemv(&input, &weights, &mut output, 8, 4);

        assert!(output.iter().all(|&x| x == 0),
            "All-zero weights must give zero output: {:?}", output);
    }

    #[test]
    fn test_gemv_identity_plus_one_weights() {
        // 1 output, 4 inputs, all weights = +1
        let raw_weights = vec![1i8; 4];
        let weights     = pack_weights(&raw_weights);
        let input       = vec![10i8, 20, 30, 40];
        let mut output  = vec![0i32; 1];

        ternary_gemv(&input, &weights, &mut output, 4, 1);

        // output[0] = 10 + 20 + 30 + 40 = 100
        assert_eq!(output[0], 100);
    }

    #[test]
    fn test_gemv_all_negative_one_weights() {
        let raw_weights = vec![-1i8; 4];
        let weights     = pack_weights(&raw_weights);
        let input       = vec![10i8, 20, 30, 40];
        let mut output  = vec![0i32; 1];

        ternary_gemv(&input, &weights, &mut output, 4, 1);

        // output[0] = -(10 + 20 + 30 + 40) = -100
        assert_eq!(output[0], -100);
    }

    #[test]
    fn test_gemv_mixed_weights() {
        // Weights: +1, -1, 0, +1
        // Input:    10,  20, 30,  40
        // output = 10 - 20 + 0 + 40 = 30
        let raw_weights = vec![1i8, -1, 0, 1];
        let weights     = pack_weights(&raw_weights);
        let input       = vec![10i8, 20, 30, 40];
        let mut output  = vec![0i32; 1];

        ternary_gemv(&input, &weights, &mut output, 4, 1);

        assert_eq!(output[0], 30);
    }

    #[test]
    fn test_gemv_multiple_output_neurons() {
        // 2 outputs, 2 inputs
        // W[0] = [+1, -1]  → out[0] = in[0] - in[1]
        // W[1] = [-1, +1]  → out[1] = -in[0] + in[1]
        let raw_weights = vec![1i8, -1, -1, 1]; // row-major
        let weights     = pack_weights(&raw_weights);
        let input       = vec![3i8, 7];
        let mut output  = vec![0i32; 2];

        ternary_gemv(&input, &weights, &mut output, 2, 2);

        assert_eq!(output[0],  3 - 7); // = -4
        assert_eq!(output[1], -3 + 7); // = +4
    }

    #[test]
    fn test_gemv_matches_naive_implementation() {
        let mut rng = make_rng();
        let in_sz   = 16;
        let out_sz  = 8;

        let raw_weights: Vec<i8> = (0..in_sz * out_sz)
            .map(|_| match rng.gen_index(3) { 0 => -1, 1 => 0, _ => 1 } as i8)
            .collect();
        let input: Vec<i8> = (0..in_sz)
            .map(|_| (rng.gen_index(255) as i8).wrapping_sub(127))
            .collect();

        // Naive reference
        let mut expected = vec![0i32; out_sz];
        for i in 0..out_sz {
            for j in 0..in_sz {
                expected[i] += (input[j] as i32) * (raw_weights[i * in_sz + j] as i32);
            }
        }

        // Ternary GEMV
        let packed = pack_weights(&raw_weights);
        let mut got = vec![0i32; out_sz];
        ternary_gemv(&input, &packed, &mut got, in_sz, out_sz);

        assert_eq!(got, expected, "ternary_gemv output doesn't match naive reference");
    }

    // =========================================================================
    // QUANTIZATION FUNCTIONS
    // =========================================================================

    #[test]
    fn test_f32_to_i8_clamps_above_one() {
        assert_eq!(f32_to_i8(2.0),  127i8);
        assert_eq!(f32_to_i8(99.0), 127i8);
    }

    #[test]
    fn test_f32_to_i8_clamps_below_neg_one() {
        assert_eq!(f32_to_i8(-2.0),  -127i8);
        assert_eq!(f32_to_i8(-99.0), -127i8);
    }

    #[test]
    fn test_f32_to_i8_zero_maps_to_zero() {
        assert_eq!(f32_to_i8(0.0), 0i8);
    }

    #[test]
    fn test_i8_to_f32_range() {
        let x = 127i8;
        let f = i8_to_f32(x);
        assert!((f - 1.0).abs() < 0.01, "i8_to_f32(127) should ≈ 1.0, got {}", f);

        let x = -127i8;
        let f = i8_to_f32(x);
        assert!((f + 1.0).abs() < 0.01, "i8_to_f32(-127) should ≈ -1.0, got {}", f);
    }

    // =========================================================================
    // TERNARY LAYER
    // =========================================================================

    #[test]
    fn test_ternary_layer_random_forward_hidden_output_length() {
        let mut rng = make_rng();
        let layer = TernaryLayer::random(16, 8, Activation::ReluQuantize, &mut rng);
        let input = vec![10i8; 16];
        let output = layer.forward_hidden(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_ternary_layer_relu_output_non_negative() {
        let mut rng = make_rng();
        let layer = TernaryLayer::random(16, 8, Activation::ReluQuantize, &mut rng);
        let input = vec![100i8; 16]; // Strong positive input.
        let output = layer.forward_hidden(&input);

        // ReLU quantized output must all be >= 0
        assert!(output.iter().all(|&x| x >= 0),
            "ReLU layer output must be non-negative: {:?}", output);
    }

    #[test]
    fn test_ternary_layer_final_output_length() {
        let mut rng = make_rng();
        let layer = TernaryLayer::random(8, FEATURE_SIZE, Activation::TanhF32, &mut rng);
        let input = vec![50i8; 8];
        let output = layer.forward_final(&input);
        assert_eq!(output.len(), FEATURE_SIZE);
    }

    #[test]
    fn test_ternary_layer_final_output_in_tanh_range() {
        let mut rng = make_rng();
        let layer = TernaryLayer::random(8, FEATURE_SIZE, Activation::TanhF32, &mut rng);
        let input = vec![127i8; 8]; // Maximum input.
        let output = layer.forward_final(&input);

        for v in output {
            assert!(
                v >= -1.0 && v <= 1.0,
                "TanhF32 output must be in [-1.0, 1.0]: {}", v
            );
        }
    }

    #[test]
    fn test_ternary_layer_memory_bytes_is_reasonable() {
        let mut rng = make_rng();
        let layer = TernaryLayer::random(64, 32, Activation::TanhF32, &mut rng);

        // 64 × 32 = 2048 weights → ceil(2048/4) = 512 bytes
        // + 32 scales × 4 = 128 bytes
        // + 32 biases × 4 = 128 bytes
        // Total = 768 bytes
        let expected_weight_bytes = packed_byte_count(64 * 32);
        assert_eq!(layer.weight_bytes(), expected_weight_bytes);

        let total = layer.memory_bytes();
        assert!(total < 1024, "Layer should fit in <1KB: {} bytes", total);
    }

    // =========================================================================
    // TERNARY ENCODER
    // =========================================================================

    #[test]
    fn test_encoder_default_small_builds_correctly() {
        let mut rng     = make_rng();
        let config      = TernaryEncoderConfig::default_small();
        let encoder     = TernaryEncoder::random(config, &mut rng);

        assert_eq!(encoder.depth(), 2);
    }

    #[test]
    fn test_encoder_encode_output_length_equals_feature_size() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
        let raw     = vec![0.5_f32; 64];
        let output  = encoder.encode_f32(&raw);
        assert_eq!(output.len(), FEATURE_SIZE);
    }

    #[test]
    fn test_encoder_output_values_are_in_tanh_range() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
        let raw     = vec![1.0_f32; 64];
        let output  = encoder.encode_f32(&raw);

        for v in output {
            assert!(
                v >= -1.0 && v <= 1.0,
                "Encoder output must be in tanh range [-1.0, 1.0]: {}", v
            );
        }
    }

    #[test]
    fn test_encoder_output_is_finite() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
        let raw     = vec![0.3_f32; 64];
        let output  = encoder.encode_f32(&raw);

        for v in output {
            assert!(v.is_finite(), "Encoder output component is not finite: {}", v);
        }
    }

    #[test]
    fn test_encoder_short_input_is_padded() {
        let mut rng = make_rng();
        let config  = TernaryEncoderConfig::default_small(); // input_size = 64
        let encoder = TernaryEncoder::random(config, &mut rng);

        // Input shorter than config.input_size — must not panic.
        let raw    = vec![0.5_f32; 10]; // Only 10, but encoder expects 64.
        let output = encoder.encode_f32(&raw);

        assert_eq!(output.len(), FEATURE_SIZE);
        for v in output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_encoder_long_input_is_truncated() {
        let mut rng = make_rng();
        let config  = TernaryEncoderConfig::default_small(); // input_size = 64
        let encoder = TernaryEncoder::random(config, &mut rng);

        // Input longer than config.input_size — must not panic.
        let raw    = vec![0.5_f32; 200];
        let output = encoder.encode_f32(&raw);

        assert_eq!(output.len(), FEATURE_SIZE);
    }

    #[test]
    fn test_encoder_is_deterministic() {
        let mut rng = BrainRng::from_seed(99);
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
        let raw     = vec![0.7_f32; 64];

        let out1 = encoder.encode_f32(&raw);
        let out2 = encoder.encode_f32(&raw);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(),
                "Encoder must be deterministic for same input: {} vs {}", a, b
            );
        }
    }

    #[test]
    fn test_encoder_different_inputs_give_different_outputs() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);

        let raw_a = vec![1.0_f32; 64];
        let raw_b = vec![0.0_f32; 64];

        let out_a = encoder.encode_f32(&raw_a);
        let out_b = encoder.encode_f32(&raw_b);

        let any_diff = out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff,
            "Different inputs must produce different encoder outputs"
        );
    }

    #[test]
    fn test_encoder_memory_is_compact() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);

        // 64→64 layer: ceil(4096/4)=1024B + 2×64×4=512B = 1536B
        // 64→32 layer: ceil(2048/4)= 512B + 2×32×4=256B = 768B
        // Total ≈ 2304 bytes ≈ 2.25 KB
        let total = encoder.total_memory_bytes();
        assert!(
            total < 4096,
            "Default small encoder should fit in <4KB: {} bytes", total
        );
    }

    // =========================================================================
    // INTEGRATION WITH SOVEREIGN
    // =========================================================================

    #[test]
    fn test_encoder_frame_feeds_into_sovereign_tick() {
        let mut rng     = BrainRng::from_seed(42);
        let config      = TernaryEncoderConfig::default_small();
        let encoder     = TernaryEncoder::random(config, &mut rng);
        let mut sovereign = Sovereign::new(42);

        // Encode raw vision data through the ternary cortex.
        let raw_vision = vec![0.8_f32; 64];
        let frame      = encoder.encode_frame(&[(Role::Vision, &raw_vision)]);

        // Feed into the sovereign tick — must not panic.
        let result = sovereign.update(&frame, 0.0);

        assert!(PROCEDURAL_ACTIONS.contains(&result.action));
        for v in result.gestalt {
            assert!(v.is_finite(), "Gestalt not finite after ternary-encoded tick: {}", v);
        }
    }

    #[test]
    fn test_encoder_encode_into_frame_inserts_correct_role() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
        let raw     = vec![0.5_f32; 64];
        let mut frame = SensoryInput::new();

        encoder.encode_into_frame(Role::Danger, &raw, &mut frame);

        assert!(frame.contains_key(&Role::Danger));
        assert!(!frame.contains_key(&Role::Vision));
    }

    // =========================================================================
    // WEIGHT SPARSITY
    // =========================================================================

    #[test]
    fn test_weight_sparsity_between_zero_and_one() {
        let mut rng = make_rng();
        let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
        let sparsity = encoder.weight_sparsity();

        assert!(
            sparsity >= 0.0 && sparsity <= 1.0,
            "Weight sparsity must be in [0.0, 1.0]: {}", sparsity
        );
    }

    #[test]
    fn test_all_zero_weights_give_full_sparsity() {
        // Build a layer where all weights are 0
        let raw   = vec![0i8; 4 * FEATURE_SIZE];
        let layer = TernaryLayer::from_raw_weights(
            &raw, 4, FEATURE_SIZE,
            vec![1.0; FEATURE_SIZE], vec![0.0; FEATURE_SIZE],
            Activation::TanhF32,
        );
        let config  = TernaryEncoderConfig::minimal(4);
        let encoder = TernaryEncoder::from_layers(vec![layer], config);

        assert_eq!(encoder.weight_sparsity(), 1.0);
    }
}
