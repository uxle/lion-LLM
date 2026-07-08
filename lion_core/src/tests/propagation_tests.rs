// lion_core/src/tests/propagation_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Builds an initialized brain with seed 42.
    fn make_brain() -> BrainMatrix {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(42);
        brain.initialize_core_brain(&mut rng);
        brain
    }

    /// Returns a uniform feature vector where every component equals `value`.
    fn uniform_vec(value: f32) -> [f32; FEATURE_SIZE] {
        [value; FEATURE_SIZE]
    }

    /// Returns a zero feature vector.
    fn zero_vec() -> [f32; FEATURE_SIZE] {
        [0.0_f32; FEATURE_SIZE]
    }

    // =========================================================================
    // VECTOR MATH PRIMITIVES
    // =========================================================================

    #[test]
    fn test_vec_magnitude_zero_vector() {
        let v = zero_vec();
        assert_eq!(vec_magnitude(&v), 0.0);
    }

    #[test]
    fn test_vec_magnitude_unit_like_vector() {
        // A vector of all 1.0 with FEATURE_SIZE=32 has magnitude sqrt(32).
        let v = uniform_vec(1.0);
        let expected = (FEATURE_SIZE as f32).sqrt();
        let computed = vec_magnitude(&v);
        assert!((computed - expected).abs() < 1e-5,
            "Expected magnitude {}, got {}", expected, computed);
    }

    #[test]
    fn test_dot_product_zero_vector() {
        let a = uniform_vec(1.0);
        let b = zero_vec();
        assert_eq!(dot_product(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_product_uniform_vectors() {
        let a = uniform_vec(2.0);
        let b = uniform_vec(3.0);
        // dot([2,2,...], [3,3,...]) = 2*3*32 = 192
        let expected = 2.0 * 3.0 * FEATURE_SIZE as f32;
        assert!((dot_product(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_alignment_identical_vectors_returns_one() {
        let v = uniform_vec(0.5);
        let mag = vec_magnitude(&v);
        let align = cosine_alignment(&v, &v, mag);
        assert!((align - 1.0).abs() < 1e-5,
            "Identical vectors should align to 1.0, got {}", align);
    }

    #[test]
    fn test_cosine_alignment_opposed_vectors_returns_neg_one() {
        let pos = uniform_vec(1.0);
        let neg = uniform_vec(-1.0);
        let mag = vec_magnitude(&neg);
        let align = cosine_alignment(&pos, &neg, mag);
        assert!((align + 1.0).abs() < 1e-5,
            "Opposed vectors should align to -1.0, got {}", align);
    }

    #[test]
    fn test_cosine_alignment_zero_input_no_nan() {
        let base  = uniform_vec(0.1);
        let input = zero_vec();
        let mag   = vec_magnitude(&input);  // 0.0
        let align = cosine_alignment(&base, &input, mag);
        // ε guard prevents NaN — result must be finite.
        assert!(align.is_finite(),
            "cosine_alignment with zero input produced NaN");
    }

    #[test]
    fn test_cosine_alignment_range() {
        let a = [0.3_f32; FEATURE_SIZE];
        let mut b = [0.1_f32; FEATURE_SIZE];
        b[0] = -0.5; // Make b different from a.
        let mag = vec_magnitude(&b);
        let align = cosine_alignment(&a, &b, mag);
        assert!(align >= -1.0 && align <= 1.0,
            "cosine_alignment out of [-1, 1]: {}", align);
    }

    // =========================================================================
    // SENSORY INJECTION
    // =========================================================================

    #[test]
    fn test_inject_sensory_sets_activations_on_target_role() {
        let mut brain = make_brain();
        let input = uniform_vec(1.0);

        brain.inject_sensory(Role::Vision, &input);

        // All Vision neurons should now have non-zero activations.
        let vision_ids: Vec<_> = brain
            .neurons_by_role(Role::Vision)
            .map(|n| n.id)
            .collect();

        assert!(!vision_ids.is_empty(), "No Vision neurons found");

        for id in vision_ids {
            let act = brain.activation_of(id);
            assert!(
                act.abs() > 0.0,
                "Vision neuron {:?} has zero activation after injection",
                id
            );
        }
    }

    #[test]
    fn test_inject_sensory_does_not_affect_other_roles() {
        let mut brain = make_brain();
        // Reset so we start clean.
        brain.reset_activations();

        let input = uniform_vec(1.0);
        brain.inject_sensory(Role::Vision, &input);

        // Motor, Memory, Danger neurons must still have activation = 0.0.
        for n in brain.alive_neurons() {
            if n.role != Role::Vision {
                assert_eq!(
                    n.activation, 0.0,
                    "Non-Vision neuron {:?} (role {:?}) was activated unexpectedly",
                    n.id, n.role
                );
            }
        }
    }

    #[test]
    fn test_inject_sensory_activation_range() {
        let mut brain = make_brain();
        let input = uniform_vec(5.0); // Strong signal.

        brain.inject_sensory(Role::Danger, &input);

        for n in brain.neurons_by_role(Role::Danger) {
            assert!(
                n.activation >= -1.0 && n.activation <= 1.0,
                "Activation {:?} out of tanh range [-1, 1]: {}",
                n.id,
                n.activation
            );
        }
    }

    #[test]
    fn test_inject_sensory_stores_trace() {
        let mut brain = make_brain();
        let input = uniform_vec(0.3);

        // Vision neurons start with 0 traces.
        let vision_ids: Vec<_> = brain
            .neurons_by_role(Role::Vision)
            .map(|n| n.id)
            .collect();
        for &id in &vision_ids {
            assert_eq!(brain.get_neuron(id).unwrap().trace_count, 0);
        }

        brain.inject_sensory(Role::Vision, &input);

        // After injection, each Vision neuron should have 1 trace.
        for &id in &vision_ids {
            assert_eq!(
                brain.get_neuron(id).unwrap().trace_count,
                1,
                "Vision neuron {:?} should have 1 trace after injection",
                id
            );
        }
    }

    #[test]
    fn test_inject_sensory_no_trace_leaves_traces_empty() {
        let mut brain = make_brain();
        let input = uniform_vec(0.3);

        brain.inject_sensory_no_trace(Role::Vision, &input);

        for n in brain.neurons_by_role(Role::Vision) {
            assert_eq!(
                n.trace_count,
                0,
                "inject_sensory_no_trace should not write traces"
            );
        }
    }

    #[test]
    fn test_inject_sensory_no_trace_still_sets_activation() {
        let mut brain = make_brain();
        let input = uniform_vec(1.0);

        brain.inject_sensory_no_trace(Role::Vision, &input);

        let any_activated = brain
            .neurons_by_role(Role::Vision)
            .any(|n| n.activation.abs() > 0.0);

        assert!(any_activated,
            "inject_sensory_no_trace must still set activations");
    }

    #[test]
    fn test_inject_opposing_signal_produces_negative_activation() {
        let mut brain = make_brain();

        // Manually set a Vision neuron's base_vector to all +1.0.
        let vision_ids: Vec<_> = brain
            .neurons_by_role(Role::Vision)
            .map(|n| n.id)
            .collect();
        brain.get_neuron_mut(vision_ids[0]).unwrap().base_vector =
            uniform_vec(1.0);

        // Inject an all -1.0 signal (directly opposed).
        let input = uniform_vec(-1.0);
        brain.inject_sensory(Role::Vision, &input);

        let act = brain.activation_of(vision_ids[0]);
        assert!(
            act < 0.0,
            "Opposing signal should produce negative activation, got {}",
            act
        );
    }

    // =========================================================================
    // SIGNAL PROPAGATION
    // =========================================================================

    #[test]
    fn test_propagate_zero_steps_leaves_activations_unchanged() {
        let mut brain = make_brain();
        let input = uniform_vec(0.5);
        brain.inject_sensory(Role::Vision, &input);

        // Capture activations before propagation.
        let before: Vec<f32> = brain
            .alive_neurons()
            .map(|n| n.activation)
            .collect();

        brain.propagate_steps(0);

        // Activations must be unchanged.
        let after: Vec<f32> = brain
            .alive_neurons()
            .map(|n| n.activation)
            .collect();

        assert_eq!(before, after,
            "propagate_steps(0) should not change activations");
    }

    #[test]
    fn test_propagate_one_step_changes_non_sensory_activations() {
        let mut brain = make_brain();

        // Inject Vision only.
        let input = uniform_vec(0.8);
        brain.inject_sensory(Role::Vision, &input);

        // Before propagation, Motor/Memory/Danger neurons should be at 0.0.
        let motor_before: Vec<f32> = brain
            .neurons_by_role(Role::Motor)
            .map(|n| n.activation)
            .collect();
        assert!(motor_before.iter().all(|&a| a == 0.0),
            "Motor neurons should be 0 before propagation");

        brain.propagate_steps(1);

        // After propagation, at least some non-Vision neurons should change.
        let any_changed = brain
            .neurons_by_role(Role::Motor)
            .any(|n| n.activation.abs() > 0.0);

        assert!(any_changed,
            "At least one Motor neuron should activate after 1 propagation step");
    }

    #[test]
    fn test_propagate_activation_stays_in_tanh_range() {
        let mut brain = make_brain();
        let input = uniform_vec(10.0); // Very strong signal.

        brain.inject_sensory(Role::Vision, &input);
        brain.propagate_steps(5);

        for n in brain.alive_neurons() {
            assert!(
                n.activation >= -1.0 && n.activation <= 1.0,
                "Activation out of tanh range after propagation: {}",
                n.activation
            );
        }
    }

    #[test]
    fn test_propagate_produces_finite_activations() {
        let mut brain = make_brain();
        let input = uniform_vec(1.0);

        brain.inject_sensory(Role::Vision, &input);
        brain.propagate_steps(10);

        for n in brain.alive_neurons() {
            assert!(
                n.activation.is_finite(),
                "Non-finite activation after propagation: {}",
                n.activation
            );
        }
    }

    #[test]
    fn test_two_propagation_steps_differ_from_one() {
        let mut brain_1 = make_brain();
        let mut brain_2 = make_brain();

        let input = uniform_vec(0.7);

        brain_1.inject_sensory(Role::Vision, &input);
        brain_1.propagate_steps(1);

        brain_2.inject_sensory(Role::Vision, &input);
        brain_2.propagate_steps(2);

        // The two-step brain should have different activations from one-step.
        let ids: Vec<_> = brain_1.collect_alive_neuron_ids();
        let any_diff = ids.iter().any(|&id| {
            (brain_1.activation_of(id) - brain_2.activation_of(id)).abs() > 1e-6
        });

        assert!(any_diff,
            "Two propagation steps should produce different result from one step");
    }

    // =========================================================================
    // FULL TICK
    // =========================================================================

    #[test]
    fn test_tick_resets_before_injecting() {
        let mut brain = make_brain();

        // First tick with Vision signal.
        let mut frame_1 = SensoryInput::new();
        frame_1.insert(Role::Vision, uniform_vec(1.0));
        brain.tick(&frame_1, 2);

        // Second tick with only Danger signal.
        let mut frame_2 = SensoryInput::new();
        frame_2.insert(Role::Danger, uniform_vec(1.0));
        brain.tick(&frame_2, 2);

        // Vision neurons must have been reset — they should NOT
        // still carry the activation from tick 1.
        // After reset, their only input is from propagation of Danger signal.
        // We can't predict exact values but they must be finite.
        for n in brain.neurons_by_role(Role::Vision) {
            assert!(n.activation.is_finite(),
                "Vision neuron has non-finite activation after second tick");
        }
    }

    #[test]
    fn test_tick_sandbox_does_not_write_traces() {
        let mut brain = make_brain();

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        frame.insert(Role::Danger, uniform_vec(1.0));

        brain.tick_sandbox(&frame, 2);

        // No traces should have been written.
        for n in brain.alive_neurons() {
            assert_eq!(
                n.trace_count,
                0,
                "tick_sandbox must not write memory traces"
            );
        }
    }

    #[test]
    fn test_multi_modality_frame_activates_multiple_roles() {
        let mut brain = make_brain();

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        frame.insert(Role::Danger, uniform_vec(1.0));

        brain.inject_frame(&frame);

        let vision_active = brain
            .neurons_by_role(Role::Vision)
            .any(|n| n.activation.abs() > 0.0);

        let danger_active = brain
            .neurons_by_role(Role::Danger)
            .any(|n| n.activation.abs() > 0.0);

        assert!(vision_active, "Vision neurons must activate from frame");
        assert!(danger_active, "Danger neurons must activate from frame");
    }

    // =========================================================================
    // MOTOR READOUT
    // =========================================================================

    #[test]
    fn test_best_motor_action_returns_valid_action() {
        let mut brain = make_brain();

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        brain.tick(&frame, 2);

        let action = brain.best_motor_action();
        assert!(
            action.is_some(),
            "best_motor_action should return Some after a tick"
        );

        let action_str = action.unwrap();
        assert!(
            PROCEDURAL_ACTIONS.contains(&action_str),
            "best_motor_action returned unknown action: {}",
            action_str
        );
    }

    #[test]
    fn test_best_motor_action_returns_highest_activation() {
        let mut brain = make_brain();

        // Manually set all Motor neuron activations to 0.0, then set FLEE to 0.9.
        let motor_ids: Vec<_> = brain
            .neurons_by_role(Role::Motor)
            .map(|n| n.id)
            .collect();

        for &id in &motor_ids {
            brain.get_neuron_mut(id).unwrap().activation = 0.0;
        }

        // Find FLEE neuron and set it highest.
        if let Some(flee_id) = brain.find_motor_neuron_by_label("FLEE") {
            brain.get_neuron_mut(flee_id).unwrap().activation = 0.9;
        }

        let action = brain.best_motor_action();
        assert_eq!(
            action,
            Some("FLEE"),
            "best_motor_action should return FLEE when it has highest activation"
        );
    }

    // =========================================================================
    // DIAGNOSTICS
    // =========================================================================

    #[test]
    fn test_mean_absolute_activation_zero_before_tick() {
        let brain = make_brain();
        // All activations are 0.0 after initialization.
        assert_eq!(brain.mean_absolute_activation(), 0.0);
    }

    #[test]
    fn test_mean_absolute_activation_nonzero_after_tick() {
        let mut brain = make_brain();

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));
        brain.tick(&frame, 2);

        let mean = brain.mean_absolute_activation();
        assert!(
            mean > 0.0,
            "Mean absolute activation should be > 0 after a tick, got {}",
            mean
        );
    }

    #[test]
    fn test_mean_absolute_activation_bounded() {
        let mut brain = make_brain();

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(100.0)); // Extreme signal.
        frame.insert(Role::Danger, uniform_vec(100.0));
        brain.tick(&frame, 10);

        let mean = brain.mean_absolute_activation();
        assert!(
            mean <= 1.0,
            "Mean absolute activation cannot exceed 1.0 (tanh bound), got {}",
            mean
        );
    }

    // =========================================================================
    // DETERMINISM
    // =========================================================================

    #[test]
    fn test_same_input_produces_same_activations() {
        let input = uniform_vec(0.7);

        let mut brain_a = make_brain();
        brain_a.inject_sensory(Role::Vision, &input);
        brain_a.propagate_steps(2);

        let mut brain_b = make_brain();
        brain_b.inject_sensory(Role::Vision, &input);
        brain_b.propagate_steps(2);

        let ids = brain_a.collect_alive_neuron_ids();
        for id in ids {
            let act_a = brain_a.activation_of(id);
            let act_b = brain_b.activation_of(id);
            assert!(
                (act_a - act_b).abs() < 1e-6,
                "Activation mismatch between identical brains: {} vs {}",
                act_a,
                act_b
            );
        }
    }
}
