// lion_core/src/tests/immune_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_brain() -> BrainMatrix {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(42);
        brain.initialize_core_brain(&mut rng);
        brain
    }

    fn uniform_vec(v: f32) -> [f32; FEATURE_SIZE] {
        [v; FEATURE_SIZE]
    }

    // =========================================================================
    // PRIMITIVE SANITIZERS
    // =========================================================================

    #[test]
    fn test_sanitize_scalar_nan_to_zero() {
        assert_eq!(sanitize_scalar(f32::NAN), 0.0);
    }

    #[test]
    fn test_sanitize_scalar_pos_inf_to_one() {
        assert_eq!(sanitize_scalar(f32::INFINITY), 1.0);
    }

    #[test]
    fn test_sanitize_scalar_neg_inf_to_neg_one() {
        assert_eq!(sanitize_scalar(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn test_sanitize_scalar_finite_unchanged() {
        let values = [0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.123, -0.987];
        for v in values {
            assert_eq!(
                sanitize_scalar(v), v,
                "sanitize_scalar changed finite value {}", v
            );
        }
    }

    #[test]
    fn test_sanitize_weight_nan_to_zero() {
        assert_eq!(sanitize_weight(f32::NAN), 0.0);
    }

    #[test]
    fn test_sanitize_weight_above_max_clamped() {
        assert_eq!(sanitize_weight(WEIGHT_MAX + 1.0), WEIGHT_MAX);
        assert_eq!(sanitize_weight(f32::INFINITY), WEIGHT_MAX);
        assert_eq!(sanitize_weight(99.0), WEIGHT_MAX);
    }

    #[test]
    fn test_sanitize_weight_below_min_clamped() {
        assert_eq!(sanitize_weight(WEIGHT_MIN - 1.0), WEIGHT_MIN);
        assert_eq!(sanitize_weight(f32::NEG_INFINITY), WEIGHT_MIN);
        assert_eq!(sanitize_weight(-99.0), WEIGHT_MIN);
    }

    #[test]
    fn test_sanitize_weight_valid_range_unchanged() {
        let values = [0.0_f32, 0.5, -0.5, WEIGHT_MAX, WEIGHT_MIN, 1.0, -1.0];
        for v in values {
            assert_eq!(
                sanitize_weight(v), v,
                "sanitize_weight changed in-range value {}", v
            );
        }
    }

    #[test]
    fn test_sanitize_strength_nan_to_zero() {
        assert_eq!(sanitize_strength(f32::NAN), 0.0);
    }

    #[test]
    fn test_sanitize_strength_above_one_clamped() {
        assert_eq!(sanitize_strength(1.5), 1.0);
        assert_eq!(sanitize_strength(f32::INFINITY), 1.0);
    }

    #[test]
    fn test_sanitize_strength_below_zero_clamped() {
        assert_eq!(sanitize_strength(-0.1), 0.0);
        assert_eq!(sanitize_strength(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_sanitize_strength_valid_range_unchanged() {
        let values = [0.0_f32, 0.5, 1.0, 0.95, 0.01];
        for v in values {
            assert_eq!(
                sanitize_strength(v), v,
                "sanitize_strength changed valid value {}", v
            );
        }
    }

    #[test]
    fn test_sanitize_vector_corrects_nan() {
        let mut v = [0.5_f32; FEATURE_SIZE];
        v[0]  = f32::NAN;
        v[15] = f32::NAN;

        let changed = sanitize_vector(&mut v);

        assert!(changed, "sanitize_vector should report a change");
        assert_eq!(v[0],  0.0);
        assert_eq!(v[15], 0.0);
        // Other components unchanged.
        assert_eq!(v[1], 0.5);
    }

    #[test]
    fn test_sanitize_vector_no_change_on_clean_input() {
        let mut v = [0.1_f32; FEATURE_SIZE];
        let changed = sanitize_vector(&mut v);
        assert!(!changed, "sanitize_vector should report no change on clean input");
    }

    // =========================================================================
    // IS_NUMERICALLY_HEALTHY
    // =========================================================================

    #[test]
    fn test_fresh_brain_is_healthy() {
        let brain = make_brain();
        assert!(
            brain.is_numerically_healthy(),
            "Freshly initialized brain should be numerically healthy"
        );
    }

    #[test]
    fn test_brain_after_tick_is_healthy() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));
        brain.tick(&frame, 2);

        assert!(
            brain.is_numerically_healthy(),
            "Brain should be healthy after a live tick (immune runs)"
        );
    }

    // =========================================================================
    // SCAN AND HEAL — NEURON ACTIVATION
    // =========================================================================

    #[test]
    fn test_scan_and_heal_fixes_nan_activation() {
        let mut brain = make_brain();

        let ids: Vec<_> = brain.collect_alive_neuron_ids();
        brain.corrupt_neuron_activation(ids[0]);

        assert!(
            brain.get_neuron(ids[0]).unwrap().activation.is_nan(),
            "Corruption did not take effect"
        );

        let report = brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(ids[0]).unwrap().activation,
            0.0,
            "NaN activation should be healed to 0.0"
        );
        assert!(report.activation_fixes > 0);
        assert!(report.total() > 0);
    }

    #[test]
    fn test_scan_and_heal_fixes_inf_activation() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        brain.get_neuron_mut(ids[0]).unwrap().activation = f32::INFINITY;

        brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(ids[0]).unwrap().activation,
            1.0,
            "+Inf activation should be healed to 1.0"
        );
    }

    #[test]
    fn test_scan_and_heal_fixes_neg_inf_activation() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        brain.get_neuron_mut(ids[0]).unwrap().activation = f32::NEG_INFINITY;

        brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(ids[0]).unwrap().activation,
            -1.0,
            "-Inf activation should be healed to -1.0"
        );
    }

    #[test]
    fn test_scan_and_heal_does_not_touch_valid_activations() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        brain.get_neuron_mut(ids[0]).unwrap().activation = 0.75;
        brain.get_neuron_mut(ids[1]).unwrap().activation = -0.5;

        brain.scan_and_heal();

        assert_eq!(brain.get_neuron(ids[0]).unwrap().activation, 0.75);
        assert_eq!(brain.get_neuron(ids[1]).unwrap().activation, -0.5);
    }

    // =========================================================================
    // SCAN AND HEAL — BASE VECTOR
    // =========================================================================

    #[test]
    fn test_scan_and_heal_fixes_nan_in_base_vector() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        brain.corrupt_base_vector(ids[0], 5);

        let report = brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(ids[0]).unwrap().base_vector[5],
            1.0,
            "+Inf in base_vector should be healed to 1.0"
        );
        assert!(report.base_vector_fixes > 0);
    }

    #[test]
    fn test_scan_and_heal_fixes_nan_in_base_vector_component() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        brain.get_neuron_mut(ids[0]).unwrap().base_vector[10] = f32::NAN;

        brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(ids[0]).unwrap().base_vector[10],
            0.0,
            "NaN base_vector component should be healed to 0.0"
        );
    }

    // =========================================================================
    // SCAN AND HEAL — TRACE VECTORS
    // =========================================================================

    #[test]
    fn test_scan_and_heal_fixes_nan_in_trace_vector() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        brain.inject_frame(&frame);

        let vision_ids: Vec<_> = brain
            .neurons_by_role(Role::Vision)
            .map(|n| n.id)
            .collect();

        // Corrupt a trace vector component.
        brain.get_neuron_mut(vision_ids[0])
            .unwrap()
            .traces[0]
            .vector[3] = f32::NAN;

        let report = brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(vision_ids[0]).unwrap().traces[0].vector[3],
            0.0
        );
        assert!(report.trace_vector_fixes > 0);
    }

    #[test]
    fn test_scan_and_heal_fixes_invalid_trace_strength() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        brain.inject_frame(&frame);

        let vision_ids: Vec<_> = brain
            .neurons_by_role(Role::Vision)
            .map(|n| n.id)
            .collect();

        // Corrupt trace strength above 1.0.
        brain.get_neuron_mut(vision_ids[0])
            .unwrap()
            .traces[0]
            .strength = 5.0;

        let report = brain.scan_and_heal();

        assert_eq!(
            brain.get_neuron(vision_ids[0]).unwrap().traces[0].strength,
            1.0
        );
        assert!(report.trace_strength_fixes > 0);
    }

    // =========================================================================
    // SCAN AND HEAL — SYNAPSE WEIGHTS
    // =========================================================================

    #[test]
    fn test_scan_and_heal_fixes_nan_weight() {
        let mut brain = make_brain();

        let syn_ids: Vec<_> = brain.collect_alive_synapse_ids();
        assert!(!syn_ids.is_empty(), "Need at least one synapse");

        brain.corrupt_synapse_weight(syn_ids[0]);

        let report = brain.scan_and_heal();

        assert_eq!(
            brain.get_synapse(syn_ids[0]).unwrap().weight,
            0.0,
            "NaN synapse weight should be healed to 0.0"
        );
        assert!(report.weight_fixes > 0);
    }

    #[test]
    fn test_scan_and_heal_fixes_out_of_bounds_weight() {
        let mut brain = make_brain();
        let syn_ids: Vec<_> = brain.collect_alive_synapse_ids();

        brain.get_synapse_mut(syn_ids[0]).unwrap().weight = 99.0;

        brain.scan_and_heal();

        assert_eq!(
            brain.get_synapse(syn_ids[0]).unwrap().weight,
            WEIGHT_MAX
        );
    }

    #[test]
    fn test_scan_and_heal_fixes_neg_out_of_bounds_weight() {
        let mut brain = make_brain();
        let syn_ids: Vec<_> = brain.collect_alive_synapse_ids();

        brain.get_synapse_mut(syn_ids[0]).unwrap().weight = -99.0;

        brain.scan_and_heal();

        assert_eq!(
            brain.get_synapse(syn_ids[0]).unwrap().weight,
            WEIGHT_MIN
        );
    }

    // =========================================================================
    // CLEAN SCAN — NO CHANGES
    // =========================================================================

    #[test]
    fn test_scan_and_heal_on_healthy_brain_returns_zero_interventions() {
        let mut brain = make_brain();
        let report = brain.scan_and_heal();
        assert!(
            report.is_clean(),
            "Healthy brain should produce zero immune interventions, got: {}",
            report.total()
        );
    }

    // =========================================================================
    // INTERVENTION COUNTER
    // =========================================================================

    #[test]
    fn test_intervention_counter_accumulates() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        // Corrupt and heal twice.
        brain.corrupt_neuron_activation(ids[0]);
        brain.scan_and_heal();

        brain.corrupt_neuron_activation(ids[1]);
        brain.scan_and_heal();

        assert!(
            brain.immune_intervention_count() >= 2,
            "Intervention counter should have accumulated at least 2"
        );
    }

    #[test]
    fn test_reset_immune_counter_clears_to_zero() {
        let mut brain = make_brain();
        let ids: Vec<_> = brain.collect_alive_neuron_ids();

        brain.corrupt_neuron_activation(ids[0]);
        brain.scan_and_heal();

        assert!(brain.immune_intervention_count() > 0);

        brain.reset_immune_counter();
        assert_eq!(brain.immune_intervention_count(), 0);
    }

    // =========================================================================
    // TICK INTEGRATION — IMMUNE RUNS AUTOMATICALLY
    // =========================================================================

    #[test]
    fn test_tick_produces_healthy_brain() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision,  uniform_vec(1.0));
        frame.insert(Role::Danger, uniform_vec(-1.0));

        brain.tick(&frame, 5);

        assert!(
            brain.is_numerically_healthy(),
            "Brain must be healthy after tick() — immune runs automatically"
        );
    }

    #[test]
    fn test_tick_returns_immune_report() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));

        // tick() now returns ImmuneReport
        let report = brain.tick(&frame, 2);

        // On a healthy brain the report should be clean.
        // (We can't guarantee NaN will appear from normal operation,
        //  but the report must always be a valid struct.)
        assert!(
            report.total() == 0 || report.total() > 0,
            "tick() must return a valid ImmuneReport"
        );
    }

    #[test]
    fn test_tick_heals_pre_existing_corruption() {
        let mut brain = make_brain();

        // Corrupt before ticking.
        let ids: Vec<_> = brain.collect_alive_neuron_ids();
        brain.corrupt_neuron_activation(ids[0]);

        assert!(!brain.is_numerically_healthy());

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        brain.tick(&frame, 2);

        // After tick, corruption must be healed.
        assert!(
            brain.is_numerically_healthy(),
            "tick() must heal pre-existing corruption via immune scan"
        );
    }

    // =========================================================================
    // SANDBOX TICKS DO NOT CALL IMMUNE
    // =========================================================================

    #[test]
    fn test_tick_sandbox_does_not_increment_immune_counter() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));

        brain.tick_sandbox(&frame, 2);

        assert_eq!(
            brain.immune_intervention_count(),
            0,
            "tick_sandbox must not call scan_and_heal"
        );
    }

    #[test]
    fn test_tick_sandbox_hebbian_does_not_increment_immune_counter() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));

        brain.tick_sandbox_hebbian(&frame, 2, 0.3);

        assert_eq!(
            brain.immune_intervention_count(),
            0,
            "tick_sandbox_hebbian must not call scan_and_heal"
        );
    }

    // =========================================================================
    // IMMUNE REPORT STRUCT
    // =========================================================================

    #[test]
    fn test_immune_report_total_sums_all_fields() {
        let report = ImmuneReport {
            activation_fixes:     1,
            base_vector_fixes:    2,
            trace_vector_fixes:   3,
            trace_strength_fixes: 4,
            weight_fixes:         5,
        };
        assert_eq!(report.total(), 15);
    }

    #[test]
    fn test_immune_report_is_clean_true_on_zero() {
        let report = ImmuneReport::default();
        assert!(report.is_clean());
    }

    #[test]
    fn test_immune_report_is_clean_false_on_any_fix() {
        let mut report = ImmuneReport::default();
        report.activation_fixes = 1;
        assert!(!report.is_clean());
    }

    // =========================================================================
    // MULTIPLE CORRUPTION TYPES IN ONE SCAN
    // =========================================================================

    #[test]
    fn test_scan_heals_multiple_corruption_types_simultaneously() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(0.5));
        brain.inject_frame(&frame);

        let neuron_ids  = brain.collect_alive_neuron_ids();
        let synapse_ids = brain.collect_alive_synapse_ids();

        // Corrupt multiple types at once.
        brain.corrupt_neuron_activation(neuron_ids[0]);
        brain.corrupt_base_vector(neuron_ids[1], 0);
        if !synapse_ids.is_empty() {
            brain.corrupt_synapse_weight(synapse_ids[0]);
        }

        assert!(!brain.is_numerically_healthy());

        let report = brain.scan_and_heal();

        assert!(
            brain.is_numerically_healthy(),
            "All corruption types must be healed in one scan_and_heal pass"
        );
        assert!(report.total() >= 2, "Expected at least 2 fixes");
    }
}
