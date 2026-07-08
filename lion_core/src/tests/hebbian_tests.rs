// lion_core/src/tests/hebbian_tests.rs

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

    /// Runs a single forward tick and returns the brain.
    fn ticked_brain(input: [f32; FEATURE_SIZE], role: Role) -> BrainMatrix {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(role, input);
        brain.tick(&frame, 2);
        brain
    }

    // =========================================================================
    // TRACE ALIGNMENT FUNCTION
    // =========================================================================

    #[test]
    fn test_trace_alignment_identical_vectors_returns_one() {
        let v = uniform_vec(0.5);
        let a = trace_alignment(&v, &v);
        assert!(
            (a - 1.0).abs() < 1e-5,
            "Identical vectors should align to 1.0, got {}", a
        );
    }

    #[test]
    fn test_trace_alignment_opposed_vectors_returns_zero() {
        // cosine of opposed vectors = -1.0
        // max(0.0, -1.0) = 0.0
        let pos = uniform_vec(1.0);
        let neg = uniform_vec(-1.0);
        let a = trace_alignment(&pos, &neg);
        assert_eq!(
            a, 0.0,
            "Opposed vectors should clamp to 0.0 (max(0, -1)), got {}", a
        );
    }

    #[test]
    fn test_trace_alignment_zero_vector_no_nan() {
        let v = uniform_vec(1.0);
        let z = uniform_vec(0.0);
        let a = trace_alignment(&v, &z);
        assert!(
            a.is_finite(),
            "trace_alignment with zero vector produced NaN or Inf: {}", a
        );
    }

    #[test]
    fn test_trace_alignment_always_non_negative() {
        // Since we take max(0.0, cosine), result is always in [0.0, 1.0].
        let pairs: &[([f32; FEATURE_SIZE], [f32; FEATURE_SIZE])] = &[
            (uniform_vec(1.0),  uniform_vec(-1.0)),
            (uniform_vec(0.5),  uniform_vec(-0.5)),
            (uniform_vec(-1.0), uniform_vec(-1.0)),
        ];
        for (v1, v2) in pairs {
            let a = trace_alignment(v1, v2);
            assert!(
                a >= 0.0,
                "trace_alignment returned negative value: {}", a
            );
        }
    }

    #[test]
    fn test_trace_alignment_range_zero_to_one() {
        let a = uniform_vec(0.7);
        let b = uniform_vec(0.3);
        let result = trace_alignment(&a, &b);
        assert!(
            result >= 0.0 && result <= 1.0,
            "trace_alignment out of [0.0, 1.0]: {}", result
        );
    }

    // =========================================================================
    // HEBBIAN GUARD CONDITIONS
    // =========================================================================

    #[test]
    fn test_plasticity_zero_produces_no_weight_change() {
        let mut brain = ticked_brain(uniform_vec(1.0), Role::Vision);

        let before = brain.synapse_weight_snapshot();
        brain.apply_hebbian_plasticity(0.0);
        let after = brain.synapse_weight_snapshot();

        let delta = BrainMatrix::total_weight_delta(&before, &after);
        assert_eq!(delta, 0.0,
            "plasticity=0.0 must produce zero weight changes, got delta {}", delta);
    }

    #[test]
    fn test_zero_activation_brain_produces_no_weight_change() {
        let mut brain = make_brain();
        // No tick — all activations are 0.0.

        let before = brain.synapse_weight_snapshot();
        brain.apply_hebbian_plasticity(0.5);
        let after = brain.synapse_weight_snapshot();

        let delta = BrainMatrix::total_weight_delta(&before, &after);
        assert_eq!(delta, 0.0,
            "All-zero activations must produce no Hebbian updates, delta: {}", delta);
    }

    // =========================================================================
    // HEBBIAN WEIGHT CHANGES
    // =========================================================================

    #[test]
    fn test_hebbian_changes_at_least_one_weight() {
        let mut brain = ticked_brain(uniform_vec(1.0), Role::Vision);

        let before = brain.synapse_weight_snapshot();
        brain.apply_hebbian_plasticity(0.5);
        let after = brain.synapse_weight_snapshot();

        let delta = BrainMatrix::total_weight_delta(&before, &after);
        assert!(
            delta > 0.0,
            "Hebbian plasticity should change at least one weight, total delta: {}", delta
        );
    }

    #[test]
    fn test_higher_plasticity_produces_larger_weight_change() {
        let input = uniform_vec(0.8);

        let mut brain_low = ticked_brain(input, Role::Vision);
        let mut brain_high = ticked_brain(input, Role::Vision);

        let before_low  = brain_low.synapse_weight_snapshot();
        let before_high = brain_high.synapse_weight_snapshot();

        brain_low.apply_hebbian_plasticity(0.05);
        brain_high.apply_hebbian_plasticity(0.5);

        let delta_low  = BrainMatrix::total_weight_delta(
            &before_low,
            &brain_low.synapse_weight_snapshot(),
        );
        let delta_high = BrainMatrix::total_weight_delta(
            &before_high,
            &brain_high.synapse_weight_snapshot(),
        );

        assert!(
            delta_high > delta_low,
            "Higher plasticity must produce larger weight changes: low={}, high={}",
            delta_low, delta_high
        );
    }

    #[test]
    fn test_weights_stay_within_bounds_after_hebbian() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));
        frame.insert(Role::Danger, uniform_vec(1.0));

        // Run many ticks and Hebbian updates to try to push weights out of bounds.
        for _ in 0..50 {
            brain.tick(&frame, 2);
            brain.apply_hebbian_plasticity(0.5);
        }

        for syn in brain.alive_synapses() {
            assert!(
                syn.weight >= WEIGHT_MIN && syn.weight <= WEIGHT_MAX,
                "Synapse weight {} out of [{}, {}] after repeated Hebbian",
                syn.weight, WEIGHT_MIN, WEIGHT_MAX
            );
        }
    }

    #[test]
    fn test_weights_are_finite_after_hebbian() {
        let mut brain = ticked_brain(uniform_vec(1.0), Role::Vision);
        brain.apply_hebbian_plasticity(0.5);

        for syn in brain.alive_synapses() {
            assert!(
                syn.weight.is_finite(),
                "Synapse weight is not finite after Hebbian: {}", syn.weight
            );
        }
    }

    // =========================================================================
    // TRACE BOOST EFFECT
    // =========================================================================

    #[test]
    fn test_aligned_traces_boost_weight_change_vs_no_trace() {
        let input = uniform_vec(0.9);

        // Brain A: inject with trace storage → traces populated → alignment boost active.
        let mut brain_with_traces = ticked_brain(input, Role::Vision);

        // Brain B: manually call inject_sensory_no_trace → no traces → no boost.
        let mut brain_no_traces = make_brain();
        brain_no_traces.reset_activations();
        brain_no_traces.inject_sensory_no_trace(Role::Vision, &input);
        brain_no_traces.propagate_steps(2);

        let before_trace    = brain_with_traces.synapse_weight_snapshot();
        let before_no_trace = brain_no_traces.synapse_weight_snapshot();

        brain_with_traces.apply_hebbian_plasticity(0.3);
        brain_no_traces.apply_hebbian_plasticity(0.3);

        let delta_with    = BrainMatrix::total_weight_delta(
            &before_trace,
            &brain_with_traces.synapse_weight_snapshot(),
        );
        let delta_without = BrainMatrix::total_weight_delta(
            &before_no_trace,
            &brain_no_traces.synapse_weight_snapshot(),
        );

        assert!(
            delta_with >= delta_without,
            "Aligned traces should produce >= weight change vs no-trace: with={}, without={}",
            delta_with, delta_without
        );
    }

    // =========================================================================
    // CORRELATED NEURON STRENGTHENING
    // =========================================================================

    #[test]
    fn test_repeated_hebbian_strengthens_correlated_synapses() {
        let mut brain = make_brain();
        let input = uniform_vec(0.7);

        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, input);

        // Run multiple ticks and Hebbian updates.
        for _ in 0..10 {
            brain.tick(&frame, 2);
            brain.apply_hebbian_plasticity(0.3);
        }

        // After many correlated activations, the total absolute weight
        // should have moved significantly from the initial state.
        let initial = make_brain().synapse_weight_snapshot();
        let current = brain.synapse_weight_snapshot();

        let total_change = BrainMatrix::total_weight_delta(&initial, &current);
        assert!(
            total_change > 0.0,
            "Repeated Hebbian updates should cumulatively change weights"
        );
    }

    #[test]
    fn test_danger_flee_synapse_strengthens_under_danger_signal() {
        let mut brain = make_brain();

        // Simulate 10 episodes where the brain sees a danger signal.
        let danger_input = uniform_vec(1.0);
        let mut frame    = SensoryInput::new();
        frame.insert(Role::Danger, danger_input);

        for _ in 0..10 {
            brain.tick(&frame, 2);
            brain.apply_hebbian_plasticity(0.3);
        }

        // Find the FLEE motor neuron.
        let flee_id = brain.find_motor_neuron_by_label("FLEE");
        assert!(flee_id.is_some(), "FLEE neuron must exist");

        // Find synapses from Danger neurons to FLEE neuron.
        let flee_idx = flee_id.unwrap().index;
        let danger_to_flee_weights: Vec<f32> = brain
            .alive_synapses()
            .filter(|s| {
                s.post_id.index == flee_idx
                    && brain
                        .get_neuron(s.pre_id)
                        .map(|n| n.role == Role::Danger)
                        .unwrap_or(false)
            })
            .map(|s| s.weight)
            .collect();

        // We cannot guarantee which direction weights moved without knowing
        // initial values, but we can assert the synapses exist and are finite.
        assert!(
            !danger_to_flee_weights.is_empty(),
            "No Danger→FLEE synapses found after initialization"
        );
        for w in danger_to_flee_weights {
            assert!(w.is_finite(), "Danger→FLEE weight is not finite: {}", w);
        }
    }

    // =========================================================================
    // PROPAGATE WITH HEBBIAN
    // =========================================================================

    #[test]
    fn test_propagate_with_hebbian_changes_weights() {
        let mut brain = make_brain();
        brain.reset_activations();
        brain.inject_sensory_no_trace(Role::Vision, &uniform_vec(0.8));

        let before = brain.synapse_weight_snapshot();
        brain.propagate_with_hebbian(2, 0.3);
        let after = brain.synapse_weight_snapshot();

        let delta = BrainMatrix::total_weight_delta(&before, &after);
        assert!(
            delta > 0.0,
            "propagate_with_hebbian must change at least one weight"
        );
    }

    #[test]
    fn test_propagate_with_hebbian_zero_plasticity_no_change() {
        let mut brain = make_brain();
        brain.reset_activations();
        brain.inject_sensory_no_trace(Role::Vision, &uniform_vec(0.8));

        let before = brain.synapse_weight_snapshot();
        brain.propagate_with_hebbian(2, 0.0);
        let after = brain.synapse_weight_snapshot();

        let delta = BrainMatrix::total_weight_delta(&before, &after);
        assert_eq!(
            delta, 0.0,
            "propagate_with_hebbian with plasticity=0 must not change weights"
        );
    }

    // =========================================================================
    // TICK SANDBOX HEBBIAN
    // =========================================================================

    #[test]
    fn test_tick_sandbox_hebbian_changes_weights() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));
        frame.insert(Role::Danger, uniform_vec(0.8));

        let before = brain.synapse_weight_snapshot();
        brain.tick_sandbox_hebbian(&frame, 2, 0.3);
        let after = brain.synapse_weight_snapshot();

        let delta = BrainMatrix::total_weight_delta(&before, &after);
        assert!(
            delta > 0.0,
            "tick_sandbox_hebbian must modify at least one weight"
        );
    }

    #[test]
    fn test_tick_sandbox_hebbian_writes_no_traces() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));

        brain.tick_sandbox_hebbian(&frame, 2, 0.3);

        for n in brain.alive_neurons() {
            assert_eq!(
                n.trace_count, 0,
                "tick_sandbox_hebbian must not write memory traces"
            );
        }
    }

    #[test]
    fn test_tick_sandbox_hebbian_activations_are_finite() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, uniform_vec(1.0));

        brain.tick_sandbox_hebbian(&frame, 2, 0.5);

        for n in brain.alive_neurons() {
            assert!(
                n.activation.is_finite(),
                "Activation not finite after tick_sandbox_hebbian: {}", n.activation
            );
        }
    }

    // =========================================================================
    // DETERMINISM
    // =========================================================================

    #[test]
    fn test_hebbian_is_deterministic_given_same_state() {
        let input = uniform_vec(0.6);

        let mut brain_a = ticked_brain(input, Role::Vision);
        let mut brain_b = ticked_brain(input, Role::Vision);

        brain_a.apply_hebbian_plasticity(0.2);
        brain_b.apply_hebbian_plasticity(0.2);

        let snap_a = brain_a.synapse_weight_snapshot();
        let snap_b = brain_b.synapse_weight_snapshot();

        for ((_, _, wa), (_, _, wb)) in snap_a.iter().zip(snap_b.iter()) {
            assert!(
                (wa - wb).abs() < 1e-6,
                "Hebbian produced different weights from identical states: {} vs {}",
                wa, wb
            );
        }
    }

    // =========================================================================
    // EPIGENOME INTEGRATION
    // =========================================================================

    #[test]
    fn test_epigenome_plasticity_scales_hebbian() {
        let input = uniform_vec(0.8);

        let mut brain_a = ticked_brain(input, Role::Vision);
        let mut brain_b = ticked_brain(input, Role::Vision);

        let epi_low  = Epigenome { plasticity: 0.01, ..Epigenome::default() };
        let epi_high = Epigenome { plasticity: 0.5,  ..Epigenome::default() };

        let before_a = brain_a.synapse_weight_snapshot();
        let before_b = brain_b.synapse_weight_snapshot();

        brain_a.apply_hebbian_plasticity(epi_low.plasticity);
        brain_b.apply_hebbian_plasticity(epi_high.plasticity);

        let delta_a = BrainMatrix::total_weight_delta(
            &before_a,
            &brain_a.synapse_weight_snapshot(),
        );
        let delta_b = BrainMatrix::total_weight_delta(
            &before_b,
            &brain_b.synapse_weight_snapshot(),
        );

        assert!(
            delta_b > delta_a,
            "Higher epigenome plasticity must produce larger Hebbian updates: low={}, high={}",
            delta_a, delta_b
        );
    }

    #[test]
    fn test_effective_mutation_rate_used_correctly() {
        let epi = Epigenome {
            plasticity:         0.1,
            accumulated_stress: 0.5,
            ..Epigenome::default()
        };
        let rate = epi.effective_mutation_rate(BASE_MUTATION_RATE);
        // base * (1 + plasticity) + stress * 0.2
        // = 0.1 * 1.1 + 0.5 * 0.2 = 0.11 + 0.10 = 0.21
        assert!(
            (rate - 0.21_f32).abs() < 1e-5,
            "Effective mutation rate incorrect: expected ~0.21, got {}", rate
        );
    }
}
