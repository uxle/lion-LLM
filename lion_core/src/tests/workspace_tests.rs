// lion_core/src/tests/workspace_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    fn make_brain() -> BrainMatrix {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(42);
        brain.initialize_core_brain(&mut rng);
        brain
    }

    // ── gather_consciousness ──────────────────────────────────────────────────

    #[test]
    fn test_gestalt_is_finite_after_zero_activations() {
        let brain = make_brain();
        let (gestalt, _) = gather_consciousness(&brain, WORKSPACE_TOP_K);
        for v in gestalt {
            assert!(v.is_finite(), "Gestalt component not finite: {}", v);
        }
    }

    #[test]
    fn test_gestalt_is_normalized_after_activation() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, [1.0_f32; FEATURE_SIZE]);
        brain.tick(&frame, 2);

        let (gestalt, _) = gather_consciousness(&brain, WORKSPACE_TOP_K);

        let norm: f32 = gestalt.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Norm should be ~1.0 (or 0.0 if all activations are zero).
        assert!(
            (norm - 1.0).abs() < 1e-5 || norm == 0.0,
            "Gestalt norm should be ~1.0 or 0.0, got {}", norm
        );
    }

    #[test]
    fn test_top_k_ids_are_valid_neurons() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, [0.5_f32; FEATURE_SIZE]);
        brain.tick(&frame, 2);

        let (_, top_k_ids) = gather_consciousness(&brain, WORKSPACE_TOP_K);

        for id in top_k_ids {
            assert!(
                brain.is_valid_neuron(id),
                "gather_consciousness returned invalid neuron id {:?}", id
            );
        }
    }

    #[test]
    fn test_top_k_count_at_most_k() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, [1.0_f32; FEATURE_SIZE]);
        brain.tick(&frame, 2);

        let (_, top_k_ids) = gather_consciousness(&brain, WORKSPACE_TOP_K);

        assert!(
            top_k_ids.len() <= WORKSPACE_TOP_K,
            "gather_consciousness returned {} ids, expected at most {}",
            top_k_ids.len(), WORKSPACE_TOP_K
        );
    }

    #[test]
    fn test_danger_signal_influences_gestalt_differently_than_vision() {
        let mut brain_v = make_brain();
        let mut brain_d = make_brain();

        let mut frame_v = SensoryInput::new();
        frame_v.insert(Role::Vision, [1.0_f32; FEATURE_SIZE]);
        brain_v.tick(&frame_v, 2);

        let mut frame_d = SensoryInput::new();
        frame_d.insert(Role::Danger, [1.0_f32; FEATURE_SIZE]);
        brain_d.tick(&frame_d, 2);

        let (gestalt_v, _) = gather_consciousness(&brain_v, WORKSPACE_TOP_K);
        let (gestalt_d, _) = gather_consciousness(&brain_d, WORKSPACE_TOP_K);

        let diff: f32 = gestalt_v
            .iter()
            .zip(gestalt_d.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.0,
            "Different modalities should produce different gestalts"
        );
    }

    // ── extract_action ────────────────────────────────────────────────────────

    #[test]
    fn test_exploit_returns_known_action() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, [0.5_f32; FEATURE_SIZE]);
        brain.tick(&frame, 2);

        let action = best_motor_action(&brain);
        assert!(
            PROCEDURAL_ACTIONS.contains(&action),
            "best_motor_action returned unknown action: {}", action
        );
    }

    #[test]
    fn test_extract_action_force_exploit_is_deterministic() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, [1.0_f32; FEATURE_SIZE]);
        brain.tick(&frame, 2);

        let epi = Epigenome { exploration_drive: 1.0, ..Epigenome::default() };
        let mut rng = BrainRng::from_seed(0);

        // Force exploit — should ignore exploration_drive=1.0.
        let action_a = extract_action(&brain, &epi, &mut rng, true);
        let action_b = extract_action(&brain, &epi, &mut rng, true);

        assert_eq!(
            action_a, action_b,
            "force_exploit must produce deterministic results"
        );
    }

    #[test]
    fn test_extract_action_exploration_returns_valid_action() {
        let mut brain = make_brain();
        let mut frame = SensoryInput::new();
        frame.insert(Role::Vision, [1.0_f32; FEATURE_SIZE]);
        brain.tick(&frame, 2);

        let epi = Epigenome { exploration_drive: 1.0, ..Epigenome::default() };
        let mut rng = BrainRng::from_seed(0);

        for _ in 0..50 {
            let action = extract_action(&brain, &epi, &mut rng, false);
            assert!(
                PROCEDURAL_ACTIONS.contains(&action),
                "explore action not in PROCEDURAL_ACTIONS: {}", action
            );
        }
    }

    #[test]
    fn test_action_to_label_round_trip() {
        for &action in PROCEDURAL_ACTIONS {
            let label = action_to_label(action);
            assert_eq!(label.as_str(), action);
        }
    }
}
