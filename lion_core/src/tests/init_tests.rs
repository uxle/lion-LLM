// lion_core/src/tests/init_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // Helper: produce a freshly initialized brain with a fixed seed.
    fn make_brain() -> BrainMatrix {
        let mut brain = BrainMatrix::new();
        let mut rng = BrainRng::from_seed(42);
        brain.initialize_core_brain(&mut rng);
        brain
    }

    // -------------------------------------------------------------------------
    // TEST: Neuron count after initialization
    // -------------------------------------------------------------------------

    #[test]
    fn test_initial_neuron_count_is_correct() {
        let brain = make_brain();
        // 3 Vision + 3 Memory + 3 Danger + 4 Motor = 13
        assert_eq!(brain.alive_neuron_count(), INITIAL_NEURON_COUNT);
        assert_eq!(brain.alive_neuron_count(), 13);
    }

    // -------------------------------------------------------------------------
    // TEST: Role distribution
    // -------------------------------------------------------------------------

    #[test]
    fn test_vision_neuron_count() {
        let brain = make_brain();
        let count = brain.neurons_by_role(Role::Vision).count();
        assert_eq!(count, NEURONS_PER_ROLE);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_memory_neuron_count() {
        let brain = make_brain();
        let count = brain.neurons_by_role(Role::Memory).count();
        assert_eq!(count, NEURONS_PER_ROLE);
    }

    #[test]
    fn test_danger_neuron_count() {
        let brain = make_brain();
        let count = brain.neurons_by_role(Role::Danger).count();
        assert_eq!(count, NEURONS_PER_ROLE);
    }

    #[test]
    fn test_motor_neuron_count() {
        let brain = make_brain();
        let count = brain.neurons_by_role(Role::Motor).count();
        assert_eq!(count, MOTOR_NEURON_COUNT);
        assert_eq!(count, 4);
    }

    // -------------------------------------------------------------------------
    // TEST: Motor neurons have correct action labels
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_procedural_actions_present() {
        let brain = make_brain();
        for &action in PROCEDURAL_ACTIONS {
            let found = brain
                .neurons_by_role(Role::Motor)
                .any(|n| {
                    n.action_label
                        .map(|l| l.as_str() == action)
                        .unwrap_or(false)
                });
            assert!(
                found,
                "Action '{}' not found among Motor neurons",
                action
            );
        }
    }

    #[test]
    fn test_motor_neurons_have_action_labels() {
        let brain = make_brain();
        for n in brain.neurons_by_role(Role::Motor) {
            assert!(
                n.action_label.is_some(),
                "Motor neuron {:?} has no action label",
                n.id
            );
        }
    }

    #[test]
    fn test_non_motor_neurons_have_no_action_label() {
        let brain = make_brain();
        for n in brain.alive_neurons() {
            if n.role != Role::Motor {
                assert!(
                    n.action_label.is_none(),
                    "Non-motor neuron {:?} should not have an action label",
                    n.id
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // TEST: Base vectors are non-zero and within range
    // -------------------------------------------------------------------------

    #[test]
    fn test_base_vectors_within_init_range() {
        let brain = make_brain();
        for n in brain.alive_neurons() {
            for &v in &n.base_vector {
                assert!(
                    v >= BASE_VECTOR_INIT_MIN && v <= BASE_VECTOR_INIT_MAX,
                    "base_vector component {} out of range [{}, {}]",
                    v, BASE_VECTOR_INIT_MIN, BASE_VECTOR_INIT_MAX
                );
            }
        }
    }

    #[test]
    fn test_no_neuron_has_zero_base_vector() {
        let brain = make_brain();
        for n in brain.alive_neurons() {
            let magnitude: f32 = n.base_vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                magnitude > 0.0,
                "Neuron {:?} has a zero base_vector — cosine alignment will divide by zero",
                n.id
            );
        }
    }

    // -------------------------------------------------------------------------
    // TEST: Synapses exist and have valid endpoints
    // -------------------------------------------------------------------------

    #[test]
    fn test_synapses_created_after_init() {
        let brain = make_brain();
        // With 13 neurons and ~50% connection probability,
        // expected synapse count ≈ 13 * 12 * 0.5 = 78.
        // We only assert > 0 to avoid flakiness from RNG variation.
        assert!(
            brain.alive_synapse_count() > 0,
            "No synapses were created during initialization"
        );
    }

    #[test]
    fn test_all_synapse_endpoints_are_valid_neurons() {
        let brain = make_brain();
        for s in brain.alive_synapses() {
            assert!(
                brain.is_valid_neuron(s.pre_id),
                "Synapse has invalid pre_id {:?}",
                s.pre_id
            );
            assert!(
                brain.is_valid_neuron(s.post_id),
                "Synapse has invalid post_id {:?}",
                s.post_id
            );
        }
    }

    #[test]
    fn test_no_self_loop_synapses() {
        let brain = make_brain();
        for s in brain.alive_synapses() {
            assert_ne!(
                s.pre_id, s.post_id,
                "Self-loop synapse found at {:?}",
                s.pre_id
            );
        }
    }

    #[test]
    fn test_synapse_weights_within_init_range() {
        let brain = make_brain();
        for s in brain.alive_synapses() {
            assert!(
                s.weight >= INITIAL_WEIGHT_MIN && s.weight <= INITIAL_WEIGHT_MAX,
                "Synapse weight {} out of range [{}, {}]",
                s.weight, INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX
            );
        }
    }

    // -------------------------------------------------------------------------
    // TEST: Determinism — same seed produces identical brains
    // -------------------------------------------------------------------------

    #[test]
    fn test_same_seed_produces_identical_brain() {
        let mut brain_a = BrainMatrix::new();
        let mut rng_a = BrainRng::from_seed(99);
        brain_a.initialize_core_brain(&mut rng_a);

        let mut brain_b = BrainMatrix::new();
        let mut rng_b = BrainRng::from_seed(99);
        brain_b.initialize_core_brain(&mut rng_b);

        assert_eq!(brain_a.alive_neuron_count(), brain_b.alive_neuron_count());
        assert_eq!(brain_a.alive_synapse_count(), brain_b.alive_synapse_count());

        // Compare every alive neuron's base_vector component by component.
        let ids_a: Vec<_> = brain_a.collect_alive_neuron_ids();
        let ids_b: Vec<_> = brain_b.collect_alive_neuron_ids();

        for (id_a, id_b) in ids_a.iter().zip(ids_b.iter()) {
            let na = brain_a.get_neuron(*id_a).unwrap();
            let nb = brain_b.get_neuron(*id_b).unwrap();
            for (va, vb) in na.base_vector.iter().zip(nb.base_vector.iter()) {
                assert!(
                    (va - vb).abs() < 1e-7,
                    "base_vector diverged between same-seeded brains: {} vs {}",
                    va, vb
                );
            }
        }
    }

    #[test]
    fn test_different_seeds_produce_different_brains() {
        let mut brain_a = BrainMatrix::new();
        let mut rng_a = BrainRng::from_seed(1);
        brain_a.initialize_core_brain(&mut rng_a);

        let mut brain_b = BrainMatrix::new();
        let mut rng_b = BrainRng::from_seed(2);
        brain_b.initialize_core_brain(&mut rng_b);

        // Both have the same neuron count (deterministic topology)...
        assert_eq!(brain_a.alive_neuron_count(), brain_b.alive_neuron_count());

        // ...but the base_vectors must differ somewhere.
        let ids_a: Vec<_> = brain_a.collect_alive_neuron_ids();
        let ids_b: Vec<_> = brain_b.collect_alive_neuron_ids();

        let any_diff = ids_a.iter().zip(ids_b.iter()).any(|(id_a, id_b)| {
            let na = brain_a.get_neuron(*id_a).unwrap();
            let nb = brain_b.get_neuron(*id_b).unwrap();
            na.base_vector
                .iter()
                .zip(nb.base_vector.iter())
                .any(|(va, vb)| (va - vb).abs() > 1e-7)
        });

        assert!(
            any_diff,
            "Two differently-seeded brains produced identical base_vectors — RNG is broken"
        );
    }

    // -------------------------------------------------------------------------
    // TEST: BrainRng domain generators
    // -------------------------------------------------------------------------

    #[test]
    fn test_gen_base_vector_range() {
        let mut rng = BrainRng::from_seed(0);
        for _ in 0..100 {
            let v = rng.gen_base_vector();
            for x in v {
                assert!(
                    x >= BASE_VECTOR_INIT_MIN && x <= BASE_VECTOR_INIT_MAX,
                    "gen_base_vector produced out-of-range value: {}",
                    x
                );
            }
        }
    }

    #[test]
    fn test_gen_initial_weight_range() {
        let mut rng = BrainRng::from_seed(0);
        for _ in 0..1000 {
            let w = rng.gen_initial_weight();
            assert!(
                w >= INITIAL_WEIGHT_MIN && w <= INITIAL_WEIGHT_MAX,
                "gen_initial_weight out of range: {}",
                w
            );
        }
    }

    #[test]
    fn test_gen_mutation_delta_range() {
        let mut rng = BrainRng::from_seed(0);
        for _ in 0..1000 {
            let d = rng.gen_mutation_delta();
            assert!(
                d >= MUTATION_DELTA_MIN && d <= MUTATION_DELTA_MAX,
                "gen_mutation_delta out of range: {}",
                d
            );
        }
    }

    #[test]
    fn test_gen_prob_range() {
        let mut rng = BrainRng::from_seed(0);
        for _ in 0..1000 {
            let p = rng.gen_prob();
            assert!(
                p >= 0.0 && p < 1.0,
                "gen_prob out of range: {}",
                p
            );
        }
    }

    #[test]
    fn test_gen_bool_with_prob_always_true_at_one() {
        let mut rng = BrainRng::from_seed(0);
        for _ in 0..100 {
            assert!(rng.gen_bool_with_prob(1.0));
        }
    }

    #[test]
    fn test_gen_bool_with_prob_never_true_at_zero() {
        let mut rng = BrainRng::from_seed(0);
        for _ in 0..100 {
            assert!(!rng.gen_bool_with_prob(0.0));
        }
    }

    #[test]
    fn test_choose_returns_element_from_slice() {
        let mut rng = BrainRng::from_seed(0);
        let options = ["WANDER", "FORAGE", "FLEE", "ATTACK"];
        for _ in 0..100 {
            let choice = rng.choose(&options);
            assert!(
                options.contains(choice),
                "choose returned value not in slice: {}",
                choice
            );
        }
    }

    // -------------------------------------------------------------------------
    // TEST: find_motor_neuron_by_label
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_motor_neuron_by_label_returns_correct_id() {
        let brain = make_brain();
        for &action in PROCEDURAL_ACTIONS {
            let id = brain.find_motor_neuron_by_label(action);
            assert!(
                id.is_some(),
                "find_motor_neuron_by_label('{}') returned None",
                action
            );
            let neuron = brain.get_neuron(id.unwrap()).unwrap();
            assert_eq!(neuron.role, Role::Motor);
            assert_eq!(neuron.action_label.unwrap().as_str(), action);
        }
    }

    #[test]
    fn test_find_motor_neuron_by_label_returns_none_for_unknown() {
        let brain = make_brain();
        let id = brain.find_motor_neuron_by_label("UNKNOWN_ACTION");
        assert!(id.is_none());
    }

    // -------------------------------------------------------------------------
    // TEST: collect_alive_neuron_ids
    // -------------------------------------------------------------------------

    #[test]
    fn test_collect_alive_neuron_ids_matches_count() {
        let brain = make_brain();
        let ids = brain.collect_alive_neuron_ids();
        assert_eq!(ids.len(), brain.alive_neuron_count());
    }

    #[test]
    fn test_all_collected_ids_are_valid() {
        let brain = make_brain();
        for id in brain.collect_alive_neuron_ids() {
            assert!(
                brain.is_valid_neuron(id),
                "collect_alive_neuron_ids returned invalid id {:?}",
                id
            );
        }
    }

    // -------------------------------------------------------------------------
    // TEST: Free slot bookkeeping after initialization
    // -------------------------------------------------------------------------

    #[test]
    fn test_free_neuron_slots_decrease_after_init() {
        let brain = make_brain();
        assert_eq!(
            brain.free_neuron_capacity(),
            MAX_NEURONS - INITIAL_NEURON_COUNT
        );
    }

    #[test]
    fn test_free_synapse_slots_decrease_after_init() {
        let brain = make_brain();
        let used = brain.alive_synapse_count();
        assert_eq!(brain.free_synapse_capacity(), MAX_SYNAPSES - used);
    }

    // -------------------------------------------------------------------------
    // TEST: Clone independence after initialization
    // -------------------------------------------------------------------------

    #[test]
    fn test_initialized_brain_clone_is_independent() {
        let original = make_brain();
        let mut cloned = original.clone();

        let ids = cloned.collect_alive_neuron_ids();
        cloned.get_neuron_mut(ids[0]).unwrap().activation = 99.0;

        // Original must be unmodified.
        assert_eq!(
            original.get_neuron(ids[0]).unwrap().activation,
            0.0
        );
    }
}
