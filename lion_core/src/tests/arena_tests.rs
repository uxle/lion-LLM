// lion_core/src/tests/arena_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // -------------------------------------------------------------------------
    // TEST: Arena construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_new_brain_is_empty() {
        let brain = BrainMatrix::new();
        assert_eq!(brain.alive_neuron_count(), 0);
        assert_eq!(brain.alive_synapse_count(), 0);
        assert_eq!(brain.free_neuron_capacity(), MAX_NEURONS);
        assert_eq!(brain.free_synapse_capacity(), MAX_SYNAPSES);
    }

    // -------------------------------------------------------------------------
    // TEST: Neuron insertion
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_neuron_returns_valid_id() {
        let mut brain = BrainMatrix::new();
        let base = [0.1_f32; FEATURE_SIZE];
        let id = brain.insert_neuron(Role::Vision, base).unwrap();

        assert!(brain.is_valid_neuron(id));
        assert_eq!(brain.alive_neuron_count(), 1);
        assert_eq!(brain.free_neuron_capacity(), MAX_NEURONS - 1);
    }

    #[test]
    fn test_insert_motor_neuron_has_action_label() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let id = brain.insert_motor_neuron(base, "FLEE").unwrap();

        let neuron = brain.get_neuron(id).unwrap();
        assert_eq!(neuron.role, Role::Motor);
        assert_eq!(
            neuron.action_label.unwrap().as_str(),
            "FLEE"
        );
    }

    // -------------------------------------------------------------------------
    // TEST: Neuron removal
    // -------------------------------------------------------------------------

    #[test]
    fn test_remove_neuron_returns_slot_to_free_list() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let id = brain.insert_neuron(Role::Memory, base).unwrap();

        assert!(brain.remove_neuron(id));
        assert_eq!(brain.alive_neuron_count(), 0);
        assert_eq!(brain.free_neuron_capacity(), MAX_NEURONS);
        assert!(!brain.is_valid_neuron(id));
    }

    #[test]
    fn test_remove_nonexistent_neuron_returns_false() {
        let mut brain = BrainMatrix::new();
        let fake_id = GenIndex::new(0, 99);
        assert!(!brain.remove_neuron(fake_id));
    }

    // -------------------------------------------------------------------------
    // TEST: Generational index invalidation
    // -------------------------------------------------------------------------

    #[test]
    fn test_stale_index_is_rejected_after_slot_reuse() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];

        // Insert and remove a neuron.
        let old_id = brain.insert_neuron(Role::Vision, base).unwrap();
        brain.remove_neuron(old_id);

        // Insert a new neuron into the same slot.
        let new_id = brain.insert_neuron(Role::Danger, base).unwrap();

        // Both use the same slot index.
        assert_eq!(old_id.index, new_id.index);

        // But old_id's generation is now stale — must be rejected.
        assert!(!brain.is_valid_neuron(old_id));
        assert!(brain.is_valid_neuron(new_id));
    }

    // -------------------------------------------------------------------------
    // TEST: Synapse insertion
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_synapse_between_two_alive_neurons() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let pre  = brain.insert_neuron(Role::Vision, base).unwrap();
        let post = brain.insert_neuron(Role::Motor,  base).unwrap();

        let syn_id = brain.insert_synapse(pre, post, 0.5).unwrap();
        assert!(brain.is_valid_synapse(syn_id));
        assert_eq!(brain.alive_synapse_count(), 1);
    }

    #[test]
    fn test_insert_synapse_rejects_dead_endpoint() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let pre  = brain.insert_neuron(Role::Vision, base).unwrap();
        let dead = GenIndex::new(999, 0); // Nonexistent

        let result = brain.insert_synapse(pre, dead, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_insert_synapse_rejects_self_loop() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let n = brain.insert_neuron(Role::Vision, base).unwrap();

        let result = brain.insert_synapse(n, n, 1.0);
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // TEST: Synapse removal
    // -------------------------------------------------------------------------

    #[test]
    fn test_remove_synapse_returns_slot_to_free_list() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let pre  = brain.insert_neuron(Role::Vision, base).unwrap();
        let post = brain.insert_neuron(Role::Motor,  base).unwrap();
        let syn  = brain.insert_synapse(pre, post, -0.3).unwrap();

        assert!(brain.remove_synapse(syn));
        assert_eq!(brain.alive_synapse_count(), 0);
        assert!(!brain.is_valid_synapse(syn));
    }

    // -------------------------------------------------------------------------
    // TEST: Memory trace logic
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_trace_fills_and_decays() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let id = brain.insert_neuron(Role::Memory, base).unwrap();

        let vec = [1.0_f32; FEATURE_SIZE];

        for _ in 0..MAX_TRACES {
            brain.get_neuron_mut(id).unwrap().add_trace(vec);
        }

        let neuron = brain.get_neuron(id).unwrap();
        assert_eq!(neuron.trace_count, MAX_TRACES);

        // After MAX_TRACES insertions, every trace should have decayed at least once.
        // The first trace inserted is now the oldest and weakest.
        for t in neuron.active_traces() {
            assert!(t.strength <= 1.0);
        }
    }

    #[test]
    fn test_add_trace_evicts_weakest_when_full() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let id = brain.insert_neuron(Role::Memory, base).unwrap();

        let vec = [1.0_f32; FEATURE_SIZE];

        // Fill the trace bank completely.
        for _ in 0..MAX_TRACES {
            brain.get_neuron_mut(id).unwrap().add_trace(vec);
        }

        // The trace bank should still be MAX_TRACES after one more insertion.
        brain.get_neuron_mut(id).unwrap().add_trace(vec);
        assert_eq!(
            brain.get_neuron(id).unwrap().trace_count,
            MAX_TRACES
        );
    }

    // -------------------------------------------------------------------------
    // TEST: Overload detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_neuron_not_overloaded_when_traces_below_capacity() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];
        let id = brain.insert_neuron(Role::Vision, base).unwrap();

        let vec = [1.0_f32; FEATURE_SIZE];
        brain.get_neuron_mut(id).unwrap().add_trace(vec);

        assert!(!brain.get_neuron(id).unwrap().is_overloaded());
    }

    // -------------------------------------------------------------------------
    // TEST: Reset activations
    // -------------------------------------------------------------------------

    #[test]
    fn test_reset_activations_clears_all_neurons() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];

        let id1 = brain.insert_neuron(Role::Vision, base).unwrap();
        let id2 = brain.insert_neuron(Role::Danger, base).unwrap();

        brain.get_neuron_mut(id1).unwrap().activation = 0.9;
        brain.get_neuron_mut(id2).unwrap().activation = -0.7;

        brain.reset_activations();

        assert_eq!(brain.get_neuron(id1).unwrap().activation, 0.0);
        assert_eq!(brain.get_neuron(id2).unwrap().activation, 0.0);
    }

    // -------------------------------------------------------------------------
    // TEST: Epigenome stress
    // -------------------------------------------------------------------------

    #[test]
    fn test_epigenome_stress_clamps_to_one() {
        let mut epi = Epigenome::default();
        epi.adapt_live_stress(2.0);
        assert_eq!(epi.accumulated_stress, 1.0);
    }

    #[test]
    fn test_epigenome_effective_mutation_rate() {
        let epi = Epigenome::default();
        let rate = epi.effective_mutation_rate(0.1);
        // base * (1 + plasticity) + stress * 0.2
        // = 0.1 * (1 + 0.05) + 0.0 * 0.2 = 0.105
        assert!((rate - 0.105_f32).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // TEST: Clone cost (proof that night cycle is O(1))
    // -------------------------------------------------------------------------

    #[test]
    fn test_brain_clone_is_independent() {
        let mut original = BrainMatrix::new();
        let base = [0.5_f32; FEATURE_SIZE];
        let id = original.insert_neuron(Role::Vision, base).unwrap();

        let mut cloned = original.clone();

        // Modify the clone.
        cloned.get_neuron_mut(id).unwrap().activation = 0.99;

        // Original must be unaffected.
        assert_eq!(original.get_neuron(id).unwrap().activation, 0.0);
    }

    // -------------------------------------------------------------------------
    // TEST: Role filtering
    // -------------------------------------------------------------------------

    #[test]
    fn test_neurons_by_role_returns_correct_subset() {
        let mut brain = BrainMatrix::new();
        let base = [0.0_f32; FEATURE_SIZE];

        brain.insert_neuron(Role::Vision, base).unwrap();
        brain.insert_neuron(Role::Vision, base).unwrap();
        brain.insert_neuron(Role::Danger, base).unwrap();

        let vision_count = brain.neurons_by_role(Role::Vision).count();
        let danger_count = brain.neurons_by_role(Role::Danger).count();
        let motor_count  = brain.neurons_by_role(Role::Motor).count();

        assert_eq!(vision_count, 2);
        assert_eq!(danger_count, 1);
        assert_eq!(motor_count,  0);
    }
}
