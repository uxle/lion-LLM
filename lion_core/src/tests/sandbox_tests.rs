// lion_core/src/tests/sandbox_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_sovereign() -> Sovereign {
        Sovereign::new(42)
    }

    fn danger_frame() -> SensoryInput {
        let mut f = SensoryInput::new();
        f.insert(Role::Danger, [1.0_f32; FEATURE_SIZE]);
        f
    }

    fn vision_frame() -> SensoryInput {
        let mut f = SensoryInput::new();
        f.insert(Role::Vision, [0.5_f32; FEATURE_SIZE]);
        f
    }

    /// Runs a waking day of N ticks and returns the sovereign.
    fn run_day(sovereign: &mut Sovereign, ticks: usize) {
        let mut reward = 0.0_f32;
        for t in 0..ticks {
            let frame = if t % 3 == 0 { danger_frame() } else { vision_frame() };
            let result = sovereign.update(&frame, reward);
            reward = if result.action == "FLEE" { 0.5 } else { -1.0 };
        }
    }

    // =========================================================================
    // FITNESS EVALUATION
    // =========================================================================

    #[test]
    fn test_evaluate_fitness_empty_buffer_returns_complexity_penalty() {
        let mut brain    = BrainMatrix::new();
        let mut rng      = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);
        let empty_buffer = EpisodicBuffer::default_capacity();

        let fitness = evaluate_fitness(&mut brain, &empty_buffer);

        // No episode rewards, only complexity penalty.
        assert!(fitness <= 0.0, "Empty buffer fitness should be <= 0.0 (complexity cost): {}", fitness);
    }

    #[test]
    fn test_evaluate_fitness_returns_finite_value() {
        let mut s = make_sovereign();
        run_day(&mut s, 10);
        s.flush_pending_episode(0.0);

        let fitness = evaluate_fitness(&mut s.brain.clone(), &s.episodic_buffer);
        assert!(fitness.is_finite(), "Fitness must be finite: {}", fitness);
    }

    #[test]
    fn test_evaluate_fitness_does_not_write_traces() {
        let mut s = make_sovereign();
        run_day(&mut s, 5);
        s.flush_pending_episode(0.0);

        let mut eval_brain = s.brain.clone();
        // Clear preexisting traces before evaluating sandbox fitness
        for n in eval_brain.neurons.iter_mut() {
            n.trace_count = 0;
        }

        evaluate_fitness(&mut eval_brain, &s.episodic_buffer);

        // Sandbox evaluation must not write memory traces.
        for n in eval_brain.alive_neurons() {
            assert_eq!(n.trace_count, 0,
                "evaluate_fitness must not write memory traces to child brain"
            );
        }
    }

    #[test]
    fn test_evaluate_fitness_is_deterministic() {
        let mut s = make_sovereign();
        run_day(&mut s, 8);
        s.flush_pending_episode(0.5);

        let f1 = evaluate_fitness(&mut s.brain.clone(), &s.episodic_buffer);
        let f2 = evaluate_fitness(&mut s.brain.clone(), &s.episodic_buffer);

        assert!(
            (f1 - f2).abs() < 1e-10,
            "evaluate_fitness must be deterministic: {} vs {}", f1, f2
        );
    }

    // =========================================================================
    // WEIGHT PERTURBATION
    // =========================================================================

    #[test]
    fn test_mutate_graph_changes_at_least_one_weight() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        let weights_before: Vec<f32> = brain
            .alive_synapses()
            .map(|s| s.weight)
            .collect();

        mutate_graph(&mut brain, &mut rng, 1.0); // 100% mutation rate.

        let weights_after: Vec<f32> = brain
            .alive_synapses()
            .map(|s| s.weight)
            .collect();

        let any_changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);

        assert!(any_changed, "mutate_graph must change at least one weight at mut_rate=1.0");
    }

    #[test]
    fn test_mutate_graph_zero_rate_changes_nothing() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        // Set all synapse weights above prune threshold to prevent pruning during mutate_graph.
        for s in brain.synapses.iter_mut() {
            if s.alive {
                s.weight = 0.8;
            }
        }

        let weights_before: Vec<f32> = brain
            .alive_synapses()
            .map(|s| s.weight)
            .collect();

        mutate_graph(&mut brain, &mut rng, 0.0); // 0% mutation rate.

        let weights_after: Vec<f32> = brain
            .alive_synapses()
            .map(|s| s.weight)
            .collect();

        let any_changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);

        assert!(!any_changed, "mutate_graph with rate=0.0 must not change any weights");
    }

    #[test]
    fn test_weights_stay_bounded_after_mutation() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        for _ in 0..10 {
            mutate_graph(&mut brain, &mut rng, 1.0);
        }

        for s in brain.alive_synapses() {
            assert!(
                s.weight >= WEIGHT_MIN && s.weight <= WEIGHT_MAX,
                "Synapse weight out of bounds after mutation: {}", s.weight
            );
        }
    }

    // =========================================================================
    // SYNAPSE PRUNING
    // =========================================================================

    #[test]
    fn test_prune_removes_near_zero_synapses() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        // Manually set all synapse weights to near-zero.
        for s in brain.synapses.iter_mut() {
            if s.alive {
                s.weight = 0.01; // Below SYNAPSE_PRUNE_THRESHOLD = 0.05
            }
        }

        let count_before = brain.alive_synapse_count();
        mutate_graph(&mut brain, &mut rng, 0.0); // Only pruning, no weight mutation.
        let count_after = brain.alive_synapse_count();

        assert!(count_after < count_before,
            "Pruning should remove near-zero synapses: before={}, after={}",
            count_before, count_after
        );
    }

    // =========================================================================
    // MITOSIS
    // =========================================================================

    #[test]
    fn test_mitosis_increases_neuron_count() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        // Saturate a Vision neuron's trace bank to trigger overload.
        let vision_ids: Vec<_> = brain.neurons_by_role(Role::Vision).map(|n| n.id).collect();
        let target = vision_ids[0];
        for _ in 0..MAX_TRACES {
            brain.get_neuron_mut(target).unwrap().add_trace([1.0_f32; FEATURE_SIZE]);
        }
        // Force all traces to high strength to satisfy is_overloaded().
        for i in 0..MAX_TRACES {
            brain.neurons[target.index].traces[i].strength = 0.9;
        }

        assert!(brain.get_neuron(target).unwrap().is_overloaded(),
            "Neuron should be overloaded before mitosis test"
        );

        let count_before = brain.alive_neuron_count();
        mutate_graph(&mut brain, &mut rng, 1.0); // 100% rate ensures mitosis fires.
        let count_after = brain.alive_neuron_count();

        assert!(count_after > count_before,
            "Mitosis should increase neuron count: before={}, after={}",
            count_before, count_after
        );
    }

    #[test]
    fn test_mitosis_child_has_same_role_as_parent() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        let vision_ids: Vec<_> = brain.neurons_by_role(Role::Vision).map(|n| n.id).collect();
        let target = vision_ids[0];

        // Saturate and overload.
        for _ in 0..MAX_TRACES {
            brain.get_neuron_mut(target).unwrap().add_trace([1.0_f32; FEATURE_SIZE]);
        }
        for i in 0..MAX_TRACES {
            brain.neurons[target.index].traces[i].strength = 0.9;
        }

        mutate_graph(&mut brain, &mut rng, 1.0);

        // All neurons must have valid roles.
        for n in brain.alive_neurons() {
            let _ = n.role; // Just assert it doesn't panic.
        }

        // At least one new Vision neuron should exist.
        let vision_count = brain.neurons_by_role(Role::Vision).count();
        assert!(vision_count > NEURONS_PER_ROLE,
            "Mitosis should add a new Vision neuron: count={}", vision_count
        );
    }

    #[test]
    fn test_mitosis_child_has_incremented_generation() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        let vision_ids: Vec<_> = brain.neurons_by_role(Role::Vision).map(|n| n.id).collect();
        let target = vision_ids[0];
        let parent_gen = brain.get_neuron(target).unwrap().generation;

        for _ in 0..MAX_TRACES {
            brain.get_neuron_mut(target).unwrap().add_trace([1.0_f32; FEATURE_SIZE]);
        }
        for i in 0..MAX_TRACES {
            brain.neurons[target.index].traces[i].strength = 0.9;
        }

        mutate_graph(&mut brain, &mut rng, 1.0);

        // Find any neuron with generation > parent_gen.
        let child_exists = brain
            .alive_neurons()
            .any(|n| n.generation > parent_gen);

        assert!(child_exists, "Mitosis child should have generation > parent's");
    }

    #[test]
    fn test_brain_remains_healthy_after_mutation() {
        let mut brain = BrainMatrix::new();
        let mut rng   = BrainRng::from_seed(0);
        brain.initialize_core_brain(&mut rng);

        for _ in 0..5 {
            mutate_graph(&mut brain, &mut rng, 0.5);
        }

        assert!(brain.is_numerically_healthy(),
            "Brain must remain numerically healthy after mutations"
        );
    }

    // =========================================================================
    // NIGHT CYCLE
    // =========================================================================

    #[test]
    fn test_night_cycle_empty_buffer_returns_no_winner() {
        let s   = make_sovereign();
        let mut rng = BrainRng::from_seed(0);

        let (winner, _) = run_night_cycle(
            &s.brain,
            &s.episodic_buffer,
            &mut rng,
            5,
        );

        assert!(winner.is_none(),
            "Night cycle with empty buffer must return no winner"
        );
    }

    #[test]
    fn test_night_cycle_report_has_correct_children_count() {
        let mut s = make_sovereign();
        run_day(&mut s, 10);
        s.flush_pending_episode(0.0);

        let mut rng = BrainRng::from_seed(0);
        let population = 5;
        let (_, report) = run_night_cycle(
            &s.brain,
            &s.episodic_buffer,
            &mut rng,
            population,
        );

        assert_eq!(report.children_evaluated, population);
    }

    #[test]
    fn test_night_cycle_sovereign_fitness_is_finite() {
        let mut s = make_sovereign();
        run_day(&mut s, 8);
        s.flush_pending_episode(0.5);

        let mut rng = BrainRng::from_seed(0);
        let (_, report) = run_night_cycle(
            &s.brain,
            &s.episodic_buffer,
            &mut rng,
            5,
        );

        assert!(report.sovereign_fitness.is_finite(),
            "Sovereign fitness must be finite: {}", report.sovereign_fitness
        );
    }

    // =========================================================================
    // TRIGGER SLEEP CYCLE (SOVEREIGN INTEGRATION)
    // =========================================================================

    #[test]
    fn test_trigger_sleep_cycle_flushes_pending_episode() {
        let mut s = make_sovereign();
        run_day(&mut s, 5);

        assert!(s.has_pending_episode());

        s.trigger_sleep_cycle(1.0);

        assert!(!s.has_pending_episode(),
            "trigger_sleep_cycle must flush the pending episode"
        );
    }

    #[test]
    fn test_trigger_sleep_cycle_returns_valid_report() {
        let mut s = make_sovereign();
        run_day(&mut s, 10);

        let report = s.trigger_sleep_cycle(0.0);

        assert!(report.sovereign_fitness.is_finite());
        assert_eq!(report.children_evaluated, NIGHT_CYCLE_POPULATION);
    }

    #[test]
    fn test_trigger_sleep_cycle_may_evolve_generation() {
        let mut s = make_sovereign();
        run_day(&mut s, 20);

        let gen_before = s.generation;
        let report = s.trigger_sleep_cycle(0.0);

        // Generation increments only if evolution occurred.
        if report.evolution_occurred {
            assert_eq!(s.generation, gen_before + 1,
                "Generation must increment on successful evolution"
            );
            assert_eq!(s.brain.epigenome.accumulated_stress, 0.0,
                "Stress must clear on successful evolution"
            );
        } else {
            assert_eq!(s.generation, gen_before,
                "Generation must not change when no evolution occurs"
            );
        }
    }

    #[test]
    fn test_full_day_and_night_cycle_produces_healthy_brain() {
        let mut s = make_sovereign();
        run_day(&mut s, 15);
        s.trigger_sleep_cycle(0.5);

        assert!(s.brain.is_numerically_healthy(),
            "Brain must be numerically healthy after a full day + night cycle"
        );
    }

    #[test]
    fn test_two_full_cycles_do_not_crash() {
        let mut s = make_sovereign();

        // Day 1
        run_day(&mut s, 10);
        s.trigger_sleep_cycle(0.0);

        // Day 2
        run_day(&mut s, 10);
        s.trigger_sleep_cycle(0.5);

        assert!(s.generation >= 1);
        assert!(s.brain.is_numerically_healthy());
    }
}
