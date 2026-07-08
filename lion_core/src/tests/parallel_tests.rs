// lion_core/src/tests/parallel_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_sovereign_with_history(ticks: usize) -> Sovereign {
        let mut s      = Sovereign::new(42);
        let mut reward = 0.0_f32;

        for t in 0..ticks {
            let mut frame = SensoryInput::new();
            if t % 3 == 0 {
                frame.insert(Role::Danger, [1.0_f32; FEATURE_SIZE]);
            } else {
                frame.insert(Role::Vision, [0.5_f32; FEATURE_SIZE]);
            }
            let result = s.update(&frame, reward);
            reward = if result.action == "FLEE" { 0.5 } else { -1.0 };
        }
        s.flush_pending_episode(reward);
        s
    }

    // =========================================================================
    // CHILD SEED DERIVATION
    // =========================================================================

    #[test]
    fn test_child_seed_different_for_each_index() {
        let base = 12345u64;
        let seeds: Vec<u64> = (0..10).map(|i| child_seed(base, i)).collect();

        // All seeds must be unique.
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(unique.len(), seeds.len(),
            "All child seeds must be unique");
    }

    #[test]
    fn test_child_seed_deterministic() {
        let s1 = child_seed(999, 3);
        let s2 = child_seed(999, 3);
        assert_eq!(s1, s2,
            "child_seed must be deterministic for same inputs");
    }

    #[test]
    fn test_child_seed_different_bases_give_different_seeds() {
        let a = child_seed(1, 0);
        let b = child_seed(2, 0);
        assert_ne!(a, b, "Different base seeds must produce different child seeds");
    }

    // =========================================================================
    // PARALLEL CORRECTNESS
    // =========================================================================

    #[test]
    fn test_parallel_night_cycle_empty_buffer_returns_no_winner() {
        let s = Sovereign::new(42);

        let (winner, _) = run_night_cycle_parallel(
            &s.brain,
            &s.episodic_buffer,
            s.tick,
            5,
        );

        assert!(winner.is_none(),
            "Parallel night cycle with empty buffer must return no winner");
    }

    #[test]
    fn test_parallel_report_has_correct_children_count() {
        let s = make_sovereign_with_history(15);
        let population = 6;

        let (_, report) = run_night_cycle_parallel(
            &s.brain,
            &s.episodic_buffer,
            s.tick,
            population,
        );

        assert_eq!(report.children_evaluated, population);
    }

    #[test]
    fn test_parallel_fitness_is_finite() {
        let s = make_sovereign_with_history(10);

        let (_, report) = run_night_cycle_parallel(
            &s.brain,
            &s.episodic_buffer,
            s.tick,
            5,
        );

        assert!(report.sovereign_fitness.is_finite(),
            "Sovereign fitness must be finite: {}", report.sovereign_fitness);
        assert!(report.best_child_fitness.is_finite(),
            "Best child fitness must be finite: {}", report.best_child_fitness);
    }

    #[test]
    fn test_parallel_winner_has_higher_fitness_than_sovereign() {
        // Run with enough history and large population to maximize evolution chance.
        let s = make_sovereign_with_history(30);

        let (winner, report) = run_night_cycle_parallel(
            &s.brain,
            &s.episodic_buffer,
            s.tick,
            NIGHT_CYCLE_POPULATION,
        );

        if report.evolution_occurred {
            assert!(winner.is_some());
            assert!(
                report.best_child_fitness > report.sovereign_fitness + 1e-6, // EVOLUTION_MARGIN equivalent
                "Winner must have fitness > sovereign + margin: child={}, sovereign={}",
                report.best_child_fitness, report.sovereign_fitness
            );
        } else {
            assert!(winner.is_none());
        }
    }

    #[test]
    fn test_parallel_gives_same_report_as_sequential_with_same_seeds() {
        // Build a history.
        let s = make_sovereign_with_history(8);
        let population = 4;
        let base_seed  = s.tick;

        // Sequential: run one child at a time using the same per-child seeds.
        let mut seq_best_fitness = {
            let mut eval = s.brain.clone();
            evaluate_fitness(&mut eval, &s.episodic_buffer)
        };

        for i in 0..population {
            let seed = child_seed(base_seed, i);
            let mut rng   = BrainRng::from_seed(seed);
            let mut child = s.brain.clone();
            let rate = child.epigenome.effective_mutation_rate(0.1); // BASE_MUTATION_RATE
            mutate_graph(&mut child, &mut rng, rate);
            let pd = rng.gen_plasticity_delta();
            let ed = rng.gen_exploration_delta();
            child.epigenome.mutate(pd, ed);
            let f = evaluate_fitness(&mut child, &s.episodic_buffer);
            if f > seq_best_fitness {
                seq_best_fitness = f;
            }
        }

        // Parallel: run all children concurrently.
        let (_, par_report) = run_night_cycle_parallel(
            &s.brain,
            &s.episodic_buffer,
            base_seed,
            population,
        );

        // Both should find the same best child fitness.
        assert!(
            (par_report.best_child_fitness - seq_best_fitness).abs() < 1e-9,
            "Parallel best fitness {} != sequential best fitness {}",
            par_report.best_child_fitness, seq_best_fitness
        );
    }

    #[test]
    fn test_parallel_winner_brain_is_numerically_healthy() {
        let s = make_sovereign_with_history(20);

        let (winner, _) = run_night_cycle_parallel(
            &s.brain,
            &s.episodic_buffer,
            s.tick,
            NIGHT_CYCLE_POPULATION,
        );

        if let Some(brain) = winner {
            assert!(brain.is_numerically_healthy(),
                "Winner brain must be numerically healthy after parallel night cycle");
        }
    }

    // =========================================================================
    // SOVEREIGN INTEGRATION
    // =========================================================================

    #[test]
    fn test_trigger_sleep_cycle_uses_parallel_by_default() {
        let mut s = make_sovereign_with_history(15);

        // trigger_sleep_cycle now calls run_night_cycle_parallel internally.
        let report = s.trigger_sleep_cycle(0.0);

        assert!(report.sovereign_fitness.is_finite());
        assert_eq!(report.children_evaluated, NIGHT_CYCLE_POPULATION);
        assert!(s.brain.is_numerically_healthy());
    }

    #[test]
    fn test_full_parallel_day_night_does_not_regress_generation() {
        let mut s = make_sovereign_with_history(20);
        let gen_before = s.generation;

        let report = s.trigger_sleep_cycle(0.5);

        assert!(
            s.generation >= gen_before,
            "Generation must not decrease: before={}, after={}",
            gen_before, s.generation
        );

        if report.evolution_occurred {
            assert_eq!(s.generation, gen_before + 1);
        } else {
            assert_eq!(s.generation, gen_before);
        }
    }

    #[test]
    fn test_parallel_two_consecutive_cycles_stable() {
        let mut s = make_sovereign_with_history(15);

        let report1 = s.trigger_sleep_cycle(0.0);
        assert!(s.brain.is_numerically_healthy());

        // Run a second day and night.
        let mut reward = 0.0_f32;
        for _ in 0..10 {
            let mut frame = SensoryInput::new();
            frame.insert(Role::Vision, [0.7_f32; FEATURE_SIZE]);
            let result = s.update(&frame, reward);
            reward = if result.action == "FLEE" { 0.5 } else { 0.0 };
        }
        let report2 = s.trigger_sleep_cycle(reward);

        assert!(s.brain.is_numerically_healthy());
        assert!(report1.sovereign_fitness.is_finite());
        assert!(report2.sovereign_fitness.is_finite());
    }

    // =========================================================================
    // SIMD-FRIENDLY GEMV CORRECTNESS
    // =========================================================================

    #[test]
    fn test_simd_friendly_gemv_matches_branchless() {
        let mut rng = BrainRng::from_seed(0);
        let in_sz   = 64;
        let out_sz  = 16;

        let raw: Vec<i8> = (0..in_sz * out_sz)
            .map(|_| match rng.gen_index(3) { 0 => -1i8, 1 => 0i8, _ => 1i8 })
            .collect();
        let input: Vec<i8> = (0..in_sz)
            .map(|_| (rng.gen_index(254) as i8).wrapping_sub(127))
            .collect();

        let weights = pack_weights(&raw);

        let mut out_branchless = vec![0i32; out_sz];
        let mut out_simd       = vec![0i32; out_sz];

        ternary_gemv(
            &input, &weights, &mut out_branchless, in_sz, out_sz,
        );
        ternary_gemv_auto(
            &input, &weights, &mut out_simd, in_sz, out_sz,
        );

        assert_eq!(out_branchless, out_simd,
            "SIMD-friendly GEMV must match branchless GEMV");
    }

    #[test]
    fn test_dispatch_routes_correctly_for_small_input() {
        let in_sz  = GEMV_SIMD_THRESHOLD - 1; // Below threshold → branchless
        let out_sz = 4;

        let raw: Vec<i8>  = vec![1i8; in_sz * out_sz];
        let weights       = pack_weights(&raw);
        let input: Vec<i8> = vec![10i8; in_sz];
        let mut out_dispatch   = vec![0i32; out_sz];
        let mut out_branchless = vec![0i32; out_sz];

        ternary_gemv_dispatch(&input, &weights, &mut out_dispatch, in_sz, out_sz);
        ternary_gemv(&input, &weights, &mut out_branchless, in_sz, out_sz);

        assert_eq!(out_dispatch, out_branchless);
    }

    #[test]
    fn test_dispatch_routes_correctly_for_large_input() {
        let in_sz  = GEMV_SIMD_THRESHOLD + 32; // Above threshold → SIMD-friendly
        let out_sz = 4;

        let raw: Vec<i8>   = vec![-1i8; in_sz * out_sz];
        let weights        = pack_weights(&raw);
        let input: Vec<i8> = vec![5i8; in_sz];
        let mut out_dispatch = vec![0i32; out_sz];
        let mut out_simd     = vec![0i32; out_sz];

        ternary_gemv_dispatch(&input, &weights, &mut out_dispatch, in_sz, out_sz);
        ternary_gemv_auto(&input, &weights, &mut out_simd, in_sz, out_sz);

        assert_eq!(out_dispatch, out_simd);
    }

    #[test]
    fn test_unpack_weight_row_matches_individual_unpack() {
        let raw: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, 0];
        let weights      = pack_weights(&raw);
        let mut scratch  = vec![0i8; 8];

        unpack_weight_row(&weights, 0, 8, &mut scratch);

        for j in 0..8 {
            assert_eq!(
                scratch[j],
                unpack_weight(&weights, j),
                "Row unpack mismatch at j={}", j
            );
        }
    }

    // =========================================================================
    // THREAD SAFETY — SANITY CHECKS
    // =========================================================================

    #[test]
    fn test_brain_matrix_is_send() {
        // Verify at compile time that BrainMatrix implements Send.
        fn assert_send<T: Send>() {}
        assert_send::<BrainMatrix>();
    }

    #[test]
    fn test_episodic_buffer_is_sync() {
        // Verify at compile time that EpisodicBuffer implements Sync.
        // Required for sharing across Rayon threads.
        fn assert_sync<T: Sync>() {}
        assert_sync::<EpisodicBuffer>();
    }

    #[test]
    fn test_parallel_night_cycle_with_rayon_threadpool() {
        // Explicitly set Rayon thread pool to 4 and verify correctness.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("Failed to build Rayon thread pool");

        let s = make_sovereign_with_history(10);

        let (winner, report) = pool.install(|| {
            run_night_cycle_parallel(
                &s.brain,
                &s.episodic_buffer,
                s.tick,
                8,
            )
        });

        assert!(report.sovereign_fitness.is_finite());
        assert_eq!(report.children_evaluated, 8);
        if let Some(w) = winner {
            assert!(w.is_numerically_healthy());
        }
    }
}
