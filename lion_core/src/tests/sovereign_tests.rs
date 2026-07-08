// lion_core/src/tests/sovereign_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

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

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_sovereign_starts_at_tick_zero() {
        let s = Sovereign::new(42);
        assert_eq!(s.tick, 0);
    }

    #[test]
    fn test_sovereign_starts_at_generation_one() {
        let s = Sovereign::new(42);
        assert_eq!(s.generation, 1);
    }

    #[test]
    fn test_sovereign_starts_with_empty_buffer() {
        let s = Sovereign::new(42);
        assert_eq!(s.episode_count(), 0);
    }

    #[test]
    fn test_sovereign_starts_with_no_pending_episode() {
        let s = Sovereign::new(42);
        assert!(!s.has_pending_episode());
    }

    // ── update() ─────────────────────────────────────────────────────────────

    #[test]
    fn test_update_increments_tick() {
        let mut s = Sovereign::new(42);
        s.update(&vision_frame(), 0.0);
        assert_eq!(s.tick, 1);
        s.update(&vision_frame(), 0.0);
        assert_eq!(s.tick, 2);
    }

    #[test]
    fn test_update_returns_known_action() {
        let mut s = Sovereign::new(42);
        let result = s.update(&vision_frame(), 0.0);
        assert!(
            PROCEDURAL_ACTIONS.contains(&result.action),
            "update() returned unknown action: {}", result.action
        );
    }

    #[test]
    fn test_update_creates_pending_episode() {
        let mut s = Sovereign::new(42);
        s.update(&vision_frame(), 0.0);
        assert!(s.has_pending_episode(), "Pending episode must exist after update()");
    }

    #[test]
    fn test_first_update_does_not_record_episode() {
        let mut s = Sovereign::new(42);
        s.update(&vision_frame(), 0.0);
        // First tick: no prior pending episode → nothing recorded yet.
        assert_eq!(s.episode_count(), 0);
    }

    #[test]
    fn test_second_update_records_first_episode() {
        let mut s = Sovereign::new(42);
        s.update(&vision_frame(), 0.0);       // tick 1 — creates pending
        s.update(&vision_frame(), 1.0);        // tick 2 — flushes pending with reward 1.0
        assert_eq!(s.episode_count(), 1);
    }

    #[test]
    fn test_episode_reward_matches_delayed_reward() {
        let mut s = Sovereign::new(42);
        s.update(&vision_frame(), 0.0);        // tick 1
        s.update(&vision_frame(), 0.5);        // tick 2 — tick 1's episode gets reward 0.5

        let ep = s.episodic_buffer.last().unwrap();
        assert_eq!(ep.reward_received, 0.5);
    }

    #[test]
    fn test_negative_reward_increases_stress() {
        let mut s = Sovereign::new(42);
        let stress_before = s.stress();
        s.update(&danger_frame(), 0.0);
        s.update(&danger_frame(), -1.0); // prev_reward = -1.0 → stress increases
        assert!(
            s.stress() > stress_before,
            "Negative reward should increase stress: before={}, after={}",
            stress_before, s.stress()
        );
    }

    #[test]
    fn test_positive_reward_does_not_increase_stress() {
        let mut s = Sovereign::new(42);
        let stress_before = s.stress();
        s.update(&vision_frame(), 1.0);
        // Stress should not increase from positive reward.
        assert!(
            s.stress() <= stress_before + 1e-6,
            "Positive reward must not increase stress"
        );
    }

    // ── flush_pending_episode ─────────────────────────────────────────────────

    #[test]
    fn test_flush_pending_records_final_episode() {
        let mut s = Sovereign::new(42);
        s.update(&vision_frame(), 0.0);         // Creates pending episode.

        assert!(s.has_pending_episode());
        s.flush_pending_episode(0.5);           // Flush with final reward.

        assert!(!s.has_pending_episode());
        assert_eq!(s.episode_count(), 1);
        assert_eq!(s.episodic_buffer.last().unwrap().reward_received, 0.5);
    }

    #[test]
    fn test_flush_on_empty_pending_is_safe() {
        let mut s = Sovereign::new(42);
        // No update called — no pending episode.
        s.flush_pending_episode(1.0); // Should not panic.
        assert_eq!(s.episode_count(), 0);
    }

    // ── hot_swap_brain ────────────────────────────────────────────────────────

    #[test]
    fn test_hot_swap_increments_generation() {
        let mut s = Sovereign::new(42);
        let new_brain = BrainMatrix::new();
        s.hot_swap_brain(new_brain);
        assert_eq!(s.generation, 2);
    }

    #[test]
    fn test_hot_swap_clears_stress() {
        let mut s = Sovereign::new(42);
        s.brain.epigenome.accumulated_stress = 0.9;
        let new_brain = BrainMatrix::new();
        s.hot_swap_brain(new_brain);
        assert_eq!(s.brain.epigenome.accumulated_stress, 0.0);
    }

    // ── Full day simulation ───────────────────────────────────────────────────

    #[test]
    fn test_ten_tick_day_records_nine_episodes() {
        let mut s      = Sovereign::new(42);
        let mut reward = 0.0_f32;

        for t in 1..=10 {
            let frame = if t % 3 == 0 { danger_frame() } else { vision_frame() };
            let result = s.update(&frame, reward);

            reward = if result.action == "FLEE" { 0.5 } else { -1.0 };
        }

        // 10 ticks → 9 episodes recorded (first tick has no prior pending).
        assert_eq!(s.episode_count(), 9);
    }

    #[test]
    fn test_tick_result_gestalt_is_finite() {
        let mut s = Sovereign::new(42);
        let result = s.update(&danger_frame(), 0.0);
        for &v in &result.gestalt {
            assert!(v.is_finite(), "Gestalt component not finite: {}", v);
        }
    }
}
