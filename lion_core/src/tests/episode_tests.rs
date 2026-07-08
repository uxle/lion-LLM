// lion_core/src/tests/episode_tests.rs

#[cfg(test)]
mod tests {
    use crate::*;

    fn dummy_frame() -> SensoryInput {
        let mut f = SensoryInput::new();
        f.insert(Role::Vision, [0.5_f32; FEATURE_SIZE]);
        f
    }

    fn dummy_episode(reward: f32) -> Episode {
        let mut ep = Episode::new(
            dummy_frame(),
            [0.1_f32; FEATURE_SIZE],
            ActionLabel::new("FLEE"),
            0.3,
        );
        ep.set_reward(reward);
        ep
    }

    // ── Episode struct ────────────────────────────────────────────────────────

    #[test]
    fn test_episode_created_with_zero_reward() {
        let ep = Episode::new(
            dummy_frame(),
            [0.0_f32; FEATURE_SIZE],
            ActionLabel::new("WANDER"),
            0.0,
        );
        assert_eq!(ep.reward_received, 0.0);
    }

    #[test]
    fn test_episode_set_reward() {
        let mut ep = Episode::new(
            dummy_frame(),
            [0.0_f32; FEATURE_SIZE],
            ActionLabel::new("FLEE"),
            0.1,
        );
        ep.set_reward(0.5);
        assert_eq!(ep.reward_received, 0.5);
    }

    #[test]
    fn test_episode_was_positive() {
        let ep = dummy_episode(1.0);
        assert!(ep.was_positive());
        assert!(!ep.was_negative());
    }

    #[test]
    fn test_episode_was_negative() {
        let ep = dummy_episode(-1.0);
        assert!(ep.was_negative());
        assert!(!ep.was_positive());
    }

    #[test]
    fn test_episode_action_str_matches_label() {
        for &action in PROCEDURAL_ACTIONS {
            let ep = Episode::new(
                dummy_frame(),
                [0.0_f32; FEATURE_SIZE],
                ActionLabel::new(action),
                0.0,
            );
            assert_eq!(ep.action_str(), action);
        }
    }

    // ── EpisodicBuffer ────────────────────────────────────────────────────────

    #[test]
    fn test_buffer_starts_empty() {
        let buf = EpisodicBuffer::default_capacity();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_buffer_records_episode() {
        let mut buf = EpisodicBuffer::default_capacity();
        buf.record(dummy_episode(1.0));
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_evicts_oldest_when_full() {
        let mut buf = EpisodicBuffer::new(3);
        buf.record(dummy_episode(1.0));
        buf.record(dummy_episode(2.0));
        buf.record(dummy_episode(3.0));

        assert_eq!(buf.len(), 3);

        // Recording a 4th episode should evict the first (reward 1.0).
        buf.record(dummy_episode(4.0));

        assert_eq!(buf.len(), 3);

        // The oldest episode (reward 1.0) should be gone.
        let rewards: Vec<f32> = buf.as_slice().map(|e| e.reward_received).collect();
        assert!(!rewards.contains(&1.0_f32));
        assert!(rewards.contains(&4.0_f32));
    }

    #[test]
    fn test_buffer_fifo_order() {
        let mut buf = EpisodicBuffer::new(5);
        for i in 0..5_i32 {
            buf.record(dummy_episode(i as f32));
        }

        let rewards: Vec<f32> = buf.as_slice().map(|e| e.reward_received).collect();
        assert_eq!(rewards, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_buffer_last_returns_most_recent() {
        let mut buf = EpisodicBuffer::default_capacity();
        buf.record(dummy_episode(1.0));
        buf.record(dummy_episode(2.0));
        buf.record(dummy_episode(3.0));

        assert_eq!(buf.last().unwrap().reward_received, 3.0);
    }

    #[test]
    fn test_buffer_clear_empties_history() {
        let mut buf = EpisodicBuffer::default_capacity();
        buf.record(dummy_episode(1.0));
        buf.record(dummy_episode(-1.0));
        buf.clear();
        assert!(buf.is_empty());
    }

    // ── RewardSummary ─────────────────────────────────────────────────────────

    #[test]
    fn test_reward_summary_counts() {
        let mut buf = EpisodicBuffer::default_capacity();
        buf.record(dummy_episode(1.0));
        buf.record(dummy_episode(-1.0));
        buf.record(dummy_episode(0.0));
        buf.record(dummy_episode(0.5));
        buf.record(dummy_episode(-0.5));

        let summary = buf.reward_summary();
        assert_eq!(summary.positive_count, 2);
        assert_eq!(summary.negative_count, 2);
        assert_eq!(summary.neutral_count,  1);
        assert_eq!(summary.episode_count,  5);
    }

    #[test]
    fn test_reward_summary_mean_reward() {
        let mut buf = EpisodicBuffer::default_capacity();
        buf.record(dummy_episode(2.0));
        buf.record(dummy_episode(-1.0));
        buf.record(dummy_episode(0.0));

        let summary = buf.reward_summary();
        let expected_mean = 1.0 / 3.0;
        assert!((summary.mean_reward() - expected_mean).abs() < 1e-5);
    }

    #[test]
    fn test_reward_summary_empty_buffer_mean_is_zero() {
        let buf = EpisodicBuffer::default_capacity();
        assert_eq!(buf.reward_summary().mean_reward(), 0.0);
    }
}
