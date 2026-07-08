// lion_core/benches/lion_bench.rs

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lion_core::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

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

// ── Benchmark: Sequential vs Parallel Night Cycle ────────────────────────────

fn bench_night_cycle(c: &mut Criterion) {
    let s = make_sovereign_with_history(50);

    let mut group = c.benchmark_group("NightCycle");

    for &population in &[5usize, 15, 30] {
        group.bench_with_input(
            BenchmarkId::new("sequential", population),
            &population,
            |b, &pop| {
                b.iter(|| {
                    let mut rng = BrainRng::from_seed(0);
                    run_night_cycle(
                        black_box(&s.brain),
                        black_box(&s.episodic_buffer),
                        &mut rng,
                        pop,
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", population),
            &population,
            |b, &pop| {
                b.iter(|| {
                    run_night_cycle_parallel(
                        black_box(&s.brain),
                        black_box(&s.episodic_buffer),
                        black_box(42u64),
                        pop,
                    )
                });
            },
        );
    }

    group.finish();
}

// ── Benchmark: GEMV Variants ─────────────────────────────────────────────────

fn bench_gemv(c: &mut Criterion) {
    let mut rng = BrainRng::from_seed(0);

    let mut group = c.benchmark_group("TernaryGEMV");

    for &in_sz in &[32usize, 64, 128, 256] {
        let out_sz     = FEATURE_SIZE;
        let raw: Vec<i8> = (0..in_sz * out_sz)
            .map(|_| match rng.gen_index(3) { 0 => -1, 1 => 0, _ => 1 })
            .collect::<Vec<_>>()
            .iter()
            .map(|&x| x as i8)
            .collect();
        let weights        = pack_weights(&raw);
        let input: Vec<i8> = (0..in_sz)
            .map(|_| (rng.gen_index(254) as i8).wrapping_sub(127))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("branchless", in_sz),
            &in_sz,
            |b, &sz| {
                let mut out = vec![0i32; out_sz];
                b.iter(|| {
                    ternary_gemv(
                        black_box(&input),
                        black_box(&weights),
                        black_box(&mut out),
                        sz,
                        out_sz,
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_friendly", in_sz),
            &in_sz,
            |b, &sz| {
                let mut out = vec![0i32; out_sz];
                b.iter(|| {
                    ternary_gemv_auto(
                        black_box(&input),
                        black_box(&weights),
                        black_box(&mut out),
                        sz,
                        out_sz,
                    )
                });
            },
        );
    }

    group.finish();
}

// ── Benchmark: Full Tick ──────────────────────────────────────────────────────

fn bench_sovereign_tick(c: &mut Criterion) {
    let mut s = Sovereign::new(42);
    let mut frame = SensoryInput::new();
    frame.insert(Role::Vision, [0.5_f32; FEATURE_SIZE]);

    c.bench_function("sovereign_tick", |b| {
        b.iter(|| s.update(black_box(&frame), black_box(0.0)))
    });
}

// ── Benchmark: Ternary Encoder ────────────────────────────────────────────────

fn bench_encoder(c: &mut Criterion) {
    let mut rng = BrainRng::from_seed(42);
    let encoder = TernaryEncoder::random(TernaryEncoderConfig::default_small(), &mut rng);
    let raw     = vec![0.5_f32; 64];

    c.bench_function("ternary_encoder_64→64→32", |b| {
        b.iter(|| encoder.encode_f32(black_box(&raw)))
    });
}

// ── Register benchmarks ───────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_night_cycle,
    bench_gemv,
    bench_sovereign_tick,
    bench_encoder,
);
criterion_main!(benches);
