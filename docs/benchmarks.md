# Benchmarks & Performance

This document explains available benchmarks and how to measure performance.

Running benchmarks
------------------

Benchmarks are located under `lion_core/benches/`. Run benchmarks with:

```bash
cargo bench
```

Profiling
---------

- Use `perf`, `cargo-flamegraph`, or `pprof` for CPU and allocation profiles.
- Collect wall-clock time and memory usage when running long experiments.

Performance tips
----------------

- Use release builds for realistic performance (`cargo build --release`).
- Reduce logging in hot loops and prefer aggregated metrics.
- When comparing changes, keep the same dataset and RNG seed for reproducibility.
