# Measurement contract

Multiply two row-major 1024 x 1024 f64 matrices, generated outside timing using
the repository's SmallRng seeds 1 and 2 and uniform [0, 1) distribution.
Time the complete returning API, including output allocation/zeroing, any
packing, computation, and output destruction. Two warmups, then at least 20
individual samples. Report median and p95, plus raw samples. Extreme tail
percentiles from 20 samples are descriptive maxima, not reliable tail estimates.

The untouched naive_local_mult, tiled_mult, and parallel_tiled_mult are the
reference implementations at upstream de0269d. Compare before/after within
each host. No GPU, kernel/governor changes, or termination of unrelated work.
Record toolchain differences between hosts. Optimized APIs must support
rectangular matrices and block tails, with deterministic correctness tests.
Preserve the reference functions so every result remains reproducible.

Warm resident inputs reflect the original Divan workload. This is compute
throughput, not disk or cold-cache latency. Floating-point changes, if used,
must be exposed explicitly and validated against the reference.
