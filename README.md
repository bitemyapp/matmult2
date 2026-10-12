# CPU matrix multiplication

This checkout preserves the original loops and adds faster CPU-only
implementations for row-major `f64` matrices. [PERFORMANCE.md](PERFORMANCE.md)
has results for an Apple M5 Max and a Threadripper PRO 9985WX, numerical checks,
and links to the raw measurements.

## Use

```rust
use matmult2::{FlatMatrix, CpuMultiplier, fast_mult, random_fill};
let mut a = FlatMatrix::new(1024, 1024);
let mut b = FlatMatrix::new(1024, 1024);
random_fill(&mut a, 1);
random_fill(&mut b, 2);
let product = fast_mult(&a, &b);

// Explicit worker count; construct once and reuse.
let cpu = CpuMultiplier::new(24);
let product = cpu.multiply(&a, &b);

// Preserve the naive loop's separate multiply/add and increasing-k order.
let ordered_product = cpu.multiply_ordered(&a, &b);
```

`fast_mult` uses the current Accelerate CPU BLAS on macOS 13.3 or newer.
Elsewhere it uses Rust `gemm` with a persistent pool of up to 24 workers.
That cap was measured for this project's 1024-square workload; it is not a
universal optimum. `CpuMultiplier` always uses Rust GEMM and allows explicit
worker counts. Its kernels detect CPU features at runtime, including AVX-512
on supported x86 CPUs and NEON on ARM. No GPU backend is linked or used.

All paths retain `f64` precision. `fast_mult` and `CpuMultiplier::multiply` may
use FMA and a different summation order, so their last bits can differ from the
naive reference. `multiply_ordered` preserves that reference order using packed
register tiles in safe Rust. `ordered_mult` is a smaller single-threaded example
of cache blocking and loop interchange. The new paths handle rectangular
matrices, empty dimensions, and partial tiles.

## Reproduce

```sh
cargo test --release
cargo build --release --bin bench_cpu
# Arguments: algorithm, square dimension, measured samples, workers, tile size.
target/release/bench_cpu naive 1024 30
VERIFY=1 target/release/bench_cpu fast 1024 200
VERIFY=1 target/release/bench_cpu gemm 1024 200 24
VERIFY=1 target/release/bench_cpu packed-ordered 1024 100 24
target/release/bench_cpu cold-gemm 1024 100 24

# Full sequential suites, including retuned original implementations.
python3 scripts/benchmark.py --label mac --threads 12 --baseline-threads 18
python3 scripts/benchmark.py --label wx --threads 24 --baseline-threads 48
python3 scripts/summarize.py 'results/mac-*.csv' 'results/wx-*.csv'

# The original Divan suite also includes the new implementations.
cargo bench --bench bench_mult -- fast_cpu
```

The harness generates identical inputs before timing and performs two warmups.
Timing includes allocation, packing, multiplication, and result destruction.
Inputs are never pre-packed across samples. Worker creation is recorded
separately; `cold-gemm` constructs and drops a pool on every call. `VERIFY=1`
compares every output with the scalar reference outside the timed interval,
requiring bit equality for ordered implementations.

The worker argument applies to `gemm`, `cold-gemm`, `packed-ordered`, and
`parallel`. `fast` selects its documented platform default, while `accelerate`
lets Apple select CPU threads. The CSV `threads` column records the supplied
argument; it is not a measurement of the library's active thread count.

`RUSTFLAGS='-C target-cpu=native'` can speed up the portable ordered kernel on
x86, but restricts that binary to the build machine's CPU features. It is not
needed for the headline fast-path results.
