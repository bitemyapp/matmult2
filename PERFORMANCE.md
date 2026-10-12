# CPU matrix multiplication results

Measured October 11, 2026 (America/Chicago). Upstream:
[`lukewassink/matmult2`](https://github.com/lukewassink/matmult2), commit
`de0269d`. Final measured source: `5b0cb79`. No GPU was used.

## Before and after

Multiplication of two **1024 × 1024 f64** matrices. Times include output
allocation, any input packing, computation, and output destruction.

| Machine | Original naive, median | Optimized default, median | Speedup | Optimized p95 |
|---|---:|---:|---:|---:|
| Apple M5 Max MacBook Pro | 1,329.53 ms | **4.258 ms** | **312×** | 5.757 ms |
| Threadripper PRO 9985WX | 2,750.64 ms | **0.980 ms** | **2,807×** | 1.070 ms |

The naive comparison is to `naive_local_mult`, the local accumulator version
shown in the original benchmark discussion. Each final baseline has 30 samples. Each optimized headline
combines **both** 200-sample final runs, rather than selecting the faster run.
The individual optimized medians were 3.806 and 4.429 ms on the Mac, and 0.980
and 0.979 ms on the workstation. The optimized throughput is approximately
504 GFLOP/s on the Mac and 2.19 TFLOP/s on the workstation, using 2N³ operations.

I also retuned the original parallel implementation, screening tile sizes and
thread counts before collecting another 30 samples:

| Machine | Best original parallel configuration tested | Original median | New default median | Speedup |
|---|---|---:|---:|---:|
| M5 Max | tile 8, 18 threads | 41.430 ms | 4.258 ms | **9.7×** |
| 9985WX | tile 8, 48 threads | 15.867 ms | 0.980 ms | **16.2×** |

Thus the improvement is substantial even against an original implementation
using many CPU cores. The Rust-only GEMM comparison at an equal four workers
also improved 117.534 → 11.566 ms on the Mac (**10.2×**) and 133.873 → 3.907 ms
on the workstation (**34.3×**).

## What changed

1. **Contiguous memory access.** The original inner loop reads B[k * 1024 + j],
   an 8 KB stride. Its tiling keeps that strided dot product inside each tile.
   `ordered_mult` instead traverses adjacent B and output elements and reuses
   cache-sized blocks. Independent output columns can be vectorized without
   changing the order of any dot product.
2. **Packing and register tiles.** `CpuMultiplier::multiply_ordered` packs B
   on every call and accumulates a 4 × 8 output tile in registers, reusing
   each B value across four rows. It is written in safe Rust and retains
   separate multiply/add in increasing k order.
3. **Tuned SIMD kernels.** `CpuMultiplier::multiply` uses the Rust
   [`gemm` crate](https://docs.rs/gemm/0.19.0/gemm/), with AVX-512 runtime
   dispatch enabled on x86 and NEON on ARM. It uses FMA and may reassociate
   sums. The 9985WX benefits enormously from its wide SIMD units.
4. **Reusable workers.** A `CpuMultiplier` owns its Rayon pool. `fast_mult`
   reuses a process-wide pool on non-macOS systems, capped at 24 workers for
   this workload. Sweeping through 128 workers showed that using all logical
   CPUs was slower at this size.
5. **Native Mac CPU BLAS.** On macOS, `fast_mult` uses the current Accelerate
   `cblas_dgemm$NEWLAPACK` interface, whose declaration was checked against the
   installed SDK. It beat both the older CBLAS symbol and the Rust NEON path.
   Accelerate chooses its CPU thread count; this API requires macOS 13.3+.

There is no GPU backend, precision reduction, multiplication-result cache,
precomputed transpose reused between timed samples, or omitted packing cost.

## Detailed final runs

Each cell below is **median / p95, in milliseconds**. The new default row
combines the two final runs. Other rows use their single final run.

| Implementation | M5 Max | 9985WX |
|---|---:|---:|
| Original naive, 1 thread | 1,329.534 / 1,674.788 | 2,750.638 / 2,793.985 |
| Original tiled, tile 8, 1 thread | 439.008 / 464.473 | 481.192 / 496.870 |
| Original parallel, tile 8, 4 threads | 117.534 / 138.591 | 133.873 / 141.983 |
| Best original parallel tested | 41.430 / 43.833 | 15.867 / 17.564 |
| Rust SIMD GEMM, 1 thread | 42.163 / 43.852 | 14.072 / 14.125 |
| Rust SIMD GEMM, 4 threads | 11.566 / 12.638 | 3.907 / 4.478 |
| Rust SIMD GEMM, 12 / 24 threads | 7.703 / 13.768 | 1.018 / 1.054 |
| Packed ordered, 12 / 24 threads | 11.702 / 16.654 | 13.451 / 14.727 |
| **New default** | **4.258 / 5.757** | **0.980 / 1.070** |

These rows expose the noisy Mac interval rather than substituting a better
exploratory result. Earlier sweeps reached approximately 4.8 ms for Mac Rust
GEMM and 8.8 ms for its ordered path. On x86 the ordered path responds strongly
to `-C target-cpu=native`: 29.171 ms single-threaded and 2.722 ms with 16 workers
in a separate 20-sample experiment. These are useful options when retaining
the original arithmetic order matters; they are not the headline comparison.

The original Divan harness independently measured the default at **3.463 ms**
on the Mac and **0.969 ms** on the workstation (100 samples each). Those
separate checks are retained in `mac-divan-fast.txt` and `wx-divan-fast.txt`.

## Correctness and numerical behavior

- Release tests passed on both machines: **10 tests** covering original
  kernels, new kernels, rectangular shapes, empty dimensions, odd dimensions,
  register/cache-block boundaries, invalid dimensions, and zero worker counts.
- Signed, scaled, and cancellation-heavy inputs are checked against an error
  bound proportional to epsilon × k × sum(abs(products)).
- Full 1024-square final runs compare **all 1,048,576 elements** with the
  original naive implementation, outside timing. Maximum absolute differences
  were **1.71e-13** for the Mac default and **1.25e-12** for the workstation
  default. Values are approximately 256 for this input distribution.
- Both ordered implementations preserve the naive operation order; full-size
  `VERIFY=1` checks require bit equality. The original, blocked, and packed
  17-square golden output files also have identical SHA-256 hashes.
- The fast implementations retain double precision but are **not promised
  bit-identical** to the naive loop because of FMA and reduction ordering.
- Matrix construction now rejects dimension multiplication overflow before
  allocating. FFI calls operate on validated, non-aliasing row-major buffers.
- `cargo fmt --check` passed. Clippy completed with only the upstream unused
  methods/helper and an upstream redundant test borrow warning.

## Measurement conditions and limits

The Mac reports an M5 Max with 18 CPU cores and 128 GiB RAM, on AC power, using
rustc 1.96.0 / LLVM 22.1.2. The workstation reports a 64-core / 128-thread
9985WX and approximately 250 GiB OS-reported RAM in `free`,
using rustc 1.94.1 / LLVM 21.1.8. Its existing governor was `performance`.
Comparisons are **within each host and toolchain**. No governor or kernel
settings were changed.

Both were live user machines. The workstation had two CPU-heavy background
processes; the Mac had changing compiler, browser, VM, and desktop activity.
None was stopped. Mac medians drifted across the session, which is why the
report retains p95s, both final runs, and earlier runs. These
are not measurements from an isolated performance lab.

Inputs use the repository's `SmallRng`, seeds 1 and 2, uniform in [0, 1), built
before the timer. Two warmups precede individual timed calls. Inputs remain
resident, as in the original benchmark. Each call allocates its output and
packs as necessary; the output is consumed by `black_box` and dropped before
the timer stops. Worker setup is logged separately. The CSV `threads` column
is the supplied argument; `fast` uses its platform default and Accelerate
selects its own threads.

Startup is distinct from steady state. The workstation default's first call
was approximately **21–22 ms**, including lazy setup, versus **0.98 ms** warm.
Constructing and dropping a fresh 24-worker Rust engine on every call measured
**2.294 ms** median / 3.227 ms p95. The Mac default's first calls were roughly
5.7–6.7 ms. A separate process memory check reported peak RSS of about **41 MiB**
for the Mac default and **26.4 MiB** for the workstation default, including
inputs, output, library state, and workers.

Profiling put 99.95% of the initial Mac samples inside the naive multiplication
kernel. Later profiles put active work in the Rust NEON microkernel or Apple's
CPU BLAS. Linux perf was unavailable under the existing `perf_event_paranoid=4`
policy; it was left unchanged. See [the profile ledger](results/PROFILE.md).
Low-sample extreme percentiles cannot establish reliable tail latency; raw
maxima are retained rather than advertised as p99.9 estimates.

Additional 60-sample size checks of the same default (no per-size retuning):

| Square dimension | M5 Max median | 9985WX median |
|---|---:|---:|
| 256 | 0.081 ms | 0.050 ms |
| 512 | 0.599 ms | 0.217 ms |
| 1025 | 3.525 ms | 1.101 ms |
| 2048 | 27.302 ms | 10.252 ms |

I tested tile/thread sweeps, native code generation, both Apple BLAS interfaces,
and the optional GEMM AMX feature. The latter's CPU detector does not recognize
M5, so it remained on NEON and is disabled in the delivered configuration.
The reported default was the strongest practical CPU path among these tested
choices; this is not a claim of a mathematical global optimum. Scala was not
benchmarked here, so these results do not establish a Rust-versus-JVM ranking.

## Reproduce and inspect

See [README.md](README.md) for a short summary, usage, and reproduction commands.

- [Final per-run CSV summary](results/final-summary.csv)
- [Mac final run log](results/mac-final-suite.log)
- [Workstation final run log](results/wx-final-suite.log)
- [Benchmark environment](results/ENVIRONMENT.md)
- [Scenario contract](results/SCENARIO.md) and [profile/decision ledger](results/PROFILE.md)
- [Golden hashes](results/golden-sha256.txt)

`results/*-final-*.csv` contains all final raw samples. `*-screen-*`, `*-tune-*`,
`*-native-*`, `*-amx-*`, and other files preserve exploratory experiments and
should not be mistaken for the final aggregate. This fork preserves the
upstream implementation alongside the new CPU backends.
