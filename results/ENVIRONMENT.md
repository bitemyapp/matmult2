# Benchmark environment

Measured October 11, 2026 (America/Chicago), using upstream `de0269d` and
optimized source `5b0cb79`. Code was built in Cargo's release profile without
`target-cpu=native` for the headline results. Native builds are labeled as
separate experiments in their filenames.

| | Mac | Workstation |
|---|---|---|
| CPU | Apple M5 Max, 18 cores | AMD Threadripper PRO 9985WX, 64 cores / 128 threads |
| Memory | 128 GiB | Approximately 250 GiB OS-reported RAM |
| OS kernel | Darwin 27.0.0, ARM64 | Linux 6.17.0-1032-oem, x86-64 |
| Rust | 1.96.0 | 1.94.1 |
| LLVM | 22.1.2 | 21.1.8 |
| Existing power setting | AC power | Performance governor |
| Default backend | Accelerate CPU BLAS, library-managed threads | Rust GEMM, AVX-512, 24 reusable workers |

These were live user machines. The workstation had two CPU-heavy background
processes; the Mac had varying compiler, browser, VM, and desktop activity.
No processes were stopped and no power or kernel settings were changed.
Performance comparisons are within each host, using the same compiler and
inputs for before and after. Linux profiling was blocked by the existing
`perf_event_paranoid=4` setting; CPU sampling was performed on the Mac.

Inputs were generated before timing with the repository's SmallRng, seeds
1 and 2, uniformly in [0, 1). Two warmups preceded each measured run. Timed
calls include output allocation, packing, multiplication, and destruction.
Persistent worker setup is logged separately; `cold-gemm` includes creation
and dropping of the worker pool on every call.
