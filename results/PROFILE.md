# Baseline profile and optimization ledger

Upstream: de0269d. Measurement-only harness: 918ee17.

| Rank | Location | Evidence | Cost |
|---|---|---|---|
| 1 | `naive_local_mult`, inner dot product | `mac-before-profile.json.gz` and symbol sidecar | 7,387 / 7,391 leaf samples (99.95%) |
| 2 | `random_fill` (outside timed work) | same sample | 2 samples |
| 3 | harness / standard library | same sample | 2 samples |

Source inspection: each inner iteration loads B[k * 1024 + j], an 8192-byte
stride, and updates one dependent scalar accumulator. Tiling improves cache
residency but keeps a strided reduction as the innermost loop. Timing and the
sample agree that the multiplication kernel, not RNG/allocation, is the target.
Linux perf is blocked by the existing perf_event_paranoid=4 setting; no global
setting was changed. Mac sampling supplies direct hotspot evidence.

| Candidate | Impact | Confidence | Effort | Score | Hypothesis |
|---|---:|---:|---:|---:|---|
| Interchange inner loops, contiguous row slices, cache blocking | 5 | 5 | 1 | 25 | SIMD across independent output columns, preserve k order |
| Register tiles and packed CPU GEMM | 5 | 5 | 3 | 8.3 | Reuse operands, amortize loads, use vector FMA |
| Reusable CPU thread pool and measured thread count | 4 | 4 | 2 | 8 | Original creates OS threads on every call |
| Native CPU BLAS on Apple | 5 | 4 | 2 | 10 | Test vendor CPU kernel as hardware-specific ceiling |

The unchanged reference remains public. The strict implementation must retain
separate multiply/add and increasing k order for each output. Fast APIs are
separately named and document FMA/reassociation, which already differs from
the reference's original tiled reduction. Validate those numerically rather
than claiming bit identity. Inputs and RNG seeds stay unchanged throughout.

## Later measurements and decisions

- Contiguous cache-blocked loops: 181 ms on the Mac versus 1,203 ms naive in
  the initial run. The 17-square SHA-256 matches the original, and rectangular
  boundary tests compare every bit. Per-element k order and separate
  multiply/add are preserved; independent output elements are interleaved.
- Packed register tiles: build B panels on every call, reuse each B value
  across four rows, keep 32 accumulators in registers. Preserve per-element
  arithmetic order, including tails. Worker bands are disjoint; there is no
  cross-worker reduction.
- Rust GEMM: `mac-gemm-profile.json.gz` has 9,330 samples in the NEON `x4x4`
  microkernel, 186 in LHS packing, 27 in RHS packing, and 14 in bzero. Worker
  waits/yields account for another 1,832 / 326 samples; these are not computation
  time. Removing zero initialization was rejected as an insignificant Mac
  hotspot. SIMD microkernels remain the dominant active computation.
- Thread sweeps: more workers are not automatically faster at 1024 square.
  24 gave a good 9985WX default; 12 was a reasonable Mac Rust setting. Larger
  counts are represented in the raw sweeps.
- Native compilation did not materially improve dispatched GEMM. It did
  improve the exact packed x86 kernel (about 136 to 29 ms single-threaded in
  exploratory runs), so the option is documented.
- Accelerate: the current `cblas_dgemm$NEWLAPACK` interface beat the deprecated
  CBLAS symbol. Its LP64 ABI was checked against installed SDK headers. The
  minimum macOS version is documented. No GPU API is involved.
- Experimental GEMM AMX was tested, but its detector only recognizes M1/M2/M3
  (`gemm-common` 0.19.0 `cache.rs`), falling back to NEON on M5. It is not enabled
  in the delivered dependency settings.
- Fast APIs document rounding changes. Signed/scaled/cancellation tests use a
  bound based on epsilon * k * sum(abs(products)); full 1024-square checks are
  in final logs. The legacy implementations remain available for reproduction.

The public fork includes timing samples and these profiling observations.
Raw sampling traces and machine process dumps are retained locally.
