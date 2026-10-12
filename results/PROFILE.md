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
