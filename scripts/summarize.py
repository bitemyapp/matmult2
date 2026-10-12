#!/usr/bin/env python3
"""Summarize raw bench_cpu CSVs. No third-party Python dependencies."""
import csv
import glob
import math
import statistics
import sys

writer = csv.writer(sys.stdout)
writer.writerow(["file", "algorithm", "n", "threads", "samples", "median_ms", "p95_ms", "min_ms", "max_ms", "gflops"])
paths = sorted({path for pattern in sys.argv[1:] for path in glob.glob(pattern)})
for path in paths:
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        continue
    times = sorted(float(row["ms"]) for row in rows)
    median = statistics.median(times)
    n = int(rows[0]["n"])
    writer.writerow([path, rows[0]["algorithm"], n, rows[0]["threads"], len(times),
                     f"{median:.6f}", f"{times[math.ceil(len(times)*.95)-1]:.6f}",
                     f"{times[0]:.6f}", f"{times[-1]:.6f}", f"{2*n**3/median/1e6:.3f}"])
