#!/usr/bin/env python3
"""Run the before/after suite serially; keep raw samples and stderr diagnostics."""
import argparse
import os
from pathlib import Path
import platform
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--label", required=True, help="prefix for files under results/")
p.add_argument("--threads", type=int, default=12 if platform.system() == "Darwin" else 24)
p.add_argument("--baseline-threads", type=int, default=18 if platform.system() == "Darwin" else 48)
p.add_argument("--samples", type=int, default=30)
p.add_argument("--fast-samples", type=int, default=200)
p.add_argument("--binary", default="target/release/bench_cpu")
args = p.parse_args()
root = Path(__file__).resolve().parent.parent
os.chdir(root)
Path("results").mkdir(exist_ok=True)
cases = [
    ("naive", "naive", args.samples, 1),
    ("fast-a", "fast", args.fast_samples, args.threads),
    ("tiled", "tiled", args.samples, 1),
    ("parallel4", "parallel", args.samples, 4),
    ("parallel-best", "parallel", args.samples, args.baseline_threads),
    ("gemm1", "gemm", 40, 1),
    ("gemm4", "gemm", 100, 4),
    ("gemm-best", "gemm", args.fast_samples, args.threads),
    ("ordered", "packed-ordered", 100, args.threads),
    ("cold", "cold-gemm", 100, args.threads),
    ("fast-b", "fast", args.fast_samples, args.threads),
]
for name, algorithm, samples, threads in cases:
    stem = Path("results") / f"{args.label}-{name}"
    command = [args.binary, algorithm, "1024", str(samples), str(threads), "8"]
    print(" ".join(command), flush=True)
    env = dict(os.environ)
    if algorithm in ("fast", "gemm", "packed-ordered"):
        env["VERIFY"] = "1"
    with stem.with_suffix(".csv").open("w") as out, stem.with_suffix(".log").open("w") as err:
        subprocess.run(command, stdout=out, stderr=err, env=env, check=True)
    print(stem.with_suffix(".log").read_text().strip(), flush=True)
