"""Head-to-head speed+accuracy benchmark: R imputeLCMD vs pyimputelcmd.

Runs ``impute.MinDet`` / ``impute.MinProb`` / ``impute.QRILC`` on a
synthetic 5000-protein × 12-sample left-censored matrix and reports:

* wall-clock time (mean across N runs) for each method, R vs Python
* MAE / Pearson r between Python and R imputed values
* For MinDet: exact-equality fraction (should be 100%)

The synthetic matrix is generated deterministically in NumPy and dumped
to TSV so both languages see identical input.

Usage
-----
    python examples/benchmark.py [--runs 3] [--n-proteins 5000] [--n-samples 12]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pyimputelcmd as pyi


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _make_matrix(n_proteins: int, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic log-intensity matrix with ~15-25% left-censored NaNs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(20.0, 1.0, (n_proteins, n_samples))
    X[X < 19.0] = np.nan
    return pd.DataFrame(
        X,
        index=[f"prot_{i:05d}" for i in range(n_proteins)],
        columns=[f"S{j:02d}" for j in range(n_samples)],
    )


R_BENCH_SCRIPT = r"""
suppressPackageStartupMessages({ library(imputeLCMD) })
args <- commandArgs(trailingOnly = TRUE)
input_tsv <- args[[1]]; out_dir <- args[[2]]
runs      <- as.integer(args[[3]])
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
X <- as.matrix(read.table(input_tsv, sep = "\t", header = TRUE,
                          row.names = 1, check.names = FALSE,
                          na.strings = c("NA","")))

time_call <- function(fn, ...) {
  t0 <- proc.time()[[3]]; out <- fn(...); t1 <- proc.time()[[3]]; list(time = t1 - t0, out = out)
}

times <- list(mindet = c(), minprob = c(), qrilc = c())
for (i in seq_len(runs)) {
  set.seed(0)
  t <- time_call(impute.MinDet,  X, q = 0.01); times$mindet  <- c(times$mindet,  t$time); md <- t$out
  set.seed(0)
  t <- time_call(impute.MinProb, X, q = 0.01, tune.sigma = 1.0); times$minprob <- c(times$minprob, t$time); mp <- t$out
  set.seed(0)
  t <- time_call(function(M) impute.QRILC(M, tune.sigma = 1.0)[[1]], X)
  times$qrilc <- c(times$qrilc, t$time); qr <- t$out
}

write.table(md, file.path(out_dir, "r_mindet.tsv"),  sep = "\t", quote = FALSE, col.names = NA)
write.table(mp, file.path(out_dir, "r_minprob.tsv"), sep = "\t", quote = FALSE, col.names = NA)
write.table(qr, file.path(out_dir, "r_qrilc.tsv"),   sep = "\t", quote = FALSE, col.names = NA)

cat(sprintf("R_MINDET=%.4f\n",  mean(times$mindet)))
cat(sprintf("R_MINPROB=%.4f\n", mean(times$minprob)))
cat(sprintf("R_QRILC=%.4f\n",   mean(times$qrilc)))
"""


def run_r_bench(input_tsv: Path, work: Path, runs: int) -> dict[str, float]:
    script = work / "r_bench.R"
    script.write_text(R_BENCH_SCRIPT)
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {script} {input_tsv} {work} {runs}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, check=True,
    )
    times = {}
    for line in res.stdout.splitlines():
        if "=" in line and line.startswith("R_"):
            key, val = line.split("=", 1)
            times[key.lower().removeprefix("r_")] = float(val)
    return times


def run_python_bench(df: pd.DataFrame, runs: int) -> dict[str, float]:
    fns = {
        'mindet':  lambda: pyi.impute_mindet(df, q=0.01),
        'minprob': lambda: pyi.impute_minprob(df, q=0.01, tune_sigma=1.0, seed=0),
        'qrilc':   lambda: pyi.impute_qrilc(df, tune_sigma=1.0, seed=0),
    }
    times = {}
    for name, fn in fns.items():
        elapsed = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            elapsed.append(time.perf_counter() - t0)
        times[name] = float(np.mean(elapsed))
    return times


def compare_outputs(df: pd.DataFrame, work: Path) -> dict[str, dict]:
    """Pairwise MAE / Pearson r between R and Python output (over imputed cells only)."""
    miss = df.isna().to_numpy()
    report = {}
    for name in ['mindet', 'minprob', 'qrilc']:
        py_fn = {
            'mindet':  pyi.impute_mindet,
            'minprob': lambda d: pyi.impute_minprob(d, seed=0),
            'qrilc':   lambda d: pyi.impute_qrilc(d, seed=0),
        }[name]
        py = py_fn(df).to_numpy() if name == 'mindet' else py_fn(df).to_numpy()
        r = pd.read_csv(work / f"r_{name}.tsv", sep="\t", index_col=0).to_numpy()
        py_imp = py[miss]
        r_imp = r[miss]
        mae = float(np.mean(np.abs(py_imp - r_imp)))
        pr = float(np.corrcoef(py_imp, r_imp)[0, 1]) if py_imp.size > 1 else float('nan')
        exact = float(np.allclose(py_imp, r_imp, atol=1e-12))
        report[name] = {
            'mae': mae,
            'pearson_r': pr,
            'bit_exact': exact,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-proteins", type=int, default=5000)
    ap.add_argument("--n-samples", type=int, default=12)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    df = _make_matrix(args.n_proteins, args.n_samples)
    pct_missing = df.isna().mean().mean()
    print(f"Matrix: {df.shape[0]} proteins × {df.shape[1]} samples, "
          f"{pct_missing:.1%} missing")

    input_tsv = WORK / "input.tsv"
    df.to_csv(input_tsv, sep="\t", na_rep="NA", index=True, index_label="protein")
    print(f"Input dumped to {input_tsv}")

    print(f"\n--- Python timings (mean of {args.runs} runs) ---")
    py_times = run_python_bench(df, args.runs)
    for k, v in py_times.items():
        print(f"  py {k:8s} {v*1000:8.2f} ms")

    print(f"\n--- R timings (mean of {args.runs} runs) ---")
    r_times = run_r_bench(input_tsv, WORK, args.runs)
    for k, v in r_times.items():
        print(f"  R  {k:8s} {v*1000:8.2f} ms")

    print("\n--- Speedup (R / Python) ---")
    for k in ['mindet', 'minprob', 'qrilc']:
        sp = r_times[k] / py_times[k] if py_times[k] > 0 else float('inf')
        print(f"  {k:8s} {sp:7.2f}×")

    print("\n--- Accuracy (Python vs R) ---")
    acc = compare_outputs(df, WORK)
    for k, v in acc.items():
        flag = "✓" if v['bit_exact'] else "≈"
        print(
            f"  {flag} {k:8s} MAE={v['mae']:.4g}  r={v['pearson_r']:.4f}  "
            f"bit-exact={bool(v['bit_exact'])}"
        )

    summary = {
        'shape': df.shape,
        'pct_missing': pct_missing,
        'py_times_ms': {k: v * 1000 for k, v in py_times.items()},
        'r_times_ms': {k: v * 1000 for k, v in r_times.items()},
        'speedup': {k: r_times[k] / py_times[k] for k in py_times},
        'accuracy': acc,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report → {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
