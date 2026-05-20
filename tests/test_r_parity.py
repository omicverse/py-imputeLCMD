"""R-parity tests against Bioconductor imputeLCMD.

These tests shell out to the CMAP R env to run the original
``impute.MinDet`` / ``impute.MinProb`` / ``impute.QRILC`` on the same
input matrix the Python side imputes, then compare:

* MinDet: bit-exact (deterministic algorithm — must match within FP).
* MinProb: distribution-level (KS test on imputed marginals — R and
  Python use different PRNGs so individual draws diverge).
* QRILC: regression-coefficient bit-exact (the (mu, sigma) fitted by
  OLS must match R's ``lm()``) and KS test on the truncated-normal
  draws.

If the CMAP R env or imputeLCMD is missing, these tests skip. The
fixture matrix is generated deterministically with NumPy so the same
input can be reproduced in R.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyimputelcmd import impute_mindet, impute_minprob, impute_qrilc


HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
RSCRIPT = os.environ.get(
    "PYIMPUTELCMD_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript",
)
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"


def _r_available() -> bool:
    if not Path(RSCRIPT).exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate /scratch/users/steorra/env/CMAP "
             "&& Rscript -e 'library(imputeLCMD); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or imputeLCMD not installed; set PYIMPUTELCMD_RSCRIPT.",
)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    p = tmp_path_factory.mktemp("r_parity")
    yield p


def _make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = rng.normal(20.0, 1.0, (500, 6))
    X[X < 19.0] = np.nan
    return pd.DataFrame(
        X,
        index=[f"prot_{i}" for i in range(X.shape[0])],
        columns=[f"S{j}" for j in range(X.shape[1])],
    )


def _write_input(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", na_rep="NA", index=True, index_label="protein")


def _run_r(input_tsv: Path, out_dir: Path, method: str, seed: int = 0) -> dict:
    cmd = (
        f"source {CONDA_BIN} && conda activate /scratch/users/steorra/env/CMAP "
        f"&& Rscript {R_DRIVER} {input_tsv} {out_dir} {method} {seed}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, check=True,
    )
    return {"stdout": res.stdout, "stderr": res.stderr}


def _read_r_imputed(out_dir: Path, method: str) -> pd.DataFrame:
    return pd.read_csv(
        out_dir / f"{method}_imputed.tsv",
        sep="\t", index_col=0,
    )


# --------------------------------------------------------------------------- #
# MinDet — deterministic → must match bit-for-bit                             #
# --------------------------------------------------------------------------- #
def test_mindet_bit_exact_vs_R(workdir):
    df = _make_dataset()
    input_tsv = workdir / "input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "mindet", "mindet")

    py = impute_mindet(df, q=0.01)
    r = _read_r_imputed(workdir / "mindet", "mindet")
    # Align column order (R may reorder if column names contain hyphens etc.)
    r = r[py.columns]
    np.testing.assert_allclose(
        py.to_numpy(), r.to_numpy(), rtol=0, atol=1e-12,
        err_msg="MinDet output diverges from R imputeLCMD",
    )


# --------------------------------------------------------------------------- #
# MinProb — stochastic → KS-test on each column's imputed marginal            #
# --------------------------------------------------------------------------- #
def test_minprob_distribution_matches_R(workdir):
    df = _make_dataset()
    input_tsv = workdir / "input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "minprob", "minprob")

    py = impute_minprob(df, q=0.01, tune_sigma=1.0, seed=0)
    r = _read_r_imputed(workdir / "minprob", "minprob")
    r = r[py.columns]
    miss = df.isna()
    for col in df.columns:
        m = miss[col].to_numpy()
        if not m.any():
            continue
        py_imp = py[col].to_numpy()[m]
        r_imp = r[col].to_numpy()[m]
        # KS test — accept if p > 0.01 (Bonferroni across 6 cols ⇒ p > 0.06).
        ks = stats.ks_2samp(py_imp, r_imp)
        assert ks.pvalue > 0.01, (
            f"MinProb distribution mismatch in {col}: "
            f"KS stat={ks.statistic:.3f}, p={ks.pvalue:.4g}"
        )


# --------------------------------------------------------------------------- #
# QRILC — regression coefficients must match exactly; sampled values via KS   #
# --------------------------------------------------------------------------- #
def test_qrilc_regression_coefs_match_R(workdir):
    """``lm()`` in R and our closed-form OLS must agree on (μ, σ̂) up to FP.

    The test compares the intercept / slope pair per sample directly,
    not the imputed values (which depend on the PRNG).
    """
    df = _make_dataset()
    input_tsv = workdir / "input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "qrilc", "qrilc")

    r_coefs = pd.read_csv(workdir / "qrilc" / "qrilc_coefs.tsv",
                          sep="\t", index_col=0)
    # Re-run the Python regression exactly the same way ``impute_qrilc`` does.
    py_coefs = {}
    X = df.to_numpy()
    for j, sample in enumerate(df.columns):
        col = X[:, j]
        miss = np.isnan(col)
        observed = col[~miss]
        p_NAs = miss.mean()
        upper_q = 0.99
        step_norm = (upper_q - p_NAs) / (upper_q * 100.0)
        q_norm_probs = p_NAs + 0.001 + np.arange(100) * step_norm
        q_emp_probs = 0.001 + np.arange(100) * 0.01
        q_norm = stats.norm.ppf(q_norm_probs)
        q_sample = np.quantile(observed, q_emp_probs, method='linear')
        x_bar = q_norm.mean()
        y_bar = q_sample.mean()
        slope = ((q_norm - x_bar) * (q_sample - y_bar)).sum() / ((q_norm - x_bar) ** 2).sum()
        intercept = y_bar - slope * x_bar
        py_coefs[sample] = (intercept, slope)

    for sample, (py_int, py_slope) in py_coefs.items():
        r_int = float(r_coefs.loc[sample, "intercept"])
        r_slope = float(r_coefs.loc[sample, "slope"])
        # Allow small FP slack — R's lm() and our closed-form OLS use
        # different numerical pipelines (QR vs sums-of-squares).
        assert abs(py_int - r_int) < 1e-6, (
            f"QRILC intercept mismatch in {sample}: py={py_int}, R={r_int}"
        )
        assert abs(py_slope - r_slope) < 1e-6, (
            f"QRILC slope mismatch in {sample}: py={py_slope}, R={r_slope}"
        )


def test_qrilc_sampled_distribution_matches_R(workdir):
    df = _make_dataset()
    input_tsv = workdir / "input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "qrilc", "qrilc")

    py = impute_qrilc(df, tune_sigma=1.0, seed=0)
    r = _read_r_imputed(workdir / "qrilc", "qrilc")
    r = r[py.columns]
    miss = df.isna()
    for col in df.columns:
        m = miss[col].to_numpy()
        if not m.any():
            continue
        py_imp = py[col].to_numpy()[m]
        r_imp = r[col].to_numpy()[m]
        ks = stats.ks_2samp(py_imp, r_imp)
        assert ks.pvalue > 0.01, (
            f"QRILC distribution mismatch in {col}: "
            f"KS stat={ks.statistic:.3f}, p={ks.pvalue:.4g}"
        )
