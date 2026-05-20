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

from pyimputelcmd import (
    impute_knn,
    impute_mindet,
    impute_minprob,
    impute_mle,
    impute_qrilc,
    impute_svd,
    impute_zero,
    model_selector,
)


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


# --------------------------------------------------------------------------- #
# v0.1.1 MAR-wrapper parity                                                   #
# --------------------------------------------------------------------------- #
def _make_correlated_dataset() -> pd.DataFrame:
    """Low-rank (rank-2) + noise matrix with inter-sample correlation.

    KNN / SVD / MLE only recover a *signal* when the samples are
    correlated — a purely-random matrix has a diagonal covariance and
    leaves nothing for any of these imputers to reconstruct. This
    fixture mirrors a realistic proteomics intensity matrix (a handful
    of latent factors driving co-variation across samples) so the
    R-vs-Python parity checks are meaningful.
    """
    rng = np.random.default_rng(7)
    n_prot, n_samp = 800, 6
    loadings = rng.normal(0.0, 1.0, (n_prot, 2))
    factors = rng.normal(0.0, 2.0, (2, n_samp))
    signal = loadings @ factors
    base = rng.normal(20.0, 0.5, n_prot)[:, None]
    X = base + signal + rng.normal(0.0, 0.3, (n_prot, n_samp))
    miss = rng.uniform(size=X.shape) < 0.10  # 10% MCAR
    X[miss] = np.nan
    return pd.DataFrame(
        X,
        index=[f"prot_{i}" for i in range(n_prot)],
        columns=[f"S{j}" for j in range(n_samp)],
    )


def _pearson_on_missing(py: pd.DataFrame, r: pd.DataFrame,
                        miss: pd.DataFrame) -> float:
    m = miss.to_numpy()
    return float(stats.pearsonr(py.to_numpy()[m], r.to_numpy()[m])[0])


def test_zero_bit_exact_vs_R(workdir):
    """impute.ZERO is deterministic — must match R bit-for-bit."""
    df = _make_dataset()
    input_tsv = workdir / "input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "zero", "zero")

    py = impute_zero(df)
    r = _read_r_imputed(workdir / "zero", "zero")
    r = r[py.columns]
    np.testing.assert_allclose(
        py.to_numpy(), r.to_numpy(), rtol=0, atol=1e-12,
        err_msg="impute.ZERO output diverges from R imputeLCMD",
    )


def test_knn_correlates_with_R(workdir):
    """KNN (K=10) on correlated data — Pearson r > 0.95 vs R."""
    df = _make_correlated_dataset()
    input_tsv = workdir / "corr_input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "knn", "knn")

    py = impute_knn(df, K=10)
    r = _read_r_imputed(workdir / "knn", "knn")[py.columns]
    rho = _pearson_on_missing(py, r, df.isna())
    assert rho > 0.95, f"KNN R-parity too low: Pearson r={rho:.4f}"


def test_svd_correlates_with_R(workdir):
    """SVD (K=2) on correlated data — Pearson r > 0.9 vs R."""
    df = _make_correlated_dataset()
    input_tsv = workdir / "corr_input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "svd", "svd")

    py = impute_svd(df, K=2)
    r = _read_r_imputed(workdir / "svd", "svd")[py.columns]
    rho = _pearson_on_missing(py, r, df.isna())
    assert rho > 0.9, f"SVD R-parity too low: Pearson r={rho:.4f}"


def test_mle_correlates_with_R(workdir):
    """MLE on correlated data — Pearson r > 0.9 vs R.

    We compare the deterministic conditional-mean fill (``sample=False``)
    against R's ``imp.norm`` draw — the EEM-fitted (μ, Σ) agree, so the
    conditional means track R's draws closely (EM convergence and the
    I-step PRNG differ, hence a correlation rather than bit-exact
    check).
    """
    df = _make_correlated_dataset()
    input_tsv = workdir / "corr_input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "mle", "mle")

    py = impute_mle(df, seed=0, sample=False)
    r = _read_r_imputed(workdir / "mle", "mle")[py.columns]
    rho = _pearson_on_missing(py, r, df.isna())
    assert rho > 0.9, f"MLE R-parity too low: Pearson r={rho:.4f}"


def test_model_selector_agrees_with_R(workdir):
    """model.Selector flags agree with R on > 90% of proteins."""
    # Bimodal MAR + MNAR dataset so the classifier has both classes.
    rng = np.random.default_rng(42)
    n_high, n_low, n_samples = 300, 100, 6
    X = np.empty((n_high + n_low, n_samples))
    X[:n_high] = rng.normal(25.0, 1.0, (n_high, n_samples))
    X[n_high:] = rng.normal(15.0, 0.5, (n_low, n_samples))
    X[n_high:][X[n_high:] < 14.5] = np.nan
    miss = rng.uniform(size=(n_high, n_samples)) < 0.05
    X[:n_high][miss] = np.nan
    df = pd.DataFrame(
        X,
        index=[f"prot_{i}" for i in range(X.shape[0])],
        columns=[f"S{j}" for j in range(X.shape[1])],
    )
    input_tsv = workdir / "ms_input.tsv"
    _write_input(df, input_tsv)
    _run_r(input_tsv, workdir / "ms", "model_selector")

    r_flags = pd.read_csv(
        workdir / "ms" / "model_selector_flags.tsv", sep="\t", index_col=0,
    )["is_mar"].to_numpy()
    is_mar_py, _ = model_selector(X)
    agreement = float((is_mar_py == (r_flags == 1)).mean())
    assert agreement > 0.90, (
        f"model.Selector agreement with R too low: {agreement * 100:.1f}%"
    )
