"""MCAR/MNAR model selection and synthetic missing-value injection.

Ports two utility R functions from Bioconductor imputeLCMD:

* :func:`model_selector` — ``model.Selector``: per-protein bool flag
  marking rows as MAR/MCAR (``True``) vs MNAR (``False``). Fits a
  quantile-regression line to the per-protein mean vector and locates
  a censoring threshold from the gap between the empirical CDF and the
  fitted normal CDF. Proteins whose row-mean sits above the threshold
  are MAR/MCAR; the rest are MNAR.
* :func:`insert_mvs` — ``insertMVs``: inject artificial missing entries
  into a fully-observed matrix for benchmarking imputers. Supports
  ``'MCAR'`` (uniform random), ``'MNAR'`` (below the threshold-quantile
  floor), and ``'mixed'``.

Matrix convention: rows = proteins, columns = samples. NaN missing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .imputers import ArrayLike, _as_array, _r_quantile


# --------------------------------------------------------------------------- #
# model.Selector — MCAR / MNAR classifier                                     #
# --------------------------------------------------------------------------- #

def model_selector(
    X: ArrayLike,
    upper_q: float = 0.99,
    n_points: int = 512,
) -> Tuple[np.ndarray, float]:
    r"""Classify each protein as MAR/MCAR (``True``) or MNAR (``False``).

    Faithful port of R ``imputeLCMD::model.Selector``. The procedure:

    1. Compute the per-protein mean vector ``mean_vect`` ignoring NaN.
    2. Fit a QR line ``q_mean_vect ~ q_normal`` over the
       theoretical-normal quantile grid (intercept = μ, slope = σ̂ of
       the underlying complete-data distribution).
    3. Build a support of length ``2 * n_points + n_features``
       spanning ``[μ - 4σ, μ + 4σ]`` plus the sorted mean vector and
       evaluate (a) the empirical CDF ``F̂`` of ``mean_vect``, and
       (b) the fitted Gaussian CDF ``Φ(·; μ, σ)``.
    4. The objective ``(Φ - F̂ + 1)/(F̂ + 1) - 1`` is maximised at the
       point where the empirical mass drops out of the fitted normal
       — this is the censoring threshold ``c``. Proteins with mean >
       ``c`` are flagged MAR/MCAR (``True``); the rest are MNAR.

    Parameters
    ----------
    X
        2-D matrix, rows = proteins, cols = samples. NaN missing.
    upper_q
        Upper-quantile cap for the QR grid. R default 0.99.
    n_points
        Number of support points padded on either side of the sorted
        mean vector. R default 512.

    Returns
    -------
    is_mar_or_mcar : ndarray of bool, shape (n_proteins,)
        ``True`` ⇒ MAR/MCAR; ``False`` ⇒ MNAR. Matches R
        ``model.selector[[1]] == 1``.
    censoring_thr : float
        The fitted censoring threshold. Matches R
        ``model.selector[[2]]``.
    """
    arr, _, _ = _as_array(X)
    n_features, n_samples = arr.shape
    mean_vect = np.nanmean(arr, axis=1)  # one mean per row

    # Fraction of rows with an undefined mean (all-NaN rows).
    p_NAs = float(np.isnan(mean_vect).sum()) / float(mean_vect.size)
    # Build the theoretical-normal and empirical quantile grids.
    step = (upper_q - p_NAs) / (upper_q * 100.0)
    n_grid = 100
    q_norm_probs = p_NAs + 0.001 + np.arange(n_grid) * step
    q_emp_probs = 0.001 + np.arange(n_grid) * 0.01

    q_normal = stats.norm.ppf(q_norm_probs, loc=0.0, scale=1.0)
    # R-compatible quantile (type 7, na.rm=TRUE) — same helper QRILC uses.
    q_mean_vect = np.asarray(_r_quantile(mean_vect, q_emp_probs))
    # OLS closed-form (matches R lm()).
    x_bar = q_normal.mean()
    y_bar = q_mean_vect.mean()
    denom = float(np.sum((q_normal - x_bar) ** 2))
    if denom == 0.0:
        slope = 0.0
        intercept = y_bar
    else:
        slope = float(np.sum((q_normal - x_bar) * (q_mean_vect - y_bar)) / denom)
        intercept = float(y_bar - slope * x_bar)
    mean_CDD = intercept
    sd_CDD = slope

    # Build the support: pad below the observed min and above the
    # observed max with ``n_points`` linearly-spaced points; sandwich
    # the sorted mean vector in between (R semantics).
    mv_clean = mean_vect[~np.isnan(mean_vect)]
    min_obs = float(mv_clean.min())
    max_obs = float(mv_clean.max())
    min_support = mean_CDD - 4.0 * sd_CDD
    max_support = mean_CDD + 4.0 * sd_CDD
    sorted_mean = np.sort(mv_clean)
    support = np.concatenate([
        np.linspace(min_support, min_obs, n_points),
        sorted_mean,
        np.linspace(max_obs, max_support, n_points),
    ])

    # Empirical CDF (right-continuous step) evaluated at the support.
    # ``np.searchsorted(sorted_mean, support, side='right') / N`` is
    # the standard ``ecdf`` formula.
    N = sorted_mean.size
    cdf_OD = np.searchsorted(sorted_mean, support, side='right') / float(N)
    if sd_CDD <= 0.0 or not np.isfinite(sd_CDD):
        # Degenerate fit — fall back to "all MAR".
        thr = min_support - 1.0
        is_mar = (mean_vect > thr) & ~np.isnan(mean_vect)
        return is_mar, float(thr)
    cdf_CD = stats.norm.cdf(support, loc=mean_CDD, scale=sd_CDD)
    diff_cdf = cdf_CD - cdf_OD
    obj_fnc = ((diff_cdf + 1.0) / (cdf_OD + 1.0) - 1.0) * 10.0
    if obj_fnc.max() > 0:
        censoring_thr = float(support[int(np.argmax(obj_fnc))])
    else:
        censoring_thr = float(min_support - 1.0)
    # R: ``model.selector[which(mean.vect > censoring.thr)] = 1``.
    # NaN mean rows → never flagged MAR.
    is_mar = (mean_vect > censoring_thr) & ~np.isnan(mean_vect)
    return is_mar, censoring_thr


# --------------------------------------------------------------------------- #
# insertMVs — synthetic MV injection for benchmarking                         #
# --------------------------------------------------------------------------- #

def insert_mvs(
    X: ArrayLike,
    n_mv: int = 200,
    mode: str = 'MCAR',
    threshold_quantile: float = 0.1,
    mnar_rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> ArrayLike:
    r"""Inject artificial missing values into a fully-observed matrix.

    A pragmatic Python equivalent of R ``insertMVs``. The R original
    is parameterised by ``(mean.THR, sd.THR, MNAR.rate)`` and a
    randomly-drawn threshold matrix; we expose the simpler and more
    common knob — ``threshold_quantile`` of the *empirical* matrix —
    so callers don't have to know the data's underlying mean / sd.

    Parameters
    ----------
    X
        2-D array / DataFrame; assumed fully observed.
    n_mv
        Total number of NaN entries to inject. For ``mode='mixed'``,
        ``mnar_rate * n_mv`` are MNAR and the rest MCAR.
    mode
        ``'MCAR'`` — pick ``n_mv`` cells uniformly at random.
        ``'MNAR'`` — pick ``n_mv`` cells whose value is below
        ``quantile(X, threshold_quantile)``.
        ``'mixed'`` — combine both per ``mnar_rate``.
    threshold_quantile
        Quantile of ``X`` below which a cell is eligible for MNAR
        masking.
    mnar_rate
        For ``mode='mixed'``: fraction of injected MVs that should be
        MNAR. Defaults to 0.5.
    seed
        NumPy PRNG seed for reproducibility.

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X`` with ``n_mv`` NaNs injected.
    """
    arr, ri, ci = _as_array(X)
    rng = np.random.default_rng(seed)
    n_total = arr.size
    mode_key = mode.lower().strip()
    if mode_key not in ('mcar', 'mnar', 'mixed'):
        raise ValueError(f"mode must be 'MCAR'/'MNAR'/'mixed', got {mode!r}")
    n_mv = min(int(n_mv), n_total)
    if mnar_rate is None:
        mnar_rate = 0.5

    flat = arr.ravel()
    finite_idx = np.flatnonzero(~np.isnan(flat))
    if mode_key == 'mcar':
        chosen = rng.choice(finite_idx, size=min(n_mv, finite_idx.size), replace=False)
    elif mode_key == 'mnar':
        thr = np.quantile(flat[finite_idx], threshold_quantile)
        elig = finite_idx[flat[finite_idx] <= thr]
        if elig.size == 0:
            chosen = np.array([], dtype=int)
        else:
            chosen = rng.choice(elig, size=min(n_mv, elig.size), replace=False)
    else:  # mixed
        n_mnar = int(round(mnar_rate * n_mv))
        n_mcar = n_mv - n_mnar
        thr = np.quantile(flat[finite_idx], threshold_quantile)
        mnar_pool = finite_idx[flat[finite_idx] <= thr]
        if mnar_pool.size == 0:
            mnar_chosen = np.array([], dtype=int)
        else:
            mnar_chosen = rng.choice(
                mnar_pool, size=min(n_mnar, mnar_pool.size), replace=False,
            )
        remaining = np.setdiff1d(finite_idx, mnar_chosen, assume_unique=False)
        mcar_chosen = rng.choice(
            remaining, size=min(n_mcar, remaining.size), replace=False,
        )
        chosen = np.concatenate([mnar_chosen, mcar_chosen])
    flat[chosen] = np.nan
    out = flat.reshape(arr.shape)
    if ri is not None:
        return pd.DataFrame(out, index=ri, columns=ci)
    return out
