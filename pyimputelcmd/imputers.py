"""
Pure-Python port of imputeLCMD (Lazar 2016 JPR).

Implements per-sample (column-wise) missing-value imputation for
label-free proteomics intensity matrices:

* ``impute_mindet``  — Deterministic Minimum (det-MinDet)
* ``impute_minprob`` — Probabilistic Minimum (MinProb)
* ``impute_qrilc``   — Quantile Regression Imputation of Left-Censored
                       data (QRILC)

Matrix convention (R-native): rows = features (proteins / peptides),
columns = samples. NaN marks missing. See ``impute`` for a single
dispatcher used by the omicverse wrapper.

References
----------
Lazar C., Gatto L., Ferro M., Bruley C., Burger T.
*Accounting for the Multiple Natures of Missing Values in Label-Free
Quantitative Proteomics Data Sets to Compare the Performance of
Normalization Strategies.*
J Proteome Res. 2016, 15 (4): 1116-1125.
DOI: 10.1021/acs.jproteome.5b00981
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


ArrayLike = Union[np.ndarray, pd.DataFrame]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _as_array(X: ArrayLike) -> Tuple[np.ndarray, Optional[pd.Index], Optional[pd.Index]]:
    """Accept ndarray or DataFrame, return (float ndarray, row_index, col_index)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, copy=True), X.index, X.columns
    arr = np.asarray(X, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"`X` must be 2-D (features × samples), got shape {arr.shape}")
    return arr, None, None


def _restore(
    arr: np.ndarray,
    row_index: Optional[pd.Index],
    col_index: Optional[pd.Index],
) -> ArrayLike:
    """Wrap the imputed ndarray back into a DataFrame if the input was a DataFrame."""
    if row_index is not None:
        return pd.DataFrame(arr, index=row_index, columns=col_index)
    return arr


def _r_quantile(values: np.ndarray, probs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """R-compatible ``quantile(x, probs, type=7, na.rm=TRUE)``.

    NumPy's ``method='linear'`` matches R's default quantile type 7
    bit-for-bit. NaNs are stripped (analogue of ``na.rm=TRUE``).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return np.quantile(values, probs, method='linear')


# --------------------------------------------------------------------------- #
# MinDet                                                                      #
# --------------------------------------------------------------------------- #

def impute_mindet(X: ArrayLike, q: float = 0.01) -> ArrayLike:
    r"""Deterministic minimum imputation (MinDet).

    For each sample (column), the missing values are replaced by the
    ``q``-quantile of that sample's observed values. ``q = 0.01`` (R
    default) places the imputed value at the 1st-percentile floor, a
    conservative estimate of the detection-limit for left-censored
    intensities.

    This routine is fully deterministic and reproduces R imputeLCMD's
    ``impute.MinDet`` bit-for-bit (modulo IEEE-754 rounding) given the
    same input.

    Parameters
    ----------
    X
        2-D array or DataFrame with rows = proteins, columns = samples.
        Missing values must be ``NaN``.
    q
        Lower-quantile probability used as the imputation floor.

    Returns
    -------
    ndarray | DataFrame
        Same shape and (if DataFrame) same index/columns as ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[10.0, np.nan, 12.0],
    ...               [np.nan, 8.0, 9.0]])
    >>> impute_mindet(X, q=0.5)
    array([[10. , 8.5, 12. ],
           [11. , 8. ,  9. ]])
    """
    arr, ri, ci = _as_array(X)
    n_features, n_samples = arr.shape
    for j in range(n_samples):
        col = arr[:, j]
        miss = np.isnan(col)
        if miss.any():
            floor = _r_quantile(col[~miss], q)
            arr[miss, j] = floor
    return _restore(arr, ri, ci)


# --------------------------------------------------------------------------- #
# MinProb                                                                     #
# --------------------------------------------------------------------------- #

def impute_minprob(
    X: ArrayLike,
    q: float = 0.01,
    tune_sigma: float = 1.0,
    seed: Optional[int] = None,
) -> ArrayLike:
    r"""Probabilistic minimum imputation (MinProb).

    For each sample (column), missing values are drawn from a Gaussian
    centred on the sample's ``q``-quantile with standard deviation
    ``sd(observed) * tune_sigma`` — i.e. random jitter below the
    detection threshold rather than the flat floor of
    :func:`impute_mindet`. Matches R imputeLCMD ``impute.MinProb`` in
    distribution; bit-exact RNG parity with R is not possible because
    the two languages use different PRNGs, so callers needing
    deterministic output should set ``seed``.

    Parameters
    ----------
    X
        2-D array or DataFrame with rows = proteins, columns = samples.
    q
        Lower-quantile probability for the imputation mean (R default
        0.01).
    tune_sigma
        Multiplier on the per-sample observed standard deviation. The
        R default 1.0 reproduces ``impute.MinProb`` exactly. Use a
        smaller value (e.g. 0.3) for tighter clouds around the
        detection limit.
    seed
        Optional integer seed for NumPy's PCG64 generator; pass to make
        the output deterministic. ``None`` (default) draws from a
        fresh RNG.

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(20, 1, (200, 4))
    >>> X[X < 19] = np.nan  # introduce left-censored missing
    >>> imputed = impute_minprob(X, q=0.01, seed=0)
    >>> np.isnan(imputed).any()
    np.False_
    """
    arr, ri, ci = _as_array(X)
    rng = np.random.default_rng(seed)
    _, n_samples = arr.shape
    for j in range(n_samples):
        col = arr[:, j]
        miss = np.isnan(col)
        n_miss = int(miss.sum())
        if n_miss == 0:
            continue
        observed = col[~miss]
        q_min = _r_quantile(observed, q)
        # R ``sd`` uses N-1 normalization → numpy ``ddof=1``.
        q_sigma = float(np.std(observed, ddof=1)) * tune_sigma
        arr[miss, j] = rng.normal(loc=q_min, scale=q_sigma, size=n_miss)
    return _restore(arr, ri, ci)


# --------------------------------------------------------------------------- #
# QRILC                                                                       #
# --------------------------------------------------------------------------- #

def impute_qrilc(
    X: ArrayLike,
    tune_sigma: float = 1.0,
    seed: Optional[int] = None,
    upper_q: float = 0.99,
) -> ArrayLike:
    r"""Quantile-Regression Imputation of Left-Censored data (QRILC).

    Faithful Python port of ``impute.QRILC`` from Bioconductor
    imputeLCMD. Per-sample procedure:

    1. ``p_NAs`` = fraction of missing values in the sample.
    2. Build the theoretical-normal quantile grid
       ``seq(p_NAs + 0.001, upper_q + 0.001, step)`` where
       ``step = (upper_q - p_NAs) / (upper_q * 100)`` (≈100 points).
    3. Build the empirical-sample quantile grid
       ``seq(0.001, upper_q + 0.001, 0.01)`` — fixed step 0.01, also
       100 points.
    4. OLS-regress empirical quantiles on theoretical quantiles. The
       intercept estimates ``mean.CDD`` (μ) and the slope estimates
       ``sd.CDD`` of the underlying (uncensored) normal.
    5. Sample the missing entries from a truncated normal with mean
       μ, upper bound ``qnorm(p_NAs + 0.001, μ, sd.CDD)``, and — to
       mirror the R reference exactly — sampling variance
       ``sd.CDD * tune_sigma`` (R's tmvtnorm interprets the ``sigma``
       argument as the *covariance*, not the standard deviation; this
       is a documented imputeLCMD quirk that we replicate to keep
       output distribution-equivalent).

    QRILC matches the R imputeLCMD reference at the level of fitted
    ``(intercept, slope)`` parameters (up to FP); the drawn imputed
    values match in distribution (KS-equivalent) but not bit-for-bit,
    because tmvtnorm uses a Gibbs sampler whereas SciPy uses
    inverse-CDF on a different PRNG.

    Parameters
    ----------
    X
        2-D array or DataFrame, rows = proteins, columns = samples.
    tune_sigma
        Multiplier on the fitted-slope-based sampling variance. R
        default 1.0.
    seed
        Optional NumPy PRNG seed.
    upper_q
        Upper-quantile cap for the regression. R default 0.99.

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(20, 1, (500, 4))
    >>> X[X < 19] = np.nan
    >>> imputed = impute_qrilc(X, tune_sigma=1.0, seed=0)
    >>> np.isnan(imputed).any()
    np.False_
    """
    arr, ri, ci = _as_array(X)
    rng = np.random.default_rng(seed)
    n_features, n_samples = arr.shape
    for j in range(n_samples):
        col = arr[:, j]
        miss = np.isnan(col)
        n_miss = int(miss.sum())
        if n_miss == 0:
            continue
        observed = col[~miss]
        if observed.size < 3:
            arr[miss, j] = float(np.min(observed))
            continue

        p_NAs = float(n_miss) / float(col.size)
        # Theoretical-normal quantile grid: R
        # ``seq(pNAs+0.001, upper.q+0.001, (upper.q-pNAs)/(upper.q*100))``.
        # The step is chosen so the grid has exactly 100 points.
        step_norm = (upper_q - p_NAs) / (upper_q * 100.0)
        n_grid = 100
        q_norm_probs = p_NAs + 0.001 + np.arange(n_grid) * step_norm
        # Empirical-sample quantile grid: R
        # ``seq(0.001, upper.q+0.001, 0.01)`` — fixed step 0.01, 100 pts.
        q_emp_probs = 0.001 + np.arange(n_grid) * 0.01

        # Convert to quantiles.
        q_norm = stats.norm.ppf(q_norm_probs, loc=0.0, scale=1.0)
        if not np.all(np.isfinite(q_norm)):
            arr[miss, j] = float(np.min(observed))
            continue
        q_sample = _r_quantile(observed, q_emp_probs)

        # OLS: q_sample = intercept + slope * q_norm. Closed form matches R lm().
        x_bar = float(np.mean(q_norm))
        y_bar = float(np.mean(q_sample))
        denom = float(np.sum((q_norm - x_bar) ** 2))
        if denom == 0.0:
            arr[miss, j] = float(np.min(observed))
            continue
        slope = float(np.sum((q_norm - x_bar) * (q_sample - y_bar)) / denom)
        intercept = y_bar - slope * x_bar

        mean_CDD = intercept
        sd_CDD = slope  # R does NOT take abs(); a negative slope leads
                         # to a negative variance below and bails out.

        # Mirror the R quirk: ``sigma = sd.CDD * tune.sigma`` is the
        # variance (rtmvnorm's ``sigma`` is a covariance matrix). The
        # sampling-distribution standard deviation is therefore
        # ``sqrt(sd_CDD * tune_sigma)``.
        sampling_var = sd_CDD * tune_sigma
        if sampling_var <= 0 or not np.isfinite(sampling_var):
            arr[miss, j] = mean_CDD
            continue
        sampling_std = float(np.sqrt(sampling_var))

        # Upper truncation bound uses ``sd = sd_CDD`` (the std, not the
        # sampling std) — another imputeLCMD quirk; matches R exactly.
        if sd_CDD <= 0:
            arr[miss, j] = mean_CDD
            continue
        upper = stats.norm.ppf(p_NAs + 0.001, loc=mean_CDD, scale=sd_CDD)
        # SciPy truncnorm: a, b in standardized units of the *sampling*
        # distribution N(mean_CDD, sampling_std^2).
        a = -np.inf
        b = (upper - mean_CDD) / sampling_std
        samples = stats.truncnorm.rvs(
            a=a, b=b, loc=mean_CDD, scale=sampling_std,
            size=n_miss, random_state=rng,
        )
        arr[miss, j] = samples
    return _restore(arr, ri, ci)


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #

def _lazy_wrapper(name: str):
    """Return a thin shim that imports a wrapper imputer on first call.

    Lets the dispatcher list extended methods (``zero``/``knn``/...)
    without forcing :mod:`wrappers` and :mod:`model_selector` to load
    at import time (avoiding scikit-learn as a hard dep on import).
    """
    def _shim(X, **kw):
        from . import wrappers
        return getattr(wrappers, name)(X, **kw)
    _shim.__name__ = name
    return _shim


_DISPATCH = {
    'mindet':   impute_mindet,
    'minprob':  impute_minprob,
    'qrilc':    impute_qrilc,
    'zero':     _lazy_wrapper('impute_zero'),
    'mle':      _lazy_wrapper('impute_mle'),
    'knn':      _lazy_wrapper('impute_knn'),
    'svd':      _lazy_wrapper('impute_svd'),
    'mar':      _lazy_wrapper('impute_mar'),
    'mar_mnar': _lazy_wrapper('impute_mar_mnar'),
}


def impute(X: ArrayLike, method: str = 'qrilc', **kwargs) -> ArrayLike:
    r"""Dispatch left-censored imputation by method name.

    A thin wrapper that selects the right imputer via the ``method``
    keyword. Handy when the imputation choice should be a parameter
    (sweeps, agent tool calls, configuration files).

    Supported ``method`` values:

    * ``'mindet'``, ``'minprob'``, ``'qrilc'`` — left-censored MNAR
      imputers (Lazar 2016).
    * ``'zero'``  — replace NaN with 0 (``impute.ZERO``).
    * ``'mle'``   — MVN-EM MLE (``impute.wrapper.MLE``).
    * ``'knn'``   — per-protein KNN (``impute.wrapper.KNN``).
    * ``'svd'``   — iterative SVD (``impute.wrapper.SVD``).
    * ``'mar'``   — :func:`pyimputelcmd.wrappers.impute_mar` (needs
      ``mcar_mask``).
    * ``'mar_mnar'`` — combined MAR+MNAR pipeline (needs
      ``mcar_mask``).

    Parameters
    ----------
    X
        2-D matrix (features × samples) with NaN for missing values.
    method
        See the list above.
    **kwargs
        Forwarded to the chosen imputer (e.g. ``q``, ``tune_sigma``,
        ``seed``, ``K``, ``mcar_mask``).

    Returns
    -------
    ndarray | DataFrame
        Same shape and (if DataFrame) same index/columns as ``X``.

    Examples
    --------
    >>> imputed = impute(X, method='qrilc', tune_sigma=1.0, seed=0)
    >>> imputed = impute(X, method='knn', K=15)
    """
    key = method.lower().strip()
    if key not in _DISPATCH:
        raise ValueError(
            f"method must be one of {sorted(_DISPATCH)}, got {method!r}"
        )
    return _DISPATCH[key](X, **kwargs)
