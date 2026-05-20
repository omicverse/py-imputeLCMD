"""MAR/MNAR wrappers ported from Bioconductor imputeLCMD.

Mirrors the R API one-to-one:

* :func:`impute_zero`   — ``impute.ZERO`` (replace NaN with 0).
* :func:`impute_knn`    — ``impute.wrapper.KNN`` (per-protein KNN via
  Euclidean distance over samples, courtesy of
  :class:`sklearn.impute.KNNImputer`).
* :func:`impute_mle`    — ``impute.wrapper.MLE`` (multivariate-normal
  EM imputation; lightweight pure-NumPy implementation of the
  ``norm::imp.norm`` pipeline).
* :func:`impute_svd`    — ``impute.wrapper.SVD`` (iterative rank-K SVD
  imputation, Stacklies 2007 / ``pcaMethods::pca(method='svdImpute')``).
* :func:`impute_mar`    — ``impute.MAR``: dispatch MAR/MCAR rows to a
  MAR imputer (MLE/KNN/SVD), leave MNAR rows untouched.
* :func:`impute_mar_mnar` — ``impute.MAR.MNAR``: apply a MAR imputer to
  MCAR-flagged rows, then run a MNAR imputer (QRILC/MinDet/MinProb) on
  the whole matrix.

Matrix convention is R-native: rows = proteins/peptides, columns =
samples; missing entries are ``NaN``.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .imputers import (
    ArrayLike,
    _as_array,
    _restore,
    impute_mindet,
    impute_minprob,
    impute_qrilc,
)


# --------------------------------------------------------------------------- #
# impute.ZERO                                                                 #
# --------------------------------------------------------------------------- #

def impute_zero(X: ArrayLike) -> ArrayLike:
    r"""Replace every ``NaN`` with ``0`` (matches R ``impute.ZERO``).

    Trivial imputer; useful as a sanity check / null baseline. Returns
    a ``DataFrame`` if the input was one, preserving index / columns.
    """
    arr, ri, ci = _as_array(X)
    arr[np.isnan(arr)] = 0.0
    return _restore(arr, ri, ci)


# --------------------------------------------------------------------------- #
# impute.wrapper.KNN                                                          #
# --------------------------------------------------------------------------- #

def impute_knn(X: ArrayLike, K: int = 10) -> ArrayLike:
    r"""Per-protein KNN imputation (R ``impute.wrapper.KNN``).

    Wraps :class:`sklearn.impute.KNNImputer` with ``n_neighbors=K`` to
    mimic Bioconductor's ``impute::impute.knn``: for each protein with
    missing values, the ``K`` nearest *proteins* (Euclidean distance
    over the samples for which both rows are observed) are averaged to
    fill in the missing entries.

    R uses ``K=15`` by default; this port defaults to ``K=10`` to match
    the R help-page default of ``impute.knn``. The signature mirrors
    ``impute.wrapper.KNN(dataSet.mvs, K)`` — pass an explicit ``K`` for
    the MAR/MNAR-pipeline value of 15.

    Parameters
    ----------
    X
        2-D array / DataFrame, rows = proteins, cols = samples.
    K
        Number of nearest proteins to average.

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X``.
    """
    try:
        from sklearn.impute import KNNImputer
    except ImportError as err:  # pragma: no cover - optional dep
        raise ImportError(
            "impute_knn requires scikit-learn. Install with "
            "`pip install scikit-learn` or `pip install pyimputelcmd[knn]`."
        ) from err
    arr, ri, ci = _as_array(X)
    n_features = arr.shape[0]
    if K > n_features:
        K = max(1, n_features - 1)
    # KNNImputer treats rows as samples, so to find nearest *proteins*
    # (R-style row-wise KNN) we pass the matrix as-is — sklearn finds
    # K nearest rows and averages their values for the missing columns.
    imputer = KNNImputer(n_neighbors=K, weights='uniform')
    out = imputer.fit_transform(arr)
    return _restore(out.astype(float, copy=False), ri, ci)


# --------------------------------------------------------------------------- #
# impute.wrapper.MLE — multivariate-normal EM (norm::imp.norm)                #
# --------------------------------------------------------------------------- #

def impute_mle(
    X: ArrayLike,
    max_iter: int = 200,
    tol: float = 1e-4,
    ridge: float = 1e-6,
    seed: Optional[int] = None,
    sample: bool = True,
) -> ArrayLike:
    r"""Multivariate-normal MLE imputation (R ``impute.wrapper.MLE``).

    Port of the ``norm::imp.norm`` pipeline used by R imputeLCMD:
    proteins (rows) are treated as samples from a multivariate normal
    over the ``n_samples`` columns; the mean vector and covariance are
    fit by EM over the observed entries (``em.norm``) and missing
    entries are then drawn from the conditional Gaussian
    ``N(μ_m + Σ_mo Σ_oo⁻¹(x_o − μ_o), Σ_m|o)`` given the row's
    observed entries (``imp.norm``, one I-step of data augmentation).

    Setting ``sample=False`` substitutes the conditional **mean** for
    the draw — useful when deterministic output is required and the
    extra variance from sampling is not.

    Parameters
    ----------
    X
        2-D array (rows = proteins, cols = samples) with NaN missing.
    max_iter
        Maximum EM iterations.
    tol
        Convergence threshold on the max-abs change in the mean
        vector between iterations.
    ridge
        Ridge added to the diagonal of the covariance to keep the
        Cholesky stable on small / collinear matrices.
    seed
        NumPy PRNG seed; passed through to the I-step draws.
    sample
        If True (default, mirrors R ``imp.norm``), draw from the
        conditional Gaussian; if False, use the conditional mean.

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X``; missing entries imputed from the fitted
        MVN.
    """
    arr, ri, ci = _as_array(X)
    n_features, n_samples = arr.shape
    mask = np.isnan(arr)
    if not mask.any():
        return _restore(arr, ri, ci)

    # Initial fill with column means (so the EM has a complete X to
    # start from).
    col_means_init = np.nanmean(arr, axis=0)
    # Guard against fully-NaN columns (degenerate but possible).
    col_means_init = np.where(np.isnan(col_means_init), 0.0, col_means_init)
    X_filled = np.where(mask, col_means_init, arr).astype(float, copy=False)

    mu = X_filled.mean(axis=0)
    centered = X_filled - mu
    cov = (centered.T @ centered) / max(n_features - 1, 1)
    cov += ridge * np.eye(n_samples)

    prev_mu = mu.copy()
    for _ in range(max_iter):
        # E-step: fill with conditional means while we're estimating
        # parameters (sampling at this stage adds variance and slows
        # EM convergence).
        for i in range(n_features):
            mi = mask[i]
            if not mi.any():
                continue
            if mi.all():
                X_filled[i] = mu
                continue
            obs = ~mi
            cov_oo = cov[np.ix_(obs, obs)]
            cov_mo = cov[np.ix_(mi, obs)]
            try:
                solve = np.linalg.solve(cov_oo, X_filled[i, obs] - mu[obs])
            except np.linalg.LinAlgError:
                solve = np.linalg.lstsq(
                    cov_oo, X_filled[i, obs] - mu[obs], rcond=None,
                )[0]
            X_filled[i, mi] = mu[mi] + cov_mo @ solve

        # M-step: re-fit mu, cov from the filled matrix.
        mu = X_filled.mean(axis=0)
        centered = X_filled - mu
        cov = (centered.T @ centered) / max(n_features - 1, 1)
        cov += ridge * np.eye(n_samples)

        if np.max(np.abs(mu - prev_mu)) < tol:
            break
        prev_mu = mu.copy()

    # I-step: draw missing entries from the predictive conditional
    # Gaussian using the fitted (μ, Σ). This is what R imp.norm does.
    rng = np.random.default_rng(seed)
    out = arr.copy()
    for i in range(n_features):
        mi = mask[i]
        if not mi.any():
            continue
        if mi.all():
            cond_mean = mu
            if sample:
                out[i] = rng.multivariate_normal(cond_mean, cov)
            else:
                out[i] = cond_mean
            continue
        obs = ~mi
        cov_oo = cov[np.ix_(obs, obs)]
        cov_mo = cov[np.ix_(mi, obs)]
        cov_mm = cov[np.ix_(mi, mi)]
        try:
            inv_oo_x = np.linalg.solve(cov_oo, arr[i, obs] - mu[obs])
        except np.linalg.LinAlgError:
            inv_oo_x = np.linalg.lstsq(
                cov_oo, arr[i, obs] - mu[obs], rcond=None,
            )[0]
        cond_mean = mu[mi] + cov_mo @ inv_oo_x
        if sample:
            try:
                inv_oo_mo = np.linalg.solve(cov_oo, cov_mo.T)
            except np.linalg.LinAlgError:
                inv_oo_mo = np.linalg.lstsq(cov_oo, cov_mo.T, rcond=None)[0]
            cond_cov = cov_mm - cov_mo @ inv_oo_mo
            # Symmetrise + ridge for numerical stability.
            cond_cov = 0.5 * (cond_cov + cond_cov.T) + ridge * np.eye(cond_cov.shape[0])
            out[i, mi] = rng.multivariate_normal(cond_mean, cond_cov)
        else:
            out[i, mi] = cond_mean
    return _restore(out, ri, ci)


# --------------------------------------------------------------------------- #
# impute.wrapper.SVD — iterative rank-K SVD imputation                        #
# --------------------------------------------------------------------------- #

def impute_svd(
    X: ArrayLike,
    K: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    center: bool = True,
) -> ArrayLike:
    r"""Iterative rank-``K`` SVD imputation (R ``impute.wrapper.SVD``).

    Ports ``pcaMethods::pca(method='svdImpute')`` (Stacklies *et al.*,
    2007). Initial fill = per-column mean; then repeatedly:

    1. Centre (optional) and take the truncated rank-K SVD of the
       filled matrix.
    2. Re-impute the missing entries from the rank-K reconstruction.

    Iteration stops once the per-iteration change in the imputed
    values drops below ``tol``.

    Parameters
    ----------
    X
        2-D array, rows = proteins, cols = samples. NaN missing.
    K
        SVD rank. R default ``K = 2``.
    max_iter
        Hard cap on iterations.
    tol
        Convergence threshold on the relative change in the missing
        entries between iterations.
    center
        If True, mean-centre the columns before each SVD step (mirrors
        the ``pcaMethods`` default).

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X``; missing entries filled from the rank-K
        reconstruction.
    """
    arr, ri, ci = _as_array(X)
    mask = np.isnan(arr)
    if not mask.any():
        return _restore(arr, ri, ci)

    col_means = np.nanmean(arr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_filled = np.where(mask, col_means, arr).astype(float, copy=True)
    K = max(1, min(K, *X_filled.shape))

    prev_missing = X_filled[mask].copy()
    for _ in range(max_iter):
        if center:
            mu = X_filled.mean(axis=0)
            Xc = X_filled - mu
        else:
            mu = np.zeros(X_filled.shape[1])
            Xc = X_filled
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        recon = (U[:, :K] * S[:K]) @ Vt[:K, :] + mu
        X_filled = np.where(mask, recon, arr)
        cur_missing = X_filled[mask]
        denom = np.linalg.norm(prev_missing) + 1e-12
        if np.linalg.norm(cur_missing - prev_missing) / denom < tol:
            break
        prev_missing = cur_missing.copy()
    return _restore(X_filled, ri, ci)


# --------------------------------------------------------------------------- #
# impute.MAR — apply a MAR imputer to MCAR-flagged rows only                   #
# --------------------------------------------------------------------------- #

_MAR_METHODS = {
    'mle': impute_mle,
    'knn': lambda X, **kw: impute_knn(X, K=kw.get('K', 15)),
    'svd': lambda X, **kw: impute_svd(X, K=kw.get('K', 2)),
}


def impute_mar(
    X: ArrayLike,
    mcar_mask: np.ndarray,
    method: str = 'mle',
    **kwargs,
) -> ArrayLike:
    r"""MAR-only imputation (R ``impute.MAR``).

    Applies a MAR imputer (``MLE`` / ``KNN`` / ``SVD``) to the rows
    flagged as MCAR by :func:`model_selector` (i.e. where
    ``mcar_mask[i] is True``). Rows flagged as MNAR remain untouched
    (still ``NaN``); a downstream MNAR imputer (e.g. QRILC) must fill
    them.

    Parameters
    ----------
    X
        2-D array, rows = proteins, cols = samples.
    mcar_mask
        Boolean 1-D array, length = ``X.shape[0]``. ``True`` flags MAR
        / MCAR rows (i.e. ``model.selector[[1]] == 1`` in R).
    method
        ``'mle'``, ``'knn'``, or ``'svd'``. Matches R
        ``method = "MLE"|"KNN"|"SVD"``.
    **kwargs
        Forwarded to the chosen imputer (e.g. ``K`` for KNN / SVD).

    Returns
    -------
    ndarray | DataFrame
        Same shape as ``X``. MNAR rows untouched.
    """
    key = method.lower().strip()
    if key not in _MAR_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_MAR_METHODS)}, got {method!r}"
        )
    arr, ri, ci = _as_array(X)
    mcar_mask = np.asarray(mcar_mask, dtype=bool)
    if mcar_mask.shape != (arr.shape[0],):
        raise ValueError(
            f"mcar_mask shape {mcar_mask.shape} does not match X rows {arr.shape[0]}"
        )
    if not mcar_mask.any():
        return _restore(arr, ri, ci)
    sub = arr[mcar_mask, :]
    sub_imputed = _MAR_METHODS[key](sub, **kwargs)
    if isinstance(sub_imputed, pd.DataFrame):
        sub_imputed = sub_imputed.to_numpy()
    arr[mcar_mask, :] = sub_imputed
    return _restore(arr, ri, ci)


# --------------------------------------------------------------------------- #
# impute.MAR.MNAR — combined MAR + MNAR pipeline                              #
# --------------------------------------------------------------------------- #

_MNAR_METHODS = {
    'qrilc':   lambda X, **kw: impute_qrilc(X, tune_sigma=kw.get('tune_sigma', 0.3),
                                            seed=kw.get('seed', None)),
    'mindet':  lambda X, **kw: impute_mindet(X, q=kw.get('q', 0.01)),
    'minprob': lambda X, **kw: impute_minprob(X, q=kw.get('q', 0.01),
                                              tune_sigma=kw.get('tune_sigma', 1.0),
                                              seed=kw.get('seed', None)),
}


def impute_mar_mnar(
    X: ArrayLike,
    mcar_mask: np.ndarray,
    method_mar: str = 'mle',
    method_mnar: str = 'qrilc',
    mar_kwargs: Optional[dict] = None,
    mnar_kwargs: Optional[dict] = None,
) -> ArrayLike:
    r"""Combined MAR + MNAR imputation (R ``impute.MAR.MNAR``).

    Mirrors the R two-stage pipeline:

    1. Apply ``method_mar`` (MLE / KNN / SVD) to the MCAR-flagged rows
       via :func:`impute_mar`.
    2. Apply ``method_mnar`` (QRILC / MinDet / MinProb) to the **full
       matrix** to fill any remaining NaNs (i.e. the MNAR rows).

    R defaults: ``method.MAR = "KNN"``, ``method.MNAR = "QRILC"`` with
    ``tune.sigma = 0.3``.

    Parameters
    ----------
    X
        2-D matrix, rows = proteins, cols = samples.
    mcar_mask
        Boolean 1-D array, length = ``X.shape[0]``. ``True`` ⇒ MAR/MCAR.
    method_mar, method_mnar
        Imputer names; see :func:`impute_mar` and module-level
        ``_MNAR_METHODS`` for the supported names.
    mar_kwargs, mnar_kwargs
        Optional kwargs forwarded to the MAR / MNAR imputer.

    Returns
    -------
    ndarray | DataFrame
        Fully imputed matrix (no NaN).
    """
    mar_kwargs = dict(mar_kwargs or {})
    mnar_kwargs = dict(mnar_kwargs or {})
    mnar_key = method_mnar.lower().strip()
    if mnar_key not in _MNAR_METHODS:
        raise ValueError(
            f"method_mnar must be one of {sorted(_MNAR_METHODS)}, got {method_mnar!r}"
        )
    # Step 1: MAR imputer on MCAR rows.
    partial = impute_mar(X, mcar_mask, method=method_mar, **mar_kwargs)
    # Step 2: MNAR imputer on whatever still has NaN.
    return _MNAR_METHODS[mnar_key](partial, **mnar_kwargs)
