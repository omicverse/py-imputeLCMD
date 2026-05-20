"""Smoke tests for :func:`model_selector` — synthetic MAR + MNAR data.

We build a matrix with two abundance regimes:

* High-abundance proteins (mean ≫ LOD) with light MCAR holes.
* Low-abundance proteins (mean ≈ LOD) with heavy left-censoring → MNAR.

The model selector should flag the second regime as MNAR (``False``)
and leave the first as MAR (``True``).
"""
from __future__ import annotations

import numpy as np

from pyimputelcmd import model_selector


def _make_bimodal(seed=42, n_high=300, n_low=100, n_samples=6):
    rng = np.random.default_rng(seed)
    X = np.empty((n_high + n_low, n_samples))
    X[:n_high] = rng.normal(25.0, 1.0, (n_high, n_samples))
    # Low-abundance: mean ~15, with hard censoring below 14.5.
    X[n_high:] = rng.normal(15.0, 0.5, (n_low, n_samples))
    X[n_high:][X[n_high:] < 14.5] = np.nan
    # MCAR on the high regime: 5% random missing.
    miss = rng.uniform(size=(n_high, n_samples)) < 0.05
    X[:n_high][miss] = np.nan
    return X, n_high, n_low


def test_model_selector_returns_bool_and_float():
    X, _, _ = _make_bimodal()
    is_mar, thr = model_selector(X)
    assert is_mar.dtype == bool
    assert isinstance(thr, float)
    assert is_mar.shape == (X.shape[0],)


def test_model_selector_flags_high_abundance_as_mar():
    X, n_high, _ = _make_bimodal()
    is_mar, _ = model_selector(X)
    # At least 90% of high-abundance proteins should be flagged MAR.
    assert is_mar[:n_high].mean() > 0.90


def test_model_selector_flags_low_abundance_as_mnar():
    X, n_high, n_low = _make_bimodal()
    is_mar, _ = model_selector(X)
    # At least 80% of low-abundance proteins should be flagged MNAR.
    assert (~is_mar[n_high:]).mean() > 0.80


def test_model_selector_threshold_sits_between_modes():
    X, n_high, _ = _make_bimodal()
    is_mar, thr = model_selector(X)
    # Threshold should land between the low and high regime means.
    assert 14.5 < thr < 25.0


def test_model_selector_no_missing_returns_all_mar():
    # If no missingness, ALL rows are above the (very low) threshold.
    rng = np.random.default_rng(0)
    X = rng.normal(20.0, 1.0, (300, 6))
    is_mar, _ = model_selector(X)
    assert is_mar.sum() == X.shape[0]
