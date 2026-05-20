"""Basic sanity checks for the three imputers — no R needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyimputelcmd import (
    impute,
    impute_mindet,
    impute_minprob,
    impute_qrilc,
)


def _make_lcmd_matrix(seed: int = 0, n_proteins: int = 500, n_samples: int = 6,
                      mean: float = 20.0, sd: float = 1.0,
                      lod: float = 19.0) -> np.ndarray:
    """500 × 6 log-intensity matrix with ~25% left-censored NaNs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(mean, sd, (n_proteins, n_samples))
    X[X < lod] = np.nan
    return X


def test_no_missing_after_mindet():
    X = _make_lcmd_matrix()
    out = impute_mindet(X)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


def test_no_missing_after_minprob():
    X = _make_lcmd_matrix()
    out = impute_minprob(X, seed=0)
    assert not np.isnan(out).any()


def test_no_missing_after_qrilc():
    X = _make_lcmd_matrix()
    out = impute_qrilc(X, seed=0)
    assert not np.isnan(out).any()


def test_mindet_matches_quantile():
    # Hand-checked: with q=0.5 the imputation floor equals the median
    # of observed values per column.
    X = np.array([
        [10.0, np.nan, 12.0],
        [np.nan, 8.0, 9.0],
        [11.0, 7.0, np.nan],
    ])
    out = impute_mindet(X, q=0.5)
    assert out[1, 0] == pytest.approx(np.median([10.0, 11.0]))
    assert out[0, 1] == pytest.approx(np.median([8.0, 7.0]))
    assert out[2, 2] == pytest.approx(np.median([12.0, 9.0]))


def test_minprob_imputes_below_observed_floor_on_average():
    # MinProb draws around the q=0.01 quantile of observed values, so
    # the mean imputed value should sit near the bottom of the
    # observed distribution.
    X = _make_lcmd_matrix(seed=0)
    mask = np.isnan(X)
    out = impute_minprob(X, q=0.01, seed=0)
    for j in range(X.shape[1]):
        if not mask[:, j].any():
            continue
        obs = X[~mask[:, j], j]
        imp_mean = out[mask[:, j], j].mean()
        # imputed mean should be close to the lower 1st-percentile of observed
        assert imp_mean < float(np.quantile(obs, 0.10))


def test_minprob_seed_reproducibility():
    X = _make_lcmd_matrix()
    out1 = impute_minprob(X, seed=42)
    out2 = impute_minprob(X, seed=42)
    np.testing.assert_array_equal(out1, out2)


def test_qrilc_seed_reproducibility():
    X = _make_lcmd_matrix()
    out1 = impute_qrilc(X, seed=42)
    out2 = impute_qrilc(X, seed=42)
    np.testing.assert_array_equal(out1, out2)


def test_qrilc_imputed_below_observed_mean():
    # QRILC samples from a truncated normal whose upper bound is
    # ``qnorm(p_NAs + 0.001, mu, sigma)`` — i.e. left of the
    # estimated detection threshold. Imputed values should therefore
    # be (on average) substantially lower than the observed mean.
    X = _make_lcmd_matrix(seed=0)
    mask = np.isnan(X)
    out = impute_qrilc(X, seed=0)
    for j in range(X.shape[1]):
        if not mask[:, j].any():
            continue
        obs = X[~mask[:, j], j]
        imp = out[mask[:, j], j]
        assert imp.mean() < obs.mean()
        # Imputed values must all sit below the (overestimated) upper
        # truncation bound; pick a generous slack of 2σ above
        # observed-mean for portability.
        assert (imp < obs.mean() + 2.0 * obs.std()).all()


def test_dataframe_roundtrip_preserves_index_columns():
    X = _make_lcmd_matrix(seed=1)
    df = pd.DataFrame(
        X,
        index=[f"prot_{i}" for i in range(X.shape[0])],
        columns=[f"S{j}" for j in range(X.shape[1])],
    )
    out = impute_qrilc(df, seed=0)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == list(df.index)
    assert list(out.columns) == list(df.columns)


def test_dispatch_routes_correctly():
    X = _make_lcmd_matrix(seed=2)
    out1 = impute(X, method='mindet', q=0.5)
    out2 = impute_mindet(X, q=0.5)
    np.testing.assert_array_equal(out1, out2)


def test_dispatch_invalid_method_raises():
    X = _make_lcmd_matrix()
    with pytest.raises(ValueError, match="method must be one of"):
        impute(X, method='nope')


def test_minprob_zero_missing_is_noop():
    rng = np.random.default_rng(0)
    X = rng.normal(20, 1, (50, 3))  # no NaNs
    out = impute_minprob(X, seed=0)
    np.testing.assert_array_equal(out, X)


def test_qrilc_zero_missing_is_noop():
    rng = np.random.default_rng(0)
    X = rng.normal(20, 1, (50, 3))
    out = impute_qrilc(X, seed=0)
    np.testing.assert_array_equal(out, X)
