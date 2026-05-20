"""Smoke tests for the v0.1.1 MAR/MNAR wrappers — no R needed.

Covers :func:`impute_zero`, :func:`impute_mle`, :func:`impute_knn`,
:func:`impute_svd`, :func:`impute_mar`, :func:`impute_mar_mnar`,
:func:`insert_mvs`, :func:`generate_expression_data`, :func:`pep2prot`
and :func:`generate_rollup_map`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyimputelcmd import (
    generate_expression_data,
    generate_rollup_map,
    impute,
    impute_knn,
    impute_mar,
    impute_mar_mnar,
    impute_mle,
    impute_svd,
    impute_zero,
    insert_mvs,
    model_selector,
    pep2prot,
)


def _make_matrix(seed=0, n=200, m=6, mean=20.0, sd=1.0, lod=19.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(mean, sd, (n, m))
    X[X < lod] = np.nan
    return X


# --------------------------------------------------------------------------- #
# impute_zero                                                                 #
# --------------------------------------------------------------------------- #
def test_impute_zero_replaces_nan_with_zero():
    X = np.array([[1.0, np.nan], [np.nan, 2.0]])
    out = impute_zero(X)
    assert not np.isnan(out).any()
    assert out[0, 1] == 0.0
    assert out[1, 0] == 0.0
    # Observed entries unchanged.
    assert out[0, 0] == 1.0
    assert out[1, 1] == 2.0


def test_impute_zero_dataframe_preserves_labels():
    df = pd.DataFrame([[1.0, np.nan]], index=['p0'], columns=['s0', 's1'])
    out = impute_zero(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ['p0']
    assert list(out.columns) == ['s0', 's1']


# --------------------------------------------------------------------------- #
# impute_knn                                                                  #
# --------------------------------------------------------------------------- #
def test_impute_knn_no_nan_and_shape_preserved():
    X = _make_matrix()
    out = impute_knn(X, K=10)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


def test_impute_knn_deterministic():
    # sklearn KNNImputer is deterministic — same input ⇒ identical output.
    X = _make_matrix(seed=1)
    a = impute_knn(X, K=15)
    b = impute_knn(X, K=15)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# impute_mle                                                                  #
# --------------------------------------------------------------------------- #
def test_impute_mle_no_nan_and_shape_preserved():
    X = _make_matrix()
    out = impute_mle(X)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


def test_impute_mle_deterministic_with_seed():
    # impute_mle draws from the predictive Gaussian (R imp.norm
    # I-step) — deterministic only when a seed is fixed.
    X = _make_matrix(seed=2)
    a = impute_mle(X, seed=7)
    b = impute_mle(X, seed=7)
    np.testing.assert_array_equal(a, b)


def test_impute_mle_conditional_mean_is_deterministic():
    # sample=False uses the conditional mean — deterministic without
    # a seed.
    X = _make_matrix(seed=2)
    a = impute_mle(X, sample=False)
    b = impute_mle(X, sample=False)
    np.testing.assert_array_equal(a, b)


def test_impute_mle_observed_entries_unchanged():
    X = _make_matrix(seed=3)
    mask = np.isnan(X)
    out = impute_mle(X, seed=0)
    np.testing.assert_array_equal(out[~mask], X[~mask])


# --------------------------------------------------------------------------- #
# impute_svd                                                                  #
# --------------------------------------------------------------------------- #
def test_impute_svd_no_nan_and_shape_preserved():
    X = _make_matrix()
    out = impute_svd(X, K=2)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


def test_impute_svd_deterministic():
    X = _make_matrix(seed=4)
    a = impute_svd(X, K=2)
    b = impute_svd(X, K=2)
    np.testing.assert_array_equal(a, b)


def test_impute_svd_observed_entries_unchanged():
    X = _make_matrix(seed=5)
    mask = np.isnan(X)
    out = impute_svd(X, K=2)
    np.testing.assert_array_equal(out[~mask], X[~mask])


# --------------------------------------------------------------------------- #
# impute_mar / impute_mar_mnar                                                #
# --------------------------------------------------------------------------- #
def test_impute_mar_leaves_mnar_rows_alone():
    X = _make_matrix(seed=6)
    # Force a known mask: first half MAR, second half MNAR.
    mcar_mask = np.zeros(X.shape[0], dtype=bool)
    mcar_mask[:X.shape[0] // 2] = True
    out = impute_mar(X, mcar_mask, method='mle')
    # MAR rows have no NaN after imputation.
    assert not np.isnan(out[:X.shape[0] // 2]).any()
    # MNAR rows: NaN preserved.
    nan_locations = np.isnan(X[X.shape[0] // 2:])
    np.testing.assert_array_equal(np.isnan(out[X.shape[0] // 2:]), nan_locations)


def test_impute_mar_mnar_fully_imputes():
    X = _make_matrix(seed=7)
    mcar_mask = np.zeros(X.shape[0], dtype=bool)
    mcar_mask[:X.shape[0] // 2] = True
    out = impute_mar_mnar(X, mcar_mask, method_mar='knn', method_mnar='qrilc',
                          mar_kwargs={'K': 10}, mnar_kwargs={'seed': 0})
    assert out.shape == X.shape
    assert not np.isnan(out).any()


@pytest.mark.parametrize('method', ['mle', 'knn', 'svd'])
def test_impute_mar_mnar_with_each_mar_method(method):
    X = _make_matrix(seed=8)
    mcar_mask = np.ones(X.shape[0], dtype=bool)
    out = impute_mar_mnar(X, mcar_mask, method_mar=method, method_mnar='mindet',
                          mnar_kwargs={'q': 0.01})
    assert not np.isnan(out).any()


# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('method', ['zero', 'mle', 'knn', 'svd'])
def test_dispatcher_routes_to_wrapper(method):
    X = _make_matrix(seed=9)
    out = impute(X, method=method)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


def test_dispatcher_lists_all_wrappers():
    from pyimputelcmd.imputers import _DISPATCH
    for name in ('mindet', 'minprob', 'qrilc', 'zero', 'mle', 'knn',
                 'svd', 'mar', 'mar_mnar'):
        assert name in _DISPATCH


# --------------------------------------------------------------------------- #
# insert_mvs                                                                   #
# --------------------------------------------------------------------------- #
def test_insert_mvs_mcar_count():
    rng = np.random.default_rng(0)
    X = rng.normal(20, 1, (50, 6))
    out = insert_mvs(X, n_mv=30, mode='MCAR', seed=0)
    assert int(np.isnan(out).sum()) == 30


def test_insert_mvs_mnar_only_low_values():
    rng = np.random.default_rng(1)
    X = rng.normal(20, 1, (100, 6))
    out = insert_mvs(X, n_mv=20, mode='MNAR', threshold_quantile=0.2, seed=1)
    mask = np.isnan(out)
    # All injected entries should sit below the empirical 20% quantile
    # of the original matrix.
    thr = np.quantile(X, 0.2)
    assert (X[mask] <= thr).all()


def test_insert_mvs_mixed_count():
    rng = np.random.default_rng(2)
    X = rng.normal(20, 1, (100, 6))
    out = insert_mvs(X, n_mv=40, mode='mixed', mnar_rate=0.5,
                     threshold_quantile=0.5, seed=2)
    assert 35 <= int(np.isnan(out).sum()) <= 40  # allow tiny slack


def test_insert_mvs_seed_reproducible():
    rng = np.random.default_rng(3)
    X = rng.normal(20, 1, (60, 4))
    a = insert_mvs(X, n_mv=15, mode='MCAR', seed=99)
    b = insert_mvs(X, n_mv=15, mode='MCAR', seed=99)
    # Same NaN pattern.
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


# --------------------------------------------------------------------------- #
# generate_expression_data                                                    #
# --------------------------------------------------------------------------- #
def test_generate_expression_data_shape_and_labels():
    syn = generate_expression_data(
        n_features=100, n_samples1=3, n_samples2=4,
        n_features_up=10, n_features_down=15, seed=0,
    )
    assert syn.data.shape == (100, 7)
    assert syn.conditions.tolist() == [1, 1, 1, 2, 2, 2, 2]
    assert int((syn.label_features == 1).sum()) == 10
    assert int((syn.label_features == 2).sum()) == 15
    assert int((syn.label_features == 3).sum()) == 75


def test_generate_expression_data_seed_reproducible():
    a = generate_expression_data(50, 3, 3, seed=42)
    b = generate_expression_data(50, 3, 3, seed=42)
    np.testing.assert_array_equal(a.data, b.data)


def test_generate_expression_data_de_effect_size():
    """Up-DE rows should have higher condition-1 mean than condition-2."""
    syn = generate_expression_data(
        n_features=50, n_samples1=10, n_samples2=10,
        n_features_up=20, n_features_down=0,
        mean_diff_abund=3.0, sd_diff_abund=0.1, seed=0,
    )
    up = syn.data[:20]
    cond1_mean = up[:, syn.conditions == 1].mean()
    cond2_mean = up[:, syn.conditions == 2].mean()
    assert cond1_mean > cond2_mean + 2.0


# --------------------------------------------------------------------------- #
# pep2prot / generate_rollup_map                                              #
# --------------------------------------------------------------------------- #
def test_generate_rollup_map_dataframe():
    df = pd.DataFrame({'pep': ['p1', 'p2', 'p3'], 'prot': ['A', 'A', 'B']})
    m = generate_rollup_map(df)
    assert list(m.columns) == ['peptide', 'protein']
    assert m.shape == (3, 2)


def test_generate_rollup_map_vector():
    m = generate_rollup_map(['A', 'A', 'B', 'C'])
    assert m['protein'].tolist() == ['A', 'B', 'C'] or \
           sorted(m['protein'].unique().tolist()) == ['A', 'B', 'C']


def test_pep2prot_median_rollup():
    pep = pd.DataFrame(
        [[10.0, 11.0],
         [12.0, 13.0],
         [100.0, 101.0]],
        index=['p1', 'p2', 'p3'], columns=['s1', 's2'],
    )
    mapping = pd.DataFrame({'peptide': ['p1', 'p2', 'p3'],
                            'protein': ['A', 'A', 'B']})
    out = pep2prot(pep, mapping, method='median')
    assert out.shape == (2, 2)
    # Protein A is the median of [p1, p2] = (11, 12).
    assert out.loc['A', 's1'] == pytest.approx(11.0)
    assert out.loc['A', 's2'] == pytest.approx(12.0)
    # Protein B is the single peptide p3.
    assert out.loc['B', 's1'] == 100.0


def test_pep2prot_vector_mapping():
    pep = pd.DataFrame([[10.0], [12.0], [100.0]],
                       index=['p1', 'p2', 'p3'], columns=['s1'])
    out = pep2prot(pep, ['A', 'A', 'B'])
    assert out.shape == (2, 1)
    assert out.loc['A', 's1'] == pytest.approx(11.0)


@pytest.mark.parametrize('method', ['mean', 'sum', 'max', 'min'])
def test_pep2prot_alternative_aggregators(method):
    pep = pd.DataFrame([[1.0], [2.0], [3.0]],
                       index=['p1', 'p2', 'p3'], columns=['s1'])
    out = pep2prot(pep, ['A', 'A', 'A'], method=method)
    expected = {'mean': 2.0, 'sum': 6.0, 'max': 3.0, 'min': 1.0}[method]
    assert out.loc['A', 's1'] == pytest.approx(expected)
