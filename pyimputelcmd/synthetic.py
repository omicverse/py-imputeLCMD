"""Synthetic proteomics data generator (R ``generate.ExpressionData``).

A direct port of the imputeLCMD helper used to simulate two-condition
expression matrices with a controllable number of differentially-
abundant features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SyntheticExpressionData:
    """Container for synthetic two-condition expression data.

    Mirrors the unnamed list returned by R ``generate.ExpressionData``:

    * ``data`` — ``n_features × n_samples`` log-intensity matrix.
    * ``conditions`` — length-``n_samples`` vector of integer labels
      (1 = condition A, 2 = condition B).
    * ``label_features`` — length-``n_features`` vector marking each
      protein as up-DE (1), down-DE (2), or null (3).
    """

    data: np.ndarray
    conditions: np.ndarray
    label_features: np.ndarray

    def __iter__(self):
        # Tuple-style unpacking to mirror R's list return.
        yield self.data
        yield self.conditions
        yield self.label_features


def generate_expression_data(
    n_features: int,
    n_samples1: int,
    n_samples2: int,
    n_features_up: int = 0,
    n_features_down: int = 0,
    mean_samples: float = 0.0,
    sd_samples: float = 1.0,
    mean_dyn_range: float = 22.0,
    sd_dyn_range: float = 3.0,
    mean_diff_abund: float = 2.0,
    sd_diff_abund: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticExpressionData:
    r"""Generate synthetic two-condition proteomics expression data.

    Faithful port of R ``imputeLCMD::generate.ExpressionData``:

    1. Draw ``n_features × n_samples`` Gaussian noise
       ``N(mean_samples, sd_samples^2)``.
    2. Draw a per-feature ``means`` vector
       ``N(mean_dyn_range, sd_dyn_range^2)`` to spread proteins across
       the dynamic range.
    3. Add up-DE / down-DE coefficients
       ``N(mean_diff_abund, sd_diff_abund^2)`` to the first
       ``n_features_up`` rows in condition 1 and the next
       ``n_features_down`` rows in condition 2.

    Parameters
    ----------
    n_features
        Total number of proteins (rows).
    n_samples1, n_samples2
        Replicate counts for conditions A and B.
    n_features_up, n_features_down
        Counts of up- / down-DE proteins.
    mean_samples, sd_samples
        Per-cell technical noise.
    mean_dyn_range, sd_dyn_range
        Per-protein abundance offset.
    mean_diff_abund, sd_diff_abund
        Differential-abundance effect size.
    seed
        NumPy PRNG seed for reproducibility.

    Returns
    -------
    SyntheticExpressionData
        ``data`` (matrix), ``conditions`` (1/2 labels),
        ``label_features`` (1=up, 2=down, 3=null).
    """
    if n_features_up + n_features_down > n_features:
        raise ValueError(
            f"n_features_up + n_features_down ({n_features_up + n_features_down}) "
            f"exceeds n_features ({n_features})"
        )
    rng = np.random.default_rng(seed)
    n_samples = n_samples1 + n_samples2
    data = rng.normal(mean_samples, sd_samples, (n_features, n_samples))
    feat_means = rng.normal(mean_dyn_range, sd_dyn_range, n_features)
    data = data + feat_means[:, None]
    conditions = np.concatenate([
        np.ones(n_samples1, dtype=int),
        np.full(n_samples2, 2, dtype=int),
    ])
    cond1 = conditions == 1
    cond2 = conditions == 2
    if n_features_up > 0:
        de_up = rng.normal(
            mean_diff_abund, sd_diff_abund, (n_features_up, n_samples1),
        )
        data[:n_features_up, cond1] = data[:n_features_up, cond1] + de_up
    if n_features_down > 0:
        de_down = rng.normal(
            mean_diff_abund, sd_diff_abund, (n_features_down, n_samples2),
        )
        s = n_features_up
        e = n_features_up + n_features_down
        data[s:e, cond2] = data[s:e, cond2] + de_down
    label_features = np.full(n_features, 3, dtype=int)
    label_features[:n_features_up] = 1
    label_features[n_features_up:n_features_up + n_features_down] = 2
    return SyntheticExpressionData(
        data=data, conditions=conditions, label_features=label_features,
    )
