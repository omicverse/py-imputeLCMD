"""
pyimputelcmd: Pure-Python port of Bioconductor imputeLCMD.

Left-censored missing-value imputation for label-free proteomics
intensity matrices. Full v0.1.1 API mirrors R imputeLCMD one-to-one:

Left-censored MNAR imputers (Lazar 2016, JPR)
---------------------------------------------
* :func:`impute_mindet`  — deterministic ``q``-quantile floor
* :func:`impute_minprob` — Gaussian draws below the ``q``-quantile
* :func:`impute_qrilc`   — quantile-regression-fit truncated normal

MAR imputers (wrappers)
-----------------------
* :func:`impute_zero` — replace NaN with 0
* :func:`impute_mle`  — MVN-EM MLE (mirrors ``norm::imp.norm``)
* :func:`impute_knn`  — per-protein KNN via Euclidean distance
* :func:`impute_svd`  — iterative rank-K SVD (Stacklies 2007)

Combined / model-aware
----------------------
* :func:`impute_mar`      — apply a MAR imputer to MCAR-flagged rows
* :func:`impute_mar_mnar` — combined MAR + MNAR pipeline

Model selection & utilities
---------------------------
* :func:`model_selector`           — MCAR vs MNAR classifier
* :func:`insert_mvs`               — inject synthetic MVs for benchmarking
* :func:`generate_expression_data` — synthetic two-condition data
* :func:`pep2prot`                 — peptide → protein roll-up
* :func:`generate_rollup_map`      — build a peptide→protein roll-up table

Single dispatcher
-----------------
``impute(X, method=…)`` selects between all imputers and is the entry
point used by ``omicverse.protein.pp.impute``. Matrix convention
(R-native): rows = proteins / peptides, columns = samples. NaN marks
missing.

Quick-start
-----------
>>> from pyimputelcmd import impute_qrilc
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(20, 1, (500, 4))         # 500 proteins × 4 samples
>>> X[X < 19] = np.nan                       # left-censored MNAR
>>> X_imp = impute_qrilc(X, tune_sigma=1.0, seed=0)
>>> np.isnan(X_imp).any()
np.False_

References
----------
Lazar, C., Gatto, L., Ferro, M., Bruley, C., Burger, T.
*Accounting for the Multiple Natures of Missing Values in Label-Free
Quantitative Proteomics Data Sets to Compare the Performance of
Normalization Strategies.*
J Proteome Res. 2016, 15 (4): 1116-1125.
DOI: 10.1021/acs.jproteome.5b00981
"""
from __future__ import annotations

from .imputers import (
    impute,
    impute_mindet,
    impute_minprob,
    impute_qrilc,
)
from .wrappers import (
    impute_zero,
    impute_knn,
    impute_mle,
    impute_svd,
    impute_mar,
    impute_mar_mnar,
)
from .model_selector import model_selector, insert_mvs
from .synthetic import generate_expression_data, SyntheticExpressionData
from .rollup import pep2prot, generate_rollup_map

__version__ = "0.1.1"

__all__ = [
    # Core (v0.1)
    "impute",
    "impute_mindet",
    "impute_minprob",
    "impute_qrilc",
    # MAR wrappers (v0.1.1)
    "impute_zero",
    "impute_knn",
    "impute_mle",
    "impute_svd",
    "impute_mar",
    "impute_mar_mnar",
    # Model selection + synthetic + roll-up (v0.1.1)
    "model_selector",
    "insert_mvs",
    "generate_expression_data",
    "SyntheticExpressionData",
    "pep2prot",
    "generate_rollup_map",
]
