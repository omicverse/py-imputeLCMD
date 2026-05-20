"""
pyimputelcmd: Pure-Python port of Bioconductor imputeLCMD.

Left-censored missing-value imputation for label-free proteomics
intensity matrices. Three imputers are provided, all faithful
re-implementations of the R reference (Lazar 2016, JPR):

* :func:`impute_mindet`  — deterministic ``q``-quantile floor
* :func:`impute_minprob` — Gaussian draws below the ``q``-quantile
* :func:`impute_qrilc`   — quantile-regression-fit truncated normal

A single :func:`impute` dispatcher selects between them via
``method='mindet' | 'minprob' | 'qrilc'`` and is the entry point used
by ``omicverse.protein.pp.impute``.

Matrix convention (R-native): rows = proteins / peptides, columns =
samples. NaN marks missing.

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

__version__ = "0.1.0"

__all__ = [
    "impute",
    "impute_mindet",
    "impute_minprob",
    "impute_qrilc",
]
