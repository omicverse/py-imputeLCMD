# pyimputelcmd

A **pure-Python port of [Bioconductor imputeLCMD](https://www.bioconductor.org/packages/release/bioc/html/imputeLCMD.html)** (Lazar et al., *J Proteome Res* 2016) for left-censored missing-value imputation in label-free LC-MS/MS proteomics data.

- **Full imputeLCMD API** — all imputers (`MinDet`, `MinProb`, `QRILC`, `ZERO`, `MLE`, `KNN`, `SVD`, `MAR`, `MAR.MNAR`), the `model.Selector` MCAR/MNAR classifier, synthetic-data and roll-up helpers
- **No `rpy2`**, no R install — everything in NumPy / SciPy / pandas / scikit-learn
- Bit-for-bit reproduction of the R reference for the deterministic `MinDet` / `ZERO`; distribution-level (KS) parity for the stochastic `MinProb` / `QRILC`; high Pearson-correlation parity for `KNN` / `SVD` / `MLE` (R-parity tests in `tests/test_r_parity.py`)
- AnnData-friendly: accepts `np.ndarray` or `pd.DataFrame` (rows = proteins, columns = samples; preserves index/columns)
- A single `impute(X, method=…)` dispatcher for the omicverse wrapper

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse) (`omicverse.protein.pp.impute`). All algorithmic work is developed upstream in omicverse and synced here for users who want the imputers without the full omicverse stack.

## Install

```bash
pip install pyimputelcmd
```

## Quick start

```python
import numpy as np
from pyimputelcmd import impute, impute_mindet, impute_minprob, impute_qrilc

rng = np.random.default_rng(0)
X = rng.normal(20.0, 1.0, (500, 6))    # 500 proteins × 6 samples
X[X < 19.0] = np.nan                   # left-censored MNAR (~16% missing)

# Three R-parity imputers — all accept the same (X, …) signature
out_md = impute_mindet(X)                     # 1st-percentile floor (q=0.01)
out_mp = impute_minprob(X, seed=0)            # Gaussian below the floor
out_qr = impute_qrilc(X, seed=0)              # truncated normal, QR-fit mu/sigma

# Single dispatcher (preferred for omicverse / config-driven workflows)
out = impute(X, method='qrilc', tune_sigma=1.0, seed=0)
```

## Functional API (mirrors R one-to-one)

### Imputers

| Python | R counterpart | Notes |
|---|---|---|
| `impute_mindet(X, q=0.01)` | `impute.MinDet` | Deterministic — bit-exact match |
| `impute_minprob(X, q=0.01, tune_sigma=1.0, seed=None)` | `impute.MinProb` | Stochastic; KS-equivalent to R |
| `impute_qrilc(X, tune_sigma=1.0, seed=None, upper_q=0.99)` | `impute.QRILC` | Stochastic; OLS-fit (μ, σ) match R `lm()` exactly |
| `impute_zero(X)` | `impute.ZERO` | Deterministic — bit-exact match |
| `impute_mle(X, max_iter=200, tol=1e-4, seed=None, sample=True)` | `impute.wrapper.MLE` | MVN-EM + I-step draw (`norm::imp.norm`); Pearson r ≈ 0.98 vs R |
| `impute_knn(X, K=10)` | `impute.wrapper.KNN` | Per-protein KNN (`sklearn.KNNImputer`); Pearson r > 0.99 vs R |
| `impute_svd(X, K=2)` | `impute.wrapper.SVD` | Iterative rank-K SVD (Stacklies 2007); Pearson r > 0.99 vs R |
| `impute_mar(X, mcar_mask, method='mle')` | `impute.MAR` | Apply a MAR imputer to MCAR-flagged rows |
| `impute_mar_mnar(X, mcar_mask, method_mar='mle', method_mnar='qrilc')` | `impute.MAR.MNAR` | Combined MAR + MNAR pipeline |
| `impute(X, method=…)` | — | Dispatcher used by `omicverse.protein.pp.impute` |

### Model selection & utilities

| Python | R counterpart | Notes |
|---|---|---|
| `model_selector(X)` → `(is_mar, censoring_thr)` | `model.Selector` | MCAR/MNAR classifier; 100% flag agreement with R |
| `insert_mvs(X, n_mv=200, mode='MCAR', …)` | `insertMVs` | Inject synthetic MVs for benchmarking |
| `generate_expression_data(n_features, n_samples1, n_samples2, …)` | `generate.ExpressionData` | Synthetic two-condition data |
| `pep2prot(peptide_data, rollup_map, method='median')` | `pep2prot` | Peptide → protein roll-up |
| `generate_rollup_map(mapping)` | `generate.RollUpMap` | Build a peptide → protein roll-up table |

## Matrix orientation

The R imputeLCMD package uses **rows = proteins / peptides, columns = samples**, and so does this port. AnnData users should transpose first:

```python
import anndata as ad
adata = ad.read_h5ad("proteins.h5ad")          # cells × proteins (AnnData layout)
X = adata.X.T                                  # proteins × samples
imputed = impute(X, method='qrilc')
adata.X = imputed.T
```

## Reproducing the R reference exactly

`tests/r_reference_driver.R` invokes the original R imputeLCMD functions on the same input matrix dumped by the Python side. `tests/test_r_parity.py` then checks:

1. **MinDet / ZERO** — `np.allclose(py, R, atol=1e-12)` (bit-exact deterministic)
2. **MinProb** — KS test per column on the imputed marginal (p > 0.01)
3. **QRILC** — closed-form OLS intercept/slope agree with R `lm()` to 1e-6, and the truncated-normal draws pass a KS test against R `rtmvnorm` (Gibbs)
4. **KNN / SVD / MLE** — Pearson correlation against R on a realistic correlated (low-rank + noise) matrix: KNN r > 0.99, SVD r > 0.99, MLE r ≈ 0.98
5. **model.Selector** — per-protein MCAR/MNAR flags agree with R (100% on the bimodal fixture)

```bash
# Run the R-parity tests (needs the CMAP env or env vars)
PYIMPUTELCMD_RSCRIPT=/path/to/Rscript pytest tests/test_r_parity.py -v
```

## Coverage of the R imputeLCMD API

**100% function coverage** — all 14 functions exported by Bioconductor
imputeLCMD are ported.

| R function | Status |
|---|---|
| `impute.MinDet`, `impute.MinProb`, `impute.QRILC` | ✅ v0.1 |
| `impute.ZERO` | ✅ v0.1.1 |
| `impute.wrapper.MLE` / `impute.wrapper.KNN` / `impute.wrapper.SVD` | ✅ v0.1.1 |
| `impute.MAR` / `impute.MAR.MNAR` | ✅ v0.1.1 |
| `model.Selector` (MCAR/MNAR classifier) | ✅ v0.1.1 |
| `insertMVs`, `generate.ExpressionData` | ✅ v0.1.1 |
| `pep2prot`, `generate.RollUpMap` | ✅ v0.1.1 |

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: `omicverse.protein.pp.impute`
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original imputeLCMD paper:

> Lazar, C., Gatto, L., Ferro, M., Bruley, C., Burger, T. **Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare the Performance of Normalization Strategies.** *J Proteome Res* 15, 1116–1125 (2016). DOI: 10.1021/acs.jproteome.5b00981

and acknowledge omicverse / this repo for the Python port.

## License

GPL-3 — matches the upstream Bioconductor package.
