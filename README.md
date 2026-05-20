# pyimputelcmd

A **pure-Python port of [Bioconductor imputeLCMD](https://www.bioconductor.org/packages/release/bioc/html/imputeLCMD.html)** (Lazar et al., *J Proteome Res* 2016) for left-censored missing-value imputation in label-free LC-MS/MS proteomics data.

- Three canonical imputers — `impute.MinDet`, `impute.MinProb`, `impute.QRILC`
- **No `rpy2`**, no R install — everything in NumPy / SciPy / pandas
- Bit-for-bit reproduction of the R reference for the deterministic `MinDet`; distribution-level parity for the stochastic `MinProb` / `QRILC` (R-parity tests in `tests/test_r_parity.py`)
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

| Python | R counterpart | Notes |
|---|---|---|
| `impute_mindet(X, q=0.01)` | `impute.MinDet(dataSet.mvs, q=0.01)` | Deterministic — bit-exact match |
| `impute_minprob(X, q=0.01, tune_sigma=1.0, seed=None)` | `impute.MinProb(dataSet.mvs, q, tune.sigma)` | Stochastic; KS-equivalent to R |
| `impute_qrilc(X, tune_sigma=1.0, seed=None, upper_q=0.99)` | `impute.QRILC(dataSet.mvs, tune.sigma)` | Stochastic; OLS-fit (μ, σ) match R `lm()` exactly |
| `impute(X, method=…)` | — | Dispatcher used by `omicverse.protein.pp.impute` |

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

`tests/r_reference_driver.R` invokes the original `impute.MinDet` / `impute.MinProb` / `impute.QRILC` on the same input matrix dumped by the Python side. `tests/test_r_parity.py` then checks:

1. **MinDet** — `np.allclose(py, R, atol=1e-12)` (bit-exact deterministic)
2. **MinProb** — KS test per column on the imputed marginal (p > 0.01)
3. **QRILC** — closed-form OLS intercept/slope agree with R `lm()` to 1e-6, and the truncated-normal draws pass a KS test against R `rtmvnorm` (Gibbs)

```bash
# Run the R-parity tests (needs the CMAP env or env vars)
PYIMPUTELCMD_RSCRIPT=/path/to/Rscript pytest tests/test_r_parity.py -v
```

## What's NOT included (yet)

| R function | Status |
|---|---|
| `impute.MinDet`, `impute.MinProb`, `impute.QRILC` | ✅ v0.1 |
| `model.Selector` (MCAR/MNAR classifier) | ⏳ planned (v0.2) |
| `impute.MAR` / `impute.wrapper.KNN` / `impute.wrapper.SVD` | ⏳ planned (v0.2) |
| `impute.wrapper.MLE` | ⏳ planned (v0.2) |

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
