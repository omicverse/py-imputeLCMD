"""Peptide-to-protein roll-up helpers (R ``pep2prot`` / ``generate.RollUpMap``).

Two small utilities for collapsing peptide-level intensity matrices to
protein-level. Mirrors the R API but accepts the more idiomatic
pandas-style inputs.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# generate.RollUpMap                                                          #
# --------------------------------------------------------------------------- #

def generate_rollup_map(
    mapping: Union[pd.DataFrame, Sequence],
    peptide_col: Optional[str] = None,
    protein_col: Optional[str] = None,
) -> pd.DataFrame:
    r"""Normalise a peptide → protein mapping into a roll-up index.

    Equivalent to the *table* returned by R ``generate.RollUpMap`` —
    a frame of unique ``(peptide, protein)`` pairs that downstream
    :func:`pep2prot` can group on.

    Accepts:

    * a 2-column :class:`pandas.DataFrame` (default: first column is
      the peptide, second is the protein), or
    * a 1-D ``pep.prot.Map`` vector (length-``n_peptides`` protein ID
      per peptide row, matching the R return value of
      ``generate.RollUpMap``).

    Parameters
    ----------
    mapping
        DataFrame or vector linking peptides to proteins.
    peptide_col, protein_col
        Column names if ``mapping`` is a DataFrame and the default
        positional convention does not apply.

    Returns
    -------
    DataFrame
        Two columns: ``peptide`` and ``protein``. One row per
        (peptide, protein) edge.
    """
    if isinstance(mapping, pd.DataFrame):
        if peptide_col is None:
            peptide_col = mapping.columns[0]
        if protein_col is None:
            protein_col = mapping.columns[1]
        out = pd.DataFrame({
            'peptide': mapping[peptide_col].to_numpy(),
            'protein': mapping[protein_col].to_numpy(),
        })
    else:
        arr = np.asarray(list(mapping))
        # R-style: a vector of protein IDs, one per peptide row.
        out = pd.DataFrame({
            'peptide': np.arange(len(arr)),
            'protein': arr,
        })
    return out.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# pep2prot                                                                    #
# --------------------------------------------------------------------------- #

def pep2prot(
    peptide_data: pd.DataFrame,
    rollup_map: Union[pd.DataFrame, np.ndarray, Sequence],
    intensity_cols: Optional[Iterable[str]] = None,
    peptide_col: Optional[str] = None,
    protein_col: Optional[str] = None,
    method: str = 'median',
) -> pd.DataFrame:
    r"""Collapse a peptide-level frame to one row per protein.

    Mirrors R ``pep2prot(pep.Expr.Data, rollup.map)`` but with a
    pandas-friendly signature. The roll-up aggregator is configurable
    (R hard-codes the median); this port additionally exposes
    ``'sum'``, ``'mean'``, ``'max'``, and ``'min'``.

    Parameters
    ----------
    peptide_data
        DataFrame indexed by peptide (or with a peptide ID column,
        passed via ``peptide_col``). Numeric columns are the
        intensity values.
    rollup_map
        Either a roll-up DataFrame from :func:`generate_rollup_map`
        (two columns ``peptide`` and ``protein``) or a 1-D vector of
        protein IDs aligned to ``peptide_data`` rows (R semantics).
    intensity_cols
        Columns of ``peptide_data`` to roll up. Defaults to all
        numeric columns.
    peptide_col, protein_col
        Names used when ``rollup_map`` is a DataFrame.
    method
        Aggregator: ``'median'`` (R default), ``'sum'``, ``'mean'``,
        ``'max'``, ``'min'``.

    Returns
    -------
    DataFrame
        One row per unique protein, indexed by protein ID. NaN
        peptide values are skipped (``na.rm = TRUE`` in R).
    """
    method_key = method.lower().strip()
    # Pandas GroupBy.median()/mean()/min()/max() drop NaN by default
    # (matching R's ``na.rm = TRUE``). ``sum`` accepts ``min_count=1`` so
    # an all-NaN group rolls up to NaN instead of 0.
    aggregators = {
        'median': lambda g: g.median(),
        'mean':   lambda g: g.mean(),
        'sum':    lambda g: g.sum(min_count=1),
        'max':    lambda g: g.max(),
        'min':    lambda g: g.min(),
    }
    if method_key not in aggregators:
        raise ValueError(
            f"method must be one of {sorted(aggregators)}, got {method!r}"
        )
    df = peptide_data.copy()
    # Normalise the mapping to a per-row protein ID series aligned
    # with ``df``.
    if isinstance(rollup_map, pd.DataFrame):
        pcol = peptide_col or (
            'peptide' if 'peptide' in rollup_map.columns else rollup_map.columns[0]
        )
        rcol = protein_col or (
            'protein' if 'protein' in rollup_map.columns else rollup_map.columns[1]
        )
        mapping = pd.Series(
            rollup_map[rcol].to_numpy(),
            index=rollup_map[pcol].to_numpy(),
        )
        proteins = mapping.reindex(df.index).to_numpy()
    else:
        arr = np.asarray(list(rollup_map))
        if arr.size != df.shape[0]:
            raise ValueError(
                f"rollup_map length {arr.size} != peptide_data rows {df.shape[0]}"
            )
        proteins = arr

    if intensity_cols is None:
        intensity_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    intensity_cols = list(intensity_cols)
    df = df[intensity_cols].copy()
    df['__protein__'] = proteins
    # Drop peptides whose protein ID is NaN (no mapping).
    df = df[~df['__protein__'].isna()]
    grouped = df.groupby('__protein__', sort=False)[intensity_cols]
    out = aggregators[method_key](grouped)
    out.index.name = 'protein'
    return out
