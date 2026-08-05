"""de_countbased.py — support logic for count-based differential expression.

Pure transforms that (a) prepare a gene-level count matrix + colData for the
R (DESeq2 / edgeR) bridge and (b) consume its per-gene adjusted p-values to
re-derive the gene-DE flag and the 'silent' fraction.  No R is called here and
no analysis statistics are computed here — this module is the deterministic,
unit-tested glue around de_countbased.R.

The count-based DE is the negative-binomial analogue of the Kruskal-Wallis
gene-level test in run_landscape.py: for each lncRNA gene, is its isoform-summed
expression different across *any* of the tissues (DESeq2 LRT ~tissue vs ~1;
edgeR quasi-likelihood ANODEV over the tissue coefficients).  Everything else
(the switching test, the analyzed gene set, the samples/tissues) is unchanged,
so a change in the 'silent' fraction is attributable to the DE method alone.
"""
import re
from collections import OrderedDict
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from isoform_metrics import classify_visibility

_REP_RE = re.compile(r"_rep\d*$")


# ── donor / replicate ID parsing ────────────────────────────────────────────

def derive_donor(sample_id: str) -> str:
    """GTEx donor ID = first two dash-delimited fields (e.g. 'GTEX-1192X')."""
    return "-".join(sample_id.split("-")[:2])


def is_technical_replicate(sample_id: str) -> bool:
    """True if the sample ID carries a trailing technical-replicate token (_rep[N])."""
    return bool(_REP_RE.search(sample_id))


def strip_rep(sample_id: str) -> str:
    """Remove a trailing _rep[N] token; base sample IDs are returned unchanged."""
    return _REP_RE.sub("", sample_id)


# ── transcript → gene count aggregation ─────────────────────────────────────

def aggregate_tx_to_gene(
    tx_counts_df: pd.DataFrame,
    tx2gene: Dict[str, str],
    gene_set: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Sum transcript-level counts to gene level.

    Parameters
    ----------
    tx_counts_df : DataFrame
        Index = transcript_id, columns = sample_id, integer counts.
    tx2gene : dict {transcript_id: gene_id}
        Only transcripts present in this map are used (others are dropped).
    gene_set : iterable of gene_id, optional
        If given, restrict the output to these genes (e.g. the analyzed
        multi-isoform gene set from the landscape).

    Returns
    -------
    DataFrame indexed by gene_id (integer counts), samples as columns.
    """
    keep_tx = [tx for tx in tx_counts_df.index if tx in tx2gene]
    sub = tx_counts_df.loc[keep_tx]
    genes = pd.Index([tx2gene[tx] for tx in keep_tx], name="gene_id")
    gene_counts = sub.groupby(genes).sum()
    if gene_set is not None:
        gene_set = set(gene_set)
        gene_counts = gene_counts.loc[[g for g in gene_counts.index if g in gene_set]]
    # Counts are integer by construction; enforce dtype so DESeq2/edgeR are happy.
    return gene_counts.round().astype(np.int64)


# ── colData (sample → tissue / donor / replicate) ───────────────────────────

def build_coldata(sample_ids: Iterable[str], sample2tissue: Dict[str, str]) -> pd.DataFrame:
    """Build a colData table for the DE models.

    Returns a DataFrame with columns [sample, tissue, donor, is_rep], one row
    per sample present in sample2tissue (order = iteration order of sample_ids).
    """
    rows = []
    for s in sample_ids:
        if s not in sample2tissue:
            continue
        rows.append(
            {
                "sample": s,
                "tissue": sample2tissue[s],
                "donor": derive_donor(s),
                "is_rep": is_technical_replicate(s),
            }
        )
    return pd.DataFrame(rows, columns=["sample", "tissue", "donor", "is_rep"])


# ── technical-replicate collapse ────────────────────────────────────────────

def collapse_replicates(gene_counts_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical replicates by summing counts of columns that share a
    base sample ID (i.e. '<id>' and '<id>_rep' are summed into one '<id>' column).

    Column order follows first appearance of each base ID.  Removes the
    pseudoreplication that would otherwise inflate DE power for the 8 GTEx
    long-read technical replicates.
    """
    groups: "OrderedDict[str, list]" = OrderedDict()
    for col in gene_counts_df.columns:
        base = strip_rep(col)
        groups.setdefault(base, []).append(col)
    data = {base: gene_counts_df[cols].sum(axis=1) for base, cols in groups.items()}
    out = pd.DataFrame(data, index=gene_counts_df.index)
    return out.astype(gene_counts_df.values.dtype)


# ── consuming DESeq2 / edgeR adjusted p-values ──────────────────────────────

def countbased_de_flags(qvals, alpha: float = 0.05) -> Dict[str, bool]:
    """Map per-gene adjusted p-values to {gene: is_DE}.

    A NaN adjusted p-value (DESeq2 independent filtering / all-zero gene) is
    treated as *not* DE — the conservative choice, since an undetectable gene
    should not be counted as differentially expressed.
    """
    q = pd.Series(qvals, dtype="float64")
    flags: Dict[str, bool] = {}
    for gene, val in q.items():
        flags[gene] = bool(val < alpha) if pd.notna(val) else False
    return flags


# ── silent fraction from DE + switch flags ──────────────────────────────────

def silent_fraction(
    gene_de_map: Dict[str, bool],
    switch_sig_map: Dict[str, bool],
) -> Dict[str, float]:
    """Re-classify visibility and return silent-fraction summary counts.

    Iterates over the switching-test gene set (`switch_sig_map`); a gene absent
    from `gene_de_map` is treated as not-DE.  Uses the same classify_visibility
    definition as the main pipeline, so only the DE input differs.
    """
    n_silent = n_visible = 0
    for gene, sig in switch_sig_map.items():
        de = bool(gene_de_map.get(gene, False))
        vis = classify_visibility(de, bool(sig))
        if vis == "silent":
            n_silent += 1
        elif vis == "visible":
            n_visible += 1
    n_switch_sig = n_silent + n_visible
    fraction = (n_silent / n_switch_sig) if n_switch_sig > 0 else float("nan")
    return {
        "n_silent": n_silent,
        "n_visible": n_visible,
        "n_switch_sig": n_switch_sig,
        "fraction": fraction,
    }
