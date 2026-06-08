"""sensitivity_de_all_tissues.py — Reviewer (gpt-5 R3 MAJOR#1) sensitivity check.

The primary analysis runs the gene-level Kruskal-Wallis DE test across each
gene's *valid tissues* (tissues with >= MIN_EXPR_SAMPLES_PER_TISSUE expressed
samples), retaining all samples (zeros included) within those tissues. A reviewer
asked whether restricting the DE test to valid tissues — rather than running it
across ALL 9 tissues — inflates the silent fraction (a gene absent from a whole
tissue would carry a strong gene-level signal that the valid-tissue test cannot
see). This script re-runs the gene-level DE test across ALL 9 tissues (every
sample, zeros retained) and recomputes the silent fraction among the *same*
switch-significant genes, then prints the comparison.

It also reports the small-permutation-pool check (Gemini R3 2.4): how many
switch-significant / silent genes had < 10 expressed samples.

Run from project root:
    python code/sensitivity_de_all_tissues.py
"""

import sys
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_landscape import (  # noqa: E402
    compute_landscape,
    load_lncrna_tx2gene,
    load_longread_tpm,
    sample_to_tissue,
    GTF_PATH,
    TPM_PATH,
    ATTR_PATH,
    MIN_SAMPLE_GENE_TPM,
    MIN_EXPR_SAMPLES_PER_TISSUE,
)

VIS_PATH = _ROOT / "data/processed/visibility_longread.tsv"


def de_flags_all_tissues(analyzed_genes, gene_tpm_sum, tissue_all_sample_idx, all_tissues):
    """KW + BH across ALL tissues (every sample, zeros retained)."""
    kw_pvals = []
    for g in analyzed_genes:
        gene_sum = gene_tpm_sum[g]
        groups = [gene_sum[tissue_all_sample_idx[t]] for t in all_tissues]
        if len(groups) >= 2 and all(len(x) >= 1 for x in groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _stat, p = stats.kruskal(*groups)
                except Exception:
                    p = 1.0
            if p is None or math.isnan(p):
                p = 1.0
        else:
            p = 1.0
        kw_pvals.append(p)
    _, qvals, _, _ = multipletests(kw_pvals, alpha=0.05, method="fdr_bh")
    return {g: bool(q < 0.05) for g, q in zip(analyzed_genes, qvals)}


def main():
    print("Loading inputs (GTF, long-read TPM, sample->tissue) ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))
    tpm_df = load_longread_tpm(str(TPM_PATH))
    s2t = sample_to_tissue(tpm_df.columns.tolist(), str(ATTR_PATH), min_samples_per_tissue=6)
    kept_samples = list(s2t.keys())
    tpm_df = tpm_df[kept_samples]

    print("Computing landscape (this reloads the long-read TPM matrix) ...")
    (isi_df, analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
     gene_if_mat, gene_tpm_sum, kept_samples, tissues_used,
     tissue_all_sample_idx) = compute_landscape(
        tpm_df, tx2gene, s2t,
        min_sample_gene_tpm=MIN_SAMPLE_GENE_TPM,
        min_expr_samples_per_tissue=MIN_EXPR_SAMPLES_PER_TISSUE,
    )
    print(f"  analyzed genes: {len(analyzed_genes):,}")
    print(f"  tissues used (all): {len(tissues_used)} -> {sorted(tissues_used)}")

    # Load the saved primary visibility (switch_sig + primary gene_de + small_perm)
    vis = pd.read_csv(VIS_PATH, sep="\t")
    vis_base = {row.gene_id.split('.')[0]: row for row in vis.itertuples(index=False)}
    # analyzed_genes carry version suffix; map via base id
    switch_sig = {}
    primary_de = {}
    small_perm = {}
    n_expr_map = {}
    for g in analyzed_genes:
        base = g.split('.')[0]
        r = vis_base.get(base)
        if r is not None:
            switch_sig[g] = bool(r.switch_sig)
            primary_de[g] = bool(r.gene_de)
            small_perm[g] = bool(r.small_perm)
            n_expr_map[g] = int(r.n_expr)

    # Alternative DE across all 9 tissues
    print("Recomputing gene-level DE across ALL 9 tissues (zeros retained) ...")
    de_all = de_flags_all_tissues(analyzed_genes, gene_tpm_sum,
                                  tissue_all_sample_idx, tissues_used)

    sig_genes = [g for g in analyzed_genes if switch_sig.get(g, False)]
    n_sig = len(sig_genes)

    # Primary silent fraction (recomputed from saved table, sanity)
    silent_primary = [g for g in sig_genes if not primary_de.get(g, False)]
    # All-9-tissue silent fraction
    silent_all = [g for g in sig_genes if not de_all.get(g, False)]

    print("\n" + "=" * 68)
    print("SENSITIVITY: gene-level DE across valid tissues vs ALL 9 tissues")
    print("=" * 68)
    print(f"Switch-significant genes (unchanged):       {n_sig}")
    print(f"  PRIMARY (DE on valid tissues):  silent = {len(silent_primary)}"
          f"  ({len(silent_primary)/n_sig*100:.1f}%)")
    print(f"  SENSITIVITY (DE on all 9 tissues): silent = {len(silent_all)}"
          f"  ({len(silent_all)/n_sig*100:.1f}%)")
    # how many primary-silent become visible under all-9
    flipped = [g for g in silent_primary if de_all.get(g, False)]
    print(f"  primary-silent genes that become DE (visible) under all-9: {len(flipped)}")
    print(f"  direction: all-9 silent fraction is "
          f"{'LOWER (more conservative)' if len(silent_all) < len(silent_primary) else 'higher/equal'}")

    print("\n" + "-" * 68)
    print("SMALL-PERMUTATION-POOL CHECK (Gemini R3 2.4)")
    print("-" * 68)
    n_smallperm_total = sum(1 for g in analyzed_genes if small_perm.get(g, False))
    smallperm_sig = [g for g in sig_genes if small_perm.get(g, False)]
    smallperm_silent = [g for g in silent_primary if small_perm.get(g, False)]
    print(f"genes flagged small_perm (n_expr<10): {n_smallperm_total}")
    print(f"  of which switch-significant:        {len(smallperm_sig)}")
    print(f"  of which silent (primary):          {len(smallperm_silent)}")
    if smallperm_sig:
        ne = [n_expr_map[g] for g in smallperm_sig]
        print(f"  n_expr of small_perm switch-sig genes: min={min(ne)} median={int(np.median(ne))} max={max(ne)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
