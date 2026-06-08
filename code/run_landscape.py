"""run_landscape.py — Phase 2 lncRNA isoform-switching landscape analysis.

Run from project root:
    python code/run_landscape.py [--B <int>]

Options:
    --B <int>   Number of permutations for the switching test (default 10000).

Outputs (data/processed/):
    lnc_isi_longread.tsv   — per-gene lnc-ISI table
    visibility_longread.tsv — per-gene visibility classification
    phase2_summary.txt      — headline numbers for the paper
"""
import argparse
import math
import sys
import time
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# -- resolve imports whether run as `python code/run_landscape.py` or via pytest
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from isoform_metrics import isoform_fraction, lnc_isi, classify_visibility
from data_io import load_lncrna_tx2gene, load_longread_tpm, sample_to_tissue

# ── Paths ──────────────────────────────────────────────────────────────────────
GTF_PATH   = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
TPM_PATH   = _ROOT / "data/raw/longread/quantification_gencode.tpm.txt.gz"
ATTR_PATH  = _ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"
PROC_DIR   = _ROOT / "data/processed"
PROC_DIR.mkdir(exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_SAMPLE_GENE_TPM          = 1.0   # TPM floor for a gene to be "expressed" in a sample
MIN_EXPR_SAMPLES_PER_TISSUE  = 3     # min expressed samples for a tissue to be "valid"
B_PERMUTATIONS               = 10000  # permutation count (10k: finer FDR resolution per reviewer request)


# ── Reusable analysis functions ────────────────────────────────────────────────

def compute_landscape(
    tx_tpm_df: pd.DataFrame,
    tx2gene: Dict[str, str],
    sample2tissue: Dict[str, str],
    min_sample_gene_tpm: float = 1.0,
    min_expr_samples_per_tissue: int = 3,
):
    """Compute per-gene tissue-level IF and lnc-ISI.

    Parameters
    ----------
    tx_tpm_df : DataFrame
        Index = transcript_id, columns = sample_id, values = TPM.
    tx2gene : dict
        {transcript_id: gene_id}
    sample2tissue : dict
        {sample_id: tissue_label}
    min_sample_gene_tpm : float
        Per-sample gene TPM floor for the sample to be "expressed".
    min_expr_samples_per_tissue : int
        Minimum expressed samples a tissue needs to be "valid".

    Returns
    -------
    Tuple of:
        isi_df          : DataFrame (gene_id, n_isoforms, n_valid_tissues, lnc_isi)
        analyzed_genes  : list of gene_id strings
        gene_valid_tissue_map : {gene: list[tissue]}
        gene_expressed_samples : {gene: {tissue: np.ndarray of sample indices}}
        gene_if_mat    : {gene: np.ndarray (n_iso, n_samp)}
        gene_tpm_sum   : {gene: np.ndarray (n_samp,)}
        kept_samples   : list of sample_ids (in column order of tx_tpm_df)
        tissues_used   : sorted list of tissue labels
        tissue_all_sample_idx : {tissue: np.ndarray of ALL sample indices in that tissue}
    """
    kept_samples = list(sample2tissue.keys())
    tissues_used = sorted(set(sample2tissue.values()))

    # Restrict TPM to kept samples
    tpm_df = tx_tpm_df[kept_samples]

    # Restrict to lncRNA transcripts present in the TPM file
    lncrna_tx_in_data = [tx for tx in tpm_df.index if tx in tx2gene]
    tpm_df = tpm_df.loc[lncrna_tx_in_data]

    # Build gene -> list of transcript IDs
    gene_to_txs: Dict[str, List[str]] = defaultdict(list)
    for tx in lncrna_tx_in_data:
        gene_to_txs[tx2gene[tx]].append(tx)

    # Keep genes with >= 2 transcripts
    multi_isoform_genes = {g: txs for g, txs in gene_to_txs.items() if len(txs) >= 2}

    sample_arr = np.array(kept_samples)
    tissue_arr = np.array([sample2tissue[s] for s in kept_samples])

    # Per-sample gene TPM and IF matrices
    gene_tpm_mat = {}
    for g, txs in multi_isoform_genes.items():
        gene_tpm_mat[g] = tpm_df.loc[txs].values.astype(np.float32)

    gene_tpm_sum = {}
    for g, mat in gene_tpm_mat.items():
        gene_tpm_sum[g] = mat.sum(axis=0)

    gene_if_mat = {}
    for g, mat in gene_tpm_mat.items():
        col_totals = mat.sum(axis=0, keepdims=True)
        denom = np.where(col_totals > 0, col_totals, 1.0)
        gene_if_mat[g] = mat / denom

    # Build a map of ALL sample indices per tissue (used by the DE test to include zeros)
    tissue_all_sample_idx: Dict[str, np.ndarray] = {}
    for tissue in tissues_used:
        t_mask = (tissue_arr == tissue)
        tissue_all_sample_idx[tissue] = np.where(t_mask)[0]

    # Tissue-level IF and lnc-ISI
    records = []
    analyzed_genes = []
    gene_valid_tissue_map = {}
    gene_expressed_samples = {}

    for g in multi_isoform_genes:
        mat_if  = gene_if_mat[g]
        mat_sum = gene_tpm_sum[g]
        n_iso   = mat_if.shape[0]

        tissue_if = {}
        tissue_expr_idx = {}

        for tissue in tissues_used:
            t_mask = (tissue_arr == tissue)
            t_idx  = np.where(t_mask)[0]
            expr_mask = mat_sum[t_idx] >= min_sample_gene_tpm
            expr_idx  = t_idx[expr_mask]

            if expr_mask.sum() < min_expr_samples_per_tissue:
                continue

            mean_if = mat_if[:, expr_idx].mean(axis=1)
            s = mean_if.sum()
            if s > 0:
                mean_if = mean_if / s

            tissue_if[tissue]       = mean_if
            tissue_expr_idx[tissue] = expr_idx

        if len(tissue_if) < 2:
            continue

        isi = lnc_isi(tissue_if)
        n_valid = len(tissue_if)
        records.append({
            'gene_id':         g,
            'n_isoforms':      n_iso,
            'n_valid_tissues': n_valid,
            'lnc_isi':         isi,
        })
        analyzed_genes.append(g)
        gene_valid_tissue_map[g]   = list(tissue_if.keys())
        gene_expressed_samples[g]  = tissue_expr_idx

    isi_df = pd.DataFrame(records)
    return (
        isi_df,
        analyzed_genes,
        gene_valid_tissue_map,
        gene_expressed_samples,
        gene_if_mat,
        gene_tpm_sum,
        kept_samples,
        tissues_used,
        tissue_all_sample_idx,
    )


def gene_de_flags(
    analyzed_genes: List[str],
    gene_valid_tissue_map: Dict[str, List[str]],
    gene_expressed_samples: Dict[str, Dict],
    gene_tpm_sum: Dict[str, np.ndarray],
    tissue_all_sample_idx: Dict[str, np.ndarray],
) -> Dict[str, bool]:
    """Run Kruskal-Wallis + BH across tissues, return {gene: is_DE (q<0.05)}.

    For each gene, the KW test uses ALL samples assigned to the gene's valid
    tissues (including samples where gene TPM is below the expression threshold,
    i.e. biological zeros are retained).  This raises DE power and reduces the
    fraction of switching genes that are classified as 'silent'.

    Note: 'valid tissues' are still determined by the >= MIN_EXPR_SAMPLES_PER_TISSUE
    expressed-sample criterion (unchanged).  Only the KW input vectors change.

    Includes nan-guard: identical groups → p=1.0 (avoids propagating nan into
    multipletests).
    """
    kw_pvals = []
    for g in analyzed_genes:
        gene_sum      = gene_tpm_sum[g]
        valid_tissues = gene_valid_tissue_map[g]
        # Use ALL samples of each valid tissue (zeros included)
        groups = [gene_sum[tissue_all_sample_idx[t]] for t in valid_tissues]

        if len(groups) >= 2 and all(len(g_) >= 1 for g_ in groups):
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

    _, kw_qvals, _, _ = multipletests(kw_pvals, alpha=0.05, method='fdr_bh')
    return {g: bool(q < 0.05) for g, q in zip(analyzed_genes, kw_qvals)}


def _jsd_batch(p_batch: np.ndarray, q_batch: np.ndarray) -> np.ndarray:
    """Vectorized JSD (log2 base, 0..1) for B pairs of probability vectors.

    Matches the precision of the scalar jsd() function in isoform_metrics.py,
    which casts inputs to float64 (numpy dtype=float default).

    Parameters
    ----------
    p_batch : np.ndarray, shape (B, n_iso)
    q_batch : np.ndarray, shape (B, n_iso)

    Returns
    -------
    np.ndarray, shape (B,) — JSD values in [0, 1].
    """
    # Cast to float64 to match isoform_metrics.jsd which does np.asarray(p, dtype=float)
    p_batch = np.asarray(p_batch, dtype=np.float64)
    q_batch = np.asarray(q_batch, dtype=np.float64)
    m = 0.5 * (p_batch + q_batch)
    # KL(p || m) for each row, handling zeros with safe log
    with np.errstate(divide='ignore', invalid='ignore'):
        log2_p_m = np.where(p_batch > 0, np.log2(np.where(p_batch > 0, p_batch, 1.0) /
                                                   np.where(m > 0, m, 1.0)), 0.0)
        log2_q_m = np.where(q_batch > 0, np.log2(np.where(q_batch > 0, q_batch, 1.0) /
                                                   np.where(m > 0, m, 1.0)), 0.0)
    kl_p = (p_batch * log2_p_m).sum(axis=1)
    kl_q = (q_batch * log2_q_m).sum(axis=1)
    return 0.5 * kl_p + 0.5 * kl_q


def switching_pvalues(
    analyzed_genes: List[str],
    gene_valid_tissue_map: Dict[str, List[str]],
    gene_expressed_samples: Dict[str, Dict],
    gene_if_mat: Dict[str, np.ndarray],
    obs_isi_map: Dict[str, float],
    B: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Label-permutation test for isoform switching + BH, return {gene: q-value}.

    Vectorized implementation: generates all B permutations at once per gene,
    computes group-mean IF vectors via array indexing, and evaluates all pairwise
    JSD values with batched numpy operations — no Python-level per-permutation loop.

    Same null distribution: permute tissue labels among expressed samples,
    preserving per-tissue group sizes.
    Same p-value formula: (1 + #{perm >= obs}) / (B + 1).
    Same seed=42 and BH correction as before.
    """
    rng = np.random.default_rng(seed=seed)
    switch_pvals = []

    for g in analyzed_genes:
        mat_if        = gene_if_mat[g]          # (n_iso, n_samples_total)
        observed      = obs_isi_map[g]
        valid_tissues = gene_valid_tissue_map[g]
        n_iso         = mat_if.shape[0]

        # Gather expressed-sample indices in tissue order
        tissue_expr = gene_expressed_samples[g]
        expr_samp_indices: List[int] = []
        tissue_group_sizes: List[int] = []
        for tissue in valid_tissues:
            idxs = tissue_expr[tissue]
            expr_samp_indices.extend(idxs.tolist())
            tissue_group_sizes.append(len(idxs))

        expr_samp_indices_arr = np.array(expr_samp_indices, dtype=np.intp)
        n_expr = len(expr_samp_indices_arr)

        # if_expr: (n_iso, n_expr) — keep float32 to match original arithmetic
        if_expr = mat_if[:, expr_samp_indices_arr]  # float32 (same dtype as gene_if_mat)

        # Generate all B permutations using the same rng.permutation calls as the
        # original implementation, so the RNG stream — and therefore p-values — are
        # identical for the same seed.  Stacking into a matrix lets all downstream
        # computation stay vectorized.
        perm_indices = np.empty((B, n_expr), dtype=np.intp)
        for b in range(B):
            perm_indices[b] = rng.permutation(n_expr)

        # Gather permuted IF columns for all B permutations.
        # if_expr[:, perm_indices] broadcasts to (n_iso, B, n_expr).
        # Transpose to (B, n_iso, n_expr) for the slicing below.
        perm_if = if_expr[:, perm_indices].transpose(1, 0, 2)  # (B, n_iso, n_expr)

        # Compute per-tissue mean IF for each permutation: (B, n_tissues, n_iso)
        n_tissues = len(valid_tissues)
        # Pre-allocate with the same dtype as if_expr (float32) to match original arithmetic
        perm_means = np.empty((B, n_tissues, n_iso), dtype=if_expr.dtype)
        start = 0
        for t_idx, size in enumerate(tissue_group_sizes):
            chunk = perm_if[:, :, start:start + size]  # (B, n_iso, size)
            mean_chunk = chunk.mean(axis=2)             # (B, n_iso)
            # Normalize so each row sums to 1 (or stays 0 if all-zero)
            row_sums = mean_chunk.sum(axis=1, keepdims=True)  # (B, 1)
            # Avoid divide-by-zero: where sum=0 keep as 0
            safe_sums = np.where(row_sums > 0, row_sums, 1.0)
            perm_means[:, t_idx, :] = mean_chunk / safe_sums  # (B, n_iso)
            start += size

        # Compute max pairwise JSD across tissue pairs for each of the B permutations
        # tissue_pairs: all C(n_tissues, 2) combinations
        # _jsd_batch returns float64; keep accumulator in float64 to match lnc_isi precision
        # (isoform_metrics.jsd casts inputs to dtype=float i.e. float64).
        tissue_pair_indices = list(combinations(range(n_tissues), 2))
        perm_max_jsd = np.zeros(B, dtype=np.float64)
        for ti, tj in tissue_pair_indices:
            p_batch = perm_means[:, ti, :]   # (B, n_iso)
            q_batch = perm_means[:, tj, :]   # (B, n_iso)
            jsd_vals = _jsd_batch(p_batch, q_batch)  # (B,)
            np.maximum(perm_max_jsd, jsd_vals, out=perm_max_jsd)

        count_ge = int(np.sum(perm_max_jsd >= observed))
        switch_pvals.append((1 + count_ge) / (B + 1))

    _, switch_qvals, _, _ = multipletests(switch_pvals, alpha=0.05, method='fdr_bh')
    return {g: float(q) for g, q in zip(analyzed_genes, switch_qvals)}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 lncRNA isoform-switching landscape analysis."
    )
    parser.add_argument(
        "--B",
        type=int,
        default=B_PERMUTATIONS,
        help=f"Number of permutations for the switching test (default {B_PERMUTATIONS}).",
    )
    args = parser.parse_args()
    b_perms = args.B

    t0 = time.time()

    # ── Step 1: load & restrict ──────────────────────────────────────────────
    print("[1/6] Loading data ...")

    print("  Loading lncRNA tx2gene from GTF ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))
    print(f"  {len(tx2gene):,} lncRNA transcripts / {len(set(tx2gene.values())):,} genes in GTF")

    print("  Loading long-read TPM table ...")
    tpm_df = load_longread_tpm(str(TPM_PATH))
    print(f"  TPM table: {tpm_df.shape[0]:,} transcripts × {tpm_df.shape[1]:,} samples")

    print("  Mapping samples → tissues ...")
    s2t = sample_to_tissue(tpm_df.columns.tolist(), str(ATTR_PATH), min_samples_per_tissue=6)
    kept_samples = list(s2t.keys())
    tissues_used = sorted(set(s2t.values()))
    print(f"  {len(kept_samples)} samples kept across {len(tissues_used)} tissues:")
    tc = Counter(s2t.values())
    for t in sorted(tc):
        print(f"    {tc[t]:2d}  {t}")

    # Restrict TPM to kept samples
    tpm_df = tpm_df[kept_samples]

    # ── Steps 2-3: compute_landscape ────────────────────────────────────────
    print("[2/6] Computing per-sample gene TPM and isoform fractions ...")
    print("[3/6] Computing tissue-level IF and lnc-ISI ...")

    (isi_df, analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
     gene_if_mat, gene_tpm_sum, _, _, tissue_all_sample_idx) = compute_landscape(
        tpm_df, tx2gene, s2t,
        min_sample_gene_tpm=MIN_SAMPLE_GENE_TPM,
        min_expr_samples_per_tissue=MIN_EXPR_SAMPLES_PER_TISSUE,
    )

    n_genes  = len(analyzed_genes)
    n_samp   = len(kept_samples)
    print(f"  Done: {n_genes:,} genes × {n_samp} samples computed.")
    print(f"  {n_genes:,} genes analyzed (>= 2 isoforms, >= 2 valid tissues).")

    isi_df.to_csv(PROC_DIR / "lnc_isi_longread.tsv", sep='\t', index=False)
    print(f"  Saved: data/processed/lnc_isi_longread.tsv")

    # ── Step 4: gene-level DE (Kruskal-Wallis across tissues) ────────────────
    print("[4/6] Gene-level differential expression (Kruskal-Wallis) ...")

    gene_de_map = gene_de_flags(
        analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
        gene_tpm_sum, tissue_all_sample_idx,
    )
    print(f"  {sum(gene_de_map.values()):,} / {len(analyzed_genes):,} genes DE (q < 0.05).")

    # ── Step 5: switching significance (permutation test) ────────────────────
    print(f"[5/6] Permutation test for isoform switching (B={b_perms}) ...")
    print("  (Progress reported every 500 genes)")

    t5 = time.time()
    obs_isi_map = dict(zip(analyzed_genes, isi_df['lnc_isi'].values))

    switch_qval_map = switching_pvalues(
        analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
        gene_if_mat, obs_isi_map, B=b_perms, seed=42,
    )
    switch_sig_map = {g: bool(q < 0.05) for g, q in switch_qval_map.items()}

    t5_done = time.time() - t5
    print(f"  Permutation test completed in {t5_done:.1f}s")
    print(f"  {sum(switch_sig_map.values()):,} / {len(analyzed_genes):,} genes switch-significant (q < 0.05).")

    # Count n_expr per gene for sensitivity analysis
    gene_n_expr_map = {}
    for g in analyzed_genes:
        tissue_expr = gene_expressed_samples[g]
        valid_tissues = gene_valid_tissue_map[g]
        gene_n_expr_map[g] = sum(len(tissue_expr[t]) for t in valid_tissues)

    # ── Step 6: visibility classification & summary ───────────────────────────
    print("[6/6] Classifying visibility and writing outputs ...")

    vis_records = []
    for g in analyzed_genes:
        de     = gene_de_map[g]
        sig    = switch_sig_map[g]
        vis    = classify_visibility(de, sig)
        n_expr = gene_n_expr_map[g]
        vis_records.append({
            'gene_id':     g,
            'lnc_isi':     obs_isi_map[g],
            'gene_de':     de,
            'switch_sig':  sig,
            'visibility':  vis,
            'n_expr':      n_expr,
            'small_perm':  n_expr < 10,
        })

    vis_df = pd.DataFrame(vis_records)
    vis_df.to_csv(PROC_DIR / "visibility_longread.tsv", sep='\t', index=False)
    print("  Saved: data/processed/visibility_longread.tsv")

    # Counts
    n_analyzed = len(analyzed_genes)
    n_silent   = int((vis_df['visibility'] == 'silent').sum())
    n_visible  = int((vis_df['visibility'] == 'visible').sum())
    n_gene_only = int((vis_df['visibility'] == 'gene_only').sum())
    n_none     = int((vis_df['visibility'] == 'none').sum())

    # Key headline fractions
    n_switch_sig = n_silent + n_visible
    silent_frac_among_switch = n_silent / n_switch_sig if n_switch_sig > 0 else float('nan')

    n_non_de = n_silent + n_none
    switch_frac_among_non_de = n_silent / n_non_de if n_non_de > 0 else float('nan')

    # Sensitivity: small_perm genes
    n_small_perm = int(vis_df['small_perm'].sum())
    vis_df_large = vis_df[~vis_df['small_perm']]
    n_silent_large   = int((vis_df_large['visibility'] == 'silent').sum())
    n_visible_large  = int((vis_df_large['visibility'] == 'visible').sum())
    n_switch_sig_large = n_silent_large + n_visible_large
    silent_frac_large = (n_silent_large / n_switch_sig_large
                         if n_switch_sig_large > 0 else float('nan'))

    # ISI distribution
    isi_vals = vis_df['lnc_isi'].values
    median_isi = float(np.median(isi_vals))
    p90_isi    = float(np.percentile(isi_vals, 90))
    n_isi_01   = int((isi_vals >= 0.1).sum())
    n_isi_025  = int((isi_vals >= 0.25).sum())

    total_time = time.time() - t0

    summary_lines = [
        "=" * 60,
        "Phase 2 Summary — lncRNA Isoform Switching Landscape",
        "=" * 60,
        "",
        f"Input data:",
        f"  Long-read samples used:      {len(kept_samples)}",
        f"  Tissues used:                {len(tissues_used)}",
        f"  Tissues: {', '.join(tissues_used)}",
        "",
        f"lncRNA genes analyzed:",
        f"  (>= 2 isoforms in data, >= 2 valid tissues)",
        f"  n = {n_analyzed:,}",
        "",
        f"Visibility classification counts:",
        f"  silent    (switch-sig, not DE):  {n_silent:,}",
        f"  visible   (switch-sig AND DE):   {n_visible:,}",
        f"  gene_only (DE only):             {n_gene_only:,}",
        f"  none      (neither):             {n_none:,}",
        "",
        f"KEY HEADLINE NUMBERS:",
        f"  Switch-significant genes:    {n_switch_sig:,}",
        f"  Silent fraction among switch-sig genes",
        f"    (silent / (silent+visible)):  "
        f"{n_silent}/{n_switch_sig} = {silent_frac_among_switch:.3f}",
        f"  Switch-sig fraction among non-DE genes",
        f"    (silent / (silent+none)):     "
        f"{n_silent}/{n_non_de} = {switch_frac_among_non_de:.4f}",
        "",
        f"SENSITIVITY (small_perm = n_expr < 10):",
        f"  Genes with small_perm=True:  {n_small_perm:,}",
        f"  Silent count (excl. small_perm): {n_silent_large:,}",
        f"  Silent fraction (excl. small_perm)",
        f"    (silent / (silent+visible)):  "
        f"{n_silent_large}/{n_switch_sig_large} = {silent_frac_large:.3f}",
        "",
        f"lnc-ISI distribution (all analyzed genes):",
        f"  Median ISI:                  {median_isi:.4f}",
        f"  90th percentile ISI:         {p90_isi:.4f}",
        f"  n(ISI >= 0.10):              {n_isi_01:,}",
        f"  n(ISI >= 0.25):              {n_isi_025:,}",
        "",
        f"Analysis parameters:",
        f"  MIN_SAMPLE_GENE_TPM:         {MIN_SAMPLE_GENE_TPM}",
        f"  MIN_EXPR_SAMPLES_PER_TISSUE: {MIN_EXPR_SAMPLES_PER_TISSUE}",
        f"  Permutations (B):            {b_perms}",
        f"  Permutation test runtime:    {t5_done:.1f}s",
        f"  Total runtime:               {total_time:.1f}s",
        "",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    with open(PROC_DIR / "phase2_summary.txt", 'w') as fh:
        fh.write(summary_text + "\n")

    print("\nSaved: data/processed/phase2_summary.txt")
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
