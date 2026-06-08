"""run_replication.py — Phase 4: short-read replication of the isoform-switching landscape.

Loads the pre-subsetted short-read matrix (data/processed/shortread_lncrna_subset.tsv.gz),
re-runs the SAME analysis pipeline as Phase 2, and compares results to the
long-read Phase 2 output.

Run from project root:
    python code/run_replication.py

Outputs (data/processed/):
    visibility_shortread.tsv  — per-gene visibility for short-read
    phase4_summary.txt        — replication numbers + honest verdict
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# -- resolve imports when run as `python code/run_replication.py`
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_io import load_lncrna_tx2gene
from isoform_metrics import classify_visibility
from run_landscape import (
    compute_landscape,
    gene_de_flags,
    switching_pvalues,
    MIN_SAMPLE_GENE_TPM,
    MIN_EXPR_SAMPLES_PER_TISSUE,
    B_PERMUTATIONS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
GTF_PATH       = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
PROC_DIR       = _ROOT / "data/processed"
SR_MATRIX      = PROC_DIR / "shortread_lncrna_subset.tsv.gz"
SR_S2T         = PROC_DIR / "shortread_sample2tissue.tsv"
LR_VISIBILITY  = PROC_DIR / "visibility_longread.tsv"
LR_ISI         = PROC_DIR / "lnc_isi_longread.tsv"
OUT_VISIBILITY = PROC_DIR / "visibility_shortread.tsv"
OUT_SUMMARY    = PROC_DIR / "phase4_summary.txt"


def main():
    t0 = time.time()

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("[1/5] Loading short-read subset matrix ...")
    sr_df = pd.read_csv(SR_MATRIX, sep='\t', index_col=0, compression='gzip')
    sr_df = sr_df.astype('float32')
    sr_df.index.name = 'transcript_id'
    print(f"  Matrix: {sr_df.shape[0]:,} transcripts × {sr_df.shape[1]:,} samples")

    print("  Loading sample → tissue mapping ...")
    s2t_df = pd.read_csv(SR_S2T, sep='\t')
    sample2tissue = dict(zip(s2t_df['SAMPID'], s2t_df['SMTSD']))
    # Keep only samples actually in the matrix columns
    sample2tissue = {s: t for s, t in sample2tissue.items() if s in sr_df.columns}
    print(f"  {len(sample2tissue)} samples mapped to tissues")

    print("  Loading lncRNA tx2gene from GTF ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))

    # ── Phase 2–equivalent analysis on short-read data ───────────────────────
    print("[2/5] Running landscape analysis (compute_landscape) ...")
    (isi_df, analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
     gene_if_mat, gene_tpm_sum, _, tissues_used, tissue_all_sample_idx) = compute_landscape(
        sr_df, tx2gene, sample2tissue,
        min_sample_gene_tpm=MIN_SAMPLE_GENE_TPM,
        min_expr_samples_per_tissue=MIN_EXPR_SAMPLES_PER_TISSUE,
    )
    print(f"  {len(analyzed_genes):,} genes analyzed (short-read)")

    print("[3/5] Gene-level DE (Kruskal-Wallis) ...")
    gene_de_map = gene_de_flags(
        analyzed_genes, gene_valid_tissue_map, gene_expressed_samples, gene_tpm_sum,
        tissue_all_sample_idx,
    )
    print(f"  {sum(gene_de_map.values()):,} genes DE (q<0.05)")

    print(f"[4/5] Permutation test (B={B_PERMUTATIONS}) ...")
    obs_isi_map = dict(zip(analyzed_genes, isi_df['lnc_isi'].values))
    switch_qval_map = switching_pvalues(
        analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
        gene_if_mat, obs_isi_map, B=B_PERMUTATIONS, seed=42,
    )
    switch_sig_map = {g: bool(q < 0.05) for g, q in switch_qval_map.items()}
    print(f"  {sum(switch_sig_map.values()):,} genes switch-significant (q<0.05)")

    # Visibility classification
    gene_n_expr_map = {}
    for g in analyzed_genes:
        tissue_expr = gene_expressed_samples[g]
        valid_tissues = gene_valid_tissue_map[g]
        gene_n_expr_map[g] = sum(len(tissue_expr[t]) for t in valid_tissues)

    vis_records = []
    for g in analyzed_genes:
        de    = gene_de_map[g]
        sig   = switch_sig_map[g]
        vis   = classify_visibility(de, sig)
        n_expr = gene_n_expr_map[g]
        vis_records.append({
            'gene_id':    g,
            'lnc_isi':    obs_isi_map[g],
            'gene_de':    de,
            'switch_sig': sig,
            'visibility': vis,
            'n_expr':     n_expr,
            'small_perm': n_expr < 10,
        })

    vis_sr_df = pd.DataFrame(vis_records)
    vis_sr_df.to_csv(OUT_VISIBILITY, sep='\t', index=False)
    print(f"  Saved: {OUT_VISIBILITY.relative_to(_ROOT)}")

    # ── Compare long-read vs short-read ──────────────────────────────────────
    print("[5/5] Comparing long-read vs short-read results ...")

    lr_df = pd.read_csv(LR_VISIBILITY, sep='\t')
    lr_isi_df = pd.read_csv(LR_ISI, sep='\t')

    # Genes analyzed in BOTH
    lr_genes = set(lr_df['gene_id'])
    sr_genes = set(vis_sr_df['gene_id'])
    both_genes = lr_genes & sr_genes
    n_both = len(both_genes)
    print(f"  Genes in both analyses: {n_both:,}")

    # Build aligned dataframes for comparison
    lr_both = lr_df[lr_df['gene_id'].isin(both_genes)].set_index('gene_id')
    sr_both = vis_sr_df[vis_sr_df['gene_id'].isin(both_genes)].set_index('gene_id')

    # Spearman correlation of lnc_isi
    genes_list = sorted(both_genes)
    lr_isi_vals = lr_both.loc[genes_list, 'lnc_isi'].values
    sr_isi_vals = sr_both.loc[genes_list, 'lnc_isi'].values
    spearman_r, spearman_p = stats.spearmanr(lr_isi_vals, sr_isi_vals)

    # Silent set replication
    lr_silent_genes = set(lr_df[lr_df['visibility'] == 'silent']['gene_id'])
    lr_silent_in_both = lr_silent_genes & both_genes
    n_lr_silent_in_both = len(lr_silent_in_both)

    # Of long-read silent genes (that are in both), what are they in short-read?
    sr_sig_set  = set(vis_sr_df[vis_sr_df['switch_sig'] == True]['gene_id'])
    sr_silent_set = set(vis_sr_df[vis_sr_df['visibility'] == 'silent']['gene_id'])

    n_lr_silent_sr_switch_sig = len(lr_silent_in_both & sr_sig_set)
    n_lr_silent_sr_silent     = len(lr_silent_in_both & sr_silent_set)
    frac_lr_silent_sr_switch_sig = (n_lr_silent_sr_switch_sig / n_lr_silent_in_both
                                    if n_lr_silent_in_both > 0 else float('nan'))
    frac_lr_silent_sr_silent     = (n_lr_silent_sr_silent / n_lr_silent_in_both
                                    if n_lr_silent_in_both > 0 else float('nan'))

    # Short-read headline: silent fraction among switch-sig
    sr_n_silent  = int((vis_sr_df['visibility'] == 'silent').sum())
    sr_n_visible = int((vis_sr_df['visibility'] == 'visible').sum())
    sr_n_switch_sig = sr_n_silent + sr_n_visible
    sr_silent_frac = (sr_n_silent / sr_n_switch_sig if sr_n_switch_sig > 0 else float('nan'))

    # Long-read headline for reference
    lr_n_silent  = int((lr_df['visibility'] == 'silent').sum())
    lr_n_visible = int((lr_df['visibility'] == 'visible').sum())
    lr_n_switch_sig = lr_n_silent + lr_n_visible
    lr_silent_frac = lr_n_silent / lr_n_switch_sig if lr_n_switch_sig > 0 else float('nan')

    # Jaccard of silent sets (over genes analyzed in both)
    lr_silent_in_both_set = lr_silent_genes & both_genes
    sr_silent_in_both_set = sr_silent_set & both_genes
    jaccard_intersection = len(lr_silent_in_both_set & sr_silent_in_both_set)
    jaccard_union = len(lr_silent_in_both_set | sr_silent_in_both_set)
    jaccard = jaccard_intersection / jaccard_union if jaccard_union > 0 else float('nan')

    # Short-read full counts
    sr_n_gene_only = int((vis_sr_df['visibility'] == 'gene_only').sum())
    sr_n_none      = int((vis_sr_df['visibility'] == 'none').sum())
    sr_n_analyzed  = len(vis_sr_df)

    total_time = time.time() - t0

    # ── Honest verdict ───────────────────────────────────────────────────────
    # Criteria:
    # - "replicates well":     spearman_r >= 0.5 AND frac_lr_silent_sr_switch_sig >= 0.4
    # - "partially replicates": spearman_r >= 0.3 AND frac_lr_silent_sr_switch_sig >= 0.2
    # - "poorly replicates":   otherwise
    if spearman_r >= 0.5 and frac_lr_silent_sr_switch_sig >= 0.4:
        verdict = "replicates well"
    elif spearman_r >= 0.3 and frac_lr_silent_sr_switch_sig >= 0.2:
        verdict = "partially replicates"
    else:
        verdict = "poorly replicates"

    summary_lines = [
        "=" * 70,
        "Phase 4 Summary — Short-Read Replication of lncRNA Isoform-Switching",
        "=" * 70,
        "",
        "SHORT-READ ANALYSIS (same thresholds as Phase 2):",
        f"  Samples selected:         {len(sample2tissue)} ({len(TARGET_TISSUES)} tissues, cap {CAP_PER_TISSUE}/tissue)",
        f"  Genes analyzed:           {sr_n_analyzed:,}",
        f"  Silent (switch-sig, !DE): {sr_n_silent:,}",
        f"  Visible (switch-sig, DE): {sr_n_visible:,}",
        f"  Gene-only (DE, !switch):  {sr_n_gene_only:,}",
        f"  None (neither):           {sr_n_none:,}",
        "",
        "SHORT-READ HEADLINE:",
        f"  Switch-significant genes: {sr_n_switch_sig:,}",
        f"  Silent fraction among switch-sig",
        f"    (silent/(silent+visible)):  {sr_n_silent}/{sr_n_switch_sig} = {sr_silent_frac:.3f}",
        "",
        "LONG-READ REFERENCE (Phase 2):",
        f"  Genes analyzed:           {len(lr_df):,}",
        f"  Silent fraction among switch-sig:",
        f"    {lr_n_silent}/{lr_n_switch_sig} = {lr_silent_frac:.3f}",
        "",
        "COMPARISON (genes in both: n = {n_both:,}):".format(n_both=n_both),
        f"  Spearman r (lnc-ISI, long vs short):  {spearman_r:.3f}  (p = {spearman_p:.2e})",
        "",
        "SILENT SET REPLICATION:",
        f"  Long-read silent genes analyzable in short-read: {n_lr_silent_in_both:,}",
        f"  Of those, switch-significant in short-read:",
        f"    {n_lr_silent_sr_switch_sig}/{n_lr_silent_in_both} = {frac_lr_silent_sr_switch_sig:.3f}",
        f"  Of those, SILENT in short-read (switch-sig AND !DE):",
        f"    {n_lr_silent_sr_silent}/{n_lr_silent_in_both} = {frac_lr_silent_sr_silent:.3f}",
        "",
        "JACCARD (silent sets, over genes in both):",
        f"  |intersection| = {jaccard_intersection}, |union| = {jaccard_union}",
        f"  Jaccard index = {jaccard:.3f}",
        "",
        "PARAMETERS:",
        f"  MIN_SAMPLE_GENE_TPM:         {MIN_SAMPLE_GENE_TPM}",
        f"  MIN_EXPR_SAMPLES_PER_TISSUE: {MIN_EXPR_SAMPLES_PER_TISSUE}",
        f"  Permutations (B):            {B_PERMUTATIONS}",
        f"  Total runtime:               {total_time:.1f}s",
        "",
        "VERDICT:",
        f"  {verdict}",
        "  (criteria: well = Spearman>=0.5 AND lr_silent_sw_sig_frac>=0.4;",
        "             partial = Spearman>=0.3 AND lr_silent_sw_sig_frac>=0.2;",
        "             poor = otherwise)",
        "",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    with open(OUT_SUMMARY, 'w') as fh:
        fh.write(summary_text + "\n")

    print(f"\nSaved: {OUT_SUMMARY.relative_to(_ROOT)}")
    print("Phase 4 complete.")


# make TARGET_TISSUES available at module scope for the summary
TARGET_TISSUES = [
    "Brain - Cerebellar Hemisphere",
    "Brain - Frontal Cortex (BA9)",
    "Brain - Putamen (basal ganglia)",
    "Cells - Cultured fibroblasts",
    "Heart - Atrial Appendage",
    "Heart - Left Ventricle",
    "Liver",
    "Lung",
    "Muscle - Skeletal",
]
CAP_PER_TISSUE = 50


if __name__ == "__main__":
    main()
