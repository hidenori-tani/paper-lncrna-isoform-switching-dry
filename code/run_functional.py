"""run_functional.py — Phase 3 Step 1: differential exonic regions for switching lncRNAs.

For each switch-significant lncRNA, identify per-tissue dominant isoforms and
compute the genomic intervals that differ between them (symmetric exon difference).

Run from project root:
    python code/run_functional.py

Outputs (data/processed/):
    switch_exons.tsv        — one row per (gene, tissueA, tissueB, domA, domB, n_diff_regions, diff_bp)
    switch_diff_exons.bed   — BED-like file of all differential exon regions
    phase3_step1_summary.txt — headline numbers
"""
import gzip
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -- resolve imports whether run as `python code/run_functional.py` or via pytest
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_io import load_lncrna_tx2gene, load_longread_tpm, sample_to_tissue
from isoform_metrics import isoform_fraction

# ── Paths ──────────────────────────────────────────────────────────────────────
GTF_PATH   = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
TPM_PATH   = _ROOT / "data/raw/longread/quantification_gencode.tpm.txt.gz"
ATTR_PATH  = _ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"
PROC_DIR   = _ROOT / "data/processed"
VIS_PATH   = PROC_DIR / "visibility_longread.tsv"
PROC_DIR.mkdir(exist_ok=True)

# ── Thresholds (must match Phase 2) ───────────────────────────────────────────
MIN_SAMPLE_GENE_TPM         = 1.0
MIN_EXPR_SAMPLES_PER_TISSUE = 3


# ── Interval helpers ──────────────────────────────────────────────────────────

Interval = Tuple[int, int]  # half-open [start, end)


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """Merge overlapping / adjacent half-open intervals. Returns sorted list."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged: List[Interval] = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(a: List[Interval], b: List[Interval]) -> List[Interval]:
    """Return the regions in A that are NOT covered by B (A minus B).

    Both inputs must be sorted non-overlapping half-open intervals (i.e. output
    of merge_intervals). Returns sorted non-overlapping intervals.
    """
    result: List[Interval] = []
    bi = 0
    for a_start, a_end in a:
        cur = a_start
        while bi < len(b) and b[bi][1] <= cur:
            bi += 1
        # Walk through b intervals that overlap [cur, a_end)
        j = bi
        while j < len(b) and b[j][0] < a_end:
            b_start, b_end = b[j]
            if b_start > cur:
                result.append((cur, min(b_start, a_end)))
            cur = max(cur, b_end)
            j += 1
        if cur < a_end:
            result.append((cur, a_end))
    return result


def symmetric_diff_intervals(
    a: List[Interval], b: List[Interval]
) -> List[Interval]:
    """Return (A - B) union (B - A) as sorted non-overlapping intervals.

    A and B must be sorted non-overlapping (i.e. already merged).
    """
    a_minus_b = subtract_intervals(a, b)
    b_minus_a = subtract_intervals(b, a)
    combined = a_minus_b + b_minus_a
    if not combined:
        return []
    return merge_intervals(combined)


def total_bp(intervals: List[Interval]) -> int:
    """Sum of lengths of half-open intervals."""
    return sum(end - start for start, end in intervals)


# ── GTF exon parser ───────────────────────────────────────────────────────────

def load_tx_exons(
    gtf_gz_path: str,
    tx_set: set,
) -> Dict[str, List[Tuple[str, List[Interval], str]]]:
    """Parse GENCODE GTF and build per-transcript exon lists.

    Parameters
    ----------
    gtf_gz_path : str
        Path to gzipped GTF.
    tx_set : set
        Set of transcript_ids to keep (others are skipped for speed).

    Returns
    -------
    dict { transcript_id: (chrom, merged_intervals, strand) }
        merged_intervals: sorted non-overlapping half-open [start-1, end) intervals
        (GTF coords are 1-based closed; we convert to 0-based half-open).
    """
    _tx_re = re.compile(r'transcript_id "([^"]+)"')
    tx_exons_raw: Dict[str, Tuple[str, List[Interval], str]] = {}

    open_fn = gzip.open if str(gtf_gz_path).endswith('.gz') else open
    with open_fn(gtf_gz_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'exon':
                continue
            attrs = fields[8]
            m_tx = _tx_re.search(attrs)
            if not m_tx:
                continue
            tx_id = m_tx.group(1)
            if tx_id not in tx_set:
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1   # convert 1-based closed to 0-based half-open
            end    = int(fields[4])        # GTF end is inclusive; half-open end = same value
            strand = fields[6]

            if tx_id not in tx_exons_raw:
                tx_exons_raw[tx_id] = (chrom, [], strand)
            tx_exons_raw[tx_id][1].append((start, end))

    # Merge exons per transcript
    result: Dict[str, Tuple[str, List[Interval], str]] = {}
    for tx_id, (chrom, ivs, strand) in tx_exons_raw.items():
        result[tx_id] = (chrom, merge_intervals(ivs), strand)

    return result


# ── Per-tissue dominant isoform logic ─────────────────────────────────────────

def compute_tissue_dominant_isoforms(
    txs: List[str],
    tpm_mat: np.ndarray,       # (n_iso, n_samp)
    gene_tpm_sum: np.ndarray,  # (n_samp,)
    sample_arr: np.ndarray,    # (n_samp,) tissue labels
    tissues_used: List[str],
) -> Dict[str, str]:
    """For each valid tissue, return the dominant isoform (transcript_id).

    A tissue is "valid" if >= MIN_EXPR_SAMPLES_PER_TISSUE samples have
    gene TPM >= MIN_SAMPLE_GENE_TPM. The dominant isoform in a valid tissue
    is the transcript with the highest mean isoform fraction.

    Returns
    -------
    dict { tissue: dominant_transcript_id }  (only valid tissues included)
    """
    tissue_dominant: Dict[str, str] = {}

    for tissue in tissues_used:
        t_mask = sample_arr == tissue
        t_idx  = np.where(t_mask)[0]
        expr_mask = gene_tpm_sum[t_idx] >= MIN_SAMPLE_GENE_TPM
        expr_idx  = t_idx[expr_mask]

        if expr_mask.sum() < MIN_EXPR_SAMPLES_PER_TISSUE:
            continue

        # Per-sample IF for expressed samples
        col_totals = tpm_mat[:, expr_idx].sum(axis=0, keepdims=True)
        denom = np.where(col_totals > 0, col_totals, 1.0)
        if_mat = tpm_mat[:, expr_idx] / denom  # (n_iso, n_expr)

        # Mean IF across expressed samples
        mean_if = if_mat.mean(axis=1)  # (n_iso,)

        dom_idx = int(np.argmax(mean_if))
        tissue_dominant[tissue] = txs[dom_idx]

    return tissue_dominant


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("[1/5] Loading data ...")

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
    print(f"  {len(kept_samples)} samples across {len(tissues_used)} tissues")

    tpm_df = tpm_df[kept_samples]

    # Restrict to lncRNA transcripts in TPM
    lncrna_tx_in_data = [tx for tx in tpm_df.index if tx in tx2gene]
    tpm_df = tpm_df.loc[lncrna_tx_in_data]
    print(f"  {len(lncrna_tx_in_data):,} lncRNA transcripts in TPM")

    # Build gene -> list of transcripts
    gene_to_txs: Dict[str, List[str]] = defaultdict(list)
    for tx in lncrna_tx_in_data:
        gene_to_txs[tx2gene[tx]].append(tx)

    # ── 2. Load switch-significant genes ─────────────────────────────────────
    print("[2/5] Loading switch-significant genes ...")
    vis_df = pd.read_csv(str(VIS_PATH), sep='\t')
    switch_genes = set(vis_df.loc[vis_df['switch_sig'] == True, 'gene_id'].tolist())
    print(f"  {len(switch_genes):,} switch-significant genes in visibility table")

    # Only keep those that have >= 2 isoforms in the TPM data
    switch_genes_multi = [g for g in switch_genes if len(gene_to_txs.get(g, [])) >= 2]
    print(f"  {len(switch_genes_multi):,} switch-significant genes with >= 2 isoforms in data")

    # ── 3. Parse GTF exon data ────────────────────────────────────────────────
    print("[3/5] Parsing exon intervals from GTF ...")
    # Collect all transcript IDs needed (only for switch-sig genes)
    needed_txs = set()
    for g in switch_genes_multi:
        for tx in gene_to_txs[g]:
            needed_txs.add(tx)
    print(f"  Parsing exons for {len(needed_txs):,} transcripts ...")
    tx_exons = load_tx_exons(str(GTF_PATH), needed_txs)
    print(f"  Exon data loaded for {len(tx_exons):,} transcripts")

    # ── 4. Compute dominant isoforms & differential exons ────────────────────
    print("[4/5] Computing dominant isoforms and differential exon regions ...")

    sample_arr = np.array([s2t[s] for s in kept_samples])
    tpm_np     = tpm_df.values  # full matrix (all lncRNA txs)
    tx_index   = {tx: i for i, tx in enumerate(tpm_df.index)}

    exon_rows: List[dict] = []          # rows for switch_exons.tsv
    bed_rows:  List[dict] = []          # rows for BED file

    n_processed = 0
    n_with_switch = 0

    for g in switch_genes_multi:
        txs  = gene_to_txs[g]
        idxs = [tx_index[tx] for tx in txs]
        mat  = tpm_np[idxs, :]                      # (n_iso, n_samp)
        gene_sum = mat.sum(axis=0)                   # (n_samp,)

        tissue_dom = compute_tissue_dominant_isoforms(
            txs, mat, gene_sum, sample_arr, tissues_used
        )
        n_processed += 1

        # Find all tissue pairs where the dominant isoform differs
        valid_tissues = list(tissue_dom.keys())
        if len(valid_tissues) < 2:
            continue

        gene_had_switch = False
        for tA, tB in combinations(valid_tissues, 2):
            dom_A = tissue_dom[tA]
            dom_B = tissue_dom[tB]
            if dom_A == dom_B:
                continue  # same dominant isoform — not a dominant switch

            gene_had_switch = True

            # Get exon intervals for both dominant isoforms
            if dom_A not in tx_exons or dom_B not in tx_exons:
                # No exon data (shouldn't happen if GTF is complete) — skip
                continue

            chrom_A, ivs_A, strand_A = tx_exons[dom_A]
            chrom_B, ivs_B, strand_B = tx_exons[dom_B]

            # Safety: only compare exons on same chrom+strand (should always be true within a gene)
            if chrom_A != chrom_B or strand_A != strand_B:
                continue

            diff_ivs = symmetric_diff_intervals(ivs_A, ivs_B)
            n_diff   = len(diff_ivs)
            d_bp     = total_bp(diff_ivs)
            name_tag = f"{g}|{dom_A}-vs-{dom_B}"

            exon_rows.append({
                'gene_id':        g,
                'tissueA':        tA,
                'tissueB':        tB,
                'dom_isoform_A':  dom_A,
                'dom_isoform_B':  dom_B,
                'n_diff_regions': n_diff,
                'diff_bp':        d_bp,
            })

            for iv_start, iv_end in diff_ivs:
                bed_rows.append({
                    'chrom':  chrom_A,
                    'start':  iv_start,
                    'end':    iv_end,
                    'name':   name_tag,
                    'score':  0,
                    'strand': strand_A,
                })

        if gene_had_switch:
            n_with_switch += 1

    # ── 5. Write outputs and summary ─────────────────────────────────────────
    print("[5/5] Writing outputs ...")

    exon_df = pd.DataFrame(exon_rows, columns=[
        'gene_id', 'tissueA', 'tissueB',
        'dom_isoform_A', 'dom_isoform_B',
        'n_diff_regions', 'diff_bp',
    ])
    exon_df.to_csv(PROC_DIR / "switch_exons.tsv", sep='\t', index=False)
    print(f"  Saved: data/processed/switch_exons.tsv  ({len(exon_df):,} rows)")

    bed_df = pd.DataFrame(bed_rows, columns=['chrom', 'start', 'end', 'name', 'score', 'strand'])
    bed_df.to_csv(PROC_DIR / "switch_diff_exons.bed", sep='\t', index=False, header=False)
    print(f"  Saved: data/processed/switch_diff_exons.bed  ({len(bed_df):,} rows)")

    # Summary statistics
    n_total_pairs = len(exon_df)
    diff_bp_vals  = exon_df['diff_bp'].values if n_total_pairs > 0 else np.array([])
    median_diff_bp = float(np.median(diff_bp_vals)) if len(diff_bp_vals) > 0 else 0.0
    p90_diff_bp    = float(np.percentile(diff_bp_vals, 90)) if len(diff_bp_vals) > 0 else 0.0
    n_bed_regions  = len(bed_df)

    total_time = time.time() - t0

    summary_lines = [
        "=" * 60,
        "Phase 3 Step 1 Summary — Dominant Isoform Switch & Differential Exons",
        "=" * 60,
        "",
        f"Switch-significant genes processed:          {n_processed:,}",
        f"Genes with >= 1 dominant-switch tissue pair: {n_with_switch:,}",
        f"Total dominant-switch tissue pairs:          {n_total_pairs:,}",
        "",
        f"Differential exonic bp distribution (per pair):",
        f"  Median diff_bp:     {median_diff_bp:,.0f}",
        f"  90th pct diff_bp:   {p90_diff_bp:,.0f}",
        "",
        f"Total differential exon regions (BED):       {n_bed_regions:,}",
        "",
        f"Analysis parameters:",
        f"  MIN_SAMPLE_GENE_TPM:         {MIN_SAMPLE_GENE_TPM}",
        f"  MIN_EXPR_SAMPLES_PER_TISSUE: {MIN_EXPR_SAMPLES_PER_TISSUE}",
        f"  Total runtime:               {total_time:.1f}s",
        "",
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    out_path = PROC_DIR / "phase3_step1_summary.txt"
    with open(out_path, 'w') as fh:
        fh.write(summary_text + "\n")
    print(f"\nSaved: {out_path}")
    print("Phase 3 Step 1 complete.")


if __name__ == "__main__":
    main()
