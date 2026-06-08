"""run_rbp.py — Phase 3 Step 2: RBP sequence-motif content change during isoform switching.

For each dominant-switch tissue pair (from switch_exons.tsv), compare per-motif
sequence density between the two dominant isoforms.  A random null set (random
isoform pairs from non-switching, non-DE / visibility-none genes) provides the
baseline; a separate length-matched null is computed in
code/sensitivity_rbp_lengthmatched.py.

Run from project root:
    python code/run_rbp.py

Outputs (data/processed/):
    switch_rbp.tsv            — one row per switch pair, per-motif deltas + total_abs_delta
    phase3_step2_summary.txt  — headline numbers, MWU results, per-RBP BH findings, verdict
"""

import gzip
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Paths ─────────────────────────────────────────────────────────────────────
FASTA_PATH      = _ROOT / "data/raw/gencode.v26.lncRNA_transcripts.fa.gz"
SWITCH_PATH     = _ROOT / "data/processed/switch_exons.tsv"
VIS_PATH        = _ROOT / "data/processed/visibility_longread.tsv"
LNC_ISI_PATH    = _ROOT / "data/processed/lnc_isi_longread.tsv"
PROC_DIR        = _ROOT / "data/processed"
PROC_DIR.mkdir(exist_ok=True)

# ── RBP motif set ─────────────────────────────────────────────────────────────
# Sources: ATtRACT database (Giudice et al. 2016, Bioinformatics) and
# CISBP-RNA (Ray et al. 2013, Science) consensus motifs.
# All motifs are on the RNA/transcript sequence; FASTA is DNA, so U -> T.
RBP_MOTIFS: Dict[str, str] = {
    "ELAVL1_HuR_ARE": r"ATTTA",                    # AU-rich element core
    "TIA1_Urich":     r"T{5,}",                    # U-rich (≥5 T)
    "HNRNPC_Utract":  r"T{4,}",                    # poly-U tract (≥4 T)
    "PTBP1_CUrich":   r"(?:CT){3,}|TCTTC",         # CU-rich / pyrimidine tract
    "PCBP_Crich":     r"C{4,}",                    # poly-C (PCBP1/2)
    "SRSF1":          r"GGAGGA|GAAGAA",             # ESE GA-rich
    "NOVA_YCAY":      r"[CT]CA[CT]",               # NOVA1/2
    "MBNL_YGCY":      r"[CT]GC[CT]",               # MBNL1
    "QKI_ACUAAY":     r"ACTAA[CT]",                # QKI response element
    "TDP43_UGrepeat": r"(?:TG){3,}",               # TARDBP (≥3 TG repeats)
    "CELF1_UGUU":     r"TGTT",                     # CELF1/CUGBP
    "FUS_GGUG":       r"GGTG",                     # FUS
    "IGF2BP_GGAC":    r"GGAC",                     # IGF2BP / m6A-associated
    "SAM68_UAAA":     r"T[AT]AA",                  # KHDRBS1
    "G_quad_like":    r"(?:G{3,}\w{1,7}){3,}G{3,}",  # G-quadruplex-like (nuclear retention)
}

# Compile once
_COMPILED = {name: re.compile(pat) for name, pat in RBP_MOTIFS.items()}
MOTIF_NAMES = list(RBP_MOTIFS.keys())


# ── FASTA loader ──────────────────────────────────────────────────────────────

def load_fasta(fasta_path: Path) -> Dict[str, str]:
    """Load GENCODE lncRNA FASTA into {transcript_id: seq_upper_DNA}.

    Header format: >ENST.version|ENSG.version|...|name|gene_name|length|
    The first '|'-delimited field (after '>') is the transcript ID.
    Sequences are converted to uppercase DNA (T not U).
    """
    tx_seqs: Dict[str, str] = {}
    cur_id: str = ""
    buf: List[str] = []

    open_fn = gzip.open if str(fasta_path).endswith(".gz") else open
    with open_fn(fasta_path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    tx_seqs[cur_id] = "".join(buf).upper()
                header = line[1:]  # strip '>'
                cur_id = header.split("|")[0]
                buf = []
            else:
                buf.append(line)
        if cur_id:
            tx_seqs[cur_id] = "".join(buf).upper()

    return tx_seqs


# ── Motif density ─────────────────────────────────────────────────────────────

def motif_density(seq: str, motif_name: str) -> float:
    """Compute motif density: (#non-overlapping matches / len(seq)) * 1000 (per kb).

    Uses re.findall for non-overlapping matches — appropriate for this
    relative comparison.  Returns 0.0 if seq is empty.
    """
    if not seq:
        return 0.0
    matches = _COMPILED[motif_name].findall(seq)
    return len(matches) / len(seq) * 1000.0


def all_motif_densities(seq: str) -> Dict[str, float]:
    """Return {motif_name: density_per_kb} for all RBP_MOTIFS."""
    return {name: motif_density(seq, name) for name in MOTIF_NAMES}


def pair_deltas(seqA: str, seqB: str) -> Dict[str, float]:
    """Per-motif density delta = density(B) - density(A) for all motifs."""
    dA = all_motif_densities(seqA)
    dB = all_motif_densities(seqB)
    return {name: dB[name] - dA[name] for name in MOTIF_NAMES}


def total_abs_delta(deltas: Dict[str, float]) -> float:
    """Sum of |delta| across all motifs."""
    return float(sum(abs(v) for v in deltas.values()))


# ── BH correction ─────────────────────────────────────────────────────────────

def bh_correct(pvals: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg FDR correction. Returns (reject_flags, adjusted_pvals)."""
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return list(reject), list(pvals_corrected)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── 1. Load FASTA ─────────────────────────────────────────────────────────
    print("[1/6] Loading FASTA ...")
    tx_seqs = load_fasta(FASTA_PATH)
    print(f"  {len(tx_seqs):,} transcript sequences loaded")

    # ── 2. Load switch pairs ─────────────────────────────────────────────────
    print("[2/6] Loading switch pairs ...")
    sw_df = pd.read_csv(SWITCH_PATH, sep="\t")
    print(f"  {len(sw_df):,} switch tissue pairs loaded")

    # ── 3. Compute per-motif deltas for switch pairs ──────────────────────────
    print("[3/6] Computing motif densities for switch pairs ...")
    sw_rows = []
    n_switch_missing = 0

    for _, row in sw_df.iterrows():
        txA = row["dom_isoform_A"]
        txB = row["dom_isoform_B"]
        if txA not in tx_seqs or txB not in tx_seqs:
            n_switch_missing += 1
            continue
        seqA = tx_seqs[txA]
        seqB = tx_seqs[txB]
        deltas = pair_deltas(seqA, seqB)
        tab = total_abs_delta(deltas)
        entry: Dict = {
            "gene_id":          row["gene_id"],
            "tissueA":          row["tissueA"],
            "tissueB":          row["tissueB"],
            "dom_isoform_A":    txA,
            "dom_isoform_B":    txB,
            "len_A":            len(seqA),
            "len_B":            len(seqB),
            "total_abs_delta":  tab,
        }
        for name in MOTIF_NAMES:
            entry[f"delta_{name}"] = deltas[name]
        sw_rows.append(entry)

    sw_rbp = pd.DataFrame(sw_rows)
    n_switch_pairs = len(sw_rbp)
    print(f"  {n_switch_pairs:,} switch pairs analyzed ({n_switch_missing} skipped, missing FASTA)")

    # ── 4. Build NULL set ─────────────────────────────────────────────────────
    print("[4/6] Building null (control) pairs ...")

    vis_df  = pd.read_csv(VIS_PATH, sep="\t")
    lnc_df  = pd.read_csv(LNC_ISI_PATH, sep="\t")

    # Null genes: visibility == "none", non-DE, n_isoforms >= 2 in the data
    none_genes = set(vis_df.loc[
        (vis_df["visibility"] == "none") & (vis_df["gene_de"] == False),
        "gene_id"
    ])
    # Gene -> isoform list from FASTA (we need the FASTA keys, not GTF)
    # Build gene->transcripts from lnc_isi + tx_seqs
    # We'll use the switch pairs to infer tx2gene, but for null we need FASTA ENST->ENSG mapping.
    # The FASTA header field 1 is ENSG.version — use that.
    print("  Building FASTA tx2gene map ...")
    fasta_tx2gene: Dict[str, str] = {}
    open_fn = gzip.open if str(FASTA_PATH).endswith(".gz") else open
    with open_fn(FASTA_PATH, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                parts = line[1:].rstrip().split("|")
                if len(parts) >= 2:
                    tx_id   = parts[0]
                    gene_id = parts[1]
                    fasta_tx2gene[tx_id] = gene_id

    # Build gene -> [tx_id, ...] for null-eligible genes
    from collections import defaultdict
    gene_to_null_txs: Dict[str, List[str]] = defaultdict(list)
    for tx, g in fasta_tx2gene.items():
        if g in none_genes and tx in tx_seqs:
            gene_to_null_txs[g].append(tx)

    # Keep only genes with >= 2 isoforms in FASTA
    null_eligible = {g: txs for g, txs in gene_to_null_txs.items() if len(txs) >= 2}
    print(f"  {len(null_eligible):,} null-eligible genes (visibility=none, non-DE, >=2 isoforms in FASTA)")

    rng = np.random.default_rng(42)
    null_rows = []
    null_genes_list = list(null_eligible.keys())

    # Generate up to n_switch_pairs null pairs
    max_null = n_switch_pairs
    attempts = 0
    max_attempts = max_null * 20  # safety cap

    while len(null_rows) < max_null and attempts < max_attempts:
        g = null_genes_list[rng.integers(0, len(null_genes_list))]
        txs = null_eligible[g]
        if len(txs) < 2:
            attempts += 1
            continue
        idx = rng.choice(len(txs), size=2, replace=False)
        txA, txB = txs[idx[0]], txs[idx[1]]
        seqA = tx_seqs[txA]
        seqB = tx_seqs[txB]
        deltas = pair_deltas(seqA, seqB)
        tab = total_abs_delta(deltas)
        entry = {
            "gene_id":         g,
            "dom_isoform_A":   txA,
            "dom_isoform_B":   txB,
            "len_A":           len(seqA),
            "len_B":           len(seqB),
            "total_abs_delta": tab,
        }
        for name in MOTIF_NAMES:
            entry[f"delta_{name}"] = deltas[name]
        null_rows.append(entry)
        attempts += 1

    null_rbp = pd.DataFrame(null_rows)
    n_null_pairs = len(null_rbp)
    print(f"  {n_null_pairs:,} null pairs generated")

    # ── 5. Statistics ─────────────────────────────────────────────────────────
    print("[5/6] Running statistics ...")

    sw_tab  = sw_rbp["total_abs_delta"].values
    nu_tab  = null_rbp["total_abs_delta"].values

    # Length comparison
    sw_abs_len_diff = np.abs(sw_rbp["len_B"].values - sw_rbp["len_A"].values)
    nu_abs_len_diff = np.abs(null_rbp["len_B"].values - null_rbp["len_A"].values)
    sw_med_len_diff = float(np.median(sw_abs_len_diff))
    nu_med_len_diff = float(np.median(nu_abs_len_diff))

    # Overall MWU: total_abs_delta switch vs null
    U_overall, p_overall = stats.mannwhitneyu(sw_tab, nu_tab, alternative="two-sided")
    n_sw = len(sw_tab)
    n_nu = len(nu_tab)
    # Rank-biserial correlation r = 1 - 2U / (n_sw * n_nu)
    r_overall = 1.0 - 2.0 * U_overall / (n_sw * n_nu)

    sw_med_tab = float(np.median(sw_tab))
    nu_med_tab = float(np.median(nu_tab))

    # Per-RBP: |delta| switch vs null — Mann-Whitney
    rbp_results = []
    for name in MOTIF_NAMES:
        sw_vals = np.abs(sw_rbp[f"delta_{name}"].values)
        nu_vals = np.abs(null_rbp[f"delta_{name}"].values)
        U_rbp, p_rbp = stats.mannwhitneyu(sw_vals, nu_vals, alternative="two-sided")
        r_rbp = 1.0 - 2.0 * U_rbp / (n_sw * n_nu)
        rbp_results.append({
            "motif":     name,
            "U":         U_rbp,
            "p_raw":     p_rbp,
            "r":         r_rbp,
            "sw_median": float(np.median(sw_vals)),
            "nu_median": float(np.median(nu_vals)),
        })

    rbp_df = pd.DataFrame(rbp_results)
    reject_flags, p_adj = bh_correct(rbp_df["p_raw"].tolist())
    rbp_df["p_adj_BH"] = p_adj
    rbp_df["sig_BH"]   = reject_flags

    sig_rbps = rbp_df.loc[rbp_df["sig_BH"], "motif"].tolist()

    # ── 6. Write outputs ─────────────────────────────────────────────────────
    print("[6/6] Writing outputs ...")

    # switch_rbp.tsv
    sw_rbp.to_csv(PROC_DIR / "switch_rbp.tsv", sep="\t", index=False)
    print(f"  Saved: data/processed/switch_rbp.tsv ({len(sw_rbp):,} rows)")

    # Per-pair total_abs_delta arrays for figure integrity (real data, not simulated)
    pd.DataFrame({
        "gene_id": sw_rbp["gene_id"].values,
        "total_abs_delta": sw_rbp["total_abs_delta"].values,
    }).to_csv(PROC_DIR / "rbp_total_abs_delta_switch.tsv", sep="\t", index=False)
    print(f"  Saved: data/processed/rbp_total_abs_delta_switch.tsv ({len(sw_rbp):,} rows)")

    pd.DataFrame({
        "gene_id": null_rbp["gene_id"].values,
        "total_abs_delta": null_rbp["total_abs_delta"].values,
    }).to_csv(PROC_DIR / "rbp_total_abs_delta_null.tsv", sep="\t", index=False)
    print(f"  Saved: data/processed/rbp_total_abs_delta_null.tsv ({len(null_rbp):,} rows)")

    # per-RBP table (auxiliary)
    rbp_df.to_csv(PROC_DIR / "rbp_stats.tsv", sep="\t", index=False, float_format="%.4g")
    print(f"  Saved: data/processed/rbp_stats.tsv")

    # Determine verdict
    # Criteria: overall p < 0.05 AND |r| > 0.1 = weak-moderate positive effect
    if p_overall >= 0.05:
        verdict = "no: switches do NOT change total RBP-motif content more than null (p >= 0.05)"
    elif abs(r_overall) < 0.05:
        verdict = "negligible: p < 0.05 but |r| < 0.05 — effect is statistically detectable but biologically negligible"
    elif abs(r_overall) < 0.20:
        verdict = "weak: switches change total RBP-motif content slightly more than null (p < 0.05, |r| < 0.20)"
    else:
        verdict = "moderate: switches change total RBP-motif content more than null (p < 0.05, |r| >= 0.20)"

    # Assemble summary text
    lines = [
        "=" * 70,
        "Phase 3 Step 2 Summary — RBP Sequence-Motif Change During Isoform Switching",
        "=" * 70,
        "",
        f"Switch pairs analyzed:   {n_switch_pairs:,}",
        f"Null pairs generated:    {n_null_pairs:,}",
        "",
        "─── Length difference (|len_B - len_A|) ───────────────────────────────",
        f"  Switch set median:  {sw_med_len_diff:,.0f} nt",
        f"  Null set median:    {nu_med_len_diff:,.0f} nt",
        "  (Length differences are not matched; see caveat below)",
        "",
        "─── Overall RBP-motif change (total_abs_delta, sum of |Δdensity| per kb) ─",
        f"  Switch set median:  {sw_med_tab:.4f}",
        f"  Null set median:    {nu_med_tab:.4f}",
        f"  Mann-Whitney U:     {U_overall:.1f}",
        f"  p (two-sided):      {p_overall:.4g}",
        f"  Rank-biserial r:    {r_overall:.4f}",
        "",
        "─── Per-RBP motif: |Δdensity| switch vs null (BH-corrected at α=0.05) ──",
    ]
    for _, r_row in rbp_df.iterrows():
        sig_mark = "**" if r_row["sig_BH"] else "  "
        lines.append(
            f"  {sig_mark}{r_row['motif']:<22s}  p_raw={r_row['p_raw']:.3g}  "
            f"p_adj={r_row['p_adj_BH']:.3g}  r={r_row['r']:.3f}  "
            f"sw_med={r_row['sw_median']:.4f}  nu_med={r_row['nu_median']:.4f}"
        )
    lines += [
        "",
        f"Significantly changed RBPs (BH q<0.05, {len(sig_rbps)} of {len(MOTIF_NAMES)}):",
        "  " + (", ".join(sig_rbps) if sig_rbps else "<none>"),
        "",
        "─── Length caveat ──────────────────────────────────────────────────────",
        f"  Switch pairs have median |Δlen| = {sw_med_len_diff:,.0f} nt vs",
        f"  null pairs {nu_med_len_diff:,.0f} nt. Density is per-kb so length per se",
        "  does not inflate counts, but composition can covary with length.",
        "  No length-matching was performed; interpret effect size with care.",
        "",
        "─── Verdict ────────────────────────────────────────────────────────────",
        f"  {verdict}",
        "",
        f"Runtime: {time.time() - t0:.1f}s",
        "=" * 70,
    ]

    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    out_path = PROC_DIR / "phase3_step2_summary.txt"
    with open(out_path, "w") as fh:
        fh.write(summary_text + "\n")
    print(f"\nSaved: {out_path}")
    print("Phase 3 Step 2 complete.")


if __name__ == "__main__":
    main()
