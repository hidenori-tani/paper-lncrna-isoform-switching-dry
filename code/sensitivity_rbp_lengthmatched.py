"""sensitivity_rbp_lengthmatched.py — Reviewer (gpt-5 R3 MAJOR#4) sensitivity.

The primary RBP-motif analysis compares the per-pair total |Δdensity| (sum over
15 motifs) between dominant-isoform switch pairs and a null of random isoform
pairs from non-switching genes. Switch pairs have a larger median isoform-length
difference (|Δlen|) than null pairs, so a reviewer asked whether the negative
result survives explicit length-matching.

This script builds a length-matched null: it bins the switch pairs by |Δlen|
(deciles) and samples null pairs to reproduce the switch |Δlen| distribution,
then re-runs the two-sided Mann-Whitney U test and the rank-biserial effect size
on the length-matched sets. It reuses the validated motif machinery in run_rbp.py
and does NOT overwrite any primary output.

Run from project root:
    python code/sensitivity_rbp_lengthmatched.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import gzip
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_rbp import (  # noqa: E402
    load_fasta, pair_deltas, total_abs_delta,
    FASTA_PATH, VIS_PATH,
)

SWITCH_RBP_PATH = _ROOT / "data/processed/switch_rbp.tsv"
SEED = 42
NULL_POOL_MULT = 8   # generate this many * n_switch candidate null pairs


def main():
    print("[1/4] Loading switch pairs (with lengths + total_abs_delta) ...")
    sw = pd.read_csv(SWITCH_RBP_PATH, sep="\t")
    sw_len_diff = np.abs(sw["len_B"].values - sw["len_A"].values).astype(float)
    sw_tab = sw["total_abs_delta"].values.astype(float)
    n_switch = len(sw)
    print(f"  {n_switch:,} switch pairs; median |Δlen| = {np.median(sw_len_diff):,.0f} nt")

    print("[2/4] Loading FASTA + building null-eligible gene->tx map ...")
    tx_seqs = load_fasta(FASTA_PATH)
    vis = pd.read_csv(VIS_PATH, sep="\t")
    none_genes = set(vis.loc[(vis["visibility"] == "none") & (vis["gene_de"] == False), "gene_id"])
    fasta_tx2gene = {}
    open_fn = gzip.open if str(FASTA_PATH).endswith(".gz") else open
    with open_fn(FASTA_PATH, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                parts = line[1:].rstrip().split("|")
                if len(parts) >= 2:
                    fasta_tx2gene[parts[0]] = parts[1]
    gene_to_txs = defaultdict(list)
    for tx, g in fasta_tx2gene.items():
        if g in none_genes and tx in tx_seqs:
            gene_to_txs[g].append(tx)
    null_eligible = {g: txs for g, txs in gene_to_txs.items() if len(txs) >= 2}
    print(f"  {len(null_eligible):,} null-eligible genes")

    print(f"[3/4] Generating candidate null pool (~{NULL_POOL_MULT}x switch) ...")
    rng = np.random.default_rng(SEED)
    genes = list(null_eligible.keys())
    pool_len_diff = []
    pool_tab = []
    target = n_switch * NULL_POOL_MULT
    attempts = 0
    while len(pool_tab) < target and attempts < target * 20:
        attempts += 1
        g = genes[rng.integers(0, len(genes))]
        txs = null_eligible[g]
        idx = rng.choice(len(txs), size=2, replace=False)
        a, b = txs[idx[0]], txs[idx[1]]
        sa, sb = tx_seqs[a], tx_seqs[b]
        pool_len_diff.append(abs(len(sb) - len(sa)))
        pool_tab.append(total_abs_delta(pair_deltas(sa, sb)))
    pool_len_diff = np.array(pool_len_diff, dtype=float)
    pool_tab = np.array(pool_tab, dtype=float)
    print(f"  null pool size: {len(pool_tab):,}")

    print("[4/4] Length-matching null to switch |Δlen| distribution (deciles) ...")
    # decile edges from switch |Δlen|
    edges = np.quantile(sw_len_diff, np.linspace(0, 1, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf
    matched_tab = []
    matched_len = []
    rng2 = np.random.default_rng(SEED + 1)
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        n_need = int(np.sum((sw_len_diff >= lo) & (sw_len_diff < hi)))
        pool_idx = np.where((pool_len_diff >= lo) & (pool_len_diff < hi))[0]
        if len(pool_idx) == 0:
            continue
        pick = rng2.choice(pool_idx, size=n_need, replace=(len(pool_idx) < n_need))
        matched_tab.extend(pool_tab[pick].tolist())
        matched_len.extend(pool_len_diff[pick].tolist())
    matched_tab = np.array(matched_tab, dtype=float)
    matched_len = np.array(matched_len, dtype=float)

    # check match quality (median |Δlen| of matched null bins ~ switch)
    U, p = stats.mannwhitneyu(sw_tab, matched_tab, alternative="two-sided")
    r = 1.0 - 2.0 * U / (len(sw_tab) * len(matched_tab))

    print("\n" + "=" * 66)
    print("RBP NEGATIVE RESULT — LENGTH-MATCHED NULL SENSITIVITY")
    print("=" * 66)
    print(f"switch pairs:            {len(sw_tab):,}  median total|Δdensity| = {np.median(sw_tab):.3f}")
    print(f"length-matched null:     {len(matched_tab):,}  median total|Δdensity| = {np.median(matched_tab):.3f}")
    print(f"switch median |Δlen|:    {np.median(sw_len_diff):,.0f} nt")
    print(f"matched-null median |Δlen|: {np.median(matched_len):,.0f} nt  (raw pool: {np.median(pool_len_diff):,.0f} nt)")
    print(f"Mann-Whitney U (2-sided): U={U:.1f}  p={p:.4g}")
    print(f"Rank-biserial r:          {r:.4f}  ({'negligible |r|<0.05' if abs(r) < 0.05 else 'see value'})")
    print("-" * 66)
    print("Interpretation: if |r| remains negligible after length-matching, the")
    print("negative RBP-motif result is not an artifact of the switch/null length")
    print("difference.")
    print("=" * 66)


if __name__ == "__main__":
    main()
