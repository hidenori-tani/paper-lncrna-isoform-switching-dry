"""run_de_likeforlike.py — separate the DE *model* from the DE *input*.

Peer review (three-model round 1, 2026-07-27) raised the same objection twice
independently: the headline comparison changes both the statistical model AND the
input scale (Kruskal-Wallis on TPM versus negative-binomial models on raw counts),
so the reported spread in the DTU+/DGE- fraction may be an artefact of the input
rather than of test power.

This script isolates the two factors. It runs the SAME rank test used for the
headline, on the SAME raw count matrix that DESeq2/edgeR use, after TMM
normalisation (edgeR's own normalisation) -> the only thing that differs from the
count-based models is the statistical model.

Design (everything else held fixed):
    switching calls   : unchanged (visibility_longread.tsv)
    gene set          : unchanged (the 1389 analysed genes)
    samples/tissues   : unchanged (all samples of each gene's valid tissues,
                        including sub-threshold samples -- matching gene_de_flags())
    varying           : input scale (TPM vs TMM-CPM) x model (rank vs NB)

Run from project root, after run_landscape.py and run_de_countbased.py:
    python code/run_de_likeforlike.py

Outputs (data/processed/):
    de_likeforlike.tsv          per-gene q-values and DE flags for each cell of the 2x2
    de_likeforlike_summary.txt  the 2x2 table of DTU+/DGE- fractions
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_io import (load_lncrna_tx2gene, load_longread_counts,
                     load_longread_tpm, sample_to_tissue)
from de_countbased import aggregate_tx_to_gene
from run_landscape import compute_landscape

GTF_PATH    = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
COUNTS_PATH = _ROOT / "data/raw/longread/quantification_gencode.counts.txt.gz"
TPM_PATH    = _ROOT / "data/raw/longread/quantification_gencode.tpm.txt.gz"
ATTR_PATH   = _ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"
VIS_PATH    = _ROOT / "data/processed/visibility_longread.tsv"
CB_PATH     = _ROOT / "data/processed/de_countbased.tsv"
PROC_DIR    = _ROOT / "data/processed"

ALPHA = 0.05

TMM_R = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(edgeR))
cnt <- read.delim(args[1], row.names = 1, check.names = FALSE)
y   <- DGEList(counts = as.matrix(cnt))
y   <- calcNormFactors(y, method = "TMM")
out <- cpm(y, normalized.lib.sizes = TRUE, log = FALSE)
write.table(out, args[2], sep = "\t", quote = FALSE, col.names = NA)
"""


def tmm_cpm(gene_counts: pd.DataFrame) -> pd.DataFrame:
    """TMM-normalised CPM via edgeR — the same normalisation edgeR's DE test uses."""
    with tempfile.TemporaryDirectory(prefix="tmm_") as td:
        cnt_f, out_f, r_f = (Path(td) / n for n in ("counts.tsv", "cpm.tsv", "tmm.R"))
        gene_counts.to_csv(cnt_f, sep="\t")
        r_f.write_text(TMM_R)
        p = subprocess.run(["Rscript", str(r_f), str(cnt_f), str(out_f)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"TMM normalisation failed:\n{p.stderr[-1500:]}")
        return pd.read_csv(out_f, sep="\t", index_col=0)


def rank_de_flags(matrix: pd.DataFrame, analyzed_genes, gene_valid_tissue_map,
                  sample2tissue) -> dict:
    """Kruskal-Wallis across a gene's valid tissues, using ALL samples of those
    tissues (zeros retained) — identical convention to run_landscape.gene_de_flags,
    but on whichever matrix is passed in."""
    cols = list(matrix.columns)
    tissue_idx = {}
    for t in set(sample2tissue.values()):
        tissue_idx[t] = np.array([i for i, s in enumerate(cols)
                                  if sample2tissue.get(s) == t], dtype=int)
    vals = matrix.to_numpy(dtype=float)
    row_of = {g: i for i, g in enumerate(matrix.index)}

    pvals = []
    for g in analyzed_genes:
        i = row_of.get(g)
        if i is None:
            pvals.append(1.0)
            continue
        v = vals[i]
        groups = [v[tissue_idx[t]] for t in gene_valid_tissue_map[g]
                  if len(tissue_idx.get(t, ())) > 0]
        p = 1.0
        if len(groups) >= 2:
            try:
                _s, p = stats.kruskal(*groups)
            except Exception:
                p = 1.0
            if p is None or np.isnan(p):
                p = 1.0
        pvals.append(p)
    _, q, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    return {g: bool(qq < ALPHA) for g, qq in zip(analyzed_genes, q)}, dict(zip(analyzed_genes, q))


def main():
    print("[1/5] Loading annotation, counts, TPM and prior calls ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))
    counts = load_longread_counts(str(COUNTS_PATH))
    s2t = sample_to_tissue(counts.columns.tolist(), str(ATTR_PATH), min_samples_per_tissue=6)
    kept = [s for s in counts.columns if s in s2t]
    vis = pd.read_csv(VIS_PATH, sep="\t")
    analyzed_genes = vis["gene_id"].tolist()
    switch_sig = dict(zip(vis["gene_id"], vis["switch_sig"].astype(bool)))
    kw_tpm_de = dict(zip(vis["gene_id"], vis["gene_de"].astype(bool)))
    print(f"  {len(kept)} samples / {len(set(s2t.values()))} tissues / {len(analyzed_genes)} genes")

    print("[2/5] Recovering each gene's valid tissue set from the TPM landscape ...")
    tx_tpm = load_longread_tpm(str(TPM_PATH))
    (_isi_df, land_genes, gene_valid_tissue_map, _ges, _ifm, _tpmsum,
     _kept, _tissues, _tai) = compute_landscape(tx_tpm, tx2gene, s2t)
    missing = [g for g in analyzed_genes if g not in gene_valid_tissue_map]
    if missing:
        raise RuntimeError(f"{len(missing)} analysed genes absent from the landscape")
    print(f"  valid-tissue map recovered for {len(land_genes)} genes")

    print("[3/5] Aggregating transcripts to genes and TMM-normalising the counts ...")
    gene_counts = aggregate_tx_to_gene(counts[kept], tx2gene, gene_set=set(analyzed_genes))
    gene_counts = gene_counts[kept]
    cpm = tmm_cpm(gene_counts)
    print(f"  gene_counts {gene_counts.shape} -> TMM-CPM {cpm.shape}")

    print("[4/5] Rank test on TMM-CPM (model varies, input matched to edgeR) ...")
    rank_cpm_de, q_rank_cpm = rank_de_flags(cpm, analyzed_genes, gene_valid_tissue_map, s2t)

    cb = pd.read_csv(CB_PATH, sep="\t")
    de_deseq2 = dict(zip(cb["gene_id"], cb["gene_de_deseq2"].astype(bool)))
    de_edger = dict(zip(cb["gene_id"], cb["gene_de_edger"].astype(bool)))

    print("[5/5] Recomputing the DTU+/DGE- fraction for every cell ...")
    ss = [g for g in analyzed_genes if switch_sig.get(g)]

    def blind_spot(de_map):
        silent = [g for g in ss if not de_map.get(g, False)]
        return len(silent), len(ss), 100.0 * len(silent) / len(ss), set(silent)

    cells = {
        "rank test on TPM (headline)":      kw_tpm_de,
        "rank test on TMM-CPM (like-for-like)": rank_cpm_de,
        "DESeq2 LRT on raw counts":         de_deseq2,
        "edgeR QL on raw counts":           de_edger,
    }
    rows, sets = [], {}
    for name, m in cells.items():
        n_sil, n_ss, pct, s = blind_spot(m)
        n_de = sum(1 for g in analyzed_genes if m.get(g, False))
        rows.append((name, n_de, n_sil, n_ss, pct))
        sets[name] = s

    out = pd.DataFrame({
        "gene_id": analyzed_genes,
        "switch_sig": [switch_sig.get(g, False) for g in analyzed_genes],
        "q_rank_cpm": [q_rank_cpm[g] for g in analyzed_genes],
        "de_rank_tpm": [kw_tpm_de.get(g, False) for g in analyzed_genes],
        "de_rank_cpm": [rank_cpm_de.get(g, False) for g in analyzed_genes],
        "de_deseq2": [de_deseq2.get(g, False) for g in analyzed_genes],
        "de_edger": [de_edger.get(g, False) for g in analyzed_genes],
    })
    out.to_csv(PROC_DIR / "de_likeforlike.tsv", sep="\t", index=False)

    lines = []
    lines.append("=" * 78)
    lines.append("Separating the DE model from the DE input")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Switching calls, gene set, samples and tissues are identical in every row;")
    lines.append("only the input scale and/or the statistical model change.")
    lines.append("")
    lines.append(f"{'DE specification':42s} {'genes DE':>9s} {'blind spot':>14s}")
    for name, n_de, n_sil, n_ss, pct in rows:
        lines.append(f"{name:42s} {n_de:9d} {n_sil:5d}/{n_ss:<4d} {pct:5.1f}%")
    lines.append("")
    a = sets["rank test on TPM (headline)"]
    b = sets["rank test on TMM-CPM (like-for-like)"]
    c = sets["DESeq2 LRT on raw counts"]
    jac = (lambda x, y: len(x & y) / len(x | y) if (x | y) else float("nan"))
    lines.append("Blind-spot set overlap (Jaccard):")
    lines.append(f"  rank-TPM vs rank-TMM-CPM : {jac(a, b):.3f}  (shared {len(a & b)})")
    lines.append(f"  rank-TMM-CPM vs DESeq2   : {jac(b, c):.3f}  (shared {len(b & c)})")
    lines.append(f"  rank-TPM vs DESeq2       : {jac(a, c):.3f}  (shared {len(a & c)})")
    lines.append("")
    pct_tpm = rows[0][4]
    pct_cpm = rows[1][4]
    pct_ds = rows[2][4]
    lines.append("Interpretation:")
    if pct_cpm >= 0.5 * pct_tpm:
        lines.append("  The rank test keeps a large blind spot even when fed the SAME TMM-normalised")
        lines.append("  counts the NB models use, so the spread is driven by the statistical model")
        lines.append("  (power), not by the TPM input scale.")
    else:
        lines.append("  Most of the spread disappears once the input is matched: the headline")
        lines.append("  difference is substantially an input-scale (TPM) artefact, not test power.")
        lines.append("  The manuscript's claim must be narrowed accordingly.")
    lines.append(f"  rank/TPM {pct_tpm:.1f}%  ->  rank/TMM-CPM {pct_cpm:.1f}%  ->  DESeq2 {pct_ds:.1f}%")
    txt = "\n".join(lines) + "\n"
    (PROC_DIR / "de_likeforlike_summary.txt").write_text(txt)
    print()
    print(txt)


if __name__ == "__main__":
    main()
