"""run_de_lengthoffset.py — is the count-based blind spot a transcript-length artefact?

Peer review (three-model round 2, 2026-07-27) raised this as a CRITICAL objection:
summing transcript-level counts to the gene level is known to bias gene-level DE
when isoform usage changes across samples, because reads scale with effective
length as well as with molar concentration (Soneson et al. 2015, tximport). A gene
at constant molarity that switches from a short to a long dominant isoform gains
reads, and a count-based model can call that a gene-level expression difference.
Since switch pairs here differ by a median of 666 nt, the confound is guaranteed to
be present in exactly the genes the blind spot is measured on — so the very small
count-based figures might be a length artefact rather than a power effect.

This script tests that directly. It recomputes gene-level DE with an average
transcript-length offset, the correction tximport applies: for each gene and sample
the effective length is the isoform-usage-weighted mean transcript length, and the
resulting per-gene, per-sample normalisation matrix is supplied to DESeq2
(normalizationFactors) and edgeR (offset). Everything else — genes, samples,
tissues, switching calls — is unchanged.

Run from project root:
    python code/run_de_lengthoffset.py

Outputs (data/processed/):
    de_lengthoffset.tsv          per-gene padj with and without the length offset
    de_lengthoffset_summary.txt  blind spot with and without the correction
"""
import gzip
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_io import (load_lncrna_tx2gene, load_longread_counts,
                     load_longread_tpm, sample_to_tissue)
from de_countbased import aggregate_tx_to_gene, build_coldata

GTF_PATH    = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
COUNTS_PATH = _ROOT / "data/raw/longread/quantification_gencode.counts.txt.gz"
TPM_PATH    = _ROOT / "data/raw/longread/quantification_gencode.tpm.txt.gz"
ATTR_PATH   = _ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"
VIS_PATH    = _ROOT / "data/processed/visibility_longread.tsv"
CB_PATH     = _ROOT / "data/processed/de_countbased.tsv"
PROC_DIR    = _ROOT / "data/processed"

ALPHA = 0.05

OFFSET_R = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(DESeq2); library(edgeR)})
cnt  <- as.matrix(read.delim(args[1], row.names = 1, check.names = FALSE))
cd   <- read.delim(args[2], stringsAsFactors = FALSE)
nm   <- as.matrix(read.delim(args[3], row.names = 1, check.names = FALSE))
cd$tissue <- factor(cd$tissue)
storage.mode(cnt) <- "integer"
nm <- nm[rownames(cnt), colnames(cnt), drop = FALSE]

# --- DESeq2 with tximport-style length normalisation factors -------------------
dds <- DESeqDataSetFromMatrix(cnt, colData = cd, design = ~ tissue)
normMat <- nm / exp(rowMeans(log(nm)))                  # centre each gene
sf <- estimateSizeFactorsForMatrix(cnt / normMat)
normalizationFactors(dds) <- sweep(normMat, 2, sf, "*")
dds <- DESeq(dds, test = "LRT", reduced = ~ 1, quiet = TRUE, fitType = "local")
p_deseq2 <- results(dds, independentFiltering = FALSE, cooksCutoff = FALSE)$padj

# --- edgeR with the same offset ------------------------------------------------
y <- DGEList(counts = cnt, group = cd$tissue)
lognm <- log(nm)
lognm <- lognm - rowMeans(lognm)
y <- scaleOffset(y, lognm + matrix(log(getOffset(y)), nrow(y), ncol(y), byrow = TRUE))
des <- model.matrix(~ tissue, data = cd)
y <- estimateDisp(y, des)
fit <- glmQLFit(y, des)
qlf <- glmQLFTest(fit, coef = 2:ncol(des))
p_edger <- p.adjust(qlf$table$PValue, method = "BH")

out <- data.frame(gene_id = rownames(cnt), padj_deseq2 = p_deseq2, padj_edger = p_edger)
write.table(out, args[4], sep = "\t", quote = FALSE, row.names = FALSE)
"""


def transcript_lengths(gtf_gz: Path) -> dict:
    """Mature transcript length = summed exon length, from the GENCODE GTF."""
    lens = defaultdict(int)
    with gzip.open(gtf_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            attr = f[8]
            i = attr.find('transcript_id "')
            if i < 0:
                continue
            tx = attr[i + 15:attr.index('"', i + 15)]
            lens[tx] += int(f[4]) - int(f[3]) + 1
    return dict(lens)


def main():
    print("[1/5] Loading annotation, counts, TPM ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))
    counts = load_longread_counts(str(COUNTS_PATH))
    tpm = load_longread_tpm(str(TPM_PATH))
    s2t = sample_to_tissue(counts.columns.tolist(), str(ATTR_PATH), min_samples_per_tissue=6)
    kept = [s for s in counts.columns if s in s2t]
    vis = pd.read_csv(VIS_PATH, sep="\t")
    analyzed = vis["gene_id"].tolist()
    switch_sig = dict(zip(vis["gene_id"], vis["switch_sig"].astype(bool)))

    print("[2/5] Computing transcript lengths from the GTF ...")
    txlen = transcript_lengths(GTF_PATH)
    print(f"  lengths for {len(txlen)} transcripts "
          f"(median {int(np.median(list(txlen.values())))} nt)")

    print("[3/5] Building the per-gene, per-sample effective-length matrix ...")
    gene_set = set(analyzed)
    gene_txs = defaultdict(list)
    for tx, g in tx2gene.items():
        if g in gene_set and tx in tpm.index and tx in txlen:
            gene_txs[g].append(tx)
    genes = [g for g in analyzed if len(gene_txs.get(g, [])) >= 1]

    tpm_k = tpm[kept]
    eff = np.zeros((len(genes), len(kept)), dtype=float)
    for i, g in enumerate(genes):
        txs = gene_txs[g]
        L = np.array([txlen[t] for t in txs], dtype=float)
        M = tpm_k.loc[txs].to_numpy(dtype=float)
        tot = M.sum(axis=0)
        w = np.divide(M, np.where(tot > 0, tot, 1.0))
        e = (w * L[:, None]).sum(axis=0)
        # samples where the gene is unexpressed get the gene's unweighted mean length
        e[tot <= 0] = L.mean()
        eff[i] = np.maximum(e, 1.0)
    eff_df = pd.DataFrame(eff, index=genes, columns=kept)
    rel = eff_df.max(axis=1) / eff_df.min(axis=1)
    print(f"  effective length varies across samples by a median factor of {rel.median():.2f} "
          f"(90th pct {rel.quantile(0.9):.2f}, max {rel.max():.2f})")

    print("[4/5] Re-running DESeq2 and edgeR with the length offset ...")
    gene_counts = aggregate_tx_to_gene(counts[kept], tx2gene, gene_set=gene_set)[kept]
    gene_counts = gene_counts.loc[genes]
    coldata = build_coldata(kept, s2t)
    coldata["lib_size"] = gene_counts.sum(axis=0).values
    with tempfile.TemporaryDirectory(prefix="lenoff_") as td:
        cf, df_, nf, of, rf = (Path(td) / n for n in
                               ("c.tsv", "d.tsv", "n.tsv", "o.tsv", "r.R"))
        gene_counts.round().astype(int).to_csv(cf, sep="\t")
        coldata.to_csv(df_, sep="\t", index=False)
        eff_df.to_csv(nf, sep="\t")
        rf.write_text(OFFSET_R)
        p = subprocess.run(["Rscript", str(rf), str(cf), str(df_), str(nf), str(of)],
                           capture_output=True, text=True)
        if p.stderr.strip():
            print("  R stderr:", p.stderr.strip()[-600:])
        if p.returncode != 0:
            raise RuntimeError("length-offset DE failed")
        res = pd.read_csv(of, sep="\t")

    print("[5/5] Recomputing the blind spot with and without the correction ...")
    de_ds = dict(zip(res["gene_id"], res["padj_deseq2"] < ALPHA))
    de_ed = dict(zip(res["gene_id"], res["padj_edger"] < ALPHA))
    cb = pd.read_csv(CB_PATH, sep="\t")
    old_ds = dict(zip(cb["gene_id"], cb["gene_de_deseq2"].astype(bool)))
    old_ed = dict(zip(cb["gene_id"], cb["gene_de_edger"].astype(bool)))
    ss = [g for g in genes if switch_sig.get(g)]

    def bs(m):
        sil = [g for g in ss if not m.get(g, False)]
        return len(sil), len(ss), 100.0 * len(sil) / len(ss)

    rows = [("DESeq2, summed counts (no offset)", *bs(old_ds)),
            ("DESeq2, tximport-style length offset", *bs(de_ds)),
            ("edgeR, summed counts (no offset)", *bs(old_ed)),
            ("edgeR, tximport-style length offset", *bs(de_ed))]

    out = pd.DataFrame({"gene_id": genes,
                        "switch_sig": [switch_sig.get(g, False) for g in genes],
                        "de_deseq2_nooffset": [old_ds.get(g, False) for g in genes],
                        "de_deseq2_lenoffset": [de_ds.get(g, False) for g in genes],
                        "de_edger_nooffset": [old_ed.get(g, False) for g in genes],
                        "de_edger_lenoffset": [de_ed.get(g, False) for g in genes],
                        "efflen_ratio_max_min": rel.reindex(genes).values})
    out.to_csv(PROC_DIR / "de_lengthoffset.tsv", sep="\t", index=False)

    lines = ["=" * 78,
             "Is the count-based blind spot a transcript-length artefact?",
             "=" * 78, "",
             "Gene-level counts obtained by summing transcript counts are biased when",
             "isoform usage shifts, because reads scale with effective length. The offset",
             "below is the isoform-usage-weighted mean transcript length per gene per",
             "sample, supplied to DESeq2 (normalizationFactors) and edgeR (offset).", "",
             f"Genes analysed: {len(genes)}   switch-significant: {len(ss)}",
             f"Effective length varies across samples by a median factor of {rel.median():.2f}",
             "",
             f"{'specification':44s} {'blind spot':>14s}"]
    for name, n_sil, n, pct in rows:
        lines.append(f"{name:44s} {n_sil:5d}/{n:<4d} {pct:5.1f}%")
    d_ds = rows[1][3] - rows[0][3]
    d_ed = rows[3][3] - rows[2][3]
    lines += ["", "Interpretation:"]
    if max(abs(d_ds), abs(d_ed)) >= 5.0:
        lines += ["  The length offset changes the blind spot materially, so the uncorrected",
                  "  count-based figures are partly a length artefact and must be replaced.",
                  f"  DESeq2 {rows[0][3]:.1f}% -> {rows[1][3]:.1f}%; edgeR {rows[2][3]:.1f}% -> {rows[3][3]:.1f}%."]
    else:
        lines += ["  The length offset changes the blind spot by less than 5 percentage points,",
                  "  so the small count-based figures are not explained by length-driven read",
                  "  inflation; the confound is real in principle but immaterial at this scale.",
                  f"  DESeq2 {rows[0][3]:.1f}% -> {rows[1][3]:.1f}%; edgeR {rows[2][3]:.1f}% -> {rows[3][3]:.1f}%."]
    txt = "\n".join(lines) + "\n"
    (PROC_DIR / "de_lengthoffset_summary.txt").write_text(txt)
    print()
    print(txt)


if __name__ == "__main__":
    main()
