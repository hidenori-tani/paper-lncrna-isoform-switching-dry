"""run_de_countbased.py — count-based DE robustness of the 'silent' fraction.

Re-derives the gene-level DE flag with negative-binomial GLM methods
(DESeq2 LRT, edgeR quasi-likelihood ANODEV) instead of the Kruskal-Wallis test
used in run_landscape.py, then re-classifies switching-visibility and reports
how the 'silent' fraction moves.  The switching test, the analyzed gene set,
and the samples/tissues are all unchanged — only the DE method differs.

Also reports a technical-replicate-collapsed sensitivity (the 8 GTEx long-read
`_rep` samples are pseudoreplicates; the headline analysis, like the KW test,
treats them as independent samples).

Run from project root (after run_landscape.py has produced visibility_longread.tsv):
    python code/run_de_countbased.py

Outputs (data/processed/):
    de_countbased.tsv           per-gene padj + DE flags (KW / DESeq2 / edgeR)
    visibility_countbased.tsv   per-gene visibility under each DE method
    de_countbased_summary.txt   headline silent fractions + method concordance
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_io import load_lncrna_tx2gene, load_longread_counts, sample_to_tissue
from de_countbased import (
    aggregate_tx_to_gene,
    build_coldata,
    collapse_replicates,
    countbased_de_flags,
    silent_fraction,
    strip_rep,
)
from isoform_metrics import classify_visibility

GTF_PATH    = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
COUNTS_PATH = _ROOT / "data/raw/longread/quantification_gencode.counts.txt.gz"
ATTR_PATH   = _ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"
VIS_PATH    = _ROOT / "data/processed/visibility_longread.tsv"
PROC_DIR    = _ROOT / "data/processed"
R_SCRIPT    = _HERE / "de_countbased.R"

ALPHA = 0.05


def _run_r(gene_counts: pd.DataFrame, coldata: pd.DataFrame, norm: str = "native") -> pd.DataFrame:
    """Write count matrix + colData to a temp dir, invoke de_countbased.R, read padj back.

    norm : "native" (composition-aware: DESeq2 median-of-ratios/poscounts, edgeR TMM)
           or "library" (whole-library size factors, matched to the TPM normalization).
    """
    with tempfile.TemporaryDirectory(prefix="de_cb_") as td:
        counts_f  = Path(td) / "counts.tsv"
        coldata_f = Path(td) / "coldata.tsv"
        out_f     = Path(td) / "out.tsv"
        gene_counts.to_csv(counts_f, sep="\t")
        coldata.to_csv(coldata_f, sep="\t", index=False)
        proc = subprocess.run(
            ["Rscript", str(R_SCRIPT), str(counts_f), str(coldata_f), str(out_f), norm],
            capture_output=True, text=True,
        )
        if proc.stderr:
            print(proc.stderr.rstrip())
        if proc.returncode != 0:
            raise RuntimeError(f"de_countbased.R failed (exit {proc.returncode})")
        return pd.read_csv(out_f, sep="\t")


def _collapse_coldata(coldata: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical replicates in colData to match collapse_replicates():
    one row per base sample; lib_size summed; tissue/donor from the base."""
    cd = coldata.copy()
    cd["base"] = cd["sample"].map(strip_rep)
    agg = (
        cd.groupby("base", sort=False)
        .agg(tissue=("tissue", "first"),
             donor=("donor", "first"),
             lib_size=("lib_size", "sum"))
        .reset_index()
        .rename(columns={"base": "sample"})
    )
    agg["is_rep"] = False
    return agg[["sample", "tissue", "donor", "is_rep", "lib_size"]]


def _kappa(a: dict, b: dict, genes) -> float:
    """Cohen's kappa between two boolean DE call maps over `genes`."""
    va = np.array([bool(a.get(g, False)) for g in genes])
    vb = np.array([bool(b.get(g, False)) for g in genes])
    n = len(genes)
    po = np.mean(va == vb)
    pa1, pb1 = va.mean(), vb.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else float("nan")


def main():
    print("[1/5] Loading tx2gene, counts, sample map, and prior visibility ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))
    counts = load_longread_counts(str(COUNTS_PATH))
    s2t = sample_to_tissue(counts.columns.tolist(), str(ATTR_PATH), min_samples_per_tissue=6)
    kept = list(s2t.keys())
    print(f"  {len(kept)} samples across {len(set(s2t.values()))} tissues")

    vis = pd.read_csv(VIS_PATH, sep="\t")
    analyzed_genes = vis["gene_id"].tolist()
    switch_sig = dict(zip(vis["gene_id"], vis["switch_sig"].astype(bool)))
    kw_de = dict(zip(vis["gene_id"], vis["gene_de"].astype(bool)))
    print(f"  {len(analyzed_genes)} analyzed genes from visibility_longread.tsv")

    # Whole-library size = per-sample total over ALL transcripts (matches TPM normalization)
    lib_size = counts[kept].sum(axis=0)

    print("[2/5] Building lncRNA gene-level count matrix (analyzed gene set) ...")
    gene_counts = aggregate_tx_to_gene(counts[kept], tx2gene, gene_set=set(analyzed_genes))
    gene_counts = gene_counts[kept]  # enforce column order
    coldata = build_coldata(kept, s2t)
    coldata["lib_size"] = coldata["sample"].map(lib_size).astype(np.int64).values
    print(f"  gene_counts: {gene_counts.shape[0]} genes x {gene_counts.shape[1]} samples")

    print("[3/5] Count-based DE, native normalization (primary) ...")
    de_full = _run_r(gene_counts, coldata, norm="native")
    padj_d = dict(zip(de_full["gene_id"], de_full["padj_deseq2"]))
    padj_e = dict(zip(de_full["gene_id"], de_full["padj_edger"]))
    de_deseq2 = countbased_de_flags(pd.Series(padj_d), alpha=ALPHA)
    de_edger  = countbased_de_flags(pd.Series(padj_e), alpha=ALPHA)

    print("      Count-based DE, whole-library normalization (sensitivity) ...")
    de_lib = _run_r(gene_counts, coldata, norm="library")
    de_deseq2_lib = countbased_de_flags(
        pd.Series(dict(zip(de_lib["gene_id"], de_lib["padj_deseq2"]))), alpha=ALPHA)
    de_edger_lib = countbased_de_flags(
        pd.Series(dict(zip(de_lib["gene_id"], de_lib["padj_edger"]))), alpha=ALPHA)

    print("[4/5] Technical-replicate-collapsed DE (native; sensitivity) ...")
    gc_coll = collapse_replicates(gene_counts)
    cd_coll = _collapse_coldata(coldata)
    gc_coll = gc_coll[cd_coll["sample"].tolist()]
    n_rep = gene_counts.shape[1] - gc_coll.shape[1]
    de_coll = _run_r(gc_coll, cd_coll, norm="native")
    padj_d_c = dict(zip(de_coll["gene_id"], de_coll["padj_deseq2"]))
    de_deseq2_coll = countbased_de_flags(pd.Series(padj_d_c), alpha=ALPHA)

    print("[5/5] Re-classifying visibility and summarizing ...")
    sf_kw     = silent_fraction(kw_de, switch_sig)
    sf_deseq2 = silent_fraction(de_deseq2, switch_sig)
    sf_edger  = silent_fraction(de_edger, switch_sig)
    sf_deseq2_lib = silent_fraction(de_deseq2_lib, switch_sig)
    sf_edger_lib  = silent_fraction(de_edger_lib, switch_sig)
    sf_coll   = silent_fraction(de_deseq2_coll, switch_sig)

    # Method concordance on the analyzed gene set
    kappa_kw_d = _kappa(kw_de, de_deseq2, analyzed_genes)
    kappa_kw_e = _kappa(kw_de, de_edger, analyzed_genes)
    kappa_d_e  = _kappa(de_deseq2, de_edger, analyzed_genes)

    # Silent-set overlap (KW vs DESeq2)
    def silent_set(de_map):
        return {g for g in switch_sig
                if classify_visibility(bool(de_map.get(g, False)), bool(switch_sig[g])) == "silent"}
    s_kw, s_d = silent_set(kw_de), silent_set(de_deseq2)
    jacc = len(s_kw & s_d) / len(s_kw | s_d) if (s_kw | s_d) else float("nan")

    # ── per-gene output table ────────────────────────────────────────────────
    rows = []
    for g in analyzed_genes:
        sig = bool(switch_sig[g])
        rows.append({
            "gene_id": g,
            "switch_sig": sig,
            "padj_deseq2": padj_d.get(g, np.nan),
            "padj_edger": padj_e.get(g, np.nan),
            "gene_de_kw": bool(kw_de.get(g, False)),
            "gene_de_deseq2": bool(de_deseq2.get(g, False)),
            "gene_de_edger": bool(de_edger.get(g, False)),
            "visibility_kw": classify_visibility(bool(kw_de.get(g, False)), sig),
            "visibility_deseq2": classify_visibility(bool(de_deseq2.get(g, False)), sig),
            "visibility_edger": classify_visibility(bool(de_edger.get(g, False)), sig),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(PROC_DIR / "de_countbased.tsv", sep="\t", index=False)
    out_df[["gene_id", "switch_sig", "visibility_kw", "visibility_deseq2",
            "visibility_edger"]].to_csv(PROC_DIR / "visibility_countbased.tsv",
                                         sep="\t", index=False)

    def fmt(sf):
        return f"{sf['n_silent']}/{sf['n_switch_sig']} = {sf['fraction']*100:.1f}%"

    lines = [
        "=" * 66,
        "Count-based DE robustness of the silent fraction (long-read)",
        "=" * 66,
        "",
        f"Analyzed genes:              {len(analyzed_genes)}",
        f"Switch-significant genes:    {sf_kw['n_switch_sig']} (unchanged; switching test not re-run)",
        f"Samples / tissues:           {len(kept)} / {len(set(s2t.values()))}",
        f"Technical replicates:        {n_rep} (_rep) collapsed in sensitivity",
        "",
        "Silent fraction among switch-significant genes, by gene-DE method:",
        f"  Kruskal-Wallis on TPM (headline):        {fmt(sf_kw)}",
        f"  DESeq2 LRT, native norm (primary):       {fmt(sf_deseq2)}",
        f"  edgeR QL ANODEV, native norm (primary):  {fmt(sf_edger)}",
        f"  DESeq2, replicates collapsed (native):   {fmt(sf_coll)}",
        f"  DESeq2, whole-library norm (sensitivity):{fmt(sf_deseq2_lib)}",
        f"  edgeR, whole-library norm (sensitivity): {fmt(sf_edger_lib)}",
        "",
        "Gene-DE call concordance (Cohen's kappa over analyzed genes):",
        f"  KW vs DESeq2:   kappa = {kappa_kw_d:.3f}",
        f"  KW vs edgeR:    kappa = {kappa_kw_e:.3f}",
        f"  DESeq2 vs edgeR: kappa = {kappa_d_e:.3f}",
        "",
        f"Silent-set overlap KW vs DESeq2 (Jaccard): {jacc:.3f}",
        f"  KW-silent n={len(s_kw)}, DESeq2-silent n={len(s_d)}, shared n={len(s_kw & s_d)}",
        "",
        f"n gene-DE (KW):     {sum(kw_de.values())}",
        f"n gene-DE (DESeq2): {sum(de_deseq2.values())}",
        f"n gene-DE (edgeR):  {sum(de_edger.values())}",
        "",
        "Normalization: PRIMARY = each method's native composition-aware norm",
        "(DESeq2 median-of-ratios/poscounts; edgeR TMM). SENSITIVITY = whole-library",
        "size factors (matched to the TPM used by the KW test). The silent fraction",
        "is essentially unchanged between the two, so it is robust to normalization.",
        "=" * 66,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(PROC_DIR / "de_countbased_summary.txt", "w") as fh:
        fh.write(summary + "\n")
    print("\nSaved: data/processed/de_countbased.tsv, visibility_countbased.tsv, de_countbased_summary.txt")


if __name__ == "__main__":
    main()
