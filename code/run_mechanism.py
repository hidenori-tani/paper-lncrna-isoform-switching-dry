"""run_mechanism.py — Structural basis of dominant-isoform switches.

For every dominant-isoform switch pair identified by run_functional.py
(data/processed/switch_exons.tsv), classify whether the two dominant isoforms
differ by an alternative transcription start site (alternative promoter / first
exon), an alternative transcription end site (alternative polyadenylation / last
exon), and/or a splice-junction change (exon skipping, intron retention,
alternative internal splice site). This addresses whether tissue-specific lncRNA
isoform switching differs by promoter/TSS choice versus splicing (annotation-based).

Run from project root (after run_landscape.py and run_functional.py):
    python code/run_mechanism.py

Outputs (data/processed/):
    switch_mechanism.tsv          — one row per dominant-switch pair with the three
                                    boolean flags + a priority category
    switch_mechanism_summary.txt  — headline fractions (pair-level and gene-level)
"""
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_functional import load_tx_exons       # reuse the GTF exon parser
from isoform_metrics import classify_switch_mechanism

GTF_PATH = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
PROC_DIR = _ROOT / "data/processed"
SWITCH_EXONS = PROC_DIR / "switch_exons.tsv"


def main():
    t0 = time.time()
    print("[1/3] Loading dominant-switch pairs ...")
    df = pd.read_csv(SWITCH_EXONS, sep="\t")
    print(f"  {len(df):,} dominant-switch tissue pairs")

    needed = set(df["dom_isoform_A"]) | set(df["dom_isoform_B"])
    print(f"  Parsing exon structures for {len(needed):,} dominant isoforms ...")
    tx_exons = load_tx_exons(str(GTF_PATH), needed)
    print(f"  Exon data loaded for {len(tx_exons):,} transcripts")

    print("[2/3] Classifying switch mechanism per pair ...")
    rows = []
    n_skip = 0
    for r in df.itertuples(index=False):
        A, B = r.dom_isoform_A, r.dom_isoform_B
        if A not in tx_exons or B not in tx_exons:
            n_skip += 1
            continue
        chromA, exA, strandA = tx_exons[A]
        chromB, exB, strandB = tx_exons[B]
        if strandA != strandB:        # within-gene isoforms should share strand
            n_skip += 1
            continue
        cls = classify_switch_mechanism(exA, exB, strandA)
        rows.append({
            "gene_id": r.gene_id,
            "tissueA": r.tissueA,
            "tissueB": r.tissueB,
            "dom_isoform_A": A,
            "dom_isoform_B": B,
            "alt_tss": cls["alt_tss"],
            "alt_tes": cls["alt_tes"],
            "alt_splicing": cls["alt_splicing"],
            "category": cls["category"],
            "multi": cls["multi"],
        })
    mech = pd.DataFrame(rows)
    mech.to_csv(PROC_DIR / "switch_mechanism.tsv", sep="\t", index=False)
    print(f"  Classified {len(mech):,} pairs ({n_skip:,} skipped: missing exon data / strand mismatch)")

    print("[3/3] Aggregating ...")
    n = len(mech)

    def pct(x):
        return 100.0 * x / n if n else float("nan")

    n_tss = int(mech["alt_tss"].sum())
    n_tes = int(mech["alt_tes"].sum())
    n_spl = int(mech["alt_splicing"].sum())
    n_multi = int(mech["multi"].sum())
    n_terminal = int((mech["alt_tss"] | mech["alt_tes"]).sum())
    n_splicing_only = int((mech["alt_splicing"] & ~mech["alt_tss"] & ~mech["alt_tes"]).sum())
    n_tss_only = int((mech["alt_tss"] & ~mech["alt_tes"] & ~mech["alt_splicing"]).sum())
    cat = mech["category"].value_counts().to_dict()

    # Gene-level: a gene "involves" a mechanism if any of its switch pairs do
    gg = mech.groupby("gene_id").agg(
        any_tss=("alt_tss", "any"),
        any_tes=("alt_tes", "any"),
        any_spl=("alt_splicing", "any"),
    )
    n_genes = len(gg)

    def gpct(x):
        return 100.0 * x / n_genes if n_genes else float("nan")

    g_tss = int(gg["any_tss"].sum())
    g_tes = int(gg["any_tes"].sum())
    g_spl = int(gg["any_spl"].sum())

    lines = [
        "=" * 64,
        "Switch mechanism summary — structural basis of dominant-isoform switches",
        "=" * 64,
        "",
        f"Dominant-switch tissue pairs classified: {n:,}  ({n_skip:,} skipped)",
        f"Genes with >=1 dominant-switch pair:     {n_genes:,}",
        "",
        "PAIR-LEVEL (n = %d pairs; flags are non-exclusive):" % n,
        f"  Alternative TSS / first exon (promoter):  {n_tss:,}  ({pct(n_tss):.1f}%)",
        f"  Splice-junction difference (any):         {n_spl:,}  ({pct(n_spl):.1f}%)",
        f"  Alternative TES / last exon (APA):        {n_tes:,}  ({pct(n_tes):.1f}%)",
        f"  Any terminal change (TSS or TES):         {n_terminal:,}  ({pct(n_terminal):.1f}%)",
        f"  Internal-splicing-only (no terminal chg): {n_splicing_only:,}  ({pct(n_splicing_only):.1f}%)",
        f"  Alt-TSS-only (no splicing/TES change):    {n_tss_only:,}  ({pct(n_tss_only):.1f}%)",
        f"  >1 change co-occurring (multi):           {n_multi:,}  ({pct(n_multi):.1f}%)",
        "",
        "PAIR-LEVEL mutually-exclusive category (priority TSS > splicing > TES):",
        f"  alt_TSS:            {cat.get('alt_TSS', 0):,}  ({pct(cat.get('alt_TSS', 0)):.1f}%)",
        f"  splice-junction (no alt-TSS): {cat.get('internal_splicing', 0):,}  ({pct(cat.get('internal_splicing', 0)):.1f}%)",
        f"  alt_TES:           {cat.get('alt_TES', 0):,}  ({pct(cat.get('alt_TES', 0)):.1f}%)",
        f"  identical:         {cat.get('identical', 0):,}  ({pct(cat.get('identical', 0)):.1f}%)",
        "",
        "GENE-LEVEL (n = %d genes with a dominant switch; >=1 pair involving):" % n_genes,
        f"  Alternative TSS / first exon (promoter):  {g_tss:,}  ({gpct(g_tss):.1f}%)",
        f"  Splice-junction difference (any):         {g_spl:,}  ({gpct(g_spl):.1f}%)",
        f"  Alternative TES / last exon (APA):        {g_tes:,}  ({gpct(g_tes):.1f}%)",
        "",
        f"Runtime: {time.time() - t0:.1f}s",
        "=" * 64,
    ]
    summary = "\n".join(lines)
    print()
    print(summary)
    with open(PROC_DIR / "switch_mechanism_summary.txt", "w") as fh:
        fh.write(summary + "\n")
    print(f"\nSaved: data/processed/switch_mechanism.tsv  and  switch_mechanism_summary.txt")

    # ── Figure 8: structural basis of dominant-isoform switches ──────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (publisher requirement)
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (matplotlib unavailable, skipping figure: {exc})")
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))

    labels = [
        "Alt. TSS / first exon\n(alternative promoter)",
        "Splice-junction\ndifference",
        "Alt. 3′ end\n(polyadenylation)",
        "Any terminal change\n(TSS or 3′)",
        ">1 change\n(coordinated)",
        "Internal splicing only",
    ]
    vals = [pct(n_tss), pct(n_spl), pct(n_tes), pct(n_terminal), pct(n_multi), pct(n_splicing_only)]
    ypos = range(len(labels))[::-1]
    bars = axA.barh(list(ypos), vals, color="#4C72B0")
    bars[-1].set_color("#C44E52")  # highlight "splicing only"
    axA.set_yticks(list(ypos))
    axA.set_yticklabels(labels, fontsize=8)
    axA.set_xlabel(f"% of dominant-switch pairs (n = {n:,})", fontsize=9)
    axA.set_xlim(0, 105)
    for y, v in zip(ypos, vals):
        axA.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8)
    axA.set_title("A  Mechanisms involved (non-exclusive)", loc="left", fontsize=10, fontweight="bold")

    cat_labels = ["Alt. TSS\n(promoter)", "Splice-junction\n(no alt-TSS)", "Alt. 3′ end\n(APA) only"]
    cat_vals = [pct(cat.get("alt_TSS", 0)), pct(cat.get("internal_splicing", 0)), pct(cat.get("alt_TES", 0))]
    axB.bar(cat_labels, cat_vals, color=["#4C72B0", "#55A868", "#8172B3"])
    axB.set_ylabel("% of dominant-switch pairs", fontsize=9)
    axB.set_ylim(0, 105)
    for i, v in enumerate(cat_vals):
        axB.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    axB.set_title("B  Primary category (priority TSS > splicing > 3′)", loc="left", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig_dir = _ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"Fig8_mechanism.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/Fig8_mechanism.png / .pdf")


if __name__ == "__main__":
    main()
