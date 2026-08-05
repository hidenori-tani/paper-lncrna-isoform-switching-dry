"""make_fig_countbased.py — Figure 9: count-based DE robustness of the silent fraction.

Every displayed number is recomputed from data/processed/de_countbased.tsv
(no hardcoded statistics), consistent with the rest of the figure pipeline.

Panel A: silent fraction among switch-significant lncRNAs under each gene-level
         DE method on the long-read cohort (Kruskal-Wallis, DESeq2, edgeR).
Panel B: number of analysed genes called gene-level DE by each method, showing
         the power difference that drives the silent-fraction collapse.

Run from project root (after run_de_countbased.py):
    python code/make_fig_countbased.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (publisher requirement)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
DE_TSV = _ROOT / "data/processed/de_countbased.tsv"
FIG_DIR = _ROOT / "figures"
SUB_FIG_DIR = _ROOT / "submission/genes_genomics/figures"

METHODS = [
    ("visibility_kw", "gene_de_kw", "Kruskal–Wallis\n(TPM, primary)", "#4C72B0"),
    ("visibility_deseq2", "gene_de_deseq2", "DESeq2\n(LRT)", "#C44E52"),
    ("visibility_edger", "gene_de_edger", "edgeR\n(QL ANODEV)", "#55A868"),
]


def _silent_fraction(df, vis_col):
    n_silent = int((df[vis_col] == "silent").sum())
    n_visible = int((df[vis_col] == "visible").sum())
    n_switch = n_silent + n_visible
    return n_silent, n_switch, (n_silent / n_switch if n_switch else float("nan"))


def main():
    df = pd.read_csv(DE_TSV, sep="\t")
    n_genes = len(df)

    fracs, de_counts, labels, colors = [], [], [], []
    silent_ns, switch_ns = [], []
    for vis_col, de_col, label, color in METHODS:
        n_sil, n_sw, frac = _silent_fraction(df, vis_col)
        fracs.append(frac * 100)
        silent_ns.append(n_sil)
        switch_ns.append(n_sw)
        de_counts.append(int(df[de_col].astype(bool).sum()))
        labels.append(label)
        colors.append(color)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 4.3))

    # ── Panel A: silent fraction by DE method ────────────────────────────────
    xA = range(len(labels))
    barsA = axA.bar(xA, fracs, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    for i, (b, frac, n_sil, n_sw) in enumerate(zip(barsA, fracs, silent_ns, switch_ns)):
        axA.text(b.get_x() + b.get_width() / 2, frac + 0.4,
                 f"{frac:.1f}%\n({n_sil}/{n_sw})", ha="center", va="bottom", fontsize=9)
    axA.set_xticks(list(xA))
    axA.set_xticklabels(labels, fontsize=9)
    axA.set_ylabel("Silent fraction among switch-significant lncRNAs (%)", fontsize=9.5)
    axA.set_ylim(0, max(fracs) * 1.28)
    axA.set_title("A  Silent fraction by gene-level DE method", fontsize=10.5, loc="left", fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)

    # ── Panel B: genes called DE by each method ──────────────────────────────
    xB = range(len(labels))
    barsB = axB.bar(xB, de_counts, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    for b, c in zip(barsB, de_counts):
        axB.text(b.get_x() + b.get_width() / 2, c + n_genes * 0.01,
                 f"{c}", ha="center", va="bottom", fontsize=9)
    axB.axhline(n_genes, color="grey", linestyle="--", linewidth=0.8)
    axB.text(-0.45, n_genes * 1.012, f"all analysed genes (n = {n_genes})",
             ha="left", va="bottom", fontsize=8, color="grey")
    axB.set_xticks(list(xB))
    axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylabel("lncRNA genes called gene-level DE across tissues", fontsize=9.5)
    axB.set_ylim(0, n_genes * 1.18)
    axB.set_title("B  Gene-level DE detections (long-read cohort)", fontsize=10.5, loc="left", fontweight="bold")
    axB.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"Fig9_countbased.{ext}", dpi=200, bbox_inches="tight")
    if SUB_FIG_DIR.exists():
        for ext in ("pdf", "png"):
            fig.savefig(SUB_FIG_DIR / f"Fig9_countbased.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Fig 9 silent fractions:",
          {lab.split(chr(10))[0]: f"{f:.1f}%" for lab, f in zip(labels, fracs)})
    print("Fig 9 DE counts:",
          {lab.split(chr(10))[0]: c for lab, c in zip(labels, de_counts)})
    print(f"Saved: figures/Fig9_countbased.pdf / .png (+ submission copy)")


if __name__ == "__main__":
    main()
