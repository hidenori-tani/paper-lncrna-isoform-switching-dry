"""make_fig_specification.py — the specification curve (new Fig 3).

The cohort, the annotation and the transcript-usage calls are identical for every
bar. The gene-level DE specification changes, and with it — for the pairwise bars —
the samples entering the contrast (~15 rather than 68) and, for the control bars,
the evaluable gene universe (416 genes with an alternative pair). All values are
read from data/processed/, never hard-coded.

Run from project root, after run_de_countbased.py, run_de_likeforlike.py and
run_de_aligned.py:
    python code/make_fig_specification.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
PROC = _ROOT / "data/processed"
FIGS = _ROOT / "figures"


def read_summary(path, pattern):
    txt = (PROC / path).read_text()
    m = re.search(pattern, txt)
    if not m:
        raise RuntimeError(f"pattern {pattern!r} not found in {path}")
    return m


def main():
    cb = pd.read_csv(PROC / "de_countbased.tsv", sep="\t")
    lf = pd.read_csv(PROC / "de_likeforlike.tsv", sep="\t")
    al = pd.read_csv(PROC / "de_aligned.tsv", sep="\t")

    ss = cb[cb["switch_sig"].astype(bool)]
    n_ss = len(ss)

    def frac(mask_de, n):
        n_sil = int((~mask_de).sum())
        return n_sil, n, 100.0 * n_sil / n

    lf_ss = lf[lf["switch_sig"].astype(bool)]
    rd = al[al["has_random"].astype(bool)]

    specs = [
        ("edgeR QL\nomnibus, n=68",   *frac(ss["gene_de_edger"].astype(bool), n_ss),   "#2c6fbb"),
        ("DESeq2 LRT\nomnibus, n=68", *frac(ss["gene_de_deseq2"].astype(bool), n_ss),  "#2c6fbb"),
        ("rank test, TPM\nomnibus, n=68", *frac(ss["gene_de_kw"].astype(bool), n_ss),      "#e08214"),
        ("rank test, TMM-CPM\nomnibus, n=68", *frac(lf_ss["de_rank_cpm"].astype(bool), len(lf_ss)), "#e08214"),
        ("DESeq2\naligned pair, n≈15",      *frac(rd["de_deseq2_aligned"].astype(bool), len(rd)), "#7b3294"),
        ("edgeR\naligned pair, n≈15",       *frac(rd["de_edger_aligned"].astype(bool), len(rd)),  "#7b3294"),
        ("DESeq2\nrandom pair, n≈15",       *frac(rd["de_deseq2_random"].astype(bool), len(rd)),  "#b3699e"),
        ("edgeR\nrandom pair, n≈15",        *frac(rd["de_edger_random"].astype(bool), len(rd)),   "#b3699e"),
        ("rank test\naligned pair, n≈15",   *frac(al["de_rank_aligned"].astype(bool), len(al)),   "#c46a1f"),
        ("rank test\nrandom pair, n≈15",    *frac(rd["de_rank_random"].astype(bool), len(rd)),    "#e8a35a"),
    ]
    specs.sort(key=lambda r: r[3])

    # The figure must fit a 174 mm single-column text width and stay legible at
    # final size, so the canvas IS the printed size and point sizes are literal.
    fig, ax = plt.subplots(figsize=(6.85, 4.75))       # 174 x 121 mm
    labels = [s[0] for s in specs]
    pcts = [s[3] for s in specs]
    cols = [s[4] for s in specs]
    bars = ax.bar(range(len(specs)), pcts, color=cols, edgecolor="black", linewidth=0.5)
    for i, (b, s) in enumerate(zip(bars, specs)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7,
                f"{s[3]:.1f}%\n({s[1]}/{s[2]})", ha="center", va="bottom", fontsize=5.4)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(labels, fontsize=5.6, rotation=34, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("DTU+/DGE− fraction among\nswitch-significant lncRNAs (%)", fontsize=7.5)
    ax.set_ylim(0, max(pcts) * 1.30)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title("The reported blind spot is a property of the gene-level DE specification\n"
                 "(same cohort and transcript-usage calls; contrast and sample count vary as shown)",
                 fontsize=7.8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6) for c in
               ("#2c6fbb", "#e08214", "#7b3294", "#b3699e", "#c46a1f", "#e8a35a")]
    ax.legend(handles,
              ["negative-binomial GLM, omnibus (9 tissues)",
               "rank test, omnibus (9 tissues)",
               "NB GLM, pair aligned to the switch",
               "NB GLM, random pair (matched-size control)",
               "rank test, pair aligned to the switch",
               "rank test, random pair (control)"],
              fontsize=5.4, loc="upper left", framealpha=0.95)

    # The fold-range badge must be computed WITHIN a single gene universe, otherwise
    # it silently mixes the 463-locus specifications with the 416-locus control and
    # contradicts the text (which reports 122-fold on all 463 loci).
    n_full = max(s[2] for s in specs)
    same_universe = [s[3] for s in specs if s[2] == n_full]
    lo, hi = min(same_universe), max(same_universe)
    # placed in the empty region above the low omnibus bars, clear of every bar
    ax.annotate(f"{hi / lo:.0f}-fold\n(same {n_full} loci)", xy=(0.30, 0.74),
                xycoords="axes fraction", ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="grey", lw=0.5))

    fig.subplots_adjust(bottom=0.26)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"Fig3_specification.{ext}", dpi=300)
    print(f"wrote {FIGS}/Fig3_specification.pdf/.png")
    print(f"badge range (same {n_full} loci): {lo:.1f}% to {hi:.1f}%  ({hi/lo:.0f}-fold)")
    print(f"overall across universes: {min(pcts):.1f}% to {max(pcts):.1f}%")
    for s in specs:
        print(f"  {s[0]:44s} {s[1]:4d}/{s[2]:<4d} {s[3]:5.1f}%")


if __name__ == "__main__":
    main()
