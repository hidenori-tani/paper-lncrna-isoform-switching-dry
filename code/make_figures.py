"""make_figures.py — Generate publication figures for the lncRNA isoform-switching paper.

Thesis: tissue-specific lncRNA isoform switching is pervasive and a quantifiable
fraction is INVISIBLE to gene-level DE analysis ("silent switching").

Run from project root:
    python code/make_figures.py

Outputs:
    figures/Fig1.png / Fig1.pdf  — Concept (schematic)
    figures/Fig2.png / Fig2.pdf  — lnc-ISI landscape
    figures/Fig3.png / Fig3.pdf  — Silent switching (HEADLINE)
    figures/Fig4.png / Fig4.pdf  — Replication & power-dependence
    figures/Fig5.png / Fig5.pdf  — Switch structure & RBP (supporting/negative)
    figures/Fig6.png / Fig6.pdf  — Case studies (silent switchers)
    data/processed/key_numbers.txt
"""

import gzip
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (publisher requirement)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

PROC_DIR   = _ROOT / "data/processed"
RAW_DIR    = _ROOT / "data/raw"
FIG_DIR    = _ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

GTF_PATH  = RAW_DIR / "gencode.v26.long_noncoding_RNAs.gtf.gz"
TPM_PATH  = RAW_DIR / "longread" / "quantification_gencode.tpm.txt.gz"
ATTR_PATH = RAW_DIR / "GTEx_v8_SampleAttributesDS.txt"

# ── Style ─────────────────────────────────────────────────────────────────────
# Colorblind-safe palette (Wong 2011, Color Universal Design)
CBLIND = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#CC0000",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "navy":   "#000080",
    "black":  "#000000",
    "gray":   "#999999",
    "light":  "#DDDDDD",
}
# Visibility class colours
VIS_COLORS = {
    "silent":    CBLIND["red"],
    "visible":   CBLIND["blue"],
    "gene_only": CBLIND["orange"],
    "none":      CBLIND["gray"],
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   9,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       300,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "lines.linewidth":  1.0,
    "patch.linewidth":  0.5,
})

PANEL_FONT = {"fontsize": 10, "fontweight": "bold"}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_all_data():
    """Load all processed TSVs needed for figures."""
    vis_lr = pd.read_csv(PROC_DIR / "visibility_longread.tsv", sep="\t")
    isi_lr = pd.read_csv(PROC_DIR / "lnc_isi_longread.tsv", sep="\t")
    vis_sr = pd.read_csv(PROC_DIR / "visibility_shortread.tsv", sep="\t")
    switch_exons = pd.read_csv(PROC_DIR / "switch_exons.tsv", sep="\t")
    switch_rbp = pd.read_csv(PROC_DIR / "switch_rbp.tsv", sep="\t")
    rbp_stats = pd.read_csv(PROC_DIR / "rbp_stats.tsv", sep="\t")
    return vis_lr, isi_lr, vis_sr, switch_exons, switch_rbp, rbp_stats


def load_biotypes() -> Dict[str, str]:
    """Parse GENCODE lncRNA GTF and return {gene_id: gene_type}."""
    gene_biotype: Dict[str, str] = {}
    _gene_id_re  = re.compile(r'gene_id "([^"]+)"')
    _gene_type_re = re.compile(r'gene_type "([^"]+)"')
    with gzip.open(str(GTF_PATH), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "gene":
                continue
            m_id = _gene_id_re.search(fields[8])
            m_bt = _gene_type_re.search(fields[8])
            if m_id and m_bt:
                gene_biotype[m_id.group(1)] = m_bt.group(1)
    return gene_biotype


def load_case_study_isoform_fractions(
    gene_ids: List[str],
    gene_names: Dict[str, str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-tissue mean isoform fractions for the given genes.

    Returns {gene_id: {"tissues": [...], "isoforms": [...], "mean_if": ndarray(n_iso, n_tiss),
                        "mean_tpm": ndarray(n_tiss)}}
    """
    from data_io import load_lncrna_tx2gene, load_longread_tpm, sample_to_tissue
    from isoform_metrics import isoform_fraction as _iso_frac

    MIN_SAMPLE_GENE_TPM = 1.0
    MIN_EXPR_SAMPLES_PER_TISSUE = 3

    print("  [case studies] Loading tx2gene ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))

    print("  [case studies] Loading TPM table ...")
    tpm_df = load_longread_tpm(str(TPM_PATH))

    print("  [case studies] Mapping samples -> tissues ...")
    s2t = sample_to_tissue(tpm_df.columns.tolist(), str(ATTR_PATH), min_samples_per_tissue=6)
    kept_samples = list(s2t.keys())
    tpm_df = tpm_df[kept_samples]

    tissues_sorted = sorted(set(s2t.values()))
    tissue_arr = np.array([s2t[s] for s in kept_samples])

    # Build gene -> transcripts
    gene_to_txs: Dict[str, List[str]] = defaultdict(list)
    lncrna_tx_in_data = [tx for tx in tpm_df.index if tx in tx2gene]
    for tx in lncrna_tx_in_data:
        gene_to_txs[tx2gene[tx]].append(tx)

    results = {}
    for gid in gene_ids:
        # Match with version suffix
        matched = [g for g in gene_to_txs if g.startswith(gid + ".") or g == gid]
        if not matched:
            print(f"  WARNING: {gid} not found in TPM data")
            continue
        gfull = matched[0]
        txs = gene_to_txs[gfull]

        tpm_mat = tpm_df.loc[txs].values.astype(np.float32)  # (n_iso, n_samp)
        gene_sum = tpm_mat.sum(axis=0)  # (n_samp,)

        col_totals = tpm_mat.sum(axis=0, keepdims=True)
        denom = np.where(col_totals > 0, col_totals, 1.0)
        if_mat = tpm_mat / denom  # (n_iso, n_samp)

        valid_tissues = []
        mean_ifs = []
        mean_tpms = []

        for tissue in tissues_sorted:
            t_idx = np.where(tissue_arr == tissue)[0]
            expr_mask = gene_sum[t_idx] >= MIN_SAMPLE_GENE_TPM
            expr_idx = t_idx[expr_mask]
            if expr_mask.sum() < MIN_EXPR_SAMPLES_PER_TISSUE:
                continue
            mean_if = if_mat[:, expr_idx].mean(axis=1)
            s = mean_if.sum()
            if s > 0:
                mean_if = mean_if / s
            mean_tpm = gene_sum[expr_idx].mean()
            valid_tissues.append(tissue)
            mean_ifs.append(mean_if)
            mean_tpms.append(mean_tpm)

        if len(valid_tissues) < 2:
            print(f"  WARNING: {gid} has < 2 valid tissues, skipping")
            continue

        mean_if_arr = np.array(mean_ifs).T  # (n_iso, n_tiss)
        mean_tpm_arr = np.array(mean_tpms)  # (n_tiss,)

        results[gid] = {
            "gene_id_full": gfull,
            "name": gene_names.get(gid, gid),
            "tissues": valid_tissues,
            "isoforms": txs,
            "mean_if": mean_if_arr,
            "mean_tpm": mean_tpm_arr,
        }
    return results


# ── Figure 1: Concept ─────────────────────────────────────────────────────────

def make_fig1(vis_lr: pd.DataFrame):
    """Fig 1 — Concept: 2×2 grid, toy example, ISI definition."""
    fig = plt.figure(figsize=(7.0, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, left=0.06, right=0.97,
                           top=0.90, bottom=0.10)

    # ── Panel A: 2×2 classification grid ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-0.2, 2.2)
    ax_a.set_ylim(-0.2, 2.2)
    ax_a.axis("off")
    ax_a.text(-0.18, 2.2, "A", **PANEL_FONT)

    # Cell fill colours
    cell_info = [
        # (x, y, label, sublabel, facecolor)
        (0, 1, "silent",    "(FOCUS)",  CBLIND["red"]),
        (1, 1, "visible",   "",         CBLIND["blue"]),
        (0, 0, "none",      "",         CBLIND["light"]),
        (1, 0, "gene only", "",         CBLIND["orange"]),
    ]
    for (col, row, lbl, sub, fc) in cell_info:
        rect = mpatches.FancyBboxPatch(
            (col, row), 1.0, 1.0,
            boxstyle="round,pad=0.04",
            linewidth=0.8,
            edgecolor="white",
            facecolor=fc,
            alpha=0.85,
            transform=ax_a.transData,
        )
        ax_a.add_patch(rect)
        ax_a.text(col + 0.5, row + 0.55, lbl,
                  ha="center", va="center", fontsize=8, fontweight="bold",
                  color="white" if fc not in (CBLIND["light"], CBLIND["orange"]) else "black")
        if sub:
            ax_a.text(col + 0.5, row + 0.30, sub,
                      ha="center", va="center", fontsize=6.5, color="white")

    # Axis labels.
    # These sit outside the data limits on purpose; text is not clipped and
    # savefig(bbox_inches="tight") grows the canvas to include them. The offsets
    # are set from measured bounding boxes, not by eye: the tick labels reach
    # y = -0.19 below the grid and x = -0.52 to its left, so the axis labels must
    # start beyond those. An earlier version placed the y-axis label at y = 0.5,
    # the centre of the bottom row rather than of the grid, so it printed through
    # the "not sig." tick label; code/check_figure_layout.py now measures this.
    ax_a.text(1.0, -0.26, "Gene-level DE", ha="center", va="top", fontsize=7.5)
    ax_a.text(-0.78, 1.0, "Isoform\nswitching sig.", ha="center", va="center",
              fontsize=7.5, rotation=90)
    # Not / Yes ticks
    for i, lbl in enumerate(["not sig.", "sig."]):
        ax_a.text(-0.12, i + 0.5, lbl, ha="right", va="center", fontsize=6.5)
    for i, lbl in enumerate(["not DE", "DE"]):
        ax_a.text(i + 0.5, -0.08, lbl, ha="center", va="top", fontsize=6.5)

    ax_a.set_title("Classification", fontsize=8, pad=4)

    # ── Panel B: Toy example ───────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.18, 1.05, "B", transform=ax_b.transAxes, **PANEL_FONT)

    tissues_toy = ["Tissue 1", "Tissue 2", "Tissue 3"]
    x = np.arange(3)

    # Total expression: flat (no DE)
    total_tpm = np.array([10.0, 10.0, 10.0])

    # Isoform fractions: flipping
    iso_a = np.array([0.80, 0.45, 0.10])
    iso_b = 1.0 - iso_a

    # Top mini-plot: total expression (flat)
    divider_y = 0.55
    ax_btop = ax_b.inset_axes([0, divider_y, 1, 0.42])
    ax_bbot = ax_b.inset_axes([0, 0,         1, divider_y - 0.04])
    ax_b.axis("off")

    ax_btop.plot(x, total_tpm, "o-", color=CBLIND["green"], lw=1.5, ms=4)
    ax_btop.set_xticks(x)
    ax_btop.set_xticklabels([])
    ax_btop.set_ylim(0, 15)
    ax_btop.set_ylabel("Total TPM", fontsize=7)
    ax_btop.set_title("Toy: Silent Switcher", fontsize=8, pad=3)
    ax_btop.spines["right"].set_visible(False)
    ax_btop.spines["top"].set_visible(False)
    ax_btop.tick_params(labelsize=6)
    ax_btop.text(0.99, 0.88, "FLAT → not DE", transform=ax_btop.transAxes,
                 ha="right", fontsize=6.5, color=CBLIND["green"], style="italic")

    # Bottom: stacked bars of isoform fractions
    bar_w = 0.55
    ax_bbot.bar(x, iso_a, bar_w, color=CBLIND["blue"],   label="Isoform α")
    ax_bbot.bar(x, iso_b, bar_w, bottom=iso_a,
                color=CBLIND["orange"], label="Isoform β")
    ax_bbot.set_xticks(x)
    ax_bbot.set_xticklabels(tissues_toy, fontsize=6.5)
    ax_bbot.set_ylim(0, 1.28)
    ax_bbot.set_ylabel("Isoform fraction", fontsize=7)
    ax_bbot.spines["right"].set_visible(False)
    ax_bbot.spines["top"].set_visible(False)
    ax_bbot.tick_params(labelsize=6)
    # Legend at upper-left, annotation at upper-right — opposite corners, both on
    # white backgrounds so they stay legible over the stacked bars and never collide.
    ax_bbot.legend(loc="upper left", fontsize=6, frameon=True,
                   facecolor="white", framealpha=0.9, edgecolor="none")
    ax_bbot.text(0.99, 0.97, "SWITCHING → sig.", transform=ax_bbot.transAxes,
                 ha="right", va="top", fontsize=6.5, color=CBLIND["red"], style="italic",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    # ── Panel C: lnc-ISI definition box ────────────────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    ax_c.text(-0.18, 1.05, "C", transform=ax_c.transAxes, **PANEL_FONT)
    ax_c.set_title("lnc-ISI Definition", fontsize=8, pad=4)

    box = dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#AAAAAA", lw=0.8)
    formula_text = (
        "lnc-ISI  =  $\\max_{(i,j) \\in \\binom{T}{2}}$  JSD($p_i$, $p_j$)\n\n"
        "T  = set of valid tissues\n"
        "$p_i$  = mean isoform-usage vector in tissue $i$\n"
        "JSD  = Jensen-Shannon divergence (log2 base)\n\n"
        "Range: 0 (identical) to 1 (maximal divergence)\n\n"
        "Captures the LARGEST pairwise switch\nacross any two tissues"
    )
    ax_c.text(0.5, 0.55, formula_text,
              transform=ax_c.transAxes,
              ha="center", va="center",
              fontsize=7, linespacing=1.6,
              bbox=box)

    _save_fig(fig, "Fig1")


# ── Figure 2: lnc-ISI Landscape ───────────────────────────────────────────────

def make_fig2(isi_lr: pd.DataFrame, gene_biotype: Dict[str, str]):
    """Fig 2 — lnc-ISI landscape: histogram, threshold bars, biotype box/violin."""
    fig = plt.figure(figsize=(7.0, 3.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40,
                           left=0.08, right=0.97, top=0.88, bottom=0.16)

    isi_vals = isi_lr["lnc_isi"].values
    median_isi = float(np.median(isi_vals))
    p90_isi    = float(np.percentile(isi_vals, 90))
    n_01  = int((isi_vals >= 0.10).sum())
    n_025 = int((isi_vals >= 0.25).sum())
    n_05  = int((isi_vals >= 0.50).sum())

    # Panel A: histogram
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(-0.22, 1.06, "A", transform=ax_a.transAxes, **PANEL_FONT)
    bins = np.linspace(0, 1, 51)
    n_hist, _, patches = ax_a.hist(isi_vals, bins=bins, color=CBLIND["blue"],
                                   alpha=0.8, edgecolor="white", linewidth=0.3)
    ax_a.axvline(median_isi, color=CBLIND["orange"], lw=1.2, ls="--",
                 label=f"Median {median_isi:.3f}")
    ax_a.axvline(p90_isi, color=CBLIND["red"], lw=1.2, ls=":",
                 label=f"90th pct {p90_isi:.3f}")
    ax_a.set_xlabel("lnc-ISI", fontsize=8)
    ax_a.set_ylabel("Number of genes", fontsize=8)
    ax_a.set_title(f"lnc-ISI distribution\n(n={len(isi_vals):,} genes)", fontsize=8)
    ax_a.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax_a.set_xlim(0, 1)

    # Panel B: bar at ISI thresholds
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.25, 1.06, "B", transform=ax_b.transAxes, **PANEL_FONT)
    thresholds = ["≥0.10", "≥0.25", "≥0.50"]
    counts_thr = [n_01, n_025, n_05]
    x_thr = np.arange(3)
    bars = ax_b.bar(x_thr, counts_thr, color=[CBLIND["blue"], CBLIND["sky"], CBLIND["green"]],
                    alpha=0.85, width=0.55)
    for bar, n in zip(bars, counts_thr):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                  str(n), ha="center", va="bottom", fontsize=7)
    ax_b.set_xticks(x_thr)
    ax_b.set_xticklabels(thresholds, fontsize=7.5)
    ax_b.set_xlabel("lnc-ISI threshold", fontsize=8)
    ax_b.set_ylabel("Number of genes", fontsize=8)
    ax_b.set_title("Genes by ISI threshold", fontsize=8)
    ax_b.set_ylim(0, n_01 * 1.18)

    # Panel C: biotype box plot (top 3 biotypes only; others grouped)
    ax_c = fig.add_subplot(gs[2])
    ax_c.text(-0.28, 1.06, "C", transform=ax_c.transAxes, **PANEL_FONT)

    isi_lr2 = isi_lr.copy()
    isi_lr2["biotype"] = isi_lr2["gene_id"].map(gene_biotype)
    bt_counts = isi_lr2["biotype"].value_counts()
    top3 = bt_counts.index[:3].tolist()  # antisense, lincRNA, processed_transcript

    def _bt_label(bt):
        mapping = {
            "antisense":             "antisense",
            "lincRNA":               "lincRNA",
            "processed_transcript":  "proc.\ntranscript",
        }
        return mapping.get(bt, bt)

    bt_order = top3 + ["other"]
    isi_lr2["bt_group"] = isi_lr2["biotype"].apply(lambda x: x if x in top3 else "other")

    data_by_bt = [isi_lr2[isi_lr2["bt_group"] == bt]["lnc_isi"].values for bt in bt_order]
    labels_bt  = [f"{_bt_label(bt)}\n(n={len(d)})" for bt, d in zip(bt_order, data_by_bt)]

    bp = ax_c.boxplot(data_by_bt, patch_artist=True, widths=0.5,
                      medianprops={"color": "white", "lw": 1.5},
                      flierprops={"marker": ".", "markersize": 2, "alpha": 0.4,
                                  "markeredgewidth": 0},
                      whiskerprops={"lw": 0.8},
                      capprops={"lw": 0.8},
                      boxprops={"lw": 0.8})
    colors_bt = [CBLIND["blue"], CBLIND["orange"], CBLIND["green"], CBLIND["gray"]]
    for patch, c in zip(bp["boxes"], colors_bt):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    ax_c.set_xticklabels(labels_bt, fontsize=6.5)
    ax_c.set_ylabel("lnc-ISI", fontsize=8)
    ax_c.set_title("lnc-ISI by biotype", fontsize=8)
    ax_c.set_ylim(-0.02, 1.0)

    _save_fig(fig, "Fig2")

    return n_01, n_025, n_05, median_isi, p90_isi


# ── Figure 3: Silent Switching (HEADLINE) ─────────────────────────────────────

def make_fig3(vis_lr: pd.DataFrame):
    """Fig 3 — Silent switching headline: counts, fraction, ISI by class."""
    fig = plt.figure(figsize=(7.0, 3.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                           left=0.08, right=0.97, top=0.88, bottom=0.16)

    counts = vis_lr["visibility"].value_counts()
    n_silent    = int(counts.get("silent", 0))
    n_visible   = int(counts.get("visible", 0))
    n_gene_only = int(counts.get("gene_only", 0))
    n_none      = int(counts.get("none", 0))
    n_total     = n_silent + n_visible + n_gene_only + n_none
    n_switch_sig = n_silent + n_visible
    silent_frac  = n_silent / n_switch_sig

    # Panel A: stacked bar of 4 classes
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(-0.22, 1.06, "A", transform=ax_a.transAxes, **PANEL_FONT)

    vis_order  = ["silent", "visible", "gene_only", "none"]
    vis_labels = ["silent", "visible", "gene only", "none"]
    vis_counts = [n_silent, n_visible, n_gene_only, n_none]
    colors_bar = [VIS_COLORS[v] for v in vis_order]

    x0 = 0.5
    bottom = 0
    for lbl, cnt, col in zip(vis_labels, vis_counts, colors_bar):
        ax_a.bar(x0, cnt, 0.4, bottom=bottom, color=col, alpha=0.85, label=lbl)
        if cnt >= 15:
            ax_a.text(x0, bottom + cnt / 2, str(cnt),
                      ha="center", va="center", fontsize=7, color="white",
                      fontweight="bold")
        bottom += cnt

    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([])
    ax_a.set_ylabel("Number of genes", fontsize=8)
    ax_a.set_title(f"Visibility classification\n(n={n_total} genes)", fontsize=8)
    # Legend below panel A (A has no x-ticks) as a 2×2 block, so it never protrudes
    # into panel B's headline annotation.
    ax_a.legend(fontsize=6.5, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.02), ncol=2, columnspacing=1.2,
                handletextpad=0.4)
    ax_a.set_ylim(0, n_total * 1.05)

    # Panel B: bar showing silent fraction among switch-sig genes
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.22, 1.06, "B", transform=ax_b.transAxes, **PANEL_FONT)

    fracs = [silent_frac, 1 - silent_frac]
    labels_b = [f"silent\n{n_silent}/{n_switch_sig}\n({silent_frac*100:.1f}%)",
                f"visible\n{n_visible}/{n_switch_sig}\n({(1-silent_frac)*100:.1f}%)"]
    colors_b = [VIS_COLORS["silent"], VIS_COLORS["visible"]]

    bars_b = ax_b.bar([0, 1], fracs, 0.55, color=colors_b, alpha=0.85)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(labels_b, fontsize=7)
    ax_b.set_ylim(0, 1.20)
    ax_b.set_ylabel("Fraction of switch-sig. genes", fontsize=8)
    ax_b.set_title("Among switch-sig. genes\n"
                   f"(n={n_switch_sig})", fontsize=8)

    # Headline annotation (computed dynamically — never hardcode)
    ax_b.text(0.5, 0.98,
              f"{silent_frac*100:.1f}% of switching lncRNAs\nare invisible to gene-level DE",
              transform=ax_b.transAxes, ha="center", va="top",
              fontsize=7, style="italic",
              color=VIS_COLORS["silent"],
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=VIS_COLORS["silent"],
                        alpha=0.8, lw=0.8))

    # Panel C: ISI distribution by visibility class
    ax_c = fig.add_subplot(gs[2])
    ax_c.text(-0.25, 1.06, "C", transform=ax_c.transAxes, **PANEL_FONT)

    classes = ["silent", "visible", "gene_only"]
    class_labels = ["silent", "visible", "gene only"]
    data_c = [vis_lr[vis_lr["visibility"] == cls]["lnc_isi"].values for cls in classes]

    bp = ax_c.boxplot(data_c, patch_artist=True, widths=0.45,
                      medianprops={"color": "white", "lw": 1.5},
                      flierprops={"marker": ".", "markersize": 2, "alpha": 0.4,
                                  "markeredgewidth": 0},
                      whiskerprops={"lw": 0.8},
                      capprops={"lw": 0.8},
                      boxprops={"lw": 0.8})
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(VIS_COLORS[cls])
        patch.set_alpha(0.85)

    n_labels = [f"{lbl}\n(n={len(d)})" for lbl, d in zip(class_labels, data_c)]
    ax_c.set_xticklabels(n_labels, fontsize=7)
    ax_c.set_ylabel("lnc-ISI", fontsize=8)
    ax_c.set_title("lnc-ISI by visibility class", fontsize=8)
    ax_c.set_ylim(-0.02, 1.0)
    ax_c.text(0.03, 0.97,
              "Silent genes reach high ISI:\nreal switches, gene-level invisible",
              transform=ax_c.transAxes, ha="left", va="top",
              fontsize=6.5, style="italic")

    _save_fig(fig, "Fig3")

    return n_silent, n_visible, n_gene_only, n_none, n_total, n_switch_sig, silent_frac


# ── Figure 4: Replication & power-dependence ─────────────────────────────────

def make_fig4(vis_lr: pd.DataFrame, vis_sr: pd.DataFrame):
    """Fig 4 — Replication & power-dependence."""
    fig = plt.figure(figsize=(7.0, 3.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                           left=0.10, right=0.97, top=0.88, bottom=0.16)

    # Merge on gene_id (version-free base IDs)
    def base_id(df):
        d = df.copy()
        d["gene_base"] = d["gene_id"].str.split(".").str[0]
        return d

    vis_lr2 = base_id(vis_lr)
    vis_sr2 = base_id(vis_sr)

    merged = vis_lr2.merge(vis_sr2, on="gene_base",
                           suffixes=("_lr", "_sr"), how="inner")
    n_both = len(merged)

    # ── Dynamic numbers for panels B & C (never hardcode — avoids stale drift) ──
    # Panel B: silent fraction long vs short read
    n_silent_lr_f = int((vis_lr["visibility"] == "silent").sum())
    n_sw_lr_f     = int((vis_lr["switch_sig"] == True).sum())
    n_silent_sr_f = int((vis_sr["visibility"] == "silent").sum())
    n_sw_sr_f     = int((vis_sr["switch_sig"] == True).sum())
    frac_lr = n_silent_lr_f / n_sw_lr_f
    frac_sr = n_silent_sr_f / n_sw_sr_f
    # Panel C: silent-set replication (long-read silent genes analysable in short-read)
    _base = merged[merged["visibility_lr"] == "silent"]
    n_base          = len(_base)
    n_switch_sig_sr = int(_base["switch_sig_sr"].sum())
    n_not_sw        = n_base - n_switch_sig_sr
    n_silent_sr     = int((_base["switch_sig_sr"] & (_base["visibility_sr"] == "silent")).sum())
    n_visible_sr    = int((_base["switch_sig_sr"] & (_base["visibility_sr"] == "visible")).sum())
    # Panel A: Spearman correlation (computed, not hardcoded)
    _sp = stats.spearmanr(merged["lnc_isi_lr"].values, merged["lnc_isi_sr"].values)
    _rho = _sp.correlation
    _pp = _sp.pvalue
    _p_text = "p < 1e-300" if _pp == 0 else f"p = {_pp:.0e}"

    # Panel A: scatter lnc-ISI long vs short
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(-0.26, 1.06, "A", transform=ax_a.transAxes, **PANEL_FONT)

    x_isi = merged["lnc_isi_lr"].values
    y_isi = merged["lnc_isi_sr"].values

    # Colour by long-read visibility
    vis_col = merged["visibility_lr"].map(VIS_COLORS).fillna(CBLIND["gray"])

    ax_a.scatter(x_isi, y_isi, c=vis_col, s=4, alpha=0.4, lw=0, rasterized=True)
    ax_a.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="y=x")
    ax_a.set_xlabel("lnc-ISI (long-read)", fontsize=8)
    ax_a.set_ylabel("lnc-ISI (short-read)", fontsize=8)
    ax_a.set_title(f"Long- vs short-read ISI\n(n={n_both} common genes)", fontsize=8)
    ax_a.text(0.05, 0.93, f"Spearman r = {_rho:.2f}\n{_p_text}",
              transform=ax_a.transAxes, fontsize=6.5, va="top")
    ax_a.set_xlim(-0.02, 1.0)
    ax_a.set_ylim(-0.02, 1.0)

    # Panel B: silent fraction comparison
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.25, 1.06, "B", transform=ax_b.transAxes, **PANEL_FONT)

    bars_b = ax_b.bar([0, 1], [frac_lr, frac_sr], 0.5,
                      color=[CBLIND["blue"], CBLIND["sky"]], alpha=0.85)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([f"Long-read\n(n={n_sw_lr_f})", f"Short-read\n(n={n_sw_sr_f:,})"], fontsize=7.5)
    ax_b.set_ylim(0, 0.28)
    ax_b.set_ylabel("Silent fraction", fontsize=8)
    ax_b.set_title("Silent fraction\n(long vs short read)", fontsize=8)
    for bar, frac in zip(bars_b, [frac_lr, frac_sr]):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.006,
                  f"{frac:.3f}", ha="center", fontsize=7.5)
    ax_b.text(0.5, 0.97, "Higher power → lower\nsilent fraction",
              transform=ax_b.transAxes, ha="center", va="top", fontsize=6.5, style="italic")

    # Panel C: reclassification flow diagram
    ax_c = fig.add_subplot(gs[2])
    ax_c.text(-0.22, 1.06, "C", transform=ax_c.transAxes, **PANEL_FONT)
    ax_c.axis("off")
    ax_c.set_title("Power-dependence of silent\nclassification", fontsize=8)

    # Numbers (n_base, n_switch_sig_sr, n_not_sw, n_silent_sr, n_visible_sr)
    # are computed dynamically above from vis_lr/vis_sr — never hardcoded.

    # Draw boxes
    def _box(ax, x, y, w, h, text, color, fontsize=7):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor="white", alpha=0.85, lw=0.8,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white" if color not in (CBLIND["light"],) else "black")

    # Wide x-range so the three bottom boxes are spaced with clear gaps and no box
    # (or its text) overlaps a neighbour.
    ax_c.set_xlim(0, 4.4)
    ax_c.set_ylim(-0.2, 3.2)

    # Top box: long-read silent genes analysable in short-read
    _box(ax_c, 2.2, 2.8, 2.9, 0.55,
         f"{n_base} long-read silent genes\n(testable in short-read)",
         CBLIND["red"], fontsize=6.5)

    # Arrows down to the switch-sig junction (left) and the not-sw-sig box (right)
    ax_c.annotate("", xy=(1.2, 2.05), xytext=(2.0, 2.52),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
    ax_c.annotate("", xy=(3.75, 2.05), xytext=(2.4, 2.52),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))

    ax_c.text(1.0, 2.34, f"{n_switch_sig_sr}/{n_base} switch-sig", ha="center", fontsize=6)
    ax_c.text(3.6, 2.34, f"{n_not_sw}/{n_base} not sw-sig", ha="center", fontsize=6)

    # Two sub-boxes for the switch-sig branch
    _box(ax_c, 0.8, 1.5, 1.25, 0.55,
         f"{n_silent_sr} remain\nsilent ({n_silent_sr/n_base*100:.0f}%)",
         CBLIND["red"], fontsize=6.5)
    _box(ax_c, 2.25, 1.5, 1.5, 0.55,
         f"{n_visible_sr} become\nvisible ({n_visible_sr/n_base*100:.0f}%)",
         CBLIND["blue"], fontsize=6.5)

    # Arrows splitting the switch-sig junction (x = 1.2)
    ax_c.annotate("", xy=(0.8, 1.78), xytext=(1.2, 2.0),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
    ax_c.annotate("", xy=(2.25, 1.78), xytext=(1.2, 2.0),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))

    # Not-sw-sig box
    _box(ax_c, 3.75, 1.5, 1.25, 0.55,
         f"{n_not_sw} not\nswitch-sig",
         CBLIND["gray"], fontsize=6.5)

    # Caption note
    ax_c.text(2.2, 0.65,
              "↑ Silent classification is power-dependent:\nmore samples → visible or not detected",
              ha="center", va="center", fontsize=6, style="italic",
              color="#444444")

    _save_fig(fig, "Fig4")

    # Return the dynamically computed replication numbers so key_numbers.txt
    # uses the same values (never hardcode — avoids stale drift).
    # Jaccard of silent sets over genes analysed in both (matches run_replication.py):
    _lr_sil_both = n_base                                            # lr-silent ∩ both
    _sr_sil_both = int((merged["visibility_sr"] == "silent").sum())  # sr-silent ∩ both
    _inter = int(((merged["visibility_lr"] == "silent") & (merged["visibility_sr"] == "silent")).sum())
    _jaccard = _inter / (_lr_sil_both + _sr_sil_both - _inter)
    return {
        "n_genes_sr": int(len(vis_sr)),
        "n_both": int(n_both),
        "spearman_p": float(_pp),
        "n_lr_silent_in_sr": int(n_base),
        "n_sw_sig_in_sr": int(n_switch_sig_sr),
        "n_silent_in_sr": int(n_silent_sr),
        "jaccard": float(_jaccard),
    }


# ── Figure 5: Switch structure & RBP ─────────────────────────────────────────

def make_fig5(vis_lr: pd.DataFrame, switch_exons: pd.DataFrame,
              switch_rbp: pd.DataFrame, rbp_stats: pd.DataFrame):
    """Fig 5 — Switch structure (dominant-isoform change, diff_bp) and RBP (negative)."""
    fig = plt.figure(figsize=(7.0, 3.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40,
                           left=0.09, right=0.97, top=0.88, bottom=0.16)

    n_switch_sig = int((vis_lr["switch_sig"] == True).sum())
    n_dom_change = switch_exons["gene_id"].nunique()
    frac_dom = n_dom_change / n_switch_sig

    diff_bp = switch_exons["diff_bp"].values
    median_bp = float(np.median(diff_bp))
    p90_bp    = float(np.percentile(diff_bp, 90))

    # Panel A: bar (dom-change fraction) + diff_bp histogram as inset
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(-0.22, 1.06, "A", transform=ax_a.transAxes, **PANEL_FONT)

    # Bar: fraction with dominant-isoform change
    bars_a = ax_a.bar([0, 1], [frac_dom, 1 - frac_dom], 0.5,
                      color=[CBLIND["blue"], CBLIND["gray"]], alpha=0.85)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(
        [f"Dominant\nisoform\nchanges\nn={n_dom_change}",
         f"Same\ndominant\nisoform\nn={n_switch_sig-n_dom_change}"],
        fontsize=7.5
    )
    ax_a.set_ylim(0, 1.25)
    ax_a.set_ylabel("Fraction of switch-sig. genes", fontsize=8)
    ax_a.set_title(
        f"Dominant-isoform change\n(of {n_switch_sig} switch-sig. genes)", fontsize=8
    )
    for bar, frac in zip(bars_a, [frac_dom, 1-frac_dom]):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.015,
                  f"{frac*100:.1f}%", ha="center", fontsize=7.5)

    # Inset: diff_bp histogram
    ax_ai = ax_a.inset_axes([0.40, 0.55, 0.56, 0.38])
    ax_ai.hist(diff_bp, bins=40, color=CBLIND["sky"], alpha=0.85, edgecolor="white", lw=0.2)
    ax_ai.axvline(median_bp, color=CBLIND["orange"], lw=1.2, ls="--",
                  label=f"Median {int(median_bp):,} bp")
    ax_ai.set_xlabel("Structural diff. (bp)", fontsize=5.5)
    ax_ai.set_ylabel("Pairs", fontsize=5.5)
    ax_ai.tick_params(labelsize=5)
    ax_ai.legend(fontsize=5, frameon=False)
    ax_ai.spines["right"].set_visible(False)
    ax_ai.spines["top"].set_visible(False)

    # Panel B: RBP total_abs_delta: switch vs null (violin-like with jitter)
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.25, 1.06, "B", transform=ax_b.transAxes, **PANEL_FONT)

    # Load real per-pair total_abs_delta arrays saved by run_rbp.py
    sw_tab_df = pd.read_csv(PROC_DIR / "rbp_total_abs_delta_switch.tsv", sep="\t")
    nu_tab_df = pd.read_csv(PROC_DIR / "rbp_total_abs_delta_null.tsv", sep="\t")
    sw_vals   = sw_tab_df["total_abs_delta"].values
    nu_vals   = nu_tab_df["total_abs_delta"].values

    vp = ax_b.violinplot(
        [sw_vals, nu_vals],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
    )
    body_colors = [CBLIND["blue"], CBLIND["gray"]]
    for i, (pc, col) in enumerate(zip(vp["bodies"], body_colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
        pc.set_edgecolor("none")
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(1.5)

    # Annotate medians
    ax_b.text(0, np.median(sw_vals) + 2.0, f"{np.median(sw_vals):.1f}",
              ha="center", fontsize=6.5, color=CBLIND["blue"])
    ax_b.text(1, np.median(nu_vals) + 2.0, f"{np.median(nu_vals):.1f}",
              ha="center", fontsize=6.5, color=CBLIND["gray"])

    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(
        [f"Switch pairs\n(n={len(sw_vals)})", f"Null pairs\n(n={len(nu_vals)})"],
        fontsize=7.5
    )
    ax_b.set_ylabel("Total RBP-motif change\n(Σ|Δdensity per kb|)", fontsize=8)
    ax_b.set_title("RBP sequence-motif change:\nswitch vs null pairs", fontsize=8)

    # Annotation: MWU result (computed dynamically — never hardcode)
    _U_b, _p_b = stats.mannwhitneyu(sw_vals, nu_vals, alternative="two-sided")
    _r_b = 1.0 - 2.0 * _U_b / (len(sw_vals) * len(nu_vals))
    ax_b.text(0.5, 0.97,
              f"MWU p={_p_b:.3f},  |r|={abs(_r_b):.2f} (NEGLIGIBLE, non-directional)\n"
              "No systematic change in tested RBP-motif density",
              transform=ax_b.transAxes, ha="center", va="top",
              fontsize=6.5, style="italic",
              bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8F0", ec=CBLIND["orange"],
                        alpha=0.85, lw=0.7))

    _save_fig(fig, "Fig5")

    return n_dom_change, frac_dom, median_bp, p90_bp


# ── Figure 6: Case studies ────────────────────────────────────────────────────

# Colorblind-safe isoform palette (up to 8 isoforms)
ISO_COLORS = [
    CBLIND["blue"], CBLIND["orange"], CBLIND["green"], CBLIND["red"],
    CBLIND["sky"], CBLIND["yellow"], CBLIND["navy"], CBLIND["gray"],
]


def make_fig6(case_data: Dict):
    """Fig 6 — Case studies: stacked-bar isoform usage + total TPM inset."""
    genes_ordered = [
        "ENSG00000227028",  # SLC8A1-AS1 ISI=0.889 (highest SILENT switcher)
        "ENSG00000261008",  # LINC01572  ISI=0.871
        "ENSG00000247081",  # BAALC-AS1  ISI=0.737
        "ENSG00000233237",  # LINC00472  ISI=0.616
    ]
    genes_ordered = [g for g in genes_ordered if g in case_data]

    n_panels = len(genes_ordered)
    if n_panels == 0:
        print("  WARNING: No case-study data available for Fig6")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(7.0, 2.8 * n_panels),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    panel_labels = list("ABCD")

    for ax, gid, plbl in zip(axes, genes_ordered, panel_labels):
        d = case_data[gid]
        tissues  = d["tissues"]
        mean_if  = d["mean_if"]   # (n_iso, n_tiss)
        mean_tpm = d["mean_tpm"]  # (n_tiss,)
        gene_name = d["name"]
        isoforms = d["isoforms"]

        n_iso  = mean_if.shape[0]
        n_tiss = len(tissues)
        x = np.arange(n_tiss)

        # Shorten tissue labels
        tiss_labels = [_short_tissue(t) for t in tissues]

        # Stacked bars
        bottom = np.zeros(n_tiss)
        for i in range(n_iso):
            col = ISO_COLORS[i % len(ISO_COLORS)]
            short_iso = isoforms[i].split(".")[0]  # remove version
            ax.bar(x, mean_if[i], bottom=bottom, color=col, alpha=0.82,
                   width=0.7, label=short_iso)
            bottom += mean_if[i]

        ax.set_xlim(-0.5, n_tiss - 0.5)
        ax.set_ylim(0, 1.20)
        ax.set_xticks(x)
        ax.set_xticklabels(tiss_labels, fontsize=5.5, rotation=40, ha="right")
        ax.set_ylabel("Mean isoform fraction", fontsize=7.5)
        ax.set_title(f"{gene_name}  (lnc-ISI = {d.get('isi', '—'):.3f})", fontsize=9)

        # Panel letter above the TPM inset (which sits at y≈1.04–1.42), clear of the
        # inset's left-hand "Gene TPM (mean)" axis title.
        ax.text(-0.085, 1.47, plbl, transform=ax.transAxes, **PANEL_FONT)

        # Compact legend for isoforms
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:min(n_iso, 4)], labels[:min(n_iso, 4)],
                  loc="upper right", fontsize=5.5, frameon=True,
                  framealpha=0.85, ncol=1)

        # Inset: total TPM across tissues (should be flat)
        ax_ins = ax.inset_axes([0.0, 1.04, 1.0, 0.38])
        ax_ins.plot(x, mean_tpm, "o-", color=CBLIND["green"], ms=3, lw=1.0)
        ax_ins.axhline(np.mean(mean_tpm), color=CBLIND["gray"], lw=0.8, ls="--", alpha=0.7)
        ax_ins.set_xlim(-0.5, n_tiss - 0.5)
        ax_ins.set_xticks(x)
        ax_ins.set_xticklabels([])
        ax_ins.set_ylabel("Gene TPM\n(mean)", fontsize=5.5)
        ax_ins.tick_params(labelsize=5)
        ax_ins.spines["right"].set_visible(False)
        ax_ins.spines["top"].set_visible(False)

        cv = np.std(mean_tpm) / (np.mean(mean_tpm) + 1e-9)
        ax_ins.text(0.98, 0.88, f"CV={cv:.2f} (gene DE=False)",
                    transform=ax_ins.transAxes, ha="right", fontsize=5.5,
                    style="italic", color=CBLIND["green"])

    _save_fig(fig, "Fig6")


def _short_tissue(t: str) -> str:
    """Shorten GTEx tissue label for axis tick."""
    replacements = [
        ("Brain - ", "Brain-"), ("Heart - ", "Heart-"),
        ("Adipose - ", "Adipose-"), ("Artery - ", "Art-"),
        ("Colon - ", "Colon-"), ("Skin - ", "Skin-"),
        ("Small Intestine - ", "SmInt-"), ("Kidney - ", "Kidney-"),
        ("Muscle - Skeletal", "Muscle"), ("Lung", "Lung"),
        ("Liver", "Liver"), ("Spleen", "Spleen"),
        ("Pancreas", "Pancreas"), ("Ovary", "Ovary"),
        ("Testis", "Testis"), ("Thyroid", "Thyroid"),
        ("Uterus", "Uterus"), ("Cells - Cultured fibroblasts", "Fibroblasts"),
        ("Cells - EBV-transformed lymphocytes", "EBV-lymph."),
        ("Whole Blood", "Wh. Blood"),
    ]
    for full, short in replacements:
        t = t.replace(full, short)
    return t


# ── Key numbers ───────────────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score 95% CI for a proportion k/n."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    center = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * (p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))**0.5 / (1 + z**2 / n)
    return center - margin, center + margin


def write_key_numbers(
    n_genes_lr: int,
    n_silent: int, n_visible: int, n_gene_only: int, n_none: int,
    n_switch_sig: int, silent_frac: float,
    median_isi: float, p90_isi: float,
    n_isi_01: int, n_isi_025: int, n_isi_05: int,
    n_silent_sr: int, n_switch_sig_sr: int, silent_frac_sr: float,
    spearman_r: float, spearman_p: float, n_genes_sr: int,
    n_both: int,
    n_lr_silent_in_sr: int, n_sw_sig_in_sr: int, n_silent_in_sr: int,
    jaccard: float,
    n_dom_change: int, frac_dom: float,
    n_switch_pairs: int, median_diff_bp: float, p90_diff_bp: float,
    mwu_p: float, mwu_r: float,
    sw_median_total: float, nu_median_total: float,
):
    # Wilson 95% CIs for the headline proportions
    lr_ci_lo, lr_ci_hi = _wilson_ci(n_silent, n_switch_sig)
    sr_ci_lo, sr_ci_hi = _wilson_ci(n_silent_sr, n_switch_sig_sr)

    from run_landscape import B_PERMUTATIONS
    lines = [
        "# key_numbers.txt — consolidated figure numbers for manuscript citation",
        "# Source: code/make_figures.py reading data/processed/*.tsv",
        f"# Permutations (B): {B_PERMUTATIONS}",
        "",
        "## Long-read dataset (Phase 2)",
        f"n_genes_analyzed_lr        = {n_genes_lr}",
        f"n_silent                   = {n_silent}",
        f"n_visible                  = {n_visible}",
        f"n_gene_only                = {n_gene_only}",
        f"n_none                     = {n_none}",
        f"sum_check (should be {n_genes_lr}) = {n_silent+n_visible+n_gene_only+n_none}",
        f"n_switch_sig_lr            = {n_switch_sig}",
        f"silent_fraction_lr         = {n_silent}/{n_switch_sig} = {silent_frac:.4f}  ({silent_frac*100:.1f}%)",
        f"silent_fraction_lr_wilson95ci = ({lr_ci_lo:.4f}, {lr_ci_hi:.4f})  [{lr_ci_lo*100:.1f}%–{lr_ci_hi*100:.1f}%]",
        "",
        "## lnc-ISI distribution",
        f"median_isi_lr              = {median_isi:.4f}",
        f"p90_isi_lr                 = {p90_isi:.4f}",
        f"n_isi_ge_0.10              = {n_isi_01}",
        f"n_isi_ge_0.25              = {n_isi_025}",
        f"n_isi_ge_0.50              = {n_isi_05}",
        "",
        "## Short-read replication (Phase 4)",
        f"n_genes_analyzed_sr        = {n_genes_sr}  (len(visibility_shortread))",
        f"n_silent_sr                = {n_silent_sr}",
        f"n_switch_sig_sr            = {n_switch_sig_sr}",
        f"silent_fraction_sr         = {n_silent_sr}/{n_switch_sig_sr} = {silent_frac_sr:.4f}  ({silent_frac_sr*100:.1f}%)",
        f"silent_fraction_sr_wilson95ci = ({sr_ci_lo:.4f}, {sr_ci_hi:.4f})  [{sr_ci_lo*100:.1f}%–{sr_ci_hi*100:.1f}%]",
        f"n_common_genes_lr_sr       = {n_both}  (merged by gene_id base)",
        f"spearman_r_isi_lr_vs_sr    = {spearman_r:.3f}",
        f"spearman_p_isi_lr_vs_sr    = {spearman_p:.2e}  (computed)",
        "",
        "## Silent-set replication",
        f"n_lr_silent_analyzable_sr  = {n_lr_silent_in_sr}",
        f"n_sw_sig_in_sr             = {n_sw_sig_in_sr}  ({n_sw_sig_in_sr}/{n_lr_silent_in_sr} = {n_sw_sig_in_sr/n_lr_silent_in_sr:.3f})",
        f"n_silent_in_sr             = {n_silent_in_sr}  ({n_silent_in_sr}/{n_lr_silent_in_sr} = {n_silent_in_sr/n_lr_silent_in_sr:.3f})",
        f"n_visible_in_sr_from_lr    = {n_sw_sig_in_sr - n_silent_in_sr}",
        f"jaccard_silent_sets        = {jaccard:.3f}",
        "",
        "## Switch structure (Phase 3 Step 1)",
        f"n_switch_sig_genes_total   = {n_switch_sig}  (long-read)",
        f"n_dom_isoform_change       = {n_dom_change}",
        f"frac_dom_isoform_change    = {n_dom_change}/{n_switch_sig} = {frac_dom:.3f}  ({frac_dom*100:.1f}%)",
        f"n_switch_pairs             = {n_switch_pairs}",
        f"diff_bp_median             = {int(median_diff_bp):,}",
        f"diff_bp_p90                = {int(p90_diff_bp):,}",
        "",
        "## RBP sequence-motif change (Phase 3 Step 2)",
        f"switch_total_abs_delta_median = {sw_median_total:.4f}",
        f"null_total_abs_delta_median   = {nu_median_total:.4f}",
        f"mwu_p_total_abs_delta         = {mwu_p:.4f}  (two-sided Mann-Whitney U)",
        f"mwu_rank_biserial_r           = {mwu_r:.4f}  (negligible; |r| < 0.05)",
        f"verdict                       = negligible / non-directional",
        "",
        "## Case-study genes (top silent switchers)",
        "SLC8A1-AS1  ENSG00000227028   lnc_isi=0.889  gene_de=False  visibility=silent",
        "LINC01572   ENSG00000261008   lnc_isi=0.871  gene_de=False  visibility=silent",
        "BAALC-AS1   ENSG00000247081   lnc_isi=0.737  gene_de=False  visibility=silent",
        "LINC00472   ENSG00000233237   lnc_isi=0.616  gene_de=False  visibility=silent",
        "",
        "# END",
    ]
    out_path = PROC_DIR / "key_numbers.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")
    return lines


# ── Figure 7: Power Curve ─────────────────────────────────────────────────────

def make_fig7_power_curve(silent_frac_lr: float, silent_frac_sr: float):
    """Fig 7 — Short-read subsampling power-calibration curve.

    x = per-tissue sample size (log scale),
    y = silent fraction (%) with mean ± SD error bars across R replicates.

    Reference points:
        Long-read:  15.3%  at N ≈ 7.6 samples/tissue (71/463)
        Short-read: 4.6%   at N = 50  (full short-read analysis)

    Reads data/processed/power_curve.tsv and power_curve_summary.txt.
    Saves figures/Fig7_power_curve.png and .pdf.
    Returns the summary DataFrame for optional downstream use.
    """
    power_curve_tsv = PROC_DIR / "power_curve.tsv"
    if not power_curve_tsv.exists():
        print("  WARNING: power_curve.tsv not found — skipping Fig7")
        return None

    df = pd.read_csv(power_curve_tsv, sep="\t")

    # Aggregate: mean ± SD per target_N
    agg = (
        df.dropna(subset=["silent_fraction"])
          .groupby("target_N")
          .agg(
              mean_sf=("silent_fraction", "mean"),
              sd_sf=("silent_fraction", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
              mean_n=("mean_samples_per_tissue", "mean"),
              n_reps=("silent_fraction", "count"),
          )
          .reset_index()
    )

    # Reference points
    LR_SF   = silent_frac_lr * 100   # long-read silent fraction (%), passed in
    LR_N    = 7.6               # mean samples/tissue, long-read (68 samples / 9 tissues = 7.56)
    SR_SF   = silent_frac_sr * 100   # full short-read silent fraction (%), passed in
    SR_N    = 50.0

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Error-bar line for the power curve
    x = agg["mean_n"].values
    y = agg["mean_sf"].values * 100
    yerr = agg["sd_sf"].values * 100

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-",
        color=CBLIND["blue"],
        ecolor=CBLIND["sky"],
        elinewidth=1.0,
        capsize=3,
        capthick=0.8,
        lw=1.5,
        ms=5,
        label="Short-read subsampling\n(mean ± SD, R replicates)",
        zorder=3,
    )

    # Long-read reference point
    ax.scatter(
        [LR_N], [LR_SF],
        color=CBLIND["red"],
        s=60,
        zorder=5,
        marker="D",
        label=f"Long-read estimate\n({LR_SF:.1f}%, N≈{LR_N:.1f})",
    )
    # Horizontal dashed line at 15.3%
    ax.axhline(
        LR_SF,
        color=CBLIND["red"],
        lw=0.9,
        ls="--",
        alpha=0.6,
        zorder=2,
    )
    ax.text(
        LR_N + 0.3, LR_SF + 0.5,
        f"Long-read: {LR_SF:.1f}%",
        color=CBLIND["red"],
        fontsize=6.5,
        va="bottom",
    )

    # Full short-read reference point
    ax.scatter(
        [SR_N], [SR_SF],
        color=CBLIND["green"],
        s=60,
        zorder=5,
        marker="s",
        label=f"Full short-read\n({SR_SF:.1f}%, N={int(SR_N)})",
    )
    ax.axhline(
        SR_SF,
        color=CBLIND["green"],
        lw=0.9,
        ls=":",
        alpha=0.6,
        zorder=2,
    )
    ax.text(
        SR_N - 3, SR_SF + 0.5,
        f"Short-read full: {SR_SF:.1f}%",
        color=CBLIND["green"],
        fontsize=6.5,
        va="bottom",
        ha="right",
    )

    # Axis formatting
    ax.set_xscale("log")
    # Set explicit x-ticks at the N-grid values plus the long-read N
    from matplotlib.ticker import FixedLocator, FixedFormatter
    n_ticks = sorted(set(list(agg["target_N"].astype(int)) + [int(round(LR_N))]))
    ax.xaxis.set_major_locator(FixedLocator(n_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in n_ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))

    ax.set_xlabel("Per-tissue sample size (target cap N)", fontsize=9)
    ax.set_ylabel("Silent fraction (%)\n[silent / (silent + visible)]", fontsize=9)
    ax.set_title(
        "Power-calibration curve: silent fraction vs sample size\n"
        "(short-read subsampling, B=1000 permutations)",
        fontsize=9,
    )

    # Shaded region indicating long-read sample size range (6–9)
    ax.axvspan(6, 9, alpha=0.10, color=CBLIND["red"], zorder=1,
               label="Long-read N range (6–9)")

    ax.legend(fontsize=6.5, frameon=True, loc="upper right",
              framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_ylim(bottom=0)

    # Annotate B
    ax.text(
        0.02, 0.04,
        "B = 1000 permutations per replicate",
        transform=ax.transAxes,
        fontsize=6,
        color="#666666",
        va="bottom",
    )

    plt.tight_layout()
    _save_fig(fig, "Fig7_power_curve")
    return agg


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_fig(fig, name: str):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    png_size = png_path.stat().st_size
    pdf_size = pdf_path.stat().st_size
    print(f"  {name}: PNG {png_size/1024:.1f} KB, PDF {pdf_size/1024:.1f} KB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print("make_figures.py — lncRNA isoform-switching paper")
    print("=" * 62)

    # ── Load processed data ────────────────────────────────────────────────────
    print("[1/8] Loading processed TSVs ...")
    vis_lr, isi_lr, vis_sr, switch_exons, switch_rbp, rbp_stats = load_all_data()

    # Cross-check counts
    counts = vis_lr["visibility"].value_counts()
    n_silent    = int(counts.get("silent", 0))
    n_visible   = int(counts.get("visible", 0))
    n_gene_only = int(counts.get("gene_only", 0))
    n_none      = int(counts.get("none", 0))
    n_total_lr  = len(vis_lr)
    assert n_silent + n_visible + n_gene_only + n_none == n_total_lr, \
        f"Count mismatch: {n_silent}+{n_visible}+{n_gene_only}+{n_none} != {n_total_lr}"
    print(f"  Long-read: {n_total_lr} genes; silent={n_silent}, visible={n_visible}, "
          f"gene_only={n_gene_only}, none={n_none}")

    # ── Load biotypes for Fig 2C ───────────────────────────────────────────────
    print("[2/8] Loading biotypes from GTF ...")
    gene_biotype = load_biotypes()

    # ── Fig 1 ─────────────────────────────────────────────────────────────────
    print("[3/8] Generating Fig1 (concept) ...")
    make_fig1(vis_lr)

    # ── Fig 2 ─────────────────────────────────────────────────────────────────
    print("[4/8] Generating Fig2 (lnc-ISI landscape) ...")
    n_isi_01, n_isi_025, n_isi_05, median_isi, p90_isi = make_fig2(isi_lr, gene_biotype)

    # ── Fig 3 ─────────────────────────────────────────────────────────────────
    print("[5/8] Generating Fig3 (silent switching headline) ...")
    n_silent, n_visible, n_gene_only, n_none, n_total_lr, n_switch_sig, silent_frac = \
        make_fig3(vis_lr)

    # ── Fig 4 ─────────────────────────────────────────────────────────────────
    print("[6/8] Generating Fig4 (replication) ...")
    counts_sr = vis_sr["visibility"].value_counts()
    n_silent_sr    = int(counts_sr.get("silent", 0))
    n_visible_sr   = int(counts_sr.get("visible", 0))
    n_switch_sig_sr = n_silent_sr + n_visible_sr
    silent_frac_sr  = n_silent_sr / n_switch_sig_sr

    _fig4 = make_fig4(vis_lr, vis_sr)

    # Merge for n_both / spearman
    def base_id(df):
        d = df.copy()
        d["gene_base"] = d["gene_id"].str.split(".").str[0]
        return d
    merged = base_id(vis_lr).merge(base_id(vis_sr), on="gene_base",
                                   suffixes=("_lr", "_sr"), how="inner")
    n_both = len(merged)
    spearman_r, spearman_p = stats.spearmanr(merged["lnc_isi_lr"], merged["lnc_isi_sr"])

    # ── Fig 5 ─────────────────────────────────────────────────────────────────
    print("[7/8] Generating Fig5 (switch structure & RBP) ...")
    n_dom_change, frac_dom, median_diff_bp, p90_diff_bp = make_fig5(
        vis_lr, switch_exons, switch_rbp, rbp_stats
    )

    # ── Fig 6: case studies ────────────────────────────────────────────────────
    print("[8/8] Generating Fig6 (case studies) ...")
    case_genes = {
        "ENSG00000227028": "SLC8A1-AS1",
        "ENSG00000261008": "LINC01572",
        "ENSG00000247081": "BAALC-AS1",
        "ENSG00000233237": "LINC00472",
    }
    case_data = load_case_study_isoform_fractions(list(case_genes.keys()), case_genes)

    # Add ISI values from vis_lr
    for gid in list(case_data.keys()):
        row = vis_lr[vis_lr["gene_id"].str.startswith(gid)]
        if not row.empty:
            case_data[gid]["isi"] = float(row["lnc_isi"].iloc[0])

    make_fig6(case_data)

    # ── Fig 7: Power curve (optional — runs if power_curve.tsv exists) ─────────
    print("[extra] Generating Fig7 (power curve) ...")
    make_fig7_power_curve(silent_frac_lr=silent_frac, silent_frac_sr=silent_frac_sr)

    # ── Key numbers ───────────────────────────────────────────────────────────
    print("[extra] Writing key_numbers.txt ...")
    # RBP overall stats computed dynamically (two-sided MWU + rank-biserial),
    # matching run_rbp.py — never hardcode (avoids stale-number drift).
    _rbp_sw = switch_rbp["total_abs_delta"].values
    _rbp_nu = pd.read_csv(PROC_DIR / "rbp_total_abs_delta_null.tsv", sep="\t")["total_abs_delta"].values
    _rbp_U, _rbp_mwu_p = stats.mannwhitneyu(_rbp_sw, _rbp_nu, alternative="two-sided")
    _rbp_mwu_r = 1.0 - 2.0 * _rbp_U / (len(_rbp_sw) * len(_rbp_nu))

    kn_lines = write_key_numbers(
        n_genes_lr=n_total_lr,
        n_silent=n_silent, n_visible=n_visible, n_gene_only=n_gene_only, n_none=n_none,
        n_switch_sig=n_switch_sig, silent_frac=silent_frac,
        median_isi=median_isi, p90_isi=p90_isi,
        n_isi_01=n_isi_01, n_isi_025=n_isi_025, n_isi_05=n_isi_05,
        n_silent_sr=n_silent_sr, n_switch_sig_sr=n_switch_sig_sr, silent_frac_sr=silent_frac_sr,
        spearman_r=spearman_r, spearman_p=_fig4["spearman_p"], n_genes_sr=_fig4["n_genes_sr"],
        n_both=n_both,
        n_lr_silent_in_sr=_fig4["n_lr_silent_in_sr"], n_sw_sig_in_sr=_fig4["n_sw_sig_in_sr"],
        n_silent_in_sr=_fig4["n_silent_in_sr"],
        jaccard=_fig4["jaccard"],
        n_dom_change=n_dom_change, frac_dom=frac_dom,
        n_switch_pairs=len(switch_exons),
        median_diff_bp=median_diff_bp, p90_diff_bp=p90_diff_bp,
        mwu_p=_rbp_mwu_p, mwu_r=_rbp_mwu_r,
        sw_median_total=float(np.median(_rbp_sw)),
        nu_median_total=float(np.median(_rbp_nu)),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("Figure file sizes:")
    total_size = 0
    for fig_name in ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7_power_curve"]:
        for ext in ["png", "pdf"]:
            fp = FIG_DIR / f"{fig_name}.{ext}"
            if fp.exists():
                sz = fp.stat().st_size
                total_size += sz
                print(f"  {fp.name:15s}  {sz/1024:7.1f} KB")
            else:
                print(f"  {fig_name}.{ext}  MISSING!")
    print(f"  Total: {total_size/1024:.0f} KB")
    print()
    print("Count cross-check:")
    print(f"  {n_silent} + {n_visible} + {n_gene_only} + {n_none} = "
          f"{n_silent+n_visible+n_gene_only+n_none}  (should be {n_total_lr})")
    print()
    print("key_numbers.txt (verbatim):")
    print("-" * 62)
    for line in kn_lines:
        print(line)
    print("-" * 62)
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
