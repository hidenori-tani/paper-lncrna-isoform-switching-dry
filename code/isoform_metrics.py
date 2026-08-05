"""lncRNA isoform switching core metrics (pure functions, no I/O)."""
import numpy as np
from itertools import combinations


def isoform_fraction(tpm):
    """Per-gene usage fraction (IF) of each isoform. If total is 0, return all zeros."""
    tpm = np.asarray(tpm, dtype=float)
    total = tpm.sum()
    if total <= 0:
        return np.zeros_like(tpm)
    return tpm / total


def jsd(p, q):
    """Jensen-Shannon divergence (log2 base, 0..1). p,q are probability vectors of equal length."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    def _kl(a, dist):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / dist[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def lnc_isi(if_by_tissue):
    """lncRNA Isoform Switching Index = max pairwise JSD of isoform usage across tissues (0..1).
    if_by_tissue: {tissue: IF vector (equal length)}. Returns 0.0 if fewer than 2 tissues.
    Each IF vector must sum to ~1.0 (expressed tissue) or ~0.0 (unexpressed tissue)."""
    for tissue, vec in if_by_tissue.items():
        vec = np.asarray(vec, dtype=float)
        s = vec.sum()
        if not (abs(s - 1.0) <= 1e-6 or abs(s) <= 1e-6):
            raise ValueError(
                f"Isoform-fraction vector for tissue '{tissue}' sums to {s:.6g}, "
                f"expected ~1.0 (expressed) or ~0.0 (unexpressed). "
                f"Normalize your input with isoform_fraction() before calling lnc_isi()."
            )
    tissues = list(if_by_tissue)
    if len(tissues) < 2:
        return 0.0
    return max(jsd(if_by_tissue[a], if_by_tissue[b]) for a, b in combinations(tissues, 2))


def _introns(exons):
    """Frozenset of intron coordinates (exon_i.end, exon_{i+1}.start) for sorted exons.

    Introns are the gaps between consecutive exons. By construction they never
    include the transcript's 5'-most start or 3'-most end, so the intron set is
    invariant to a pure TSS or TES (terminal) shift — it changes only when the
    splice-junction pattern changes. This is what lets classify_switch_mechanism
    separate splice-junction differences from pure terminal (promoter/3'-end) shifts.
    (Note: an alternative first/last exon changes both a terminus and its adjacent
    junction, so an intron-set difference is not by itself proof of strictly-internal
    splicing — it flags any difference in the splice-junction set.)
    """
    ex = sorted(exons)
    return frozenset((ex[i][1], ex[i + 1][0]) for i in range(len(ex) - 1))


def classify_switch_mechanism(exons_a, exons_b, strand):
    """Classify the structural basis of a dominant-isoform switch between two isoforms.

    Given the exon structures of two isoforms of the same (stranded) gene locus,
    decide whether they differ by (i) transcription start site / first exon
    (alternative promoter usage), (ii) transcription end site / last exon
    (alternative polyadenylation), and/or (iii) a splice-junction (intron-set) difference.

    Parameters
    ----------
    exons_a, exons_b : list of (start, end)
        Half-open genomic exon intervals (+-strand orientation in coordinates),
        as produced by run_functional.load_tx_exons. Need not be pre-sorted.
    strand : str
        '+' or '-'. Determines which genomic terminus is the TSS vs the TES.

    Returns
    -------
    dict with keys:
        alt_tss      : bool — the two isoforms start transcription at different
                       genomic positions (alternative promoter / alternative first exon)
        alt_tes      : bool — they end at different genomic positions
                       (alternative polyadenylation / alternative last exon)
        alt_splicing : bool — their splice-junction (intron) sets differ (exon skipping,
                       intron retention, alternative splice site). This is ANY intron-set
                       difference, which an alternative first/last exon also produces.
        category     : str — a single mutually-exclusive label assigned by priority
                       'alt_TSS' > 'internal_splicing' > 'alt_TES'; 'identical' if the
                       two structures are indistinguishable.
        multi        : bool — True if more than one of the three changes co-occurs.
    """
    if strand not in ("+", "-"):
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")
    ea = sorted(exons_a)
    eb = sorted(exons_b)
    if not ea or not eb:
        raise ValueError("exon lists must be non-empty")

    left_a, right_a = ea[0][0], ea[-1][1]
    left_b, right_b = eb[0][0], eb[-1][1]
    if strand == "+":
        tss_a, tes_a = left_a, right_a
        tss_b, tes_b = left_b, right_b
    else:  # minus strand: transcription starts at the high-coordinate end
        tss_a, tes_a = right_a, left_a
        tss_b, tes_b = right_b, left_b

    alt_tss = tss_a != tss_b
    alt_tes = tes_a != tes_b
    alt_splicing = _introns(ea) != _introns(eb)

    if not (alt_tss or alt_tes or alt_splicing):
        category = "identical"
    elif alt_tss:
        category = "alt_TSS"
    elif alt_splicing:
        category = "internal_splicing"
    else:
        category = "alt_TES"

    multi = (int(alt_tss) + int(alt_tes) + int(alt_splicing)) > 1
    return {
        "alt_tss": alt_tss,
        "alt_tes": alt_tes,
        "alt_splicing": alt_splicing,
        "category": category,
        "multi": multi,
    }


def classify_visibility(gene_de, switch_sig):
    """switching-visibility 2-axis classification.
    silent    = gene-level expression unchanged (not DE) but isoform switching significant (the paper's protagonist)
    visible   = both gene-level and isoform change
    gene_only = only gene-level changes (no switching)
    none      = neither changes"""
    if switch_sig and not gene_de:
        return "silent"
    if switch_sig and gene_de:
        return "visible"
    if gene_de and not switch_sig:
        return "gene_only"
    return "none"
