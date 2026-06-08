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
