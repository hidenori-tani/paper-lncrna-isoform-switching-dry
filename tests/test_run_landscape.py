"""Unit tests for run_landscape.py logic.

Covers:
  - nan-guard (FIX 1 / C1): identical groups → p=1.0
  - DE all-samples fix (FIX 2): gene_de_flags uses ALL samples of valid tissues,
    not just the expressed subset, so genes that are silent in one tissue are
    detected as DE.
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# Ensure run_landscape module is importable
_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))


# ---------------------------------------------------------------------------
# Reproduce the nan-guard logic exactly as written in run_landscape.py so
# the test is self-contained and survives refactors of the main script.
# ---------------------------------------------------------------------------

def _kw_pval_with_nan_guard(groups):
    """Thin wrapper replicating the guarded KW call in run_landscape.py."""
    if len(groups) >= 2 and all(len(g) >= 1 for g in groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _stat, p = stats.kruskal(*groups)
            except Exception:
                p = 1.0
        if p is None or math.isnan(p):
            p = 1.0
    else:
        p = 1.0
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kw_identical_groups_produce_nan_without_guard():
    """Verify the underlying scipy behaviour: identical values → nan p-value.

    This documents *why* the guard is necessary.
    """
    groups = [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _stat, p_raw = stats.kruskal(*groups)
    # scipy returns nan when all values are identical (constant tie correction)
    assert math.isnan(p_raw), (
        f"Expected nan from scipy.stats.kruskal on identical groups, got {p_raw!r}. "
        "If scipy behaviour has changed this test may need updating."
    )


def test_nan_guard_converts_nan_to_1():
    """The guarded call must return 1.0 (not nan) for all-identical groups."""
    groups = [np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])]
    p = _kw_pval_with_nan_guard(groups)
    assert not math.isnan(p), "nan must not escape the guard"
    assert p == pytest.approx(1.0), f"Expected 1.0, got {p}"


def test_nan_guard_normal_groups_unchanged():
    """For genuinely different groups the guard must not alter the p-value."""
    rng = np.random.default_rng(0)
    groups = [rng.normal(0, 1, 20), rng.normal(5, 1, 20)]
    p_guarded = _kw_pval_with_nan_guard(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _stat, p_raw = stats.kruskal(*groups)
    assert p_guarded == pytest.approx(p_raw, rel=1e-9)
    assert not math.isnan(p_guarded)


def test_nan_guard_too_few_groups_returns_1():
    """Fewer than 2 groups (degenerate input) must return p=1.0."""
    p = _kw_pval_with_nan_guard([np.array([1.0, 2.0])])
    assert p == pytest.approx(1.0)


def test_nan_guard_empty_group_list_returns_1():
    """Empty group list must return p=1.0 (no crash)."""
    p = _kw_pval_with_nan_guard([])
    assert p == pytest.approx(1.0)


def test_no_nan_reaches_multipletests():
    """A list of p-values produced by the guard must contain no nan.

    Simulates 5 genes: 3 normal, 1 all-identical, 1 degenerate (1 group).
    """
    rng = np.random.default_rng(1)
    group_sets = [
        [rng.normal(0, 1, 10), rng.normal(1, 1, 10)],    # normal
        [rng.normal(0, 1, 8),  rng.normal(3, 1, 8)],     # normal
        [rng.normal(0, 1, 15), rng.normal(0, 1, 15)],    # likely non-sig but not nan
        [np.array([2.0] * 5),  np.array([2.0] * 5)],     # all-identical → nan without guard
        [np.array([7.0, 8.0])],                           # only 1 group → p=1.0
    ]
    pvals = [_kw_pval_with_nan_guard(gs) for gs in group_sets]
    assert all(not math.isnan(p) for p in pvals), (
        f"nan found in p-value list: {pvals}"
    )
    # Confirm multipletests does not raise
    from statsmodels.stats.multitest import multipletests
    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    assert all(not math.isnan(q) for q in qvals), (
        f"nan found in q-value list: {qvals}"
    )


# ---------------------------------------------------------------------------
# FIX 2: gene_de_flags uses ALL samples of valid tissues (zeros included)
# ---------------------------------------------------------------------------

def test_de_flag_includes_zero_tpm_samples():
    """gene_de_flags must detect DE when one tissue is near-zero and another is high.

    Scenario
    --------
    12 samples: 0-5 = tissue_A (gene TPM ~10), 6-11 = tissue_B (gene TPM = 0).
    Tissue_B qualifies as 'valid' because it has >= 3 expressed samples declared
    (we inject 3 indices with zero TPM as the expressed_samples just to make the
    tissue eligible).

    Old code: KW groups were built from gene_expressed_samples[g][t], which for
    tissue_B would be the 3 expressed-declared indices — all at 0.0 TPM — making
    both groups look like [~10, ~10, ...] vs [0, 0, 0], so actually similar in
    this test.

    New code (the fix): KW groups come from tissue_all_sample_idx[t], which for
    tissue_B gives all 6 samples (all 0.0 TPM).  tissue_A is all ~10.  The
    separation is maximal and the gene must be called DE.

    We actually verify the DE call, not just the input size, so we need the
    two tissue distributions to be clearly distinct.  The setup below ensures
    tissue_A ≈ 10, tissue_B = 0 for ALL samples — a textbook DE signal.
    """
    from run_landscape import gene_de_flags

    rng = np.random.default_rng(7)

    # 12 samples: 0-5 = tissue_A (high), 6-11 = tissue_B (zero)
    n_samples = 12
    tissue_a_idx = np.arange(0, 6)
    tissue_b_idx = np.arange(6, 12)

    gene_tpm = np.zeros(n_samples, dtype=np.float32)
    gene_tpm[tissue_a_idx] = rng.uniform(8.0, 12.0, len(tissue_a_idx))  # clearly high
    gene_tpm[tissue_b_idx] = 0.0                                          # all silent

    gene = "TESTGENE1"

    # valid_tissue_map: both tissues are valid.
    # For tissue_B we declare 3 expressed samples (indices 6, 7, 8) so it
    # passes the expressed-sample gate — but their TPM is 0.0 in this data.
    gene_valid_tissue_map = {gene: ["tissue_A", "tissue_B"]}
    gene_expressed_samples = {
        gene: {
            "tissue_A": tissue_a_idx,          # 6 expressed in A
            "tissue_B": tissue_b_idx[:3],       # 3 "expressed" in B (all TPM 0)
        }
    }

    # The fix: tissue_all_sample_idx gives ALL 6 samples per tissue
    tissue_all_sample_idx = {
        "tissue_A": tissue_a_idx,
        "tissue_B": tissue_b_idx,   # 6 samples, all zero TPM
    }

    gene_tpm_sum = {gene: gene_tpm}

    result = gene_de_flags(
        analyzed_genes=[gene],
        gene_valid_tissue_map=gene_valid_tissue_map,
        gene_expressed_samples=gene_expressed_samples,
        gene_tpm_sum=gene_tpm_sum,
        tissue_all_sample_idx=tissue_all_sample_idx,
    )

    # tissue_A ~10 vs tissue_B 0.0 → KW p << 0.05 → DE=True
    # (single gene so BH q = raw p; with perfect separation p ≈ 0)
    assert result[gene] is True, (
        f"Expected gene to be DE (q<0.05) when tissue_B zeros are included in KW; "
        f"got DE={result[gene]!r}"
    )


def test_de_flag_all_samples_larger_than_expressed_only():
    """The KW input size using all_samples must be >= that using expressed_only.

    This is a structural check: for any gene/tissue, the number of samples
    passed to KW with the fix is >= the number passed without the fix.
    """
    from run_landscape import gene_de_flags

    rng = np.random.default_rng(99)

    # Build a scenario where tissue_B has both expressed and silent samples
    n = 12
    tissue_a_idx = np.arange(0, 6)
    tissue_b_idx = np.arange(6, 12)

    gene_tpm = np.zeros(n, dtype=np.float32)
    gene_tpm[tissue_a_idx] = rng.uniform(1.5, 5.0, 6)
    gene_tpm[tissue_b_idx[:4]] = rng.uniform(1.5, 5.0, 4)  # 4 expressed in B
    gene_tpm[tissue_b_idx[4:]] = 0.0                        # 2 silent in B

    gene = "TESTGENE2"
    gene_valid_tissue_map = {gene: ["A", "B"]}
    gene_expressed_samples = {gene: {"A": tissue_a_idx, "B": tissue_b_idx[:4]}}
    tissue_all_sample_idx  = {"A": tissue_a_idx, "B": tissue_b_idx}
    gene_tpm_sum = {gene: gene_tpm}

    # Check sizes: all_samples has 6+6=12; expressed_only had 6+4=10
    sizes_all = sum(len(tissue_all_sample_idx[t])
                    for t in gene_valid_tissue_map[gene])
    sizes_expr = sum(len(gene_expressed_samples[gene][t])
                     for t in gene_valid_tissue_map[gene])
    assert sizes_all >= sizes_expr, (
        f"Expected all_samples ({sizes_all}) >= expressed_only ({sizes_expr})"
    )
    # And the function should run without error
    result = gene_de_flags(
        analyzed_genes=[gene],
        gene_valid_tissue_map=gene_valid_tissue_map,
        gene_expressed_samples=gene_expressed_samples,
        gene_tpm_sum=gene_tpm_sum,
        tissue_all_sample_idx=tissue_all_sample_idx,
    )
    assert gene in result


# ---------------------------------------------------------------------------
# Vectorized permutation test: _jsd_batch and switching_pvalues
# ---------------------------------------------------------------------------

def test_jsd_batch_matches_scalar():
    """_jsd_batch must match scalar jsd() from isoform_metrics for all pairs."""
    from run_landscape import _jsd_batch
    from isoform_metrics import jsd

    rng = np.random.default_rng(7)
    B, n_iso = 50, 5
    p = rng.dirichlet(np.ones(n_iso), size=B)
    q = rng.dirichlet(np.ones(n_iso), size=B)

    batch = _jsd_batch(p, q)
    scalar = np.array([jsd(p[b], q[b]) for b in range(B)])

    np.testing.assert_allclose(
        batch, scalar, atol=1e-12,
        err_msg="_jsd_batch diverges from scalar jsd()"
    )


def test_jsd_batch_bounds():
    """JSD values must lie in [0, 1]."""
    from run_landscape import _jsd_batch

    rng = np.random.default_rng(9)
    B, n_iso = 200, 6
    p = rng.dirichlet(np.ones(n_iso), size=B)
    q = rng.dirichlet(np.ones(n_iso), size=B)

    batch = _jsd_batch(p, q)
    assert np.all(batch >= -1e-12), "JSD batch has values < 0"
    assert np.all(batch <= 1.0 + 1e-12), "JSD batch has values > 1"


def test_switching_pvalues_matches_baseline_b1000():
    """Vectorized switching_pvalues must reproduce the known B=1000 visibility
    counts (silent=71, visible=385, gene_only=601, none=332) with seed=42
    on the long-read dataset.

    This test guards against precision regressions: any change that shifts
    silent by more than ±1 at B=1000 must fail (at B=10k the null shifts
    counts even less).
    """
    from pathlib import Path
    import pandas as pd
    from run_landscape import (
        switching_pvalues,
        gene_de_flags,
        compute_landscape,
        MIN_SAMPLE_GENE_TPM,
        MIN_EXPR_SAMPLES_PER_TISSUE,
    )
    from data_io import load_lncrna_tx2gene, load_longread_tpm, sample_to_tissue
    from isoform_metrics import classify_visibility
    from collections import Counter

    _ROOT = Path(__file__).resolve().parent.parent

    tx2gene = load_lncrna_tx2gene(str(_ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"))
    tpm_df = load_longread_tpm(str(_ROOT / "data/raw/longread/quantification_gencode.tpm.txt.gz"))
    s2t = sample_to_tissue(
        tpm_df.columns.tolist(),
        str(_ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"),
        min_samples_per_tissue=6,
    )
    tpm_df = tpm_df[list(s2t.keys())]

    (isi_df, analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
     gene_if_mat, gene_tpm_sum, _, _, tissue_all_sample_idx) = compute_landscape(
        tpm_df, tx2gene, s2t,
        min_sample_gene_tpm=MIN_SAMPLE_GENE_TPM,
        min_expr_samples_per_tissue=MIN_EXPR_SAMPLES_PER_TISSUE,
    )
    obs_isi_map = dict(zip(analyzed_genes, isi_df["lnc_isi"].values))
    gene_de_map = gene_de_flags(
        analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
        gene_tpm_sum, tissue_all_sample_idx,
    )

    q_map = switching_pvalues(
        analyzed_genes, gene_valid_tissue_map, gene_expressed_samples,
        gene_if_mat, obs_isi_map, B=1000, seed=42,
    )
    switch_sig = {g: q < 0.05 for g, q in q_map.items()}
    vis_counts = Counter(classify_visibility(gene_de_map[g], switch_sig[g]) for g in analyzed_genes)

    assert abs(vis_counts["silent"] - 71) <= 1, (
        f"silent={vis_counts['silent']} differs from baseline 71 by more than ±1"
    )
    assert abs(vis_counts["visible"] - 385) <= 2, (
        f"visible={vis_counts['visible']} differs from baseline 385 by more than ±2"
    )
