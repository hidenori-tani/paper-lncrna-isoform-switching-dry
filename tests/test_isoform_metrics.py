"""Tests for lncRNA isoform switching core metrics."""
import numpy as np
import pytest
from isoform_metrics import isoform_fraction, jsd, lnc_isi, classify_visibility


def test_isoform_fraction_sums_to_one():
    tpm = np.array([2.0, 6.0, 2.0])
    f = isoform_fraction(tpm)
    assert np.isclose(f.sum(), 1.0)
    assert np.allclose(f, [0.2, 0.6, 0.2])


def test_isoform_fraction_all_zero_returns_zeros():
    tpm = np.array([0.0, 0.0])
    f = isoform_fraction(tpm)
    assert np.allclose(f, [0.0, 0.0])  # 0/0 defined as 0


def test_jsd_identical_is_zero():
    p = np.array([0.5, 0.5])
    assert np.isclose(jsd(p, p), 0.0)


def test_jsd_disjoint_is_one():
    p = np.array([1.0, 0.0]); q = np.array([0.0, 1.0])
    assert np.isclose(jsd(p, q), 1.0)


def test_jsd_symmetric():
    p = np.array([0.7, 0.3]); q = np.array([0.2, 0.8])
    assert np.isclose(jsd(p, q), jsd(q, p))


def test_lnc_isi_no_switch_is_zero():
    if_by_tissue = {"brain": np.array([0.5, 0.5]), "liver": np.array([0.5, 0.5])}
    assert np.isclose(lnc_isi(if_by_tissue), 0.0)


def test_lnc_isi_full_switch_is_one():
    if_by_tissue = {"brain": np.array([1.0, 0.0]), "liver": np.array([0.0, 1.0])}
    assert np.isclose(lnc_isi(if_by_tissue), 1.0)


def test_lnc_isi_is_max_pairwise_jsd():
    if_by_tissue = {
        "a": np.array([0.5, 0.5]),
        "b": np.array([0.6, 0.4]),
        "c": np.array([0.0, 1.0]),
    }
    val = lnc_isi(if_by_tissue)
    # The true max-pairwise-JSD is jsd(b, c): b=[0.6,0.4] vs c=[0,1] is more
    # divergent than a=[0.5,0.5] vs c=[0,1]. Spec listed (a,c) but (b,c) is
    # larger; we assert against the empirically verified maximum.
    expected = max(
        jsd(if_by_tissue["a"], if_by_tissue["b"]),
        jsd(if_by_tissue["a"], if_by_tissue["c"]),
        jsd(if_by_tissue["b"], if_by_tissue["c"]),
    )
    assert val == pytest.approx(expected, rel=1e-6)


def test_classify_silent():
    assert classify_visibility(gene_de=False, switch_sig=True) == "silent"


def test_classify_visible():
    assert classify_visibility(gene_de=True, switch_sig=True) == "visible"


def test_classify_none():
    assert classify_visibility(gene_de=False, switch_sig=False) == "none"


def test_classify_gene_only():
    assert classify_visibility(gene_de=True, switch_sig=False) == "gene_only"


def test_lnc_isi_rejects_unnormalized_input():
    with pytest.raises(ValueError):
        lnc_isi({"a": np.array([2.0, 0.0]), "b": np.array([0.0, 2.0])})


def test_lnc_isi_allows_all_zero_tissue():
    # gene unexpressed in one tissue (all-zero IF) is valid; should not raise
    val = lnc_isi({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 0.0])})
    assert 0.0 <= val <= 1.0
