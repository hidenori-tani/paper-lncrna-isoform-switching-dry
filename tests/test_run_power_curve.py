"""Unit tests for run_power_curve.py — per-tissue subsampling helper.

Tests the subsample_per_tissue helper:
  - Given a sample2tissue dict, drawing N per tissue returns exactly N
    samples per tissue (or all available if N > available) and preserves
    tissue labels.
  - Tissues with fewer than min_tissue_size samples are dropped.
  - Output size is correct and reproducible with the same seed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from run_power_curve import subsample_per_tissue


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_s2t(tissue_sizes: dict) -> dict:
    """Build a fake {sample_id: tissue} dict from {tissue: n_samples}."""
    s2t = {}
    counter = 0
    for tissue, n in tissue_sizes.items():
        for _ in range(n):
            s2t[f"s{counter:04d}"] = tissue
            counter += 1
    return s2t


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_draws_exactly_n_per_tissue():
    """When N <= available in each tissue, each tissue returns exactly N samples."""
    s2t = _make_s2t({"Brain": 20, "Liver": 15, "Lung": 10})
    rng = np.random.default_rng(seed=42)
    sub, actual_n = subsample_per_tissue(s2t, n=8, rng=rng, min_tissue_size=5)

    for tissue in ["Brain", "Liver", "Lung"]:
        assert actual_n[tissue] == 8, f"Expected 8 for {tissue}, got {actual_n[tissue]}"

    # Verify sample2tissue labels are preserved
    for sample, tissue in sub.items():
        assert tissue in {"Brain", "Liver", "Lung"}, f"Unknown tissue label: {tissue!r}"
        # sample must originally be in s2t with the same label
        assert s2t[sample] == tissue, "Tissue label mismatch in output"


def test_returns_all_when_n_exceeds_available():
    """When N > available for a tissue, min(N, available) samples are returned."""
    s2t = _make_s2t({"Brain": 6, "Liver": 50})
    rng = np.random.default_rng(seed=0)
    sub, actual_n = subsample_per_tissue(s2t, n=20, rng=rng, min_tissue_size=5)

    # Brain has only 6 → all 6 drawn
    assert actual_n["Brain"] == 6, f"Expected 6 (all), got {actual_n['Brain']}"
    # Liver has 50 → 20 drawn
    assert actual_n["Liver"] == 20, f"Expected 20, got {actual_n['Liver']}"


def test_drops_tissues_below_min_size():
    """Tissues with fewer samples than min_tissue_size must be absent from output."""
    # Heart has only 3 samples → should be dropped when min_tissue_size=5
    s2t = _make_s2t({"Brain": 10, "Heart": 3, "Lung": 8})
    rng = np.random.default_rng(seed=1)
    sub, actual_n = subsample_per_tissue(s2t, n=5, rng=rng, min_tissue_size=5)

    assert "Heart" not in actual_n, "Heart should have been dropped (3 < 5)"
    assert "Brain" in actual_n
    assert "Lung" in actual_n
    # Heart samples must not appear in the output
    heart_samples = {s for s, t in s2t.items() if t == "Heart"}
    for s in heart_samples:
        assert s not in sub, f"Heart sample {s} should not be in output"


def test_output_size_equals_sum_of_actual_n():
    """Total output size must equal sum of actual_n values."""
    s2t = _make_s2t({"A": 30, "B": 25, "C": 20, "D": 10})
    rng = np.random.default_rng(seed=7)
    sub, actual_n = subsample_per_tissue(s2t, n=8, rng=rng, min_tissue_size=5)

    assert len(sub) == sum(actual_n.values()), (
        f"Output size {len(sub)} != sum(actual_n) {sum(actual_n.values())}"
    )


def test_no_duplicate_samples_in_output():
    """Each sample_id must appear at most once in the output."""
    s2t = _make_s2t({"Brain": 50, "Liver": 50})
    rng = np.random.default_rng(seed=3)
    sub, _ = subsample_per_tissue(s2t, n=10, rng=rng, min_tissue_size=5)
    assert len(sub) == len(set(sub.keys())), "Duplicate sample_ids in output"


def test_reproducible_with_same_seed():
    """Two calls with the same seed and same rng state must return identical results."""
    s2t = _make_s2t({"Brain": 40, "Lung": 35, "Liver": 30})

    rng1 = np.random.default_rng(seed=99)
    sub1, actual_n1 = subsample_per_tissue(s2t, n=10, rng=rng1, min_tissue_size=5)

    rng2 = np.random.default_rng(seed=99)
    sub2, actual_n2 = subsample_per_tissue(s2t, n=10, rng=rng2, min_tissue_size=5)

    assert set(sub1.keys()) == set(sub2.keys()), "Different sample sets with same seed"
    assert actual_n1 == actual_n2


def test_tissue_labels_preserved_in_output():
    """Tissue labels in output must match the original s2t for every drawn sample."""
    s2t = _make_s2t({"Muscle": 20, "Heart": 15, "Kidney": 12})
    rng = np.random.default_rng(seed=5)
    sub, _ = subsample_per_tissue(s2t, n=7, rng=rng, min_tissue_size=5)

    for sample, tissue in sub.items():
        assert s2t[sample] == tissue, (
            f"Sample {sample}: expected tissue {s2t[sample]!r}, got {tissue!r}"
        )


def test_empty_input_returns_empty_output():
    """Empty sample2tissue dict should return empty outputs without error."""
    rng = np.random.default_rng(seed=0)
    sub, actual_n = subsample_per_tissue({}, n=10, rng=rng, min_tissue_size=5)
    assert sub == {}
    assert actual_n == {}


def test_all_tissues_below_min_size_returns_empty():
    """When all tissues are too small, both output dicts must be empty."""
    s2t = _make_s2t({"A": 2, "B": 3, "C": 1})
    rng = np.random.default_rng(seed=0)
    sub, actual_n = subsample_per_tissue(s2t, n=5, rng=rng, min_tissue_size=5)
    assert sub == {}
    assert actual_n == {}
