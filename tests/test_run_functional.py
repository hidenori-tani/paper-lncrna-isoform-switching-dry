"""Unit tests for run_functional.py — interval symmetric-difference helpers.

These tests exercise the interval arithmetic (merge, subtract, symmetric diff)
and the total_bp helper without requiring any real data files.
"""
import pytest
from run_functional import (
    merge_intervals,
    subtract_intervals,
    symmetric_diff_intervals,
    total_bp,
)


# ── merge_intervals ────────────────────────────────────────────────────────────

def test_merge_empty():
    assert merge_intervals([]) == []


def test_merge_single():
    assert merge_intervals([(10, 20)]) == [(10, 20)]


def test_merge_no_overlap():
    assert merge_intervals([(10, 20), (30, 40)]) == [(10, 20), (30, 40)]


def test_merge_adjacent():
    """Adjacent intervals [10,20) and [20,30) should merge to [10,30)."""
    assert merge_intervals([(10, 20), (20, 30)]) == [(10, 30)]


def test_merge_overlapping():
    assert merge_intervals([(10, 25), (20, 40)]) == [(10, 40)]


def test_merge_contained():
    """Inner interval fully contained in outer."""
    assert merge_intervals([(10, 50), (20, 30)]) == [(10, 50)]


def test_merge_unsorted_input():
    """merge_intervals should handle unsorted input correctly."""
    assert merge_intervals([(30, 40), (10, 20), (15, 25)]) == [(10, 25), (30, 40)]


# ── subtract_intervals ─────────────────────────────────────────────────────────

def test_subtract_empty_a():
    assert subtract_intervals([], [(10, 20)]) == []


def test_subtract_empty_b():
    """A - empty = A."""
    assert subtract_intervals([(10, 20)], []) == [(10, 20)]


def test_subtract_no_overlap():
    """B does not overlap A at all — A unchanged."""
    assert subtract_intervals([(10, 20)], [(30, 40)]) == [(10, 20)]


def test_subtract_full_cover():
    """B completely covers A — result is empty."""
    assert subtract_intervals([(10, 20)], [(5, 25)]) == []


def test_subtract_prefix():
    """B covers the front part of A."""
    result = subtract_intervals([(10, 30)], [(10, 20)])
    assert result == [(20, 30)]


def test_subtract_suffix():
    """B covers the back part of A."""
    result = subtract_intervals([(10, 30)], [(20, 30)])
    assert result == [(10, 20)]


def test_subtract_middle():
    """B punches a hole in the middle of A."""
    result = subtract_intervals([(10, 40)], [(20, 30)])
    assert result == [(10, 20), (30, 40)]


def test_subtract_multiple_b():
    """Multiple B intervals each bite a piece out of A."""
    result = subtract_intervals([(0, 100)], [(10, 20), (50, 60)])
    assert result == [(0, 10), (20, 50), (60, 100)]


# ── symmetric_diff_intervals ───────────────────────────────────────────────────

def test_symdiff_identical():
    """Identical intervals → empty symmetric difference."""
    a = [(100, 200), (300, 400)]
    b = [(100, 200), (300, 400)]
    assert symmetric_diff_intervals(a, b) == []


def test_symdiff_disjoint():
    """Completely disjoint A and B → all intervals in both."""
    a = [(100, 200)]
    b = [(300, 400)]
    result = symmetric_diff_intervals(a, b)
    assert result == [(100, 200), (300, 400)]


def test_symdiff_partial_overlap():
    """Partial overlap: A=[100,200), B=[150,250).
    Diff = [100,150) + [200,250) = 150bp total.
    """
    a = [(100, 200)]
    b = [(150, 250)]
    result = symmetric_diff_intervals(a, b)
    assert result == [(100, 150), (200, 250)]
    assert total_bp(result) == 100


def test_symdiff_spec_example():
    """Canonical example from the task spec:
    A = [(chr1,100,200), (chr1,300,400)]
    B = [(chr1,150,250)]
    Regions in A not in B: [100,150) + [300,400)
    Regions in B not in A: [200,250)
    Union: [100,150) + [200,250) + [300,400)
    Total bp = 50 + 50 + 100 = 200
    """
    # Using integer coords as in the real code (0-based half-open)
    a = [(100, 200), (300, 400)]
    b = [(150, 250)]
    result = symmetric_diff_intervals(a, b)
    expected = [(100, 150), (200, 250), (300, 400)]
    assert result == expected, f"Expected {expected}, got {result}"
    assert total_bp(result) == 200


def test_symdiff_b_inside_a():
    """B is entirely inside A (no exons outside A for B side).
    A=[100,400), B=[150,250)
    A-B = [100,150) + [250,400)  → 50 + 150 = 200 bp
    B-A = [] (B entirely inside A)
    Diff = [100,150) + [250,400) = 200 bp total
    """
    a = [(100, 400)]
    b = [(150, 250)]
    result = symmetric_diff_intervals(a, b)
    assert result == [(100, 150), (250, 400)]
    assert total_bp(result) == 200  # 50 + 150


def test_symdiff_empty_inputs():
    """Both empty → empty result."""
    assert symmetric_diff_intervals([], []) == []


def test_symdiff_a_empty():
    """A empty, B non-empty → B is the diff."""
    b = [(10, 20), (30, 40)]
    result = symmetric_diff_intervals([], b)
    assert result == b


def test_symdiff_b_empty():
    """B empty, A non-empty → A is the diff."""
    a = [(10, 20), (30, 40)]
    result = symmetric_diff_intervals(a, [])
    assert result == a


# ── total_bp ───────────────────────────────────────────────────────────────────

def test_total_bp_empty():
    assert total_bp([]) == 0


def test_total_bp_single():
    assert total_bp([(10, 20)]) == 10


def test_total_bp_multiple():
    assert total_bp([(10, 20), (30, 40), (50, 60)]) == 30
