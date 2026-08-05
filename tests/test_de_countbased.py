"""Unit tests for de_countbased.py — count-based DE (DESeq2/edgeR) support logic.

These test the PURE transform functions that prepare inputs for, and consume
outputs from, the R (DESeq2 / edgeR) bridge.  All tests are self-contained
(synthetic data); none reads the gitignored raw GTEx matrices.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parent.parent / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from de_countbased import (
    derive_donor,
    is_technical_replicate,
    strip_rep,
    aggregate_tx_to_gene,
    build_coldata,
    collapse_replicates,
    countbased_de_flags,
    silent_fraction,
)


# ── donor / replicate ID parsing ────────────────────────────────────────────

def test_derive_donor_takes_first_two_dash_fields():
    assert derive_donor("GTEX-1192X-0011-R10a-SM-4RXXZ") == "GTEX-1192X"


def test_derive_donor_ignores_rep_suffix():
    assert derive_donor("GTEX-13QJ3-0726-SM-7LDHS_rep") == "GTEX-13QJ3"


def test_is_technical_replicate_true_for_rep_suffix():
    assert is_technical_replicate("GTEX-13QJ3-0726-SM-7LDHS_rep") is True


def test_is_technical_replicate_false_for_base_sample():
    assert is_technical_replicate("GTEX-13QJ3-0726-SM-7LDHS") is False


def test_strip_rep_removes_only_trailing_rep_token():
    assert strip_rep("GTEX-13QJ3-0726-SM-7LDHS_rep") == "GTEX-13QJ3-0726-SM-7LDHS"
    assert strip_rep("GTEX-1192X-0011-R10a-SM-4RXXZ") == "GTEX-1192X-0011-R10a-SM-4RXXZ"


# ── transcript → gene count aggregation ─────────────────────────────────────

def _toy_tx_counts():
    return pd.DataFrame(
        {"s1": [10, 20, 5], "s2": [1, 2, 3]},
        index=["txA1", "txA2", "txB1"],
    )


def test_aggregate_tx_to_gene_sums_transcripts_within_gene():
    tx2gene = {"txA1": "GA", "txA2": "GA", "txB1": "GB"}
    out = aggregate_tx_to_gene(_toy_tx_counts(), tx2gene)
    assert out.loc["GA", "s1"] == 30 and out.loc["GA", "s2"] == 3
    assert out.loc["GB", "s1"] == 5 and out.loc["GB", "s2"] == 3


def test_aggregate_tx_to_gene_ignores_transcripts_without_mapping():
    tx2gene = {"txA1": "GA", "txA2": "GA"}  # txB1 unmapped
    out = aggregate_tx_to_gene(_toy_tx_counts(), tx2gene)
    assert "GB" not in out.index
    assert set(out.index) == {"GA"}


def test_aggregate_tx_to_gene_restricts_to_gene_set():
    tx2gene = {"txA1": "GA", "txA2": "GA", "txB1": "GB"}
    out = aggregate_tx_to_gene(_toy_tx_counts(), tx2gene, gene_set={"GA"})
    assert list(out.index) == ["GA"]


def test_aggregate_tx_to_gene_returns_integer_counts():
    tx2gene = {"txA1": "GA", "txA2": "GA", "txB1": "GB"}
    out = aggregate_tx_to_gene(_toy_tx_counts(), tx2gene)
    assert np.issubdtype(out.values.dtype, np.integer)


# ── coldata (sample → tissue / donor / replicate) ───────────────────────────

def test_build_coldata_has_expected_columns_and_donor():
    s2t = {
        "GTEX-1192X-0011-R10a-SM-4RXXZ": "Lung",
        "GTEX-13QJ3-0726-SM-7LDHS": "Liver",
        "GTEX-13QJ3-0726-SM-7LDHS_rep": "Liver",
    }
    cd = build_coldata(list(s2t), s2t)
    assert list(cd.columns) == ["sample", "tissue", "donor", "is_rep"]
    assert set(cd["donor"]) == {"GTEX-1192X", "GTEX-13QJ3"}


def test_build_coldata_flags_replicates():
    s2t = {
        "GTEX-13QJ3-0726-SM-7LDHS": "Liver",
        "GTEX-13QJ3-0726-SM-7LDHS_rep": "Liver",
    }
    cd = build_coldata(list(s2t), s2t).set_index("sample")
    assert cd.loc["GTEX-13QJ3-0726-SM-7LDHS_rep", "is_rep"] == True   # noqa: E712
    assert cd.loc["GTEX-13QJ3-0726-SM-7LDHS", "is_rep"] == False      # noqa: E712


# ── technical-replicate collapse ────────────────────────────────────────────

def test_collapse_replicates_sums_rep_into_base():
    gc = pd.DataFrame(
        {
            "GTEX-X-SM-1": [10, 20],
            "GTEX-Y-SM-2": [5, 6],
            "GTEX-Y-SM-2_rep": [7, 8],
        },
        index=["GA", "GB"],
    )
    out = collapse_replicates(gc)
    # the _rep column is summed into its base; the base name is retained
    assert "GTEX-Y-SM-2_rep" not in out.columns
    assert out.loc["GA", "GTEX-Y-SM-2"] == 12
    assert out.loc["GB", "GTEX-Y-SM-2"] == 14
    assert out.loc["GA", "GTEX-X-SM-1"] == 10  # untouched


def test_collapse_replicates_reduces_column_count():
    gc = pd.DataFrame(
        {"a-SM-1": [1], "b-SM-2": [1], "b-SM-2_rep": [1]},
        index=["GA"],
    )
    out = collapse_replicates(gc)
    assert out.shape[1] == 2  # a, b (b collapsed)


# ── consuming DESeq2/edgeR q-values ─────────────────────────────────────────

def test_countbased_de_flags_threshold_and_nan():
    q = pd.Series({"GA": 0.01, "GB": 0.5, "GC": float("nan")})
    flags = countbased_de_flags(q, alpha=0.05)
    assert flags["GA"] is True
    assert flags["GB"] is False
    assert flags["GC"] is False  # NaN padj (DESeq2 filtered) → not DE (conservative)


# ── silent fraction from DE + switch flags ──────────────────────────────────

def test_silent_fraction_matches_visibility_definition():
    de = {"g1": False, "g2": True, "g3": False, "g4": True}
    sig = {"g1": True, "g2": True, "g3": False, "g4": False}
    # g1 silent, g2 visible, g3 none, g4 gene_only → switch_sig = {g1,g2}
    res = silent_fraction(de, sig)
    assert res["n_silent"] == 1
    assert res["n_visible"] == 1
    assert res["n_switch_sig"] == 2
    assert res["fraction"] == pytest.approx(0.5)


def test_silent_fraction_defaults_missing_de_to_false():
    # a gene present in switch_sig but missing from the DE map counts as not-DE
    de = {"g1": True}
    sig = {"g1": True, "g2": True}
    res = silent_fraction(de, sig)
    # g1 visible, g2 silent (DE missing → False) → 1/2
    assert res["n_silent"] == 1
    assert res["n_switch_sig"] == 2
    assert res["fraction"] == pytest.approx(0.5)
