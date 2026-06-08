"""Unit tests for run_rbp.py — motif density function and U->T handling.

Tests verify:
  1. density() on known sequences with known match counts
  2. U->T motif conversion: all motifs use T (DNA), not U (RNA)
  3. pair_deltas() sign and magnitude
  4. total_abs_delta() arithmetic
  5. Edge cases (empty sequence, no match)
"""
import re
import pytest
from run_rbp import (
    RBP_MOTIFS,
    MOTIF_NAMES,
    _COMPILED,
    motif_density,
    all_motif_densities,
    pair_deltas,
    total_abs_delta,
)


# ── 1. U -> T motif handling ──────────────────────────────────────────────────

def test_motifs_use_T_not_U():
    """All motif patterns must be expressed in DNA (T), not RNA (U)."""
    for name, pat in RBP_MOTIFS.items():
        assert "U" not in pat, (
            f"Motif '{name}' contains 'U'; should be converted to 'T' for DNA matching. "
            f"Pattern: {pat!r}"
        )


def test_compiled_patterns_present():
    """All motif names must have a compiled regex."""
    for name in MOTIF_NAMES:
        assert name in _COMPILED, f"Missing compiled regex for '{name}'"


# ── 2. motif_density — known sequences ───────────────────────────────────────

def test_density_attta_known():
    """ELAVL1_HuR_ARE = ATTTA; 2 non-overlapping matches in 1000-nt sequence -> 2/1000*1000 = 2.0."""
    seq = "A" * 100 + "ATTTA" + "A" * 100 + "ATTTA" + "A" * 790
    # total length = 1000; 2 matches
    d = motif_density(seq, "ELAVL1_HuR_ARE")
    assert d == pytest.approx(2.0, abs=1e-9)


def test_density_tia1_urich():
    """TIA1_Urich = T{5,}: match TTTTT (5 T's) but not TTTT (4 T's)."""
    seq5  = "TTTTT"   # 1 match in 5 nt -> 200.0 per kb
    seq4  = "TTTT"    # 0 matches (needs >= 5)
    assert motif_density(seq5, "TIA1_Urich") == pytest.approx(200.0, abs=1e-6)
    assert motif_density(seq4, "TIA1_Urich") == pytest.approx(0.0, abs=1e-9)


def test_density_hnrnpc_utract():
    """HNRNPC_Utract = T{4,}: match TTTT (4 T's)."""
    seq4 = "TTTT"   # 1 match in 4 nt -> 250.0 per kb
    seq3 = "TTT"    # 0 matches
    assert motif_density(seq4, "HNRNPC_Utract") == pytest.approx(250.0, abs=1e-6)
    assert motif_density(seq3, "HNRNPC_Utract") == pytest.approx(0.0, abs=1e-9)


def test_density_nova_ycay():
    """NOVA_YCAY = [CT]CA[CT]: matches CCAC, TCAC, CCAT, TCAT."""
    seq = "CCACNNNNTCATNNNNACAG"   # 2 matches; 'ACAG' doesn't match ([CT] required at pos 0)
    # seq length = 20, 2 matches -> 2/20*1000 = 100.0
    # Check 'CCAC' and 'TCAT' match; 'ACAG' (starts with A) does not
    d = motif_density(seq, "NOVA_YCAY")
    assert d == pytest.approx(100.0, abs=1e-6)


def test_density_srsf1():
    """SRSF1 = GGAGGA|GAAGAA: match both alternates."""
    seq = "GGAGGANNNNNGAAGAA"   # 2 matches in 17 nt
    d = motif_density(seq, "SRSF1")
    assert d == pytest.approx(2 / 17 * 1000, abs=1e-6)


def test_density_mbnl_ygcy():
    """MBNL_YGCY = [CT]GC[CT]: matches CGCC, TGCC, CGCT, TGCT."""
    seq = "CGCCNNCGCT"   # 2 matches in 10 nt -> 200.0
    d = motif_density(seq, "MBNL_YGCY")
    assert d == pytest.approx(200.0, abs=1e-6)


def test_density_tdp43_ugrepeat():
    """TDP43_UGrepeat = (?:TG){3,}: match TGTGTG (3x TG)."""
    seq_match = "TGTGTG"   # 1 match (3 repeats) in 6 nt
    seq_short = "TGTG"     # only 2 repeats, no match
    assert motif_density(seq_match, "TDP43_UGrepeat") > 0.0
    assert motif_density(seq_short, "TDP43_UGrepeat") == pytest.approx(0.0, abs=1e-9)


def test_density_g_quad_like():
    """G_quad_like = (?:G{3,}\\w{1,7}){3,}G{3,}: require 4 G-tracts."""
    # 4 G-tracts with 2-nt linkers -> match
    seq_match = "GGGAAGGGAAGGGAAGGG"
    # Only 3 G-tracts -> no match
    seq_nomatch = "GGGAAGGGAAGGG"
    assert motif_density(seq_match, "G_quad_like") > 0.0
    assert motif_density(seq_nomatch, "G_quad_like") == pytest.approx(0.0, abs=1e-9)


def test_density_empty_seq():
    """Empty sequence returns 0.0 without error."""
    assert motif_density("", "ELAVL1_HuR_ARE") == pytest.approx(0.0, abs=1e-9)


def test_density_no_match():
    """Sequence with no matches returns 0.0."""
    seq = "G" * 500   # no ATTTA
    assert motif_density(seq, "ELAVL1_HuR_ARE") == pytest.approx(0.0, abs=1e-9)


def test_density_returns_per_kb():
    """Density is per kilobase: 1 match in 2000-nt seq -> 0.5 per kb."""
    seq = "A" * 997 + "ATTTA" + "A" * 998   # exactly 2000 nt, 1 match
    assert len(seq) == 2000
    d = motif_density(seq, "ELAVL1_HuR_ARE")
    assert d == pytest.approx(0.5, abs=1e-9)


# ── 3. all_motif_densities returns all keys ───────────────────────────────────

def test_all_motif_densities_keys():
    """all_motif_densities must return exactly MOTIF_NAMES keys."""
    densities = all_motif_densities("ATCGATCG")
    assert set(densities.keys()) == set(MOTIF_NAMES)


# ── 4. pair_deltas ────────────────────────────────────────────────────────────

def test_pair_deltas_identical():
    """Identical sequences -> all deltas are zero."""
    seq = "ATTTA" * 10 + "A" * 50
    deltas = pair_deltas(seq, seq)
    for name in MOTIF_NAMES:
        assert deltas[name] == pytest.approx(0.0, abs=1e-9), f"Non-zero delta for {name}"


def test_pair_deltas_sign():
    """Sequence B richer in ATTTA than A -> positive delta for ELAVL1_HuR_ARE."""
    seqA = "A" * 1000              # no ATTTA
    seqB = "ATTTA" * 10 + "A" * 950  # 10 matches in 1000 nt
    deltas = pair_deltas(seqA, seqB)
    assert deltas["ELAVL1_HuR_ARE"] > 0.0


def test_pair_deltas_negative():
    """Sequence B poorer in ATTTA than A -> negative delta."""
    seqA = "ATTTA" * 10 + "A" * 950  # 10 matches in 1000 nt
    seqB = "A" * 1000              # no ATTTA
    deltas = pair_deltas(seqA, seqB)
    assert deltas["ELAVL1_HuR_ARE"] < 0.0


# ── 5. total_abs_delta ────────────────────────────────────────────────────────

def test_total_abs_delta_zero():
    """total_abs_delta of all-zero dict is 0.0."""
    deltas = {name: 0.0 for name in MOTIF_NAMES}
    assert total_abs_delta(deltas) == pytest.approx(0.0, abs=1e-9)


def test_total_abs_delta_known():
    """total_abs_delta sums absolute values regardless of sign."""
    deltas = {"A": 2.0, "B": -3.0, "C": 0.0}
    assert total_abs_delta(deltas) == pytest.approx(5.0, abs=1e-9)


def test_total_abs_delta_from_pair():
    """total_abs_delta on real pair is >= 0."""
    seqA = "ATTTA" * 5 + "C" * 100
    seqB = "TTTTT" * 3 + "G" * 100
    deltas = pair_deltas(seqA, seqB)
    assert total_abs_delta(deltas) >= 0.0


# ── 6. Non-overlapping match count ────────────────────────────────────────────

def test_nonoverlapping_findall():
    """re.findall with T{4,} on TTTTTT: should find 1 match (not 3 overlapping)."""
    pat = re.compile(r"T{4,}")
    # 'TTTTTT' -> one non-overlapping match of length 6
    matches = pat.findall("TTTTTT")
    assert len(matches) == 1


def test_attta_adjacent():
    """Two adjacent non-overlapping ATTTAs: ATTTTAATTTA -> 0 (ATTTTTA doesn't match; ATTTA only)."""
    # 'ATTTTAATTTA': positions 0-4='ATTTT'(no), actually let's use clear non-overlapping
    seq = "ATTTAATTTA"   # positions 0-4: ATTTA(no leading A issue)
    # Actually: 'ATTTA' at pos 0 and 'ATTTA' at pos 5
    seq2 = "ATTTAATTTA"
    matches = _COMPILED["ELAVL1_HuR_ARE"].findall(seq2)
    assert len(matches) == 2
