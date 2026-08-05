"""Unit tests for classify_switch_mechanism (switch structural-basis classifier).

The classifier decides whether two isoforms of the same stranded locus differ by
alternative TSS (promoter), alternative TES (APA), and/or internal splicing.
Key property under test: the intron-set comparison must DECOUPLE internal splicing
from pure terminal (TSS/TES) shifts, and the TSS/TES assignment must respect strand.
"""
import pytest

from isoform_metrics import classify_switch_mechanism, _introns


def test_pure_alt_tss_plus_strand():
    # same internal/3' structure, different 5' start
    a = [(100, 200), (300, 400)]
    b = [(150, 200), (300, 400)]
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_tss"] and not r["alt_tes"] and not r["alt_splicing"]
    assert r["category"] == "alt_TSS"
    assert r["multi"] is False


def test_pure_alt_tes_plus_strand():
    a = [(100, 200), (300, 400)]
    b = [(100, 200), (300, 450)]
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_tes"] and not r["alt_tss"] and not r["alt_splicing"]
    assert r["category"] == "alt_TES"


def test_internal_splicing_exon_skip():
    # middle exon skipped; both termini identical -> pure splicing
    a = [(100, 200), (300, 400), (500, 600)]
    b = [(100, 200), (500, 600)]
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_splicing"] and not r["alt_tss"] and not r["alt_tes"]
    assert r["category"] == "internal_splicing"


def test_terminal_shift_does_not_falsely_flag_splicing():
    # ONLY the 5' start moves; intron set must be identical (decoupling property)
    a = [(100, 200), (300, 400)]
    b = [(150, 200), (300, 400)]
    assert _introns(a) == _introns(b) == frozenset({(200, 300)})
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_splicing"] is False


def test_strand_flips_tss_and_tes():
    # same coordinate pair classified opposite on + vs - strand
    a = [(100, 200), (300, 400)]
    b = [(100, 200), (300, 450)]  # high-coordinate (3') end moved
    plus = classify_switch_mechanism(a, b, "+")
    minus = classify_switch_mechanism(a, b, "-")
    assert plus["alt_tes"] and not plus["alt_tss"]      # + strand: 3' end = TES
    assert minus["alt_tss"] and not minus["alt_tes"]    # - strand: high coord = TSS


def test_identical_structure_unordered_input():
    a = [(100, 200), (300, 400)]
    b = [(300, 400), (100, 200)]  # same set, different order
    r = classify_switch_mechanism(a, b, "+")
    assert r["category"] == "identical"
    assert not (r["alt_tss"] or r["alt_tes"] or r["alt_splicing"])


def test_multi_change_tss_plus_splicing():
    a = [(100, 200), (300, 400), (500, 600)]
    b = [(150, 200), (500, 600)]  # different TSS AND skipped internal exon
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_tss"] and r["alt_splicing"]
    assert r["multi"] is True
    assert r["category"] == "alt_TSS"  # priority: TSS > splicing > TES


def test_single_exon_alt_tss():
    a = [(100, 500)]
    b = [(200, 500)]
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_tss"] and not r["alt_splicing"] and not r["alt_tes"]
    assert r["category"] == "alt_TSS"


def test_priority_splicing_over_tes():
    # internal splicing change AND a 3' end change, same TSS -> splicing wins by priority
    a = [(100, 200), (300, 400), (500, 600)]
    b = [(100, 200), (500, 650)]  # exon skip + extended last exon (TES moved)
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_splicing"] and r["alt_tes"] and not r["alt_tss"]
    assert r["category"] == "internal_splicing"


def test_alternative_first_exon_flags_both_tss_and_splicing():
    # An ALTERNATIVE first exon (not a mere TSS coordinate shift) changes both the
    # 5' terminus AND the adjacent splice junction. This documents that alt_splicing
    # means "any splice-junction-set difference", not strictly-internal splicing.
    a = [(100, 200), (300, 400), (500, 600)]
    b = [(50, 120), (300, 400), (500, 600)]  # different first exon entirely
    r = classify_switch_mechanism(a, b, "+")
    assert r["alt_tss"] is True          # TSS moved 100 -> 50
    assert r["alt_splicing"] is True     # first intron donor moved 200 -> 120
    assert r["multi"] is True
    assert r["category"] == "alt_TSS"    # priority still assigns to TSS
    # contrast: a pure first-exon TRUNCATION (same donor) must NOT flag splicing
    c = [(150, 200), (300, 400), (500, 600)]  # same first-exon end (200) as `a`
    r2 = classify_switch_mechanism(a, c, "+")
    assert r2["alt_tss"] is True and r2["alt_splicing"] is False


def test_errors():
    with pytest.raises(ValueError):
        classify_switch_mechanism([], [(1, 2)], "+")
    with pytest.raises(ValueError):
        classify_switch_mechanism([(1, 2)], [(1, 2)], "x")
