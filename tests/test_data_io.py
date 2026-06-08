"""Tests for data_io.py I/O helpers."""
import gzip
import io
import re
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from data_io import (
    load_lncrna_tx2gene,
    load_longread_tpm,
    sample_to_tissue,
    _get_longread_prefix,
    _strip_sm,
)


# ── _strip_sm ─────────────────────────────────────────────────────────────────

def test_strip_sm_removes_suffix():
    assert _strip_sm("GTEX-1192X-0011-R10a-SM-4RXXZ") == "GTEX-1192X-0011-R10a"


def test_strip_sm_leaves_prefix_intact():
    s = "GTEX-1117F-0011-R10a"
    assert _strip_sm(s) == s  # no -SM- suffix → unchanged


# ── _get_longread_prefix ──────────────────────────────────────────────────────

def test_longread_prefix_basic():
    s = "GTEX-1192X-0011-R10a-SM-4RXXZ"
    assert _get_longread_prefix(s) == "GTEX-1192X-0011-R10a"


def test_longread_prefix_rep_strip():
    """_rep suffix should be stripped before the -SM- strip."""
    s = "GTEX-S95S-0008-SM-3RQ8B_rep2"
    # Step 1: strip _rep2 -> "GTEX-S95S-0008-SM-3RQ8B"
    # Step 2: strip -SM-3RQ8B -> "GTEX-S95S-0008"
    assert _get_longread_prefix(s) == "GTEX-S95S-0008"


def test_longread_prefix_ctrl_exp_not_stripped():
    """_ctrl and _exp suffixes are NOT stripped by _get_longread_prefix
    (they don't match _rep\\d*$). The -SM- part is still stripped.
    This means these samples won't match v8 SAMPID prefixes and are
    naturally excluded by the mapping (which is the intended behaviour)."""
    s_ctrl = "GTEX-WY7C-0008-SM-3NZB5_ctrl"
    s_exp  = "GTEX-WY7C-0008-SM-3NZB5_exp"
    # After stripping: _ctrl/_exp remain, then -SM- strip runs
    # _ctrl: no -SM-... suffix after _ctrl → whole string unchanged
    # _exp:  same
    # In practice these won't match any v8 prefix → excluded from mapping
    pfx_ctrl = _get_longread_prefix(s_ctrl)
    pfx_exp  = _get_longread_prefix(s_exp)
    # They should NOT produce a clean GTEX-WY7C-0008 prefix
    assert pfx_ctrl != "GTEX-WY7C-0008"
    assert pfx_exp  != "GTEX-WY7C-0008"


def test_longread_prefix_same_donor_maps_to_same_prefix():
    """Two technical replicates of the same sample should have the same prefix."""
    s1 = "GTEX-S95S-0008-SM-3RQ8B_rep1"
    s2 = "GTEX-S95S-0008-SM-3RQ8B_rep2"
    assert _get_longread_prefix(s1) == _get_longread_prefix(s2)


# ── load_lncrna_tx2gene ───────────────────────────────────────────────────────

_SYNTHETIC_GTF = textwrap.dedent("""\
    ##description: test
    chr1\tHAVANA\tgene\t100\t500\t.\t+\t.\tgene_id "ENSG00000000001.1"; gene_type "lincRNA";
    chr1\tHAVANA\ttranscript\t100\t400\t.\t+\t.\tgene_id "ENSG00000000001.1"; transcript_id "ENST00000000001.1"; gene_type "lincRNA";
    chr1\tHAVANA\texon\t100\t300\t.\t+\t.\tgene_id "ENSG00000000001.1"; transcript_id "ENST00000000001.1";
    chr1\tHAVANA\ttranscript\t200\t500\t.\t+\t.\tgene_id "ENSG00000000001.1"; transcript_id "ENST00000000002.1"; gene_type "lincRNA";
    chr1\tHAVANA\tgene\t600\t900\t.\t+\t.\tgene_id "ENSG00000000002.1"; gene_type "lincRNA";
    chr1\tHAVANA\ttranscript\t600\t900\t.\t+\t.\tgene_id "ENSG00000000002.1"; transcript_id "ENST00000000003.1"; gene_type "lincRNA";
""")


def _write_synthetic_gtf(tmp_path, use_gz=True):
    if use_gz:
        p = tmp_path / "test.gtf.gz"
        with gzip.open(p, 'wt') as fh:
            fh.write(_SYNTHETIC_GTF)
    else:
        p = tmp_path / "test.gtf"
        p.write_text(_SYNTHETIC_GTF)
    return str(p)


def test_load_lncrna_tx2gene_count(tmp_path):
    gtf = _write_synthetic_gtf(tmp_path)
    tx2gene = load_lncrna_tx2gene(gtf)
    assert len(tx2gene) == 3  # 3 transcript lines


def test_load_lncrna_tx2gene_mapping(tmp_path):
    gtf = _write_synthetic_gtf(tmp_path)
    tx2gene = load_lncrna_tx2gene(gtf)
    assert tx2gene["ENST00000000001.1"] == "ENSG00000000001.1"
    assert tx2gene["ENST00000000002.1"] == "ENSG00000000001.1"
    assert tx2gene["ENST00000000003.1"] == "ENSG00000000002.1"


def test_load_lncrna_tx2gene_exon_lines_excluded(tmp_path):
    """Exon lines must not create spurious entries."""
    gtf = _write_synthetic_gtf(tmp_path)
    tx2gene = load_lncrna_tx2gene(gtf)
    # No duplicate keys (exon lines would create them for ENST00000000001.1)
    assert len(tx2gene) == 3


# ── load_longread_tpm ─────────────────────────────────────────────────────────

def _write_synthetic_tpm(tmp_path, use_gz=True):
    content = (
        "transcript\tSAMPLE_A\tSAMPLE_B\tSAMPLE_C\n"
        "ENST00000000001.1\t10.0\t0.0\t5.5\n"
        "ENST00000000002.1\t0.0\t20.0\t3.0\n"
    )
    if use_gz:
        p = tmp_path / "tpm.txt.gz"
        with gzip.open(p, 'wt') as fh:
            fh.write(content)
    else:
        p = tmp_path / "tpm.txt"
        p.write_text(content)
    return str(p)


def test_load_longread_tpm_shape(tmp_path):
    tpm_path = _write_synthetic_tpm(tmp_path)
    df = load_longread_tpm(tpm_path)
    assert df.shape == (2, 3)


def test_load_longread_tpm_index_name(tmp_path):
    tpm_path = _write_synthetic_tpm(tmp_path)
    df = load_longread_tpm(tpm_path)
    assert df.index.name == 'transcript_id'


def test_load_longread_tpm_values(tmp_path):
    tpm_path = _write_synthetic_tpm(tmp_path)
    df = load_longread_tpm(tpm_path)
    assert df.loc["ENST00000000001.1", "SAMPLE_A"] == pytest.approx(10.0)
    assert df.loc["ENST00000000002.1", "SAMPLE_B"] == pytest.approx(20.0)


# ── sample_to_tissue ──────────────────────────────────────────────────────────

def _write_synthetic_attr(tmp_path):
    """Minimal GTEx-style sample attributes file."""
    content = (
        "SAMPID\tSMTSD\n"
        # Tissue A: 6 matching samples
        "GTEX-AAA1-0011-R1a-SM-AAAA1\tMuscle - Skeletal\n"
        "GTEX-BBB1-0011-R1a-SM-BBBB1\tMuscle - Skeletal\n"
        "GTEX-CCC1-0011-R1a-SM-CCCC1\tMuscle - Skeletal\n"
        "GTEX-DDD1-0011-R1a-SM-DDDD1\tMuscle - Skeletal\n"
        "GTEX-EEE1-0011-R1a-SM-EEEE1\tMuscle - Skeletal\n"
        "GTEX-FFF1-0011-R1a-SM-FFFF1\tMuscle - Skeletal\n"
        # Tissue B: only 2 matching samples (below min=6, should be excluded)
        "GTEX-GGG1-0011-R1a-SM-GGGG1\tBrain - Cerebellum\n"
        "GTEX-HHH1-0011-R1a-SM-HHHH1\tBrain - Cerebellum\n"
    )
    p = tmp_path / "attrs.txt"
    p.write_text(content)
    return str(p)


def test_sample_to_tissue_keeps_min_tissue(tmp_path):
    attr_path = _write_synthetic_attr(tmp_path)
    # Long-read sample IDs matching the v8 prefixes above
    sample_ids = [
        "GTEX-AAA1-0011-R1a-SM-AAAA1",
        "GTEX-BBB1-0011-R1a-SM-BBBB1",
        "GTEX-CCC1-0011-R1a-SM-CCCC1",
        "GTEX-DDD1-0011-R1a-SM-DDDD1",
        "GTEX-EEE1-0011-R1a-SM-EEEE1",
        "GTEX-FFF1-0011-R1a-SM-FFFF1",
        "GTEX-GGG1-0011-R1a-SM-GGGG1",
        "GTEX-HHH1-0011-R1a-SM-HHHH1",
    ]
    result = sample_to_tissue(sample_ids, attr_path, min_samples_per_tissue=6)
    tissues = set(result.values())
    assert "Muscle - Skeletal" in tissues
    assert "Brain - Cerebellum" not in tissues   # only 2 samples → excluded


def test_sample_to_tissue_count(tmp_path):
    attr_path = _write_synthetic_attr(tmp_path)
    sample_ids = [
        "GTEX-AAA1-0011-R1a-SM-AAAA1",
        "GTEX-BBB1-0011-R1a-SM-BBBB1",
        "GTEX-CCC1-0011-R1a-SM-CCCC1",
        "GTEX-DDD1-0011-R1a-SM-DDDD1",
        "GTEX-EEE1-0011-R1a-SM-EEEE1",
        "GTEX-FFF1-0011-R1a-SM-FFFF1",
    ]
    result = sample_to_tissue(sample_ids, attr_path, min_samples_per_tissue=6)
    assert len(result) == 6


def test_sample_to_tissue_unmapped_excluded(tmp_path):
    attr_path = _write_synthetic_attr(tmp_path)
    sample_ids = [
        "GTEX-AAA1-0011-R1a-SM-AAAA1",
        "GTEX-ZZZ9-9999-R9a-SM-ZZZZ9",  # not in attrs → should be excluded
    ]
    result = sample_to_tissue(sample_ids, attr_path, min_samples_per_tissue=1)
    assert "GTEX-ZZZ9-9999-R9a-SM-ZZZZ9" not in result


def test_sample_to_tissue_real_sample_id_mapping():
    """Verify that the real first sample maps to the right prefix (no real file needed)."""
    # This tests the prefix logic in isolation
    real_id = "GTEX-1192X-0011-R10a-SM-4RXXZ"
    pfx = _get_longread_prefix(real_id)
    # After stripping -SM-4RXXZ → "GTEX-1192X-0011-R10a"
    assert pfx == "GTEX-1192X-0011-R10a"
    # That prefix matches what _strip_sm would give for the v8 SAMPID
    # GTEX-1192X-0011-R10a-SM-AHZ7F (hypothetical v8 entry with same donor/site)
    v8_sampid = "GTEX-1192X-0011-R10a-SM-SOMEID"
    assert _strip_sm(v8_sampid) == pfx
