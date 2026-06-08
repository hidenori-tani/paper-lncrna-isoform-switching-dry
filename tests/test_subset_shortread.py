"""Unit tests for subset_shortread.py helpers.

Tests the streaming column-selection helper and the subset writer on
synthetic data — no real GCT file required.
"""
import gzip
import io
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import helpers from the module under test
# ---------------------------------------------------------------------------
from subset_shortread import get_column_indices, stream_subset, select_samples


# ---------------------------------------------------------------------------
# Tests for get_column_indices
# ---------------------------------------------------------------------------

def test_get_column_indices_basic():
    """Selected SAMPIDs should return their correct 0-based column indices."""
    # header: col0=transcript_id, col1=gene_id, col2=S1, col3=S2, col4=S3
    header = ['transcript_id', 'gene_id', 'S1', 'S2', 'S3']
    selected = {'S1', 'S3'}
    result = get_column_indices(header, selected)
    # Should return (2, 'S1') and (4, 'S3') in some order
    indices_only = sorted(i for i, _ in result)
    names_only   = {s for _, s in result}
    assert indices_only == [2, 4]
    assert names_only == {'S1', 'S3'}


def test_get_column_indices_none_selected():
    """If no selected SAMPIDs appear in the header, return empty list."""
    header = ['transcript_id', 'gene_id', 'S1', 'S2']
    result = get_column_indices(header, {'X99', 'Y00'})
    assert result == []


def test_get_column_indices_all_selected():
    """When all SAMPIDs are selected, all data columns should be returned."""
    header = ['transcript_id', 'gene_id', 'A', 'B', 'C']
    result = get_column_indices(header, {'A', 'B', 'C'})
    assert len(result) == 3
    indices = sorted(i for i, _ in result)
    assert indices == [2, 3, 4]


def test_get_column_indices_skips_first_two():
    """Columns 0 (transcript_id) and 1 (gene_id) must never be included."""
    header = ['transcript_id', 'gene_id', 'S1']
    # Even if we add 'transcript_id' and 'gene_id' to the selected set:
    result = get_column_indices(header, {'transcript_id', 'gene_id', 'S1'})
    indices = [i for i, _ in result]
    assert 0 not in indices
    assert 1 not in indices
    assert 2 in indices  # S1 at index 2 is included


def test_get_column_indices_preserves_order():
    """Result order must follow column order in the header."""
    header = ['transcript_id', 'gene_id', 'C', 'A', 'B']
    result = get_column_indices(header, {'A', 'B', 'C'})
    names_in_order = [s for _, s in result]
    assert names_in_order == ['C', 'A', 'B']


# ---------------------------------------------------------------------------
# Tests for stream_subset
# ---------------------------------------------------------------------------

def _make_fake_gct_gz(lines: list) -> bytes:
    """Compress a list of text lines into gzip bytes (in-memory)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
        gz.write('\n'.join(lines).encode('utf-8') + b'\n')
    return buf.getvalue()


def test_stream_subset_writes_lncrna_rows_only():
    """Only rows whose transcript_id is in lncrna_tx_set should be written."""
    gct_lines = [
        '#1.2',
        '4\t3',
        'transcript_id\tgene_id\tS1\tS2\tS3',
        'ENST001\tENSG001\t1.0\t2.0\t3.0',   # lncRNA
        'ENST002\tENSG002\t4.0\t5.0\t6.0',   # NOT lncRNA
        'ENST003\tENSG003\t7.0\t8.0\t9.0',   # lncRNA
        'ENST004\tENSG004\t0.0\t0.0\t0.0',   # NOT lncRNA
    ]
    fake_gz = _make_fake_gct_gz(gct_lines)

    lncrna_set = {'ENST001', 'ENST003'}
    col_index_sampid = [(2, 'S1'), (3, 'S2'), (4, 'S3')]  # all 3 data columns

    with tempfile.NamedTemporaryFile(suffix='.gct.gz', delete=False) as tmp_in:
        tmp_in.write(fake_gz)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix='.tsv.gz', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        n = stream_subset(tmp_in_path, lncrna_set, col_index_sampid, tmp_out_path)
        assert n == 2, f"Expected 2 lncRNA rows written, got {n}"

        with gzip.open(tmp_out_path, 'rt') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        assert lines[0] == 'transcript_id\tS1\tS2\tS3'
        tx_ids = {l.split('\t')[0] for l in lines[1:]}
        assert tx_ids == {'ENST001', 'ENST003'}
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)


def test_stream_subset_column_selection():
    """Only the selected columns should appear in output."""
    gct_lines = [
        '#1.2',
        '1\t3',
        'transcript_id\tgene_id\tS1\tS2\tS3',
        'ENST001\tENSG001\t10.0\t20.0\t30.0',
    ]
    fake_gz = _make_fake_gct_gz(gct_lines)
    lncrna_set = {'ENST001'}
    # Select only S1 and S3 (indices 2 and 4), skip S2 (index 3)
    col_index_sampid = [(2, 'S1'), (4, 'S3')]

    with tempfile.NamedTemporaryFile(suffix='.gct.gz', delete=False) as tmp_in:
        tmp_in.write(fake_gz)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix='.tsv.gz', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        n = stream_subset(tmp_in_path, lncrna_set, col_index_sampid, tmp_out_path)
        assert n == 1

        with gzip.open(tmp_out_path, 'rt') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        assert lines[0] == 'transcript_id\tS1\tS3'
        data_fields = lines[1].split('\t')
        assert data_fields[0] == 'ENST001'
        assert data_fields[1] == '10.0'   # S1
        assert data_fields[2] == '30.0'   # S3 (S2 was skipped)
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)


def test_stream_subset_empty_lncrna_set():
    """If no transcripts match, output should have only a header row."""
    gct_lines = [
        '#1.2',
        '2\t2',
        'transcript_id\tgene_id\tS1\tS2',
        'ENST001\tENSG001\t1.0\t2.0',
        'ENST002\tENSG002\t3.0\t4.0',
    ]
    fake_gz = _make_fake_gct_gz(gct_lines)
    col_index_sampid = [(2, 'S1'), (3, 'S2')]

    with tempfile.NamedTemporaryFile(suffix='.gct.gz', delete=False) as tmp_in:
        tmp_in.write(fake_gz)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix='.tsv.gz', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        n = stream_subset(tmp_in_path, set(), col_index_sampid, tmp_out_path)
        assert n == 0

        with gzip.open(tmp_out_path, 'rt') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]
        # Only the header line
        assert len(lines) == 1
        assert lines[0].startswith('transcript_id')
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)


# ---------------------------------------------------------------------------
# Tests for select_samples
# ---------------------------------------------------------------------------

def test_select_samples_cap(tmp_path):
    """When tissue has more samples than cap, exactly cap samples are selected."""
    import pandas as pd
    # Create a fake attributes file with 100 samples in one tissue
    sampids = [f'GTEX-{i:04d}-0001-SM-XXXXX' for i in range(100)]
    attr_df = pd.DataFrame({
        'SAMPID': sampids,
        'SMTSD': ['Liver'] * 100,
    })
    attr_path = tmp_path / 'attrs.txt'
    attr_df.to_csv(str(attr_path), sep='\t', index=False)

    result = select_samples(str(attr_path), ['Liver'], cap=30, seed=42)
    assert len(result) == 30
    assert all(t == 'Liver' for t in result.values())


def test_select_samples_fewer_than_cap(tmp_path):
    """When tissue has fewer samples than cap, all are selected."""
    import pandas as pd
    sampids = [f'GTEX-{i:04d}-0001-SM-XXXXX' for i in range(10)]
    attr_df = pd.DataFrame({'SAMPID': sampids, 'SMTSD': ['Lung'] * 10})
    attr_path = tmp_path / 'attrs.txt'
    attr_df.to_csv(str(attr_path), sep='\t', index=False)

    result = select_samples(str(attr_path), ['Lung'], cap=50, seed=42)
    assert len(result) == 10


def test_select_samples_reproducible(tmp_path):
    """Same seed should give same result when called twice."""
    import pandas as pd
    sampids = [f'GTEX-{i:04d}-0001-SM-XXXXX' for i in range(200)]
    attr_df = pd.DataFrame({'SAMPID': sampids, 'SMTSD': ['Heart - Left Ventricle'] * 200})
    attr_path = tmp_path / 'attrs.txt'
    attr_df.to_csv(str(attr_path), sep='\t', index=False)

    r1 = select_samples(str(attr_path), ['Heart - Left Ventricle'], cap=50, seed=7)
    r2 = select_samples(str(attr_path), ['Heart - Left Ventricle'], cap=50, seed=7)
    assert set(r1.keys()) == set(r2.keys())
