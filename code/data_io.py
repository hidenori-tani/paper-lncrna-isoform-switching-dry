"""data_io.py — I/O helpers for the lncRNA isoform-switching paper.

All functions are pure I/O: they load data and return simple structures.
No analysis logic lives here.
"""
import gzip
import re
from collections import defaultdict
from typing import Dict

import pandas as pd


# ── GTF parser ────────────────────────────────────────────────────────────────

def load_lncrna_tx2gene(gtf_gz_path: str) -> Dict[str, str]:
    """Parse GENCODE lncRNA GTF and return {transcript_id: gene_id}.

    Only 'transcript' feature lines are used. Both IDs include the version
    suffix (e.g. 'ENST00000473358.1', 'ENSG00000243485.5') exactly as they
    appear in the GTF — this matches the TPM file.
    """
    tx2gene: Dict[str, str] = {}
    _tx_re = re.compile(r'transcript_id "([^"]+)"')
    _gene_re = re.compile(r'gene_id "([^"]+)"')

    open_fn = gzip.open if str(gtf_gz_path).endswith('.gz') else open

    with open_fn(gtf_gz_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'transcript':
                continue
            attrs = fields[8]
            m_tx = _tx_re.search(attrs)
            m_gene = _gene_re.search(attrs)
            if m_tx and m_gene:
                tx2gene[m_tx.group(1)] = m_gene.group(1)

    return tx2gene


# ── TPM loader ────────────────────────────────────────────────────────────────

def load_longread_tpm(tpm_gz_path: str) -> pd.DataFrame:
    """Load FLAIR / GENCODE long-read TPM table.

    Returns a DataFrame indexed by transcript_id (column 0 of the file),
    with sample IDs as columns and float TPM values.
    """
    open_fn = gzip.open if str(tpm_gz_path).endswith('.gz') else open
    with open_fn(tpm_gz_path, 'rt') as fh:
        # Read without dtype= first so the index column stays as strings.
        # Then convert value columns to float32 explicitly.
        df = pd.read_csv(fh, sep='\t', index_col=0, low_memory=False)

    df = df.astype('float32')
    df.index.name = 'transcript_id'
    return df


# ── Sample → tissue mapping ───────────────────────────────────────────────────

def _strip_sm(sampid: str) -> str:
    """Remove aliquot suffix -SM-<alphanumeric> from a sample ID."""
    return re.sub(r'-SM-[A-Za-z0-9]+$', '', sampid)


def _get_longread_prefix(sample_id: str) -> str:
    """Derive the site prefix from a long-read sample ID.

    Steps (must match how v8 SAMPIDs are stripped):
    1. Strip trailing technical-replicate suffix: _rep\\d*$
    2. Strip aliquot suffix: -SM-[A-Za-z0-9]+$
    """
    s = re.sub(r'_rep\d*$', '', sample_id)
    s = re.sub(r'-SM-[A-Za-z0-9]+$', '', s)
    return s


def sample_to_tissue(
    sample_ids,
    attr_path: str,
    min_samples_per_tissue: int = 6,
) -> Dict[str, str]:
    """Map long-read sample IDs to GTEx tissue labels, with quality filters.

    Parameters
    ----------
    sample_ids : iterable of str
        Column names from the long-read TPM file (excluding the 'transcript' header).
    attr_path : str
        Path to GTEx_v8_SampleAttributesDS.txt (TSV with SAMPID, SMTSD columns).
    min_samples_per_tissue : int
        Tissues with fewer than this many mapped samples are dropped (default 6).

    Returns
    -------
    dict {sample_id: tissue}
        Only samples that (a) map to a tissue AND (b) belong to a tissue with
        >= min_samples_per_tissue samples are included.
    """
    # Build prefix -> SMTSD from v8 attributes
    attrs = pd.read_csv(attr_path, sep='\t', usecols=['SAMPID', 'SMTSD'])
    prefix_to_tissue: Dict[str, str] = {}
    for _, row in attrs.iterrows():
        pfx = _strip_sm(row['SAMPID'])
        prefix_to_tissue[pfx] = row['SMTSD']

    # Map each long-read sample to its tissue
    raw_map: Dict[str, str] = {}
    for s in sample_ids:
        pfx = _get_longread_prefix(s)
        tissue = prefix_to_tissue.get(pfx)
        if tissue is not None:
            raw_map[s] = tissue

    # Count samples per tissue and filter
    tissue_counts: Dict[str, int] = defaultdict(int)
    for tissue in raw_map.values():
        tissue_counts[tissue] += 1

    kept_tissues = {t for t, n in tissue_counts.items() if n >= min_samples_per_tissue}

    return {s: t for s, t in raw_map.items() if t in kept_tissues}
