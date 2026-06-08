"""subset_shortread.py — Memory/disk-safe streaming subset of GTEx v8 short-read GCT.

Streams the (large, ~12 GB uncompressed) GCT gzip line-by-line, writing ONLY
lncRNA transcript rows and ONLY the selected sample columns to a small gzipped
output.  The full uncompressed file is never written to disk.

Outputs (data/processed/):
    shortread_lncrna_subset.tsv.gz   — lncRNA rows × selected samples (TSV, gzipped)
    shortread_sample2tissue.tsv      — two-column TSV: SAMPID, SMTSD

Run from project root:
    python code/subset_shortread.py
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- resolve imports when run as `python code/subset_shortread.py`
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_io import load_lncrna_tx2gene

# ── Paths ──────────────────────────────────────────────────────────────────────
GTF_PATH   = _ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
GCT_PATH   = _ROOT / "data/raw/shortread/GTEx_v8_transcript_tpm.gct.gz"
ATTR_PATH  = _ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"
PROC_DIR   = _ROOT / "data/processed"
PROC_DIR.mkdir(exist_ok=True)
OUT_MATRIX = PROC_DIR / "shortread_lncrna_subset.tsv.gz"
OUT_S2T    = PROC_DIR / "shortread_sample2tissue.tsv"

# ── Target tissues ─────────────────────────────────────────────────────────────
TARGET_TISSUES = [
    "Brain - Cerebellar Hemisphere",
    "Brain - Frontal Cortex (BA9)",
    "Brain - Putamen (basal ganglia)",
    "Cells - Cultured fibroblasts",
    "Heart - Atrial Appendage",
    "Heart - Left Ventricle",
    "Liver",
    "Lung",
    "Muscle - Skeletal",
]

CAP_PER_TISSUE = 50   # random cap per tissue
SEED           = 42


def select_samples(attr_path: str, target_tissues, cap: int, seed: int):
    """Return {SAMPID: SMTSD} for selected short-read samples.

    Steps:
    1. Load attributes; keep rows whose SMTSD is in target_tissues.
    2. For each tissue, randomly sample up to `cap` SAMPIDs (seed fixed).
    """
    attrs = pd.read_csv(attr_path, sep='\t', usecols=['SAMPID', 'SMTSD'])
    attrs = attrs[attrs['SMTSD'].isin(target_tissues)].copy()

    rng = np.random.default_rng(seed)
    selected = {}
    for tissue in target_tissues:
        subset = attrs[attrs['SMTSD'] == tissue]['SAMPID'].tolist()
        if len(subset) > cap:
            chosen = rng.choice(len(subset), size=cap, replace=False)
            subset = [subset[i] for i in sorted(chosen)]
        for s in subset:
            selected[s] = tissue
    return selected


def get_column_indices(header_fields: list, selected_sampids: set):
    """Return sorted list of (col_index, sampid) for selected SAMPIDs.

    header_fields: full split header line (index 0 = 'transcript_id',
                   index 1 = 'gene_id', index 2+ = SAMPIDs).
    selected_sampids: set of SAMPIDs to keep.
    """
    indices = []
    for i, field in enumerate(header_fields):
        if i < 2:
            continue  # skip transcript_id and gene_id columns
        if field in selected_sampids:
            indices.append((i, field))
    return indices  # list of (col_index, sampid)


def stream_subset(
    gct_gz_path: str,
    lncrna_tx_set: set,
    col_index_sampid: list,
    out_gz_path: str,
) -> int:
    """Stream the GCT, writing lncRNA rows with selected columns to out_gz_path.

    Returns number of lncRNA transcript rows written.
    """
    col_indices = [c for c, _ in col_index_sampid]
    sampids     = [s for _, s in col_index_sampid]

    n_written = 0
    with gzip.open(gct_gz_path, 'rt') as fin, gzip.open(out_gz_path, 'wt') as fout:
        # Skip the two GCT header lines
        next(fin)  # #1.2
        next(fin)  # nrows ntcols

        # Read column header line
        header_line = next(fin)
        # Write output header: transcript_id + selected sample columns
        fout.write("transcript_id\t" + "\t".join(sampids) + "\n")

        for line in fin:
            # Fast split: only need col 0 (transcript_id) + selected cols
            # Use split with no limit — data lines are wide but we index by position
            fields = line.rstrip('\n').split('\t')
            tx_id = fields[0]

            if tx_id not in lncrna_tx_set:
                continue

            # Extract selected column values
            values = [fields[i] for i in col_indices]
            fout.write(tx_id + "\t" + "\t".join(values) + "\n")
            n_written += 1

    return n_written


def main():
    print("[1/3] Loading lncRNA transcript set from GTF ...")
    tx2gene = load_lncrna_tx2gene(str(GTF_PATH))
    lncrna_tx_set = set(tx2gene.keys())
    print(f"  {len(lncrna_tx_set):,} lncRNA transcripts in GTF")

    print("[2/3] Selecting short-read samples ...")
    sample2tissue = select_samples(str(ATTR_PATH), TARGET_TISSUES, CAP_PER_TISSUE, SEED)
    selected_sampids = set(sample2tissue.keys())
    print(f"  {len(selected_sampids)} samples selected across {len(TARGET_TISSUES)} tissues:")
    from collections import Counter
    tc = Counter(sample2tissue.values())
    for t in sorted(tc):
        print(f"    {tc[t]:3d}  {t}")

    # Save sample2tissue
    s2t_df = pd.DataFrame(
        [(s, t) for s, t in sample2tissue.items()],
        columns=['SAMPID', 'SMTSD']
    )
    s2t_df.to_csv(OUT_S2T, sep='\t', index=False)
    print(f"  Saved: {OUT_S2T.relative_to(_ROOT)}")

    print("[3/3] Streaming GCT — extracting lncRNA rows for selected samples ...")
    print(f"  Input:  {GCT_PATH.relative_to(_ROOT)}")
    print(f"  Output: {OUT_MATRIX.relative_to(_ROOT)}")
    print("  (This may take a few minutes — streaming line by line ...)")

    # We need to know the column indices BEFORE streaming, so peek at header
    with gzip.open(str(GCT_PATH), 'rt') as fin:
        next(fin)  # #1.2
        next(fin)  # nrows ntcols
        header_line = next(fin).rstrip('\n')

    header_fields = header_line.split('\t')
    col_index_sampid = get_column_indices(header_fields, selected_sampids)
    found_sampids = {s for _, s in col_index_sampid}
    missing = selected_sampids - found_sampids
    if missing:
        print(f"  WARNING: {len(missing)} selected SAMPIDs not found in GCT header!")
        for s in sorted(missing)[:5]:
            print(f"    {s}")
    print(f"  Found {len(col_index_sampid)} of {len(selected_sampids)} selected SAMPIDs in GCT.")

    n_written = stream_subset(
        str(GCT_PATH), lncrna_tx_set, col_index_sampid, str(OUT_MATRIX)
    )

    print(f"  lncRNA transcript rows written: {n_written:,}")
    print("Done.")


if __name__ == "__main__":
    main()
