#!/usr/bin/env bash
# get_data.sh  — Download all raw data for the lncRNA isoform-switching paper.
# Idempotent: skips files that already exist.
# Usage: bash code/get_data.sh   (run from project root)

set -euo pipefail

mkdir -p data/raw/longread

# ── Long-read quantification (GTEx v9, FLAIR / GENCODE v26) ──────────────────

GENCODE_TPM="data/raw/longread/quantification_gencode.tpm.txt.gz"
if [ ! -f "$GENCODE_TPM" ]; then
    echo "Downloading $GENCODE_TPM ..."
    curl -L -o "$GENCODE_TPM" \
        "https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/quantification_gencode.tpm.txt.gz"
else
    echo "Already exists: $GENCODE_TPM"
fi

# Long-read GENCODE raw counts (for count-based DE: DESeq2 / edgeR robustness) ─
GENCODE_COUNTS="data/raw/longread/quantification_gencode.counts.txt.gz"
if [ ! -f "$GENCODE_COUNTS" ]; then
    echo "Downloading $GENCODE_COUNTS ..."
    curl -L -o "$GENCODE_COUNTS" \
        "https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/quantification_gencode.counts.txt.gz"
else
    echo "Already exists: $GENCODE_COUNTS"
fi

# Optional – FLAIR novel+known isoforms (sensitivity analysis, larger file)
FLAIR_TPM="data/raw/longread/quantification_flair_filter.tpm.txt.gz"
if [ ! -f "$FLAIR_TPM" ]; then
    echo "Skipping optional novel-isoform file (uncomment to download):"
    echo "  URL: https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/quantification_flair_filter.tpm.txt.gz"
    # Uncomment the next two lines to download:
    # curl -L -o "$FLAIR_TPM" \
    #     "https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/quantification_flair_filter.tpm.txt.gz"
else
    echo "Already exists: $FLAIR_TPM"
fi

# Optional – FLAIR transcript models GTF
FLAIR_GTF="data/raw/longread/flair_filter_transcripts.gtf.gz"
if [ ! -f "$FLAIR_GTF" ]; then
    echo "Skipping optional FLAIR GTF (uncomment to download):"
    echo "  URL: https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/flair_filter_transcripts.gtf.gz"
    # curl -L -o "$FLAIR_GTF" \
    #     "https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/flair_filter_transcripts.gtf.gz"
else
    echo "Already exists: $FLAIR_GTF"
fi

# README for long-read data
LR_README="data/raw/longread/README.txt"
if [ ! -f "$LR_README" ]; then
    echo "Downloading $LR_README ..."
    curl -L -o "$LR_README" \
        "https://storage.googleapis.com/adult-gtex/long-read-data/v9/long-read-RNA-seq/README.txt"
else
    echo "Already exists: $LR_README"
fi

# ── GTEx v8 sample attributes ────────────────────────────────────────────────

SAMPATTR="data/raw/GTEx_v8_SampleAttributesDS.txt"
if [ ! -f "$SAMPATTR" ]; then
    echo "Downloading $SAMPATTR ..."
    curl -L -o "$SAMPATTR" \
        "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
else
    echo "Already exists: $SAMPATTR"
fi

# ── GENCODE v26 lncRNA annotation ────────────────────────────────────────────

LNCRNA_GTF="data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"
if [ ! -f "$LNCRNA_GTF" ]; then
    echo "Downloading $LNCRNA_GTF ..."
    curl -L -o "$LNCRNA_GTF" \
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.long_noncoding_RNAs.gtf.gz"
else
    echo "Already exists: $LNCRNA_GTF"
fi

# ── GENCODE v26 lncRNA transcript FASTA (for run_rbp.py) ─────────────────────

LNCRNA_FASTA="data/raw/gencode.v26.lncRNA_transcripts.fa.gz"
if [ ! -f "$LNCRNA_FASTA" ]; then
    echo "Downloading $LNCRNA_FASTA ..."
    curl -L -o "$LNCRNA_FASTA" \
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.lncRNA_transcripts.fa.gz"
else
    echo "Already exists: $LNCRNA_FASTA"
fi

# ── GTEx v8 short-read transcript TPM (replication; subset_shortread.py) ──────
# NOTE: large file (~3.6 GB). Required only for the short-read replication /
# power-curve analyses (run_replication.py, run_power_curve.py via
# subset_shortread.py). The long-read mechanism/landscape analyses do NOT need it.

mkdir -p data/raw/shortread
SHORTREAD_TPM="data/raw/shortread/GTEx_v8_transcript_tpm.gct.gz"
if [ ! -f "$SHORTREAD_TPM" ]; then
    echo "Downloading $SHORTREAD_TPM (~3.6 GB) ..."
    curl -L -o "$SHORTREAD_TPM" \
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct.gz"
else
    echo "Already exists: $SHORTREAD_TPM"
fi

echo ""
echo "Done. All required data files are present."
