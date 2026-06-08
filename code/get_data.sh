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

echo ""
echo "Done. All required data files are present."
