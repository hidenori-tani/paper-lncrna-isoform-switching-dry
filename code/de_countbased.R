#!/usr/bin/env Rscript
# de_countbased.R — count-based differential expression for the lncRNA
# isoform-switching paper.  For each analyzed lncRNA gene, test whether its
# isoform-summed expression differs across ANY tissue, by two negative-binomial
# GLM methods:
#   DESeq2 : likelihood-ratio test, full ~tissue vs reduced ~1
#   edgeR  : quasi-likelihood ANODEV over all tissue coefficients
# This is the count-based analogue of the Kruskal-Wallis-across-tissues gene-DE
# test in run_landscape.py, used to show the 'silent' fraction is robust to the
# choice of DE method.
#
# Usage:
#   Rscript de_countbased.R <lnc_counts.tsv> <coldata.tsv> <out.tsv> [norm]
#
#   lnc_counts.tsv : header = sample IDs; first column = gene_id; integer counts.
#   coldata.tsv    : tab-sep header with columns sample, tissue, donor, is_rep, lib_size
#                    (row order need not match the count columns; it is aligned below).
#   out.tsv        : gene_id, padj_deseq2, padj_edger
#   norm           : "native" (default) or "library".
#
# Normalization modes:
#   native  (default, primary) — each method's own composition-aware normalization:
#           DESeq2 median-of-ratios size factors (type="poscounts", robust to the
#           many zeros in lncRNA counts); edgeR TMM.  This is the field-standard
#           way to run these tools and corrects for library composition differences
#           across the very different tissues.
#   library (sensitivity) — whole-library size factors (coldata$lib_size), matching
#           the total-library normalization of the TPM used by the Kruskal-Wallis
#           test, so the contrast isolates the DE *method* from the normalization.

suppressMessages({
  library(DESeq2)
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3 || length(args) > 4) {
  stop("usage: de_countbased.R <lnc_counts.tsv> <coldata.tsv> <out.tsv> [norm]")
}
counts_path  <- args[1]
coldata_path <- args[2]
out_path     <- args[3]
norm_mode    <- if (length(args) == 4) args[4] else "native"
if (!norm_mode %in% c("native", "library")) stop("norm must be 'native' or 'library'")

counts <- as.matrix(read.delim(counts_path, row.names = 1, check.names = FALSE))
storage.mode(counts) <- "integer"
coldata <- read.delim(coldata_path, check.names = FALSE, stringsAsFactors = FALSE)

# Align colData rows to count columns (exact 1:1)
if (!all(coldata$sample %in% colnames(counts))) {
  stop("coldata$sample has entries not present in the count matrix columns")
}
counts <- counts[, coldata$sample, drop = FALSE]
coldata$tissue <- factor(coldata$tissue)
if (nlevels(coldata$tissue) < 2) stop("need >= 2 tissue levels")

libsize <- as.numeric(coldata$lib_size)
if (any(!is.finite(libsize)) || any(libsize <= 0)) stop("lib_size must be positive/finite")
# Whole-library size factors (geometric-mean-centred library totals)
size_factors <- libsize / exp(mean(log(libsize)))

genes <- rownames(counts)
message(sprintf("[de_countbased.R] normalization mode: %s", norm_mode))

# ── DESeq2: LRT ~tissue vs ~1 ────────────────────────────────────────────────
dds <- DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ tissue)
if (norm_mode == "native") {
  # composition-aware median-of-ratios; poscounts tolerates the many zeros
  dds <- estimateSizeFactors(dds, type = "poscounts")
} else {
  sizeFactors(dds) <- size_factors               # whole-library normalization
}
dds <- estimateDispersions(dds, quiet = TRUE)
dds <- nbinomLRT(dds, reduced = ~ 1, quiet = TRUE)
res <- results(dds)
padj_deseq2 <- res$padj[match(genes, rownames(res))]

# ── edgeR: quasi-likelihood ANODEV over tissue coefficients ──────────────────
if (norm_mode == "native") {
  y <- DGEList(counts = counts)                  # lib.size = column sums
  y <- calcNormFactors(y)                        # TMM composition normalization
} else {
  y <- DGEList(counts = counts, lib.size = libsize)
  y$samples$norm.factors <- rep(1, ncol(counts)) # pure whole-library normalization
}
design <- model.matrix(~ tissue, data = coldata)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)
qlf <- glmQLFTest(fit, coef = 2:ncol(design))     # test ALL tissue coefficients (ANODEV)
tt <- topTags(qlf, n = Inf, sort.by = "none")$table
padj_edger <- tt$FDR[match(genes, rownames(tt))]

out <- data.frame(
  gene_id     = genes,
  padj_deseq2 = padj_deseq2,
  padj_edger  = padj_edger,
  check.names = FALSE
)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)

n_de_d <- sum(!is.na(padj_deseq2) & padj_deseq2 < 0.05)
n_de_e <- sum(!is.na(padj_edger)  & padj_edger  < 0.05)
message(sprintf("[de_countbased.R] genes tested: %d", length(genes)))
message(sprintf("[de_countbased.R] DESeq2 DE (padj<0.05): %d", n_de_d))
message(sprintf("[de_countbased.R] edgeR  DE (FDR<0.05):  %d", n_de_e))
