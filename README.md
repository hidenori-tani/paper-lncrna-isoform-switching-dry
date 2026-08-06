# The gene-level blind spot in lncRNA isoform switching

Analysis code for **"The reported gene-level blind spot in lncRNA isoform switching is specification-dependent"** (Hidenori Tani).

Transcriptomic studies routinely report a class of genes whose isoform composition changes across
conditions while their total expression does not — the differential-transcript-usage-positive,
differential-gene-expression-negative (DTU+/DGE−) case — and quote the size of that class as a
biological quantity. The label, however, is defined by a *failure to reject* a null hypothesis at
the gene level, so its prevalence is a function of the power and the specification of that
gene-level test. This repository holds the code that quantifies that dependence on GTEx long-read
and short-read RNA-seq.

The switching axis itself is measured by the **lncRNA Isoform Switching Index (lnc-ISI)**, the
maximum pairwise Jensen–Shannon divergence of mean per-tissue isoform-usage vectors, tested by a
within-gene label permutation.

## Key results

With the cohort, the annotation and the transcript-usage calls held fixed on **463
switch-significant lncRNAs** (GTEx v9 long-read, 9 tissues, 68 samples), and only the gene-level
DE specification varied:

- The reported blind-spot fraction ranges from **0.4% to 52.7% — a factor of 122** — and reaches
  **64.2%** for a control (Fig 3).
- **The test family matters, not the input scale.** A rank test gives 15.3% against 0.4% (edgeR)
  and 1.3% (DESeq2), and **17.3% on the very TMM-normalised counts the negative-binomial models
  consume**.
- **Contrast breadth matters.** Narrowing from nine tissues to each gene's switch-defining pair
  (~15 samples instead of 68) raises it to 40.4%; a **random** pair matched exactly on sample size
  raises it *further*, to 50.4% against 38.3% — so the enlargement follows the narrowing of the
  contrast, not its alignment to the switch.
- **Sample size alone reproduces the effect.** Subsampling the short-read cohort moves the
  fraction from 4.5% (N = 50) to 22.2% (N = 5) with everything else fixed (Fig 6).
- A tximport-style length offset moves the count-based figures only to 1.9% and 1.1%, excluding
  the obvious confound.
- **Throughout, the switching signal is unchanged; only the visibility label moves.**

Supporting (explicitly confirmatory) results: 268/463 (57.9%) switch-significant genes change
their dominant isoform (median differential exonic content 1738 bp); 95.6% of those switches
involve an alternative TSS / first exon and only 1.5% are genuinely internal-splicing-only, which
confirms Reyes and Huber (2018) for lncRNAs rather than reporting a new mechanism. RBP
sequence-motif density does not systematically change during switching (rank-biserial |r| = 0.033)
— a bounded negative result, reported descriptively.

## Repository layout

```
code/                          analysis pipeline (pure Python + one R script; no notebooks)
  isoform_metrics.py             lnc-ISI / JSD / visibility classification (unit-tested)
  run_landscape.py               lnc-ISI landscape + visibility (permutation test)
  run_functional.py              dominant-isoform switch + differential exons
  run_mechanism.py               structural basis: alt-TSS vs splicing vs APA (Fig 8)
  de_countbased.py/.R            count-based gene-level DE (DESeq2 LRT, edgeR QL ANODEV)
  run_de_countbased.py           count-based DE driver
  run_de_aligned.py              DE restricted to each gene's switch-defining tissue pair
  run_de_pairwise_rank.py        two-sample rank test on the same pairs
  run_de_likeforlike.py          matched-size random-pair control
  run_de_lengthoffset.py         tximport-style length-offset sensitivity
  run_rbp.py                     RBP sequence-motif change (negative result)
  run_replication.py             GTEx v8 short-read replication
  run_power_curve.py             short-read subsampling power calibration (Fig 6)
  make_figures.py                Figs 1, 2, 4, 5, 6, 7, 9
  make_fig_specification.py      Fig 3 (the DE-specification audit)
  make_fig_countbased.py         count-based DE figure
  sensitivity_*.py               DE-definition and length-matched RBP sensitivity analyses
  get_data.sh                    downloads all public input data
tests/                         pytest suite (136 tests)
data/processed/                processed tables the figures are computed from (2.2 MB)
conftest.py                    makes code/ importable bare (avoids the stdlib `code` shadow)
```

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

bash code/get_data.sh                    # GTEx long/short-read TPM + counts, GENCODE v26
python code/run_landscape.py             # B = 10,000 permutations (seed 42)
python code/run_functional.py
python code/run_mechanism.py
python code/run_rbp.py                   # needs gencode.v26.lncRNA_transcripts.fa.gz
python code/subset_shortread.py          # stream-subset the ~3.6 GB GTEx v8 TPM (9 tissues)
python code/run_replication.py
python code/run_power_curve.py
python code/run_de_countbased.py         # needs R with DESeq2 + edgeR
python code/run_de_aligned.py
python code/run_de_pairwise_rank.py
python code/run_de_likeforlike.py
python code/run_de_lengthoffset.py
python code/make_figures.py              # Figs 1, 2, 4, 5, 6, 7, 9 + key_numbers.txt
python code/make_fig_specification.py    # Fig 3
python code/sensitivity_de_all_tissues.py
python code/sensitivity_rbp_lengthmatched.py

pytest -q                                # 136 tests
```

All randomness uses a fixed seed (42). **Every statistic displayed in a figure is recomputed from
the processed data at figure-build time rather than hard-coded**, so the figures cannot drift away
from the numbers in the text.

## Data availability

All input data are public. Transcript-level expression matrices are from the **GTEx Portal**
(GTEx project dbGaP study accession phs000424; the matrices used here are open-access); the
reference annotation is **GENCODE v26** (https://www.gencodegenes.org). Exact download commands
are in [`code/get_data.sh`](code/get_data.sh).

Raw matrices are large and are **not redistributed here**. The processed tables the figures are
computed from are included in [`data/processed/`](data/processed/); the short-read replication,
the subsampling power curve and the RBP-motif tables are regenerated by the pipeline from the
downloaded inputs.

## Software

Python 3.11 (numpy, pandas, scipy, statsmodels, matplotlib, pytest) and R 4.x (DESeq2, edgeR) for
the count-based differential-expression step. Tested on macOS.

## Citation

Tani H. *The reported gene-level blind spot in lncRNA isoform switching is
specification-dependent.* (manuscript; citation to be updated on publication).

## Licence

MIT — see [LICENSE](LICENSE).
