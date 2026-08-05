"""run_de_pairwise_rank.py — complete the model x contrast design.

Round-3 review: the rank-versus-negative-binomial comparison was only made in the
omnibus setting, while the large pairwise blind spots came only from NB models, so
model family and contrast breadth were not fully separated. This adds the missing
cell: a two-sample rank test (Mann-Whitney) on each gene's own argmax-JSD tissue
pair, on the same TMM-normalised CPM used elsewhere, with a single Benjamini-Hochberg
family across genes.

Run after run_de_likeforlike.py and run_de_aligned.py:
    python code/run_de_pairwise_rank.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from data_io import (load_lncrna_tx2gene, load_longread_counts, sample_to_tissue)
from de_countbased import aggregate_tx_to_gene
from run_de_likeforlike import tmm_cpm

ALPHA = 0.05
PROC = _ROOT / "data/processed"

tx2gene = load_lncrna_tx2gene(str(_ROOT / "data/raw/gencode.v26.long_noncoding_RNAs.gtf.gz"))
counts = load_longread_counts(str(_ROOT / "data/raw/longread/quantification_gencode.counts.txt.gz"))
s2t = sample_to_tissue(counts.columns.tolist(),
                       str(_ROOT / "data/raw/GTEx_v8_SampleAttributesDS.txt"),
                       min_samples_per_tissue=6)
kept = [s for s in counts.columns if s in s2t]
al = pd.read_csv(PROC / "de_aligned.tsv", sep="\t")
genes = al["gene_id"].tolist()

gc = aggregate_tx_to_gene(counts[kept], tx2gene, gene_set=set(genes))[kept]
cpm = tmm_cpm(gc)
cols = list(cpm.columns)
tis = {t: [i for i, s in enumerate(cols) if s2t[s] == t] for t in set(s2t.values())}
vals = cpm.to_numpy(dtype=float)
row = {g: i for i, g in enumerate(cpm.index)}

def rank_p(g, ta, tb):
    i = row.get(g)
    if i is None:
        return 1.0
    a, b = vals[i][tis[ta]], vals[i][tis[tb]]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    try:
        return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        return 1.0

for tag, ca, cb in (("aligned", "tissue_a", "tissue_b"),
                    ("random", "rand_tissue_a", "rand_tissue_b")):
    sub = al.dropna(subset=[ca, cb]) if tag == "random" else al
    p = [rank_p(r["gene_id"], r[ca], r[cb]) for _, r in sub.iterrows()]
    _, q, _, _ = multipletests(p, alpha=ALPHA, method="fdr_bh")
    sil = int((q >= ALPHA).sum())
    print(f"rank test (Mann-Whitney) on TMM-CPM, {tag:8s} pair, single BH family: "
          f"{sil}/{len(sub)} = {100*sil/len(sub):.1f}%")
    al.loc[sub.index, f"de_rank_{tag}"] = q < ALPHA

al.to_csv(PROC / "de_aligned.tsv", sep="\t", index=False)
print("written back to de_aligned.tsv")
