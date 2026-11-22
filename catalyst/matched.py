"""Matched-cohort positive control: restrict to the 199 reviewed (Swiss-Prot style) entries, where curation
status is constant, so the ranking cannot win by 'this entry is well annotated'. Same blocks and protocols.
Adds two trivial baselines that use no biology: sequence length, and count of non-empty metadata fields."""
import json, re, sys
import sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

AA = "ACDEFGHIKLMNPQRSTVWY"
META = ["Pathway", "Disruption phenotype", "Glycosylation", "Proteomes"]
clean = lambda t: re.sub(r"\{.*?\}", "", str(t)) if isinstance(t, str) else ""

df = pd.read_csv((sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'))
pos_all = df.index[df["EC number"].fillna("").str.contains(r"5\.5\.1")].tolist()
reviewed = ~df.apply(lambda r: str(r["Entry Name"]).startswith(str(r["Entry"])), axis=1)
sub = df[reviewed].reset_index(drop=True)
names_all = df.loc[pos_all, "Entry Name"].tolist()
pos = sub.index[sub["Entry Name"].isin(names_all)].tolist()
names = sub.loc[pos, "Entry Name"].tolist()
I = np.array(json.load(open("ident.json"))["ident"])
order = [names_all.index(n) for n in names]
I = I[np.ix_(order, order)]
sub = sub.drop(columns=["EC number", "Catalytic activity"])
N, npos = len(sub), len(pos)
print(f"matched cohort: {N} reviewed entries, {npos} positives, {N-npos} negatives")
p = (npos - 1) / (N - 1)
print(f"random baselines: AUC 0.500, hit@10 {(1-(1-p)**10)*100:.1f}%, hit@50 {(1-(1-p)**50)*100:.1f}%")

ispos = np.zeros(N, bool); ispos[pos] = True
print(f"trivial baselines in this cohort: sequence length AUC {roc_auc_score(ispos, sub.Sequence.str.len()):.3f}, "
      f"non-empty metadata count AUC {roc_auc_score(ispos, sub[META].notna().sum(axis=1)):.3f}")

text = sub[META].apply(lambda r: " ".join(clean(v) for v in r), axis=1)
M = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(text))
seqs = sub["Sequence"].fillna("").str.upper().tolist()
comp = np.zeros((N, 20))
for i, s in enumerate(seqs):
    for j, a in enumerate(AA): comp[i, j] = s.count(a) / len(s)
comp = normalize(comp)
kmer = normalize(TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=3).fit_transform(seqs))
sm = lambda i: np.asarray((M[i] @ M.T).todense()).ravel()
ss = lambda i: 0.5 * (comp[i] @ comp.T).ravel() + 0.5 * np.asarray((kmer[i] @ kmer.T).todense()).ravel()
blocks = {"meta": sm, "seq": ss, "both": lambda i: 0.5 * (sm(i) + ss(i))}

def link(thr):
    n = len(I); par = list(range(n))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if I[i, j] >= thr:
                a, b = f(i), f(j)
                if a != b: par[a] = b
    out = {}
    for i in range(n): out.setdefault(f(i), []).append(i)
    return list(out.values())

def ev(s, qi, excl):
    mask = np.ones(N, bool); mask[qi] = False
    for e in excl: mask[e] = False
    lab = ispos[mask]; v = s[mask]
    if lab.sum() == 0: return None
    o = np.argsort(-v); r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
    a, b = int(lab.sum()), int((~lab).sum())
    return dict(best=int(r[lab].min()), auc=float((b*a + a*(a+1)/2 - r[lab].sum())/(a*b)),
                h10=r[lab].min() <= 10, left=a)

print(f"\n{'protocol':>22} {'block':>5} {'median rank':>12} {'hit@10':>8} {'AUC':>7} {'pos left':>9}")
for pname, thr in [("A leave-one-out", None)] + [(f"B family holdout t={t:.2f}", t) for t in (0.40, 0.35, 0.30)]:
    fam = {}
    if thr is not None:
        for fi, f in enumerate(link(thr)):
            for i in f: fam[pos[i]] = fi
    for bn, fn in blocks.items():
        res = [r for qi in pos
               if (r := ev(fn(qi), qi, [q for q in pos if q != qi and fam[q] == fam[qi]] if thr else []))]
        if res:
            print(f"{pname:>22} {bn:>5} {np.median([r['best'] for r in res]):12.0f} "
                  f"{np.mean([r['h10'] for r in res])*100:7.1f}% {np.mean([r['auc'] for r in res]):7.3f} "
                  f"{np.mean([r['left'] for r in res]):9.1f}")
