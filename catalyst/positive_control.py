"""
positive_control.py  --  non-circular positive control for the original enzyme-screening pipeline.

Run:  python positive_control.py /path/to/dataset.csv [ident.json]
      (ident.json = pairwise alignment identities among positives, from align_ident.py; recomputed if absent)

QUESTION: with the giveaway features removed, does the pipeline rank known Diels-Alderases above random
enzymes, and does anything survive when homologs of the query are taken off the board?

POSITIVES: the 22 entries whose EC number contains 5.5.1 (intramolecular lyases; includes the paper's
  references CGHA_CHAGB, FSA2_FUSSF, PHM7_PYRSX and its top three EQX3_GIBF5, EQX3_FUSHE, TAS3_HAPIR).
  The label is a database annotation, not an assay. NEGATIVES: the other 3,597.
REMOVED so the answer cannot leak: the EC number column and the "Catalytic activity" reaction text.
  (The paper ranked on the first substrate in that text; 3 positives share one identical reaction string
  and 2 share another, which is where its 1.00 ties came from.)
FEATURE BLOCKS, scored separately so it is visible which does the work:
  meta = pathway + disruption phenotype + glycosylation + proteome, TF-IDF over character 3-5 grams, cosine
  seq  = amino-acid composition (20 fractions) + 3-mer TF-IDF profile, cosine, averaged
         (replaces the original one-hot "which amino acids occur", which is near-all-ones and carries nothing)
  both = mean of the two
PROTOCOLS:
  A  leave-one-out: rank all other entries; a hit is any other positive.
  B  family holdout at identity threshold t: single-linkage clusters of the positives over BLOSUM62 local
     alignment identity; when querying a positive, every member of its cluster is removed, so a hit must be
     a positive from another family. Run at t = 0.40, 0.35, 0.30 (stricter = larger families removed).
CALIBRATION (measured, see background.py): positive-vs-negative identity is median 0.038 / 95th 0.098,
  positive-vs-positive median 0.256 -- so all 22 positives are homologs of each other and the whole positive
  set is ONE superfamily. Protocol B therefore tests recovery across sub-families, not across unrelated folds.
METRICS: median rank of the best remaining positive, hit@10, hit@50, mean ROC AUC over all remaining
  positives. Random baselines: AUC 0.500; hit@10 and hit@50 printed at run time.

PREDICTIONS (written before running this version):
  W1  protocol A: seq AUC > 0.9, hit@10 > 60% -- the positives are one homologous family.
  W2  protocol A: meta AUC 0.6-0.8, above chance but below seq (only 174 of 3,619 rows have a pathway).
  W3  protocol B at t = 0.30: seq hit@10 falls below half its protocol-A value; AUC falls below 0.8.
  W4  meta is nearly unchanged from A to B (metadata is not family-specific).
  W5  the paper's top three sit in the same clusters as FSA2/CGHA, so those pairings stop counting in B.
"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

AA = "ACDEFGHIKLMNPQRSTVWY"
META_COLS = ["Pathway", "Disruption phenotype", "Glycosylation", "Proteomes"]
DROPPED = ["EC number", "Catalytic activity"]
THRESHOLDS = [0.40, 0.35, 0.30]


def clean(t):
    return re.sub(r"\{.*?\}", "", str(t)) if isinstance(t, str) else ""


def identities(df, pos, cache):
    if cache and os.path.exists(cache):
        return np.array(json.load(open(cache))["ident"])
    from Bio import Align
    from Bio.Align import substitution_matrices
    al = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                               open_gap_score=-11, extend_gap_score=-1)
    seqs = df.loc[pos, "Sequence"].tolist()
    n = len(pos); I = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = seqs[i], seqs[j]
            x = al.align(a, b)[0].aligned
            m = sum(sum(1 for p, q in zip(a[u[0]:u[1]], b[v[0]:v[1]]) if p == q) for u, v in zip(x[0], x[1]))
            I[i, j] = I[j, i] = m / min(len(a), len(b))
    return I


def single_linkage(I, thr):
    n = len(I); p = list(range(n))
    def find(a):
        while p[a] != a: p[a] = p[p[a]]; a = p[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if I[i, j] >= thr:
                a, b = find(i), find(j)
                if a != b: p[a] = b
    out = {}
    for i in range(n): out.setdefault(find(i), []).append(i)
    return list(out.values())


def evaluate(scores, pos_set, qi, excluded):
    n = len(scores)
    mask = np.ones(n, bool); mask[qi] = False
    for e in excluded: mask[e] = False
    lab = np.zeros(n, bool); lab[list(pos_set)] = True
    s, lab = scores[mask], lab[mask]
    if lab.sum() == 0: return None
    order = np.argsort(-s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    best = int(ranks[lab].min()); npos, nneg = int(lab.sum()), int((~lab).sum())
    auc = (nneg * npos + npos * (npos + 1) / 2 - ranks[lab].sum()) / (npos * nneg)
    return dict(best=best, auc=float(auc), hit10=best <= 10, hit50=best <= 50, left=npos)


def main():
    path = sys.argv[1]
    cache = sys.argv[2] if len(sys.argv) > 2 else "ident.json"
    df = pd.read_csv(path)
    pos = df.index[df["EC number"].fillna("").str.contains(r"5\.5\.1")].tolist()
    names = df.loc[pos, "Entry Name"].tolist()
    I = identities(df, pos, cache)
    df = df.drop(columns=DROPPED)

    text = df[META_COLS].apply(lambda r: " ".join(clean(v) for v in r), axis=1)
    M = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(text))
    seqs = df["Sequence"].fillna("").str.upper().tolist()
    comp = np.zeros((len(seqs), 20))
    for i, s in enumerate(seqs):
        if s:
            for j, a in enumerate(AA): comp[i, j] = s.count(a) / len(s)
    comp = normalize(comp)
    kmer = normalize(TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=3).fit_transform(seqs))

    def sc_meta(i): return np.asarray((M[i] @ M.T).todense()).ravel()
    def sc_seq(i): return 0.5 * (comp[i] @ comp.T).ravel() + 0.5 * np.asarray((kmer[i] @ kmer.T).todense()).ravel()
    blocks = {"meta": sc_meta, "seq": sc_seq, "both": lambda i: 0.5 * (sc_meta(i) + sc_seq(i))}

    npos, N = len(pos), len(df)
    p = (npos - 1) / (N - 1)
    print(f"{N} entries, {npos} positives (EC 5.5.1). Features: {META_COLS} + Sequence. Dropped: {DROPPED}.")
    print(f"random baselines: AUC 0.500, hit@10 {(1-(1-p)**10)*100:.1f}%, hit@50 {(1-(1-p)**50)*100:.1f}%")
    off = I[~np.eye(npos, dtype=bool)]
    print(f"positive-vs-positive identity: median {np.median(off):.3f}, max {off.max():.3f} "
          f"(background positive-vs-negative is median 0.038, 95th 0.098 -- all positives are one superfamily)")

    rows = {}
    print(f"\n{'protocol':>22} {'block':>5} {'median rank':>12} {'hit@10':>8} {'hit@50':>8} {'AUC':>7} {'pos left':>9}")
    protocols = [("A leave-one-out", None)] + [(f"B family holdout t={t:.2f}", t) for t in THRESHOLDS]
    for pname, thr in protocols:
        fam_of = {}
        if thr is not None:
            for fi, f in enumerate(single_linkage(I, thr)):
                for i in f: fam_of[pos[i]] = fi
        for bname, fn in blocks.items():
            res = []
            for k, qi in enumerate(pos):
                excl = [q for q in pos if q != qi and fam_of[q] == fam_of[qi]] if thr is not None else []
                r = evaluate(fn(qi), set(pos), qi, excl)
                if r: res.append(r); rows[(pname, bname, names[k])] = r
            if res:
                print(f"{pname:>22} {bname:>5} {np.median([r['best'] for r in res]):12.0f} "
                      f"{np.mean([r['hit10'] for r in res])*100:7.1f}% {np.mean([r['hit50'] for r in res])*100:7.1f}% "
                      f"{np.mean([r['auc'] for r in res]):7.3f} {np.mean([r['left'] for r in res]):9.1f}")

    print("\nfamilies at t = 0.30 (single linkage over alignment identity):")
    for f in sorted(single_linkage(I, 0.30), key=len, reverse=True):
        print(f"  {len(f):2d}: {', '.join(names[i] for i in f)}")

    print("\nbest rank per query, seq block (A -> B at t=0.30), * = the paper's references or its top three:")
    star = {"CGHA_CHAGB", "FSA2_FUSSF", "PHM7_PYRSX", "EQX3_GIBF5", "EQX3_FUSHE", "TAS3_HAPIR"}
    for nm in names:
        a = rows.get(("A leave-one-out", "seq", nm)); b = rows.get(("B family holdout t=0.30", "seq", nm))
        print(f"  {nm:>12}{'*' if nm in star else ' '} {a['best']:>5} -> {b['best']:>5}"
              f"   (meta {rows[('A leave-one-out','meta',nm)]['best']:>4} -> "
              f"{rows[('B family holdout t=0.30','meta',nm)]['best']:>4})")


if __name__ == "__main__":
    main()
