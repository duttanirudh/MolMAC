"""Background identity: positives vs random negatives, to calibrate what counts as homology here."""
import sys
import numpy as np, pandas as pd, json
from Bio import Align
from Bio.Align import substitution_matrices
df = pd.read_csv((sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'))
pos = df.index[df['EC number'].fillna('').str.contains(r'5\.5\.1')].tolist()
neg = [i for i in df.index if i not in pos]
rng = np.random.default_rng(0); samp = rng.choice(neg, 200, replace=False)
al = Align.PairwiseAligner(mode='local', substitution_matrix=substitution_matrices.load('BLOSUM62'),
                           open_gap_score=-11, extend_gap_score=-1)
def ident(a, b):
    x = al.align(a, b)[0].aligned
    m = sum(sum(1 for p, q in zip(a[u[0]:u[1]], b[v[0]:v[1]]) if p == q) for u, v in zip(x[0], x[1]))
    return m / min(len(a), len(b))
vals = []
for i in pos:
    for j in samp[:60]:
        vals.append(ident(df.loc[i, 'Sequence'], df.loc[j, 'Sequence']))
vals = np.array(vals)
print(f'positive-vs-negative identity (n={len(vals)}): median {np.median(vals):.3f}, '
      f'90th {np.percentile(vals,90):.3f}, 95th {np.percentile(vals,95):.3f}, 99th {np.percentile(vals,99):.3f}, max {vals.max():.3f}')
d = json.load(open('ident.json')); I = np.array(d['ident']); off = I[~np.eye(len(I), dtype=bool)]
print(f'positive-vs-positive  (n={len(off)}): median {np.median(off):.3f}, 90th {np.percentile(off,90):.3f}, max {off.max():.3f}')
for t in [0.30, 0.35, 0.40]:
    print(f'  threshold {t}: {100*(off>=t).mean():.1f}% of positive pairs, {100*(vals>=t).mean():.2f}% of background pairs')
