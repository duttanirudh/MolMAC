"""Pairwise local-alignment identity among the 22 positives (BLOSUM62, Smith-Waterman via PairwiseAligner)."""
import sys
import numpy as np, pandas as pd, json, sys
from Bio import Align
from Bio.Align import substitution_matrices

df = pd.read_csv((sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'))
pos = df.index[df['EC number'].fillna('').str.contains(r'5\.5\.1')].tolist()
names = df.loc[pos, 'Entry Name'].tolist(); seqs = df.loc[pos, 'Sequence'].tolist()
al = Align.PairwiseAligner(mode='local', substitution_matrix=substitution_matrices.load('BLOSUM62'),
                           open_gap_score=-11, extend_gap_score=-1)
n = len(pos); I = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        a = al.align(seqs[i], seqs[j])[0]
        s = a.aligned
        m = sum(sum(1 for x, y in zip(seqs[i][p[0]:p[1]], seqs[j][q[0]:q[1]]) if x == y)
                for p, q in zip(s[0], s[1]))
        I[i, j] = I[j, i] = m / min(len(seqs[i]), len(seqs[j]))
json.dump({'names': names, 'ident': I.tolist()}, open('ident.json', 'w'))
print('pairs with alignment identity >= 0.20:')
for i in range(n):
    for j in range(i + 1, n):
        if I[i, j] >= 0.20: print(f'  {names[i]:>12} {names[j]:>12}  {I[i,j]:.2f}')
off = I[~np.eye(n, dtype=bool)]
print(f'off-diagonal identity: median {np.median(off):.3f}, 90th {np.percentile(off,90):.3f}, max {off.max():.3f}')
