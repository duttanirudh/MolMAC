"""
s2_oa.py  --  S2 of the scale-up: products-only refinement (Ogita-Aishima, basic form) started from
each front end. Question: which front-end precision lands the start inside the refiner's basin, vs N?

START: front-end projector P -> range-finder bases Co, Cv (as before) -> ONE Rayleigh-Ritz per block
       (the only diagonalizations; N/2 x N/2, once) -> full approximate eigenvector matrix X = [Co Cv].
REFINER (Ogita-Aishima basic step, float64; arXiv 1909.13709 description, denominator d_j - d_i):
   R = I - X^T X,  S = X^T H X,  d_i = s_ii / (1 - r_ii)
   e_ii = r_ii / 2
   e_ij = (s_ij + d_j r_ij) / (d_j - d_i)   if |d_j - d_i| >= delta1,  else r_ij / 2
   X <- X (I + E)        -> 4 dense N x N products per step (X^T X, H X, X^T (H X), X E)
Occupied set after each step = the N/2 columns with the smallest d (aufbau by sorting).
"Converged" = map rel. error <= 1e-5 within 10 OA steps; "failed" otherwise (incl. divergence).
Runs per (N, front end): 6 settings x 10 range-finder seeds = 60.

PREDICTIONS (written before running):
  T1  COLD fails in most runs (> 50%) at every N: one Rayleigh-Ritz from a near-random subspace
      leaves the start far outside the basin (||E|| < 1/3 required).
  T2  FP16PAIR converges in >= 95% of runs at every N, in <= 3 OA steps (quadratic convergence).
  T3  INT8 converges in most runs at N = 8, but its success rate drops with N (< 80% by N = 40).
  T4  AN8 is worse than INT8 at every N; AN10 lies between INT8 and FP16PAIR.
  T5  delta1 = 1e-3 (cluster handling on) gives fewer failures than delta1 = 1e-10 at gamma = 0.1,
      where the weakly coupled fragments give near-degenerate pairs.
"""
import itertools
import time
import numpy as np
import scale

TARGET, KOA = 1e-5, 10


def start_X(P, H, no, seed):
    N = len(H); P = P.astype(np.float64)
    rng = np.random.default_rng(seed)
    Co, _ = np.linalg.qr(P @ rng.standard_normal((N, no)))
    Q, _ = np.linalg.qr((np.eye(N) - P) @ rng.standard_normal((N, N - no)))
    Cv, _ = np.linalg.qr(Q - Co @ (Co.T @ Q))
    Co, _ = scale._rr(Co, H); Cv, _ = scale._rr(Cv, H)
    return np.hstack([Co, Cv])


def oa_step(X, H, delta1):
    R = np.eye(len(H)) - X.T @ X
    S = X.T @ (H @ X)
    d = np.diag(S) / (1.0 - np.diag(R))
    Dd = d[None, :] - d[:, None]                      # Dd[i, j] = d_j - d_i
    far = np.abs(Dd) >= delta1
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where(far, (S + d[None, :] * R) / Dd, R / 2.0)
    np.fill_diagonal(E, np.diag(R) / 2.0)
    return X + X @ E, d


def map_err(X, d, H, A, no, ref):
    order = np.argsort(d)
    o, v = order[:no], order[no:]
    Co, Cv = X[:, o], X[:, v]
    s = scale.map_from_pairs(Co, Cv, d[o], d[v], A)
    if not np.all(np.isfinite(s)): return np.inf
    return np.abs(s - ref).max() / np.abs(ref).max()


def run(P, H, A, no, ref, seed, delta1):
    X = start_X(P, H, no, seed)
    d = np.diag(X.T @ H @ X)
    for k in range(KOA + 1):
        if not np.all(np.isfinite(X)) or np.abs(X).max() > 1e3: return None
        if map_err(X, d, H, A, no, ref) <= TARGET: return k
        X, d = oa_step(X, H, delta1)
    return None


def main():
    np.seterr(all="ignore"); t0 = time.time()
    fronts = {"COLD": None, **{k: v[0] for k, v in scale.BACKENDS.items()}}
    ksets, gammas, seeds = [(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5], range(10)
    print(f"Ogita-Aishima from each front end: success rate over 60 runs (6 settings x 10 seeds), "
          f"median OA steps when converged; target {TARGET:g} within {KOA} steps")
    for delta1 in [1e-10, 1e-3]:
        print(f"\ndelta1 = {delta1:g}")
        print(f"{'N':>3} " + " ".join(f"{f:>16}" for f in fronts) + "   fail@g=0.1 (per front end)")
        for N in [8, 12, 16, 24, 32, 40]:
            no = N // 2
            sysm = [(scale.system(N, g, ks), g) for ks, g in itertools.product(ksets, gammas)]
            cells, g01 = [], []
            for f, mm in fronts.items():
                res, fails01 = [], 0
                for (H, A), g in sysm:
                    ref = scale.reference(H, A, no)
                    if mm is None:
                        a, b = scale.gersh(H); P = (b * np.eye(N) - H) / (b - a)
                    else:
                        P = scale.sp2(H, no, mm)[0]
                    for sd in seeds:
                        k = run(P, H, A, no, ref, sd, delta1); res.append(k)
                        fails01 += (k is None) and g == 0.1
                ok = [k for k in res if k is not None]
                cells.append(f"{len(ok) / len(res):5.2f} ({int(np.median(ok)) if ok else '-'} st)")
                g01.append(str(fails01))
            print(f"{N:3d} " + " ".join(f"{c:>16}" for c in cells) + "   " + "/".join(g01))
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
