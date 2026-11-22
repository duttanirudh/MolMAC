"""
no_rr.py  --  can the Ogita-Aishima start be made of matrix products only?

The s2_oa.py start does one Rayleigh-Ritz per block (two N/2 x N/2 diagonalizations). Variants here:
  RR      as before (reference)
  NONE    no diagonalization: X = [Co Cv] straight from the range finder, then OA steps
  POWER   no diagonalization; before OA, m steps of block power iteration with shifted H inside each block
          (C <- orth((s I - H) C) for the occupied block, orth((H - s' I) C) for the virtual block; QR only),
          to spread the eigenvalue estimates d_i inside each block before OA divides by their differences
OA uses the cluster threshold delta1 so near-equal d_i inside a block fall back to r_ij / 2.
60 runs per cell (6 settings x 10 seeds). Front ends: FP16PAIR, INT8 (digital), AN10 (analog, 10-bit ADC).

PREDICTIONS (written before running):
  N1  NONE fails for most runs at every N: inside a block the range-finder columns are not eigenvectors, so the
      OA formula (which assumes near-eigenvectors) diverges or stalls.
  N2  delta1 = 0.1 rescues part of NONE (cluster handling treats a whole block's near-equal estimates as one
      group), but stays below RR.
  N3  POWER with m = 20 is still below RR at N >= 24 (block power iteration converges slowly when the
      spectrum inside a block is dense).
"""
import itertools
import time
import numpy as np
import scale
from s2_oa import oa_step, map_err

TARGET, KOA = 1e-5, 30


def bases(P, no, seed):
    N = len(P); P = P.astype(np.float64); rng = np.random.default_rng(seed)
    Co, _ = np.linalg.qr(P @ rng.standard_normal((N, no)))
    Q, _ = np.linalg.qr((np.eye(N) - P) @ rng.standard_normal((N, N - no)))
    Cv, _ = np.linalg.qr(Q - Co @ (Co.T @ Q))
    return Co, Cv


def start(P, H, no, seed, mode, m=20):
    Co, Cv = bases(P, no, seed)
    if mode == "RR":
        Co, _ = scale._rr(Co, H); Cv, _ = scale._rr(Cv, H)
    elif mode == "POWER":
        a, b = scale.gersh(H); N = len(H)
        for _ in range(m):
            Co, _ = np.linalg.qr((b * np.eye(N) - H) @ Co)
            Cv, _ = np.linalg.qr((H - a * np.eye(N)) @ Cv)
            Cv, _ = np.linalg.qr(Cv - Co @ (Co.T @ Cv))
    return np.hstack([Co, Cv])


def run(P, H, A, no, ref, seed, mode, delta1):
    X = start(P, H, no, seed, mode)
    d = np.diag(X.T @ H @ X)
    for k in range(KOA + 1):
        if not np.all(np.isfinite(X)) or np.abs(X).max() > 1e3: return None
        if map_err(X, d, H, A, no, ref) <= TARGET: return k
        X, d = oa_step(X, H, delta1)
    return None


def main():
    np.seterr(all="ignore"); t0 = time.time()
    fronts = {"FP16PAIR": scale.mm_fp16pair, "INT8": scale.mm_int8, "AN10": scale.mm_analog(10)}
    cases = [("RR", 1e-10), ("NONE", 1e-10), ("NONE", 0.1), ("POWER", 1e-10), ("POWER", 0.1)]
    print(f"success over 60 runs, median OA steps; up to {KOA} OA steps")
    for N in [8, 16, 24, 40]:
        no = N // 2
        sysm = [scale.system(N, g, ks) for ks, g in itertools.product([(1, 1, 1), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5])]
        refs = [scale.reference(H, A, no) for H, A in sysm]
        for f, mm in fronts.items():
            Ps = [scale.sp2(H, no, mm)[0] for H, A in sysm]
            cells = []
            for mode, d1 in cases:
                res = [run(P, H, A, no, r, sd, mode, d1) for P, (H, A), r in zip(Ps, sysm, refs) for sd in range(10)]
                ok = [k for k in res if k is not None]
                cells.append(f"{mode}/{d1:g}: {len(ok) / len(res):.2f} ({int(np.median(ok)) if ok else '-'})")
            print(f"N={N:2d} {f:>8} | " + " | ".join(cells), flush=True)
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
