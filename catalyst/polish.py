"""
polish.py  --  can int8 SP2 be rescued by a short fp32 stage?  (butadiene + acrolein, 18 settings)

A  H-blind polish: int8 SP2 to its stall, then m more SP2 steps in fp32 (SP2 steps use X only, never H).
B  switch point: int8 products for the first n_s SP2 steps, fp32 afterwards.
C  H-aware refinement in fp32, starting from the int8 projector, k steps of:
     Rayleigh-Ritz inside the occupied and virtual blocks, then a first-order occupied-virtual
     correction Z = (Cv^T H Co) / (e_o - e_v), update, re-orthonormalize (QR).
   Same spirit as matrix-multiplication-based eigenvector refinement (Ogita-Aishima); not their algorithm.
   Uses small block diagonalizations (4x4) and a QR, done digitally.
   The sensitivity map is then computed from the refined eigenpairs (first-order perturbation formula).

PREDICTIONS (written before running):
  A1  polish makes P idempotent to the fp32 level (~1e-7), but the error in I stays at the int8 level
      (median ~0.26), because SP2 steps cannot see H (occupied-virtual error is not damped; H4).
  A2  the energy 2 Tr(P H) with the polished P has relative error ~1e-5 (second order in subspace error).
  B1  any n_s >= 1 leaves the I error >= 1e-3 (far above fp32's ~1e-6); no clean decay with later switching.
  C1  quadratic convergence: median I error <= 1e-2 after k=1, <= 1e-4 after k=2, fp32 floor (~1e-6) by k=3.
  C2  map from refined eigenpairs at k=3: median rel. error ~1e-6, 0 sign failures.
"""
import itertools
import numpy as np
from h1_acrolein import hamiltonian, N_OCC
from r123_float64 import A_OBS, reference, gershgorin
from ladder import mm_int8, mm_fp32, sp2, metrics

f32 = np.float32
SETTINGS = list(itertools.product([(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.5, 1.0, 2.0], [0.1, 0.3, 0.5]))


def I_of(P): return float(np.sum(A_OBS * 2 * P.astype(np.float64)))


def exact_I(H): return I_of(sp2(H, lambda a, b: a @ b, np.float64)[0])


def sp2_switch(H, n_s, max_iter=100):
    emin, emax = gershgorin(H)
    X = ((emax * np.eye(8) - H) / (emax - emin)).astype(f32)
    hist = []
    for it in range(1, max_iter + 1):
        mm = mm_int8 if it <= n_s else mm_fp32
        X2 = mm(X, X)
        t1, t2 = np.trace(X), np.trace(X2)
        X = X2 if abs(t2 - N_OCC) < abs(2 * t1 - t2 - N_OCC) else (2 * X - X2).astype(f32)
        hist.append(float(np.trace(X) - np.sum(X * X)))
        if it > max(4, n_s + 4) and hist[-1] >= hist[-3]:
            break
    return X


def polish_blind(X, m):
    for _ in range(m):
        X2 = mm_fp32(X, X)
        t1, t2 = np.trace(X), np.trace(X2)
        X = X2 if abs(t2 - N_OCC) < abs(2 * t1 - t2 - N_OCC) else (2 * X - X2).astype(f32)
    return X


def refine(P, H, k):
    H = H.astype(f32)
    w, V = np.linalg.eigh(P.astype(f32))
    Cv, Co = V[:, :8 - N_OCC], V[:, 8 - N_OCC:]
    for _ in range(k):
        eo, Uo = np.linalg.eigh(Co.T @ H @ Co); Co = Co @ Uo
        ev, Uv = np.linalg.eigh(Cv.T @ H @ Cv); Cv = Cv @ Uv
        Z = (Cv.T @ H @ Co) / (eo[None, :] - ev[:, None])
        Co, Cv = Co + Cv @ Z, Cv - Co @ Z.T
        Co, _ = np.linalg.qr(Co)
        Cv = Cv - Co @ (Co.T @ Cv); Cv, _ = np.linalg.qr(Cv)
    eo, Uo = np.linalg.eigh(Co.T @ H @ Co); Co = Co @ Uo
    ev, Uv = np.linalg.eigh(Cv.T @ H @ Cv); Cv = Cv @ Uv
    return (Co @ Co.T).astype(f32), Co, Cv, eo, ev


def map_from_pairs(Co, Cv, eo, ev):
    s = np.zeros(8)
    for kk in range(8):
        # H' = -e_k e_k^T ; dD = 2 sum_ij m_ji/(e_i - e_j) (c_i c_j^T + c_j c_i^T)
        M = -np.outer(Cv[kk, :], Co[kk, :])                 # m_ji = c_j^T H' c_i
        G = M / (eo[None, :] - ev[:, None])
        dD = 2 * (Co @ G.T @ Cv.T + Cv @ G @ Co.T)
        s[kk] = float(np.sum(A_OBS * dD.astype(np.float64)))
    return s


def med(x): return float(np.median(x))


def main():
    Hs = [hamiltonian(g, hO, *ks) for ks, hO, g in SETTINGS]
    Iex = [exact_I(H) for H in Hs]
    Eex = [float(2 * np.trace(sp2(H, lambda a, b: a @ b, np.float64)[0] @ H)) for H in Hs]
    X8 = [sp2(H, mm_int8, f32)[0] for H in Hs]

    print("A  H-blind fp32 SP2 polish after int8 stall (median over 18)")
    print(f"{'m':>3} {'idempotency':>12} {'rel err I':>10} {'worst I':>9} {'rel err E':>10}")
    for m in [0, 1, 2, 5, 10]:
        idem, eI, eE = [], [], []
        for H, X, Ie, Ee in zip(Hs, X8, Iex, Eex):
            P = polish_blind(X, m)
            idem.append(float(np.linalg.norm(P @ P - P)))
            eI.append(abs(I_of(P) - Ie) / abs(Ie))
            eE.append(abs(2 * float(np.trace(P.astype(np.float64) @ H)) - Ee) / abs(Ee))
        print(f"{m:3d} {med(idem):12.1e} {med(eI):10.1e} {max(eI):9.1e} {med(eE):10.1e}")

    print("\nB  int8 for the first n_s SP2 steps, then fp32 (median / worst rel err of I)")
    for n_s in [0, 1, 2, 3, 5, 8, 12, 100]:
        e = [abs(I_of(sp2_switch(H, n_s)) - Ie) / abs(Ie) for H, Ie in zip(Hs, Iex)]
        print(f"  n_s={n_s:3d}  {med(e):9.1e} / {max(e):9.1e}")

    print("\nC  H-aware fp32 refinement from the int8 projector")
    print(f"{'k':>3} {'rel err I':>10} {'worst I':>9} {'map rel':>9} {'map worst':>10} {'sign fails':>10}")
    for k in [0, 1, 2, 3, 4]:
        eI, mr, sf = [], [], 0
        for H, X, Ie in zip(Hs, X8, Iex):
            P, Co, Cv, eo, ev = refine(X, H, k)
            eI.append(abs(I_of(P) - Ie) / abs(Ie))
            mt = metrics(map_from_pairs(Co, Cv, eo, ev), reference(H))
            mr.append(mt["rel"]); sf += mt["signs"]
        print(f"{k:3d} {med(eI):10.1e} {max(eI):9.1e} {med(mr):9.1e} {max(mr):10.1e} {sf:6d}/144")


if __name__ == "__main__":
    main()
