"""
r123_float64.py  --  three routes to the catalyst sensitivity map dI/d(delta_k), float64.

System and units as in h1_acrolein.py (imported). delta_k lowers atom k's on-site energy: H_kk -= delta_k.
Observable: I = D[C1,Cb] - D[C4,Ca] = Tr(A D), with A symmetric (+1/2 at (0,4),(4,0); -1/2 at (3,5),(5,3)).

Reference: analytic first-order perturbation theory from the eigendecomposition.
R1  direct perturbed SP2: carry X' alongside X (one perturbation, i.e. one atom, per run).
R2  dual susceptibility: reverse (adjoint) pass through the stored SP2 sequence; ONE pass gives
    chi = dI/dH for all entries; atom sensitivities are -diag(chi).
    (My implementation is the adjoint of the SP2 recursion. It is equal in exact arithmetic to the
     susceptibility idea in arXiv 2409.17033; whether it matches their formulation step by step is unchecked.)
R3  re-solve SP2 with H_kk -/+ h, central differences (one weight reload per solve on analog hardware).
Spectral bounds (Gershgorin) are held fixed under perturbation for R1/R2, as is the branch sequence.

PREDICTIONS (written before running):
  Q1  R1 and R2 match the reference to <= 1e-10 in every setting.
  Q2  R3 matches to <= 1e-7 (truncation ~h^2 with h = 1e-4; rounding ~1e-16/h).
  Q3  R1 and R2 agree with each other to <= 1e-12 (the paper reports 12-digit agreement in double).
  Q4  Matrix-product counts per full 8-atom map: R2 smallest (~2x one solve), R1 ~ 8x more products
      than R2, R3 ~ 16 solves.
"""
import itertools
import numpy as np
from h1_acrolein import hamiltonian, N_OCC

A_OBS = np.zeros((8, 8))
A_OBS[0, 4] = A_OBS[4, 0] = 0.5
A_OBS[3, 5] = A_OBS[5, 3] = -0.5


def gershgorin(H):
    r = np.abs(H).sum(axis=1) - np.abs(np.diag(H))
    return float((np.diag(H) - r).min()), float((np.diag(H) + r).max())


def sp2_forward(H, bounds=None, max_iter=100):
    emin, emax = bounds if bounds else gershgorin(H)
    w = emax - emin
    X = (emax * np.eye(len(H)) - H) / w
    Xs, branches, hist, nprod = [], [], [], 0
    for it in range(1, max_iter + 1):
        Xs.append(X)
        X2 = X @ X; nprod += 1
        sq = abs(np.trace(X2) - N_OCC) < abs(2 * np.trace(X) - np.trace(X2) - N_OCC)
        X = X2 if sq else 2 * X - X2
        branches.append(sq)
        hist.append(float(np.trace(X) - np.sum(X * X)))   # Tr(X - X^2) without an extra product
        if it > 4 and hist[-1] >= hist[-3]:
            break
    return X, Xs, branches, w, nprod


def reference(H):
    e, C = np.linalg.eigh(H)
    s = np.zeros(8)
    for k in range(8):
        Hp = np.zeros((8, 8)); Hp[k, k] = -1.0
        dD = np.zeros((8, 8))
        for i in range(N_OCC):
            for j in range(N_OCC, 8):
                m = C[:, j] @ Hp @ C[:, i]
                dD += 2 * m / (e[i] - e[j]) * (np.outer(C[:, i], C[:, j]) + np.outer(C[:, j], C[:, i]))
        s[k] = np.sum(A_OBS * dD)
    return s


def r1(H):
    _, Xs, br, w, nprod = sp2_forward(H)
    s, total = np.zeros(8), nprod
    for k in range(8):
        dX = np.zeros((8, 8)); dX[k, k] = 1.0 / w          # dX0 = -dH/w, dH_kk = -1
        for X, sq in zip(Xs, br):
            sym = dX @ X + X @ dX; total += 2
            dX = sym if sq else 2 * dX - sym
        s[k] = np.sum(A_OBS * 2 * dX)
    return s, total


def r2(H):
    _, Xs, br, w, nprod = sp2_forward(H)
    Y, total = 2 * A_OBS, nprod                              # d(Tr A D)/dX_N with D = 2 X_N
    for X, sq in zip(reversed(Xs), reversed(br)):
        sym = X @ Y + Y @ X; total += 2
        Y = sym if sq else 2 * Y - sym
    chi = -Y / w                                             # dI/dH
    return -np.diag(chi), total


def r3(H, h=1e-4):
    s, total = np.zeros(8), 0
    for k in range(8):
        Hp, Hm = H.copy(), H.copy()
        Hp[k, k] -= h; Hm[k, k] += h
        Xp, *_, n1 = sp2_forward(Hp); Xm, *_, n2 = sp2_forward(Hm)
        total += n1 + n2
        s[k] = (np.sum(A_OBS * 2 * Xp) - np.sum(A_OBS * 2 * Xm)) / (2 * h)
    return s, total


def main():
    k_sets = {"std": (1.0, 1.0, 1.0), "2312": (1.1, 0.9, 1.0)}
    worst = {"R1": 0, "R2": 0, "R3": 0, "R1-R2": 0}
    print(f"{'k':>5} {'hO':>4} {'gam':>4} {'|R1-ref|':>10} {'|R2-ref|':>10} {'|R3-ref|':>10} "
          f"{'|R1-R2|':>10} {'prod R1':>8} {'prod R2':>8} {'prod R3':>8}")
    for (kn, ks), hO, g in itertools.product(k_sets.items(), [0.5, 1.0, 2.0], [0.1, 0.3, 0.5]):
        H = hamiltonian(g, hO, *ks)
        ref = reference(H)
        (s1, n1), (s2, n2), (s3, n3) = r1(H), r2(H), r3(H)
        d = {"R1": np.abs(s1 - ref).max(), "R2": np.abs(s2 - ref).max(),
             "R3": np.abs(s3 - ref).max(), "R1-R2": np.abs(s1 - s2).max()}
        for key in worst: worst[key] = max(worst[key], d[key])
        print(f"{kn:>5} {hO:4.1f} {g:4.1f} {d['R1']:10.1e} {d['R2']:10.1e} {d['R3']:10.1e} "
              f"{d['R1-R2']:10.1e} {n1:8d} {n2:8d} {n3:8d}")
    print("\nWorst over 18 settings:", {k: f"{v:.1e}" for k, v in worst.items()})
    H = hamiltonian(0.3, 1.0, 1.0, 1.0, 1.0)
    print("Map (std, hO=1, g=0.3) ref:", np.array2string(reference(H), precision=4))


if __name__ == "__main__":
    main()
