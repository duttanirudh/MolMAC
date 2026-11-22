"""
refine.py  --  H-aware fp32 refinement of an approximate projector (shared by polish/analog scripts).

Start: occupied basis from a RANGE FINDER (P @ R, fixed seeded Gaussian R, then QR) -- no full
eigendecomposition. Virtual basis from (I - P) @ R2, orthogonalized against it.
Each step: Rayleigh-Ritz inside each block (4x4 diagonalizations, done digitally), then the
first-order occupied-virtual correction Z = (Cv^T H Co) / (e_o - e_v), update, re-orthonormalize.
Known cost caveat: the 4x4 block diagonalizations are O(N_occ^3); a products-only refinement
(e.g. Ogita-Aishima) is an open item.
"""
import numpy as np
from h1_acrolein import N_OCC
from r123_float64 import A_OBS

f32 = np.float32
_rng = np.random.default_rng(7)
R_OCC = _rng.standard_normal((8, N_OCC)).astype(f32)
R_VIR = _rng.standard_normal((8, 8 - N_OCC)).astype(f32)


def initial_bases(P, seed=None):
    P = P.astype(f32)
    if seed is None:
        Ro, Rv = R_OCC, R_VIR
    else:                                                    # per-run range-finder seed
        g = np.random.default_rng(seed)
        Ro = g.standard_normal((8, N_OCC)).astype(f32); Rv = g.standard_normal((8, 8 - N_OCC)).astype(f32)
    Co, _ = np.linalg.qr(P @ Ro)
    Q, _ = np.linalg.qr((np.eye(8, dtype=f32) - P) @ Rv)
    Cv, _ = np.linalg.qr(Q - Co @ (Co.T @ Q))
    return Co, Cv


def _rr(C, H):
    e, U = np.linalg.eigh(C.T @ H @ C)
    return C @ U, e


def _rr_both(Co, Cv, H, aufbau=True, max_swaps=8):
    """Rayleigh-Ritz in both blocks; if the aufbau order is violated (highest occupied Ritz value
    above the lowest virtual one), swap that pair of vectors between blocks and redo."""
    Co, eo = _rr(Co, H); Cv, ev = _rr(Cv, H)
    for _ in range(max_swaps if aufbau else 0):
        i, j = int(np.argmax(eo)), int(np.argmin(ev))
        if eo[i] <= ev[j]:
            break
        Co, Cv = Co.copy(), Cv.copy()
        Co[:, i], Cv[:, j] = Cv[:, j].copy(), Co[:, i].copy()
        Co, eo = _rr(Co, H); Cv, ev = _rr(Cv, H)
    return Co, Cv, eo, ev


def refine(P, H, k, aufbau=True, seed=None):
    if not np.all(np.isfinite(P)):
        raise FloatingPointError("non-finite projector")
    H = H.astype(f32)
    Co, Cv = initial_bases(P, seed)
    for _ in range(k):
        Co, Cv, eo, ev = _rr_both(Co, Cv, H, aufbau)
        Z = (Cv.T @ H @ Co) / (eo[None, :] - ev[:, None])
        Co, Cv = Co + Cv @ Z, Cv - Co @ Z.T
        Co, _ = np.linalg.qr(Co)
        Cv, _ = np.linalg.qr(Cv - Co @ (Co.T @ Cv))
    Co, Cv, eo, ev = _rr_both(Co, Cv, H, aufbau)
    return (Co @ Co.T).astype(f32), Co, Cv, eo, ev


def map_from_pairs(Co, Cv, eo, ev):
    """dI/d(lower alpha_k) from eigenpairs, first-order perturbation theory."""
    s = np.zeros(8)
    for kk in range(8):
        M = -np.outer(Cv[kk, :], Co[kk, :])
        G = M / (eo[None, :] - ev[:, None])
        dD = 2 * (Co @ G.T @ Cv.T + Cv @ G @ Co.T)
        s[kk] = float(np.sum(A_OBS * dD.astype(np.float64)))
    return s
