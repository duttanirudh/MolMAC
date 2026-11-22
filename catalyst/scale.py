"""
scale.py  --  S1 of the scale-up: does the low-precision / analog front end save more refinement
steps as the system grows?  Deterministic backends only (no Monte Carlo).

FAMILY (toy, for scaling the numerics; chemically only loosely a Diels-Alder series):
  diene      = all-carbon polyene of length N/2; forming bonds at its first four atoms (C1, C4)
  dienophile = polyenal of length N/2: Cb=Ca-C=C-...-C=O (Cb terminal, O last)
  forming bonds C1-Cb and C4-Ca with coupling gamma. N = 8 reproduces butadiene + acrolein.
  Bond parameters: std (all k = 1) and the 2312.02020 set (C=C 1.1, C-C 0.9, C=O 1.0); h_O = 1.
  Occupied orbitals N/2. Observable I = p(C1-Cb) - p(C4-Ca); map = dI/d(lower alpha_k), all N atoms.

FRONT ENDS (each followed by the H-aware refinement with aufbau check, k steps, float64):
  COLD      X0 = scaled H, no SP2 (control)
  FP16PAIR  SP2 with FP16-pair products
  INT8      SP2 with int8 products (digital, exact integer accumulation)
  AN8       analog chain, 8-bit weights / 8-bit duty / 8-bit ADC (no gain, offset or noise)
  AN10      same with a 10-bit ADC
Reported per N: k at which all settings reach map rel. error <= 1e-5 ("k_all"), steps saved vs COLD,
median SP2 iterations, and MAC counts (front-end MACs; digital MACs per refinement step).

PREDICTIONS (written before running):
  S1a  COLD k_all grows with N (gap shrinks): from 4 at N = 8 to about 6-8 at N = 40.
  S1b  INT8 saves about 1 step at N = 8, growing slowly to +1..+2 by N = 40.
  S1c  AN8 is a net cost or 0 at every N; AN10 saves 1-2 steps at every N.
  S1d  SP2 iterations grow roughly like log(width/gap): from ~18 at N = 8 to ~22-26 at N = 40.
"""
import itertools
import time
import numpy as np

f16, f32 = np.float16, np.float32
TARGET, KMAX = 1e-5, 12


def system(N, gamma, ks, hO=1.0):
    k1, k2, k3 = ks
    L = N // 2
    H = np.zeros((N, N))
    for i in range(L - 1):                                   # diene polyene
        H[i, i + 1] = H[i + 1, i] = -1.0
    d = list(range(L, N))                                    # dienophile: Cb, Ca, ..., C, O
    for t in range(L - 1):
        i, j = d[t], d[t + 1]
        k = k3 if t == L - 2 else (k1 if t % 2 == 0 else k2)
        H[i, j] = H[j, i] = -k
    H[N - 1, N - 1] = -hO
    for i, j in [(0, L), (3, L + 1)]:
        H[i, j] = H[j, i] = -gamma
    A = np.zeros((N, N)); A[0, L] = A[L, 0] = 0.5; A[3, L + 1] = A[L + 1, 3] = -0.5
    return H, A


def gersh(H):
    r = np.abs(H).sum(1) - np.abs(np.diag(H))
    return float((np.diag(H) - r).min()), float((np.diag(H) + r).max())


def map_from_pairs(Co, Cv, eo, ev, A):
    """dI/d(lower alpha_k) for all k, vectorized first-order perturbation theory."""
    G = 1.0 / (eo[None, :] - ev[:, None])                     # (nv, no)
    # dD/d(lower alpha_k) = -2 sum_ij G_ji Cv_kj Co_ki (Co_i Cv_j^T + Cv_j Co_i^T)
    # I = Tr(A D) -> dI_k = -4 sum_ij G_ji Cv_kj Co_ki (Co_i^T A Cv_j)
    B = Co.T @ A @ Cv                                        # (no, nv)
    M = (G.T * B)                                            # (no, nv)
    return -4.0 * np.einsum("ki,ij,kj->k", Co, M, Cv)


def reference(H, A, no):
    e, C = np.linalg.eigh(H)
    return map_from_pairs(C[:, :no], C[:, no:], e[:no], e[no:], A)


# ---------------- products ----------------
def mm_fp16pair(X, Y):
    X, Y = X.astype(f32), Y.astype(f32)
    Xh = X.astype(f16).astype(f32); Xl = (X - Xh).astype(f16).astype(f32)
    Yh = Y.astype(f16).astype(f32); Yl = (Y - Yh).astype(f16).astype(f32)
    return Xh @ Yh + Xh @ Yl + Xl @ Yh


def _q(X, bits):
    s = float(np.abs(X).max()) or 1.0; L = 2 ** (bits - 1) - 1
    return np.round(X / s * L), s / L


def mm_int8(X, Y):
    Xq, sx = _q(X, 8); Yq, sy = _q(Y, 8)
    return ((Xq @ Yq) * (sx * sy)).astype(f32)


def mm_analog(adc_bits):
    def mm(X, Y):
        X, Y = X.astype(f32), Y.astype(f32)
        sw = float(np.abs(Y).max()) or 1.0
        W = (np.round(Y / sw * 127) / 127).astype(f32); FS = np.abs(W).sum(0) + 1e-30
        sa = float(np.abs(X).max()) or 1.0
        out = np.zeros((X.shape[0], Y.shape[1]), f32)
        for sg in (1.0, -1.0):
            dty = np.round(np.maximum(sg * X, 0) / sa * 255) / 255
            Z = dty @ W; st = 2 * FS / 2 ** adc_bits
            out += sg * np.clip(np.round(Z / st) * st, -FS, FS)
        return out * sa * sw
    return mm


BACKENDS = {"FP16PAIR": (mm_fp16pair, 1), "INT8": (mm_int8, 1),
            "AN8": (mm_analog(8), 2), "AN10": (mm_analog(10), 2)}   # (product, passes per product)


def sp2(H, no, mm, max_iter=200):
    a, b = gersh(H); N = len(H)
    X = ((b * np.eye(N) - H) / (b - a)).astype(f32); hist = []
    for it in range(1, max_iter + 1):
        X2 = mm(X, X).astype(f32)
        t1, t2 = np.trace(X), np.trace(X2)
        X = X2 if abs(t2 - no) < abs(2 * t1 - t2 - no) else (2 * X - X2).astype(f32)
        if not np.all(np.isfinite(X)) or np.abs(X).max() > 1e3: break
        hist.append(float(np.trace(X) - np.sum(X * X)))
        if it > 4 and abs(hist[-1]) >= abs(hist[-3]): break
    return X, it


# ---------------- refinement (general N; same method as refine.py) ----------------
def _rr(C, H):
    e, U = np.linalg.eigh(C.T @ H @ C); return C @ U, e


def _rr_both(Co, Cv, H):
    Co, eo = _rr(Co, H); Cv, ev = _rr(Cv, H)
    for _ in range(len(eo)):
        i, j = int(np.argmax(eo)), int(np.argmin(ev))
        if eo[i] <= ev[j]: break
        Co, Cv = Co.copy(), Cv.copy()
        Co[:, i], Cv[:, j] = Cv[:, j].copy(), Co[:, i].copy()
        Co, eo = _rr(Co, H); Cv, ev = _rr(Cv, H)
    return Co, Cv, eo, ev


def refine_all(P, H, no, A, ref, kmax=KMAX, seed=7):
    """Return the map error after k = 0..kmax refinement steps."""
    # Refinement runs in float64 (the exact digital stage). In float32 its own floor (~1e-5..4e-5 for the
    # 2312 parameter set at N >= 32) sits on the target, which would hide what the front end contributes.
    N = len(H); H = H.astype(np.float64); P = P.astype(np.float64)
    rng = np.random.default_rng(seed)
    Co, _ = np.linalg.qr(P @ rng.standard_normal((N, no)))
    Q, _ = np.linalg.qr((np.eye(N) - P) @ rng.standard_normal((N, N - no)))
    Cv, _ = np.linalg.qr(Q - Co @ (Co.T @ Q))
    errs = []
    for k in range(kmax + 1):
        Co, Cv, eo, ev = _rr_both(Co, Cv, H)
        s = map_from_pairs(Co.astype(np.float64), Cv.astype(np.float64), eo.astype(np.float64),
                           ev.astype(np.float64), A)
        errs.append(np.abs(s - ref).max() / np.abs(ref).max() if np.all(np.isfinite(s)) else np.inf)
        Z = (Cv.T @ H @ Co) / (eo[None, :] - ev[:, None])
        Co, Cv = Co + Cv @ Z, Cv - Co @ Z.T
        Co, _ = np.linalg.qr(Co); Cv, _ = np.linalg.qr(Cv - Co @ (Co.T @ Cv))
    return errs


def k_reach(errs_list, frac=1.0):
    for k in range(KMAX + 1):
        if np.mean([e[k] <= TARGET for e in errs_list]) >= frac: return k
    return None


def main():
    np.seterr(all="ignore"); t0 = time.time()
    Ns = [8, 12, 16, 24, 32, 40]
    ksets = [(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)]
    gammas = [0.1, 0.3, 0.5]
    SEEDS = range(10)
    print(f"target map rel. error <= {TARGET:g}; {len(ksets) * len(gammas)} settings per N; k_all = steps until ALL "
          f"settings pass; median [min-max] over {len(SEEDS)} range-finder seeds (99 = not by k = {KMAX})")
    print("saved = median COLD - median front end;  breakeven = front-end MACs / (saved x MACs per refinement step):"
          " how many times cheaper per MAC the front end must be just to pay for itself")
    fronts = {"COLD": None, **BACKENDS}
    for N in Ns:
        no = N // 2
        sysm = [system(N, g, ks) for ks, g in itertools.product(ksets, gammas)]
        refs = [reference(H, A, no) for H, A in sysm]
        gap = min(np.diff(np.linalg.eigvalsh(H))[no - 1] for H, _ in sysm)
        nv = N - no
        mac_step = (N * N * no + N * no * no) + (N * N * nv + N * nv * nv) + (N * N * no + nv * N * no) \
                   + N * nv * no * 2
        med, out = {}, []
        for name, spec in fronts.items():
            Ps, its = [], []
            for H, A in sysm:
                if spec is None:
                    a, b = gersh(H); Ps.append((b * np.eye(N) - H) / (b - a)); its.append(0)
                else:
                    X, it = sp2(H, no, spec[0]); Ps.append(X); its.append(it)
            ks = []
            for sd in SEEDS:
                k = k_reach([refine_all(P, H, no, A, r, seed=sd) for P, (H, A), r in zip(Ps, sysm, refs)])
                ks.append(99 if k is None else k)
            med[name] = int(np.median(ks))
            cell = f"{name}:{med[name]}[{min(ks)}-{max(ks)}]"
            if spec is not None:
                saved = med["COLD"] - med[name]
                fe_mac = int(np.median(its)) * N ** 3 * spec[1]
                be = f"{fe_mac / (saved * mac_step):.1f}x" if saved > 0 else "never"
                cell += f" saved {saved:+d} it {int(np.median(its))} be {be}"
            out.append(cell)
        print(f"N={N:2d} gap {gap:.3f} | " + " | ".join(out))
    print("\n(MACs per refinement step exclude the two N/2 x N/2 block diagonalizations per step)")
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
