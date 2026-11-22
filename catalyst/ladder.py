"""
ladder.py  --  precision ladder for R1/R2/R3 on butadiene + acrolein (18 parameter settings).

Backends (every matrix PRODUCT is done in the backend; other arithmetic in float32 unless float64):
  fp64      reference arithmetic
  fp32      float32 products and storage
  fp16pair  X = Xh + Xl (both fp16), XY ~ XhYh + XhYl + XlYh, fp32 accumulation (low-low dropped),
            mirroring the scheme in arXiv 2409.17033. No scaling, so tiny low parts can underflow in fp16.
  int8      per-matrix symmetric int8 quantization of both operands, integer accumulate, rescale to fp32.
Spectral bounds (Gershgorin) are computed digitally in fp64 for all backends.
Error metric for a sensitivity map s vs exact ref: max|s - ref| / max|ref|.
Also: sign agreement (8 atoms), Kendall tau, top-2 within the dienophile (Cb, Ca, Cc, O), error of I itself.

PREDICTIONS (written before running):
  L1  fp32: R1, R2 rel. error ~1e-6 (within 1e-7..1e-5); all signs right.
  L2  fp16pair: same order as fp32 (within 10x of fp32).
  L3  H2': at fp32 and fp16pair, R1 and R2 errors within 3x of each other (the paper's large gap
      came from thousands of orbitals; at 8 atoms I expect no meaningful difference).
  L4  int8: SP2 stalls at idempotency error ~1e-2; map rel. error 10%-50%; sign failures appear,
      mostly at gamma = 0.1 and on the smallest entries (C1, C3).
  L5  R3: error vs step h is U-shaped; best h grows as precision drops (rule h ~ eps^(1/3):
      ~4e-3 fp32, ~0.1-0.2 int8); at its best h, R3 is worse than R1/R2 at fp32.
  L6  I itself at int8: relative error > 20% at gamma = 0.1 (earlier prediction: int8 swamps I there).
"""
import itertools
import numpy as np
from h1_acrolein import hamiltonian, N_OCC
from r123_float64 import A_OBS, reference, gershgorin

f16, f32 = np.float16, np.float32


def mm_fp64(A, B): return A @ B
def mm_fp32(A, B): return (A.astype(f32) @ B.astype(f32)).astype(f32)


def mm_fp16pair(A, B):
    A, B = A.astype(f32), B.astype(f32)
    Ah = A.astype(f16).astype(f32); Al = (A - Ah).astype(f16).astype(f32)
    Bh = B.astype(f16).astype(f32); Bl = (B - Bh).astype(f16).astype(f32)
    return (Ah @ Bh + Ah @ Bl + Al @ Bh).astype(f32)


def _q8(A):
    s = float(np.abs(A).max()) / 127.0
    if s == 0.0: return np.zeros(A.shape, np.int64), 1.0
    return np.round(A / s).astype(np.int64), s


def mm_int8(A, B):
    Aq, sa = _q8(A); Bq, sb = _q8(B)
    return ((Aq @ Bq) * (sa * sb)).astype(f32)


BACKENDS = {"fp64": (mm_fp64, np.float64), "fp32": (mm_fp32, f32),
            "fp16pair": (mm_fp16pair, f32), "int8": (mm_int8, f32)}


def sp2(H, mm, dt, max_iter=100):
    emin, emax = gershgorin(H)
    w = emax - emin
    X = ((emax * np.eye(8) - H) / w).astype(dt)
    Xs, br, hist = [], [], []
    for it in range(1, max_iter + 1):
        Xs.append(X)
        X2 = mm(X, X).astype(dt)
        t1, t2 = np.trace(X), np.trace(X2)
        sq = abs(t2 - N_OCC) < abs(2 * t1 - t2 - N_OCC)
        X = X2 if sq else (2 * X - X2).astype(dt)
        br.append(sq)
        hist.append(float(np.trace(X) - np.sum(X * X)))
        if it > 4 and hist[-1] >= hist[-3]:
            break
    return X, Xs, br, w, abs(hist[-1])


def r1(H, mm, dt):
    _, Xs, br, w, _ = sp2(H, mm, dt)
    s = np.zeros(8)
    for k in range(8):
        dX = np.zeros((8, 8), dt); dX[k, k] = 1.0 / w
        for X, sq in zip(Xs, br):
            sym = (mm(dX, X) + mm(X, dX)).astype(dt)
            dX = sym if sq else (2 * dX - sym).astype(dt)
        s[k] = np.sum(A_OBS * 2 * dX.astype(np.float64))
    return s


def r2(H, mm, dt):
    _, Xs, br, w, _ = sp2(H, mm, dt)
    Y = (2 * A_OBS).astype(dt)
    for X, sq in zip(reversed(Xs), reversed(br)):
        sym = (mm(X, Y) + mm(Y, X)).astype(dt)
        Y = sym if sq else (2 * Y - sym).astype(dt)
    return np.diag(Y.astype(np.float64)) / w


def r3(H, mm, dt, h):
    s = np.zeros(8)
    for k in range(8):
        Hp, Hm = H.copy(), H.copy()
        Hp[k, k] -= h; Hm[k, k] += h
        Ip = np.sum(A_OBS * 2 * sp2(Hp, mm, dt)[0].astype(np.float64))
        Im = np.sum(A_OBS * 2 * sp2(Hm, mm, dt)[0].astype(np.float64))
        s[k] = (Ip - Im) / (2 * h)
    return s


def kendall(a, b):
    n, c = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            c += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return c / (n * (n - 1) / 2)


def metrics(s, ref):
    top = lambda v: set(4 + np.argsort(-np.abs(v[4:]))[:2])
    return dict(rel=np.abs(s - ref).max() / np.abs(ref).max(),
                signs=int(np.sum(np.sign(s) != np.sign(ref))),
                tau=kendall(s, ref), top2=top(s) == top(ref))


def main():
    settings = list(itertools.product([(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.5, 1.0, 2.0], [0.1, 0.3, 0.5]))
    hs = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
    refs = {i: reference(hamiltonian(g, hO, *ks)) for i, (ks, hO, g) in enumerate(settings)}
    print("Summary over 18 settings: median / worst rel. error; total sign failures; min Kendall tau; "
          "top-2 (dienophile) matches; SP2 idempotency floor")
    r3curve = {}
    for name, (mm, dt) in BACKENDS.items():
        res = {"R1": [], "R2": [], "R3": []}
        Ierr, Ierr01, idem = [], [], []
        curve = {h: [] for h in hs}
        for i, (ks, hO, g) in enumerate(settings):
            H = hamiltonian(g, hO, *ks); ref = refs[i]
            X, *_, floor = sp2(H, mm, dt); idem.append(floor)
            Iex = np.sum(A_OBS * 2 * sp2(H, mm_fp64, np.float64)[0])
            e = abs(np.sum(A_OBS * 2 * X.astype(np.float64)) - Iex) / abs(Iex)
            Ierr.append(e)
            if g == 0.1: Ierr01.append(e)
            res["R1"].append(metrics(r1(H, mm, dt), ref))
            res["R2"].append(metrics(r2(H, mm, dt), ref))
            for h in hs:
                curve[h].append(metrics(r3(H, mm, dt, h), ref))
        hbest = min(hs, key=lambda h: np.median([m["rel"] for m in curve[h]]))
        res["R3"] = curve[hbest]
        r3curve[name] = {h: np.median([m["rel"] for m in curve[h]]) for h in hs}
        print(f"\n[{name}]  SP2 idempotency floor median {np.median(idem):.1e};  "
              f"rel. error of I: median {np.median(Ierr):.1e}, worst at gamma=0.1 {max(Ierr01):.1e}")
        for r, ms in res.items():
            rel = [m["rel"] for m in ms]
            tag = f" (best h={hbest:g})" if r == "R3" else ""
            print(f"  {r}{tag:<16} rel {np.median(rel):.1e} / {max(rel):.1e}   sign fails "
                  f"{sum(m['signs'] for m in ms):3d}/144   min tau {min(m['tau'] for m in ms):+.2f}   "
                  f"top2 {sum(m['top2'] for m in ms):2d}/18")
    print("\nR3 median rel. error vs step h:")
    print(f"{'h':>9} " + " ".join(f"{n:>9}" for n in BACKENDS))
    for h in hs:
        print(f"{h:9g} " + " ".join(f"{r3curve[n][h]:9.1e}" for n in BACKENDS))


if __name__ == "__main__":
    main()
