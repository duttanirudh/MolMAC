"""
bits.py  --  minimum front-end precision for the Ogita-Aishima refiner, vs system size.
Deterministic analog chain (no gain / offset / noise), then the s2_oa.py pipeline
(range finder -> one Rayleigh-Ritz per block -> OA steps, float64). Success = map rel. error <= 1e-5
within 10 OA steps; 60 runs per cell (6 settings x 10 range-finder seeds).

VARIANTS
  W8D8-worst   8-bit weights, 8-bit duty, ADC b bits, ADC range = worst-case column sum sum_i|W_ij|
  W8D8-fit     same, ADC range fitted to the largest actual |column sum| of that product
               (an ideal programmable-gain stage; optimistic)
  ALLb-fit     weights, duty and ADC all b bits, fitted range
  (reference rows: INT8 / INT10 / INT12 digital, exact integer accumulation)

PREDICTIONS (written before running):
  B1  W8D8-worst needs 10-12 ADC bits for >= 95% at N = 8, and about 2 bits more at N = 40.
  B2  a fitted ADC range is worth >= 2 bits at every N.
  B3  for every variant that reaches 95%, the requirement rises by 1-3 bits from N = 8 to N = 40
      (my derivation: log2(gap_8 / gap_40) = log2(0.97 / 0.22) ~ 2.1 bits).
  B4  W8D8 variants cannot reach 95% at N >= 32 at any ADC width (8-bit operands set a floor;
      digital INT8 already gave 0.77 at N = 40); only ALLb-fit does.
"""
import itertools
import time
import numpy as np
import scale
from s2_oa import run

f32 = np.float32
BITS = [8, 10, 12, 14, 16]


def mm_chain(wb, db, ab, fit):
    Lw, Ld = 2 ** (wb - 1) - 1, 2 ** db - 1
    def mm(X, Y):
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        sw = float(np.abs(Y).max()) or 1.0
        W = np.round(Y / sw * Lw) / Lw
        sa = float(np.abs(X).max()) or 1.0
        out = np.zeros((X.shape[0], Y.shape[1]))
        for sg in (1.0, -1.0):
            dty = np.round(np.maximum(sg * X, 0) / sa * Ld) / Ld
            Z = dty @ W
            FS = (np.abs(Z).max(0) if fit else np.abs(W).sum(0)) + 1e-30
            st = 2 * FS / 2 ** ab
            out += sg * np.clip(np.round(Z / st) * st, -FS, FS)
        return (out * sa * sw).astype(f32)
    return mm


def mm_int(bits):
    L = 2 ** (bits - 1) - 1
    def mm(X, Y):
        sx = float(np.abs(X).max()) or 1.0; sy = float(np.abs(Y).max()) or 1.0
        return ((np.round(X / sx * L) @ np.round(Y / sy * L)) * (sx * sy / L / L)).astype(f32)
    return mm


def success(mm, N, ksets, gammas, seeds):
    no = N // 2; ok = 0; tot = 0
    for ks, g in itertools.product(ksets, gammas):
        H, A = scale.system(N, g, ks); ref = scale.reference(H, A, no)
        P = scale.sp2(H, no, mm)[0]
        for sd in seeds:
            ok += run(P, H, A, no, ref, sd, 1e-10) is not None; tot += 1
    return ok / tot


def main():
    np.seterr(all="ignore"); t0 = time.time()
    ksets, gammas, seeds = [(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5], range(10)
    Ns = [8, 16, 24, 32, 40]
    rows = {f"W8D8-worst": lambda b: mm_chain(8, 8, b, False),
            f"W8D8-fit": lambda b: mm_chain(8, 8, b, True),
            f"ALLb-fit": lambda b: mm_chain(b, b, b, True),
            f"INTb (digital)": lambda b: mm_int(b)}
    print("success rate (>= 0.95 marked *) by bits b; min bits for >= 0.95 at the end of each row")
    print(f"{'variant':>15} {'N':>3} " + " ".join(f"{'b=' + str(b):>7}" for b in BITS) + "   min b")
    summary = {}
    for name, mk in rows.items():
        for N in Ns:
            rates = [success(mk(b), N, ksets, gammas, seeds) for b in BITS]
            mb = next((b for b, r in zip(BITS, rates) if r >= 0.95), None)
            summary[(name, N)] = mb
            print(f"{name:>15} {N:3d} " + " ".join(f"{r:6.2f}{'*' if r >= 0.95 else ' '}" for r in rates)
                  + f"   {mb}")
        print()
    print("min bits for >= 95% vs N:")
    for name in rows:
        print(f"{name:>15}: " + "  ".join(f"N={N}:{summary[(name, N)]}" for N in Ns))
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
