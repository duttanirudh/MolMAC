"""
bits2.py  --  (a) a realistic auto-ranging ADC and (b) the bits trend beyond N = 40.

Front end: 8-bit weights, 8-bit duty per polarity, b-bit ADC, no gain/offset/noise (deterministic).
ADC range modes:
  worst   range = worst-case column sum sum_i |W_ij|                         (from bits.py)
  ideal   range = the actual max |column sum| of THIS product                (from bits.py; oracle)
  pga     range = worst-case / 2^g, g an integer gain step (0..12) per column, chosen from the PREVIOUS
          product's measured column max with 2x headroom; first product uses the worst case; a clipped
          column saturates. Practical: needs only last iteration's readings.
Pipeline: SP2 -> range finder -> one Rayleigh-Ritz per block -> Ogita-Aishima (s2_oa.run), 60 runs/cell.
Also digital integer (INTb) at N > 40.

PREDICTIONS (written before running):
  A1  pga needs at most 2 bits more than ideal at every N (power-of-two steps lose < 1 bit; headroom 1 bit).
  A2  pga needs fewer bits than worst at N >= 24.
  A3  beyond N = 40 the requirement keeps rising roughly like log2(1/gap): about +1 bit from N = 40 to 64
      for INTb and ideal (gap ratio 0.22 -> ~0.14 expected, log2 ~ 0.65).
"""
import itertools
import time
import numpy as np
import scale
from s2_oa import run

f32 = np.float32
BITS = [8, 10, 12, 14, 16]


class Chain:
    def __init__(self, ab, mode):
        self.ab, self.mode, self.prev = ab, mode, None

    def mm(self, X, Y):
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        sw = float(np.abs(Y).max()) or 1.0
        W = np.round(Y / sw * 127) / 127
        sa = float(np.abs(X).max()) or 1.0
        FSw = np.abs(W).sum(0) + 1e-30
        out = np.zeros((X.shape[0], Y.shape[1])); seen = np.zeros(Y.shape[1])
        for sg in (1.0, -1.0):
            dty = np.round(np.maximum(sg * X, 0) / sa * 255) / 255
            Z = dty @ W
            if self.mode == "worst":
                FS = FSw
            elif self.mode == "ideal":
                FS = np.abs(Z).max(0) + 1e-30
            else:
                if self.prev is None:
                    FS = FSw
                else:
                    g = np.clip(np.floor(np.log2(FSw / (2 * self.prev + 1e-30))), 0, 12)
                    FS = FSw / 2 ** g
            st = 2 * FS / 2 ** self.ab
            out += sg * np.clip(np.round(Z / st) * st, -FS, FS)
            seen = np.maximum(seen, np.abs(Z).max(0))
        self.prev = seen
        return (out * sa * sw).astype(f32)


def mm_int(bits):
    L = 2 ** (bits - 1) - 1
    def mm(X, Y):
        sx = float(np.abs(X).max()) or 1.0; sy = float(np.abs(Y).max()) or 1.0
        return ((np.round(X / sx * L) @ np.round(Y / sy * L)) * (sx * sy / L / L)).astype(f32)
    return mm


def success(make_mm, N):
    no = N // 2; ok = tot = 0
    for ks, g in itertools.product([(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5]):
        H, A = scale.system(N, g, ks); ref = scale.reference(H, A, no)
        P = scale.sp2(H, no, make_mm())[0]
        for sd in range(10):
            ok += run(P, H, A, no, ref, sd, 1e-10) is not None; tot += 1
    return ok / tot


def main():
    np.seterr(all="ignore"); t0 = time.time()
    rows = {"worst": lambda b: (lambda: Chain(b, "worst").mm),
            "ideal": lambda b: (lambda: Chain(b, "ideal").mm),
            "pga": lambda b: (lambda: Chain(b, "pga").mm),
            "INTb": lambda b: (lambda: mm_int(b))}
    Ns = [8, 16, 24, 32, 40, 48, 56, 64]
    gaps = {N: min(np.diff(np.linalg.eigvalsh(scale.system(N, g, ks)[0]))[N // 2 - 1]
                   for ks, g in itertools.product([(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5])) for N in Ns}
    print("min HOMO-LUMO gap: " + "  ".join(f"N={N}:{gaps[N]:.3f}" for N in Ns))
    print("min bits for >= 95% success (None = not by 16), success rates per bit width in brackets")
    summ = {}
    for name, mk in rows.items():
        for N in Ns:
            if name == "worst" and N > 40: continue
            r = [success(mk(b), N) for b in BITS]
            mb = next((b for b, x in zip(BITS, r) if x >= 0.95), None); summ[(name, N)] = mb
            print(f"{name:>6} N={N:2d}  min b={str(mb):>4}   [" + " ".join(f"{x:.2f}" for x in r) + "]", flush=True)
    print("\nsummary (min bits):")
    for name in rows:
        print(f"{name:>6}: " + "  ".join(f"{N}:{summ.get((name, N), '-')}" for N in Ns))
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
