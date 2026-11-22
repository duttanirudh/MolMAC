"""
analog_mc.py  --  analog behavioral model + Monte Carlo over simulated chips (butadiene + acrolein).

Run:   python analog_mc.py              (full: 200 chips; several minutes)
       python analog_mc.py --chips 3    (smoke test)

ANALOG PRODUCT MODEL for Y = A @ B  (B programmed as weights, rows of A applied as PWM activations):
  weights   B quantized to 8-bit signed (per-matrix scale) if quantization is on; each cell (i,j) has a
            static per-chip gain G_ij = 1 + sigma_g * N(0,1).
  activations  signed rows split into positive and negative parts, each applied as a duty cycle in [0,1]
            (per-matrix scale), 8-bit duty resolution if quantization is on. Two passes per product.
  readout   column sum per pass, + static per-chip column offset (fraction of column full scale
            FS_j = sum_i |W_ij|), + fresh Gaussian read noise (fraction of FS_j), then an ADC
            (8 bits over [-FS_j, FS_j]) if quantization is on.  Result = (pass+ - pass-) * scale.
  offset modes  "same": both passes read the same column -> static offset cancels in the difference.
                "split": each pass has its own static offset (e.g. separate columns or arrays).
PARAMETER STATUS: gain sigma 1.7% is the measured MolMAC/VoltMatrix-scale figure; 0.12% is the MolMAC
  oracle-calibrated figure; ALL offset and noise levels here are PLACEHOLDERS (not sourced).

PIPELINE: analog SP2 (Gershgorin bounds digital; trace-correcting; stall rule or a fixed cap) ->
  fp32 H-aware refinement, k steps (refine.py) -> sensitivity map from refined eigenpairs.
TARGET: map rel. error <= 1e-5 ("reached"). Chip seed c -> numpy default_rng(1000 + c).

PREDICTIONS (written before any run):
  M1  every mode except N10 and ALL_unc reaches the target in >= 95% of runs by k = 3.
  M2  O_same gives exactly the same results as IDEAL (static offset cancels in the two-pass difference).
  M3  N10 and ALL_unc need k >= 3, and some runs (< 20%) still miss at k = 3.
  H3' (R3 without refinement, h = 0.03; revised from v3 H3):
      O_same  -> same error as IDEAL-R3 (cancellation),  O_split -> NOT cancelled, error ~ sigma_o x (1-10),
      gain    -> rel. error ~ sigma_g x (0.3-10)  (survives differencing),
      noise   -> rel. error ~ sigma_n / h x (0.3-10).
  H4' (iteration cap, then k = 2): too few SP2 steps (4) fail (virtual leakage in P); a middle cap
      (8-12) matches or beats running to the stall; the gap widens in ALL_unc.
"""
import argparse
import itertools
import time
import numpy as np
from h1_acrolein import hamiltonian, N_OCC
from r123_float64 import A_OBS, reference, gershgorin
from refine import refine, map_from_pairs

f32 = np.float32
SETTINGS = list(itertools.product([(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.5, 1.0, 2.0], [0.1, 0.3, 0.5]))
TARGET = 1e-5

MODES = {  # name: (quant, sigma_g, sigma_o, offset_mode, sigma_n)
    "IDEAL":   (False, 0.0,    0.0,   "same",  0.0),
    "Q8":      (True,  0.0,    0.0,   "same",  0.0),
    "G_cal":   (False, 0.0012, 0.0,   "same",  0.0),
    "G_unc":   (False, 0.017,  0.0,   "same",  0.0),
    "O_same":  (False, 0.0,    0.01,  "same",  0.0),
    "O_split": (False, 0.0,    0.01,  "split", 0.0),
    "N1":      (False, 0.0,    0.0,   "same",  0.001),
    "N10":     (False, 0.0,    0.0,   "same",  0.01),
    "ALL_cal": (True,  0.0012, 0.002, "split", 0.002),
    "ALL_unc": (True,  0.017,  0.01,  "split", 0.01),
}


class Chip:
    def __init__(self, seed, quant, sg, so, omode, sn, noise_seed=None):
        static = np.random.default_rng(seed)                 # static chip: gain map and offsets
        self.quant, self.sn = quant, sn
        self.G = (1.0 + sg * static.standard_normal((8, 8))).astype(f32)
        self.o_pos = (so * static.standard_normal(8)).astype(f32)
        self.o_neg = self.o_pos if omode == "same" else (so * static.standard_normal(8)).astype(f32)
        self.rng = np.random.default_rng(seed + 10**6 if noise_seed is None else noise_seed)  # read noise

    def mm(self, A, B):
        A, B = A.astype(f32), B.astype(f32)
        sw = float(np.abs(B).max()) or 1.0
        W = np.round(B / sw * 127) / 127 if self.quant else B / sw
        W = (W * self.G).astype(f32)
        FS = np.abs(W).sum(axis=0) + 1e-30
        sa = float(np.abs(A).max()) or 1.0
        out = np.zeros((A.shape[0], B.shape[1]), f32)
        for sign, off in ((1.0, self.o_pos), (-1.0, self.o_neg)):
            duty = np.maximum(sign * A, 0) / sa
            if self.quant:
                duty = np.round(duty * 255) / 255
            Y = duty @ W + off * FS
            if self.sn:
                Y = Y + self.sn * FS * self.rng.standard_normal(Y.shape)
            if self.quant:
                step = 2 * FS / 256
                Y = np.clip(np.round(Y / step) * step, -FS, FS)
            out += sign * Y
        return (out * sa * sw).astype(f32)


def sp2_chip(H, chip, cap=None, max_iter=60):
    emin, emax = gershgorin(H)
    X = ((emax * np.eye(8) - H) / (emax - emin)).astype(f32)
    hist = []
    for it in range(1, (cap or max_iter) + 1):
        X2 = chip.mm(X, X)
        t1, t2 = np.trace(X), np.trace(X2)
        X = X2 if abs(t2 - N_OCC) < abs(2 * t1 - t2 - N_OCC) else (2 * X - X2).astype(f32)
        if not np.all(np.isfinite(X)) or np.abs(X).max() > 1e3:   # divergence guard
            break
        hist.append(float(np.trace(X) - np.sum(X * X)))
        if cap is None and it > 4 and abs(hist[-1]) >= abs(hist[-3]):
            break
    return X, it


def safe_err(X, H, k, ref):
    """refine + map; a diverged or failed refinement counts as a failure."""
    try:
        _, Co, Cv, eo, ev = refine(X, H, k)
        return map_err(map_from_pairs(Co, Cv, eo, ev), ref)
    except (np.linalg.LinAlgError, FloatingPointError):
        return np.inf, 8


def map_err(s, ref):
    if not np.all(np.isfinite(s)): return np.inf, 8
    return np.abs(s - ref).max() / np.abs(ref).max(), int(np.sum(np.sign(s) != np.sign(ref)))


def r3_chip(H, chip_args, seed, h):
    s = np.zeros(8)
    for k in range(8):
        vals = []
        for sgn in (+1, -1):
            Hp = H.copy(); Hp[k, k] -= sgn * h
            chip = Chip(seed, *chip_args, noise_seed=seed * 100 + 2 * k + (sgn > 0))  # same static chip, fresh noise
            X, _ = sp2_chip(Hp, chip)
            vals.append(float(np.sum(A_OBS * 2 * X.astype(np.float64))))
        s[k] = (vals[0] - vals[1]) / (2 * h)
    return s


def frac_se(x):
    p = float(np.mean(x)); return p, float(np.sqrt(p * (1 - p) / len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=200)
    ap.add_argument("--part", choices=["1", "2", "3", "all"], default="all")
    a = ap.parse_args()
    np.seterr(all="ignore")
    KS = range(6)
    DETERMINISTIC = {"IDEAL", "Q8", "O_same"}   # every chip identical: effective n = 18 settings
    t0 = time.time()
    Hs = [hamiltonian(g, hO, *ks) for ks, hO, g in SETTINGS]
    refs = [reference(H) for H in Hs]
    chips = range(a.chips)
    print(f"chips={a.chips}, settings={len(SETTINGS)}, target map rel. error <= {TARGET:g}")

    if a.part in ("1", "all"):
        print("\n[1] refined pipeline (aufbau check on): fraction reaching target at k = 0..5; "
              "median map error and sign failures at k = 3; median SP2 iterations. "
              "* = deterministic mode (n = 18 settings, s.e. not meaningful)")
        print(f"{'mode':>9} " + " ".join(f"{'k=' + str(k):>13}" for k in KS) + f" {'med k=3':>9} {'signs k=3':>12} {'iters':>5}")
        # CONTROL: cold start = refine directly from X0 (scaled H, zero SP2 steps). Steps saved vs this row
        # are the only measure of what the analog / low-precision stage contributes.
        reach = {k: [] for k in KS}; med3, signs = [], 0
        for H, ref in zip(Hs, refs):
            emin, emax = gershgorin(H)
            X = ((emax * np.eye(8) - H) / (emax - emin)).astype(f32)
            for k in KS:
                e, sf = safe_err(X, H, k, ref); reach[k].append(e <= TARGET)
                if k == 3: med3.append(e); signs += sf
        cells = " ".join(f"{np.mean(reach[k]):6.3f}      " for k in KS)
        print(f"{'COLD*':>9} {cells} {np.median(med3):9.1e} {f'{signs}/{8 * len(med3)}':>12} {0:5d}")
        for name, args in MODES.items():
            reach = {k: [] for k in KS}; med3, signs, iters = [], 0, []
            for c in chips:
                for H, ref in zip(Hs, refs):
                    X, it = sp2_chip(H, Chip(1000 + c, *args)); iters.append(it)
                    for k in KS:
                        e, sf = safe_err(X, H, k, ref)
                        reach[k].append(e <= TARGET)
                        if k == 3: med3.append(e); signs += sf
            cells = " ".join(f"{frac_se(reach[k])[0]:6.3f}+-{frac_se(reach[k])[1]:.3f}" for k in KS)
            tag = name + ("*" if name in DETERMINISTIC else "")
            print(f"{tag:>9} {cells} {np.median(med3):9.1e} {f'{signs}/{8 * len(med3)}':>12} {int(np.median(iters)):5d}")

    if a.part in ("2", "all"):
        print("\n[2] H3': R3 finite differences on the analog solver, NO refinement, h = 0.03 "
              "(6 settings: hO = 1; median / 90th pct map rel. error)")
        sub = [i for i, (ks, hO, g) in enumerate(SETTINGS) if hO == 1.0]
        for name in ["IDEAL", "O_same", "O_split", "G_cal", "G_unc", "N1"]:
            errs = [map_err(r3_chip(Hs[i], MODES[name], 1000 + c, 0.03), refs[i])[0] for c in chips for i in sub]
            print(f"{name:>8}  {np.median(errs):9.2e} / {np.percentile(errs, 90):9.2e}")

    if a.part in ("3", "all"):
        print("\n[3] H4': SP2 iteration cap, then k = 3 refinement "
              "(fraction reaching target; median map error; fraction diverged)")
        for name in ["ALL_cal", "ALL_unc"]:
            print(f"{name:>8}")
            for cap in [4, 6, 8, 12, 20, None]:
                ok, ee, div = [], [], 0
                for c in chips:
                    for H, ref in zip(Hs, refs):
                        X, _ = sp2_chip(H, Chip(1000 + c, *MODES[name]), cap=cap)
                        div += (not np.all(np.isfinite(X))) or np.abs(X).max() > 1e3
                        e, _ = safe_err(X, H, 3, ref); ok.append(e <= TARGET); ee.append(e)
                print(f"   cap={str(cap or 'stall'):>5}: reach {np.mean(ok):.3f}  median {np.median(ee):8.1e}  "
                      f"diverged {div / len(ok):.3f}")
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
