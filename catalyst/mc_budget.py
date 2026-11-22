"""
mc_budget.py  --  error-source precision budget for the analog front end + Ogita-Aishima refiner, vs N.

Run:   python mc_budget.py              (full: 200 chips)
       python mc_budget.py --chips 2    (smoke test)

FRONT END (configuration chosen from bits.py): 8-bit signed weights, 8-bit duty per polarity (two passes),
  12-bit ADC whose range is fitted per product to the actual (noisy) column signal (auto-ranging).
ERROR SOURCES, each swept alone, then two combinations. Offset and noise are added at the summing node in
  units of the worst-case column scale FS_j = sum_i |W_ij| (a physical signal scale: auto-ranging the ADC
  afterwards does NOT shrink them). Offsets are static per chip and separate per pass ("split").
  gain    static per-cell gain 1 + sigma_g N(0,1)        levels 0.12% (MolMAC calibrated), 0.5%, 1.7%
          (measured scale, MolMAC/VoltMatrix), 5%
  offset  static per-column, per-pass                     levels 1e-4, 1e-3, 1e-2 of FS
  noise   fresh per read                                  levels 1e-4, 1e-3, 1e-2 of FS
  Offset and noise levels are PLACEHOLDERS, not sourced.
PIPELINE: analog SP2 (stall rule + divergence guard) -> range finder -> one Rayleigh-Ritz per block ->
  Ogita-Aishima steps (float64), success = map rel. error <= 1e-5 within 10 steps (s2_oa.run).
SAMPLING: chip c uses static seed 1000 + c, noise seed 5000 + c, and range-finder seed c.
  Runs per cell = chips x 6 settings (std / 2312 bond sets x gamma 0.1, 0.3, 0.5).

PREDICTIONS (written before any run):
  P1  gain: >= 95% success at 1.7% for N = 8; the tolerable sigma_g shrinks with N, to about 0.5% at N = 40.
  P2  offset and noise: tolerable level ~1e-3 of FS at N = 8, falling 3-10x by N = 40.
  P3  combined calibrated (gain 0.12%, offset 1e-3, noise 1e-3) passes at N = 8 and 24, fails at N = 40;
      combined uncalibrated (gain 1.7%, same offset/noise) fails at N = 24 and 40.
  P4  at every N the failure rate is non-decreasing in each error level (no sweet spots).
"""
import argparse
import itertools
import time
import numpy as np
import scale
from s2_oa import run

f32 = np.float32
TARGET = 1e-5


class Chip:
    def __init__(self, N, c, sg=0.0, so=0.0, sn=0.0, adc=12):
        st = np.random.default_rng(1000 + c)
        self.G = 1.0 + sg * st.standard_normal((N, N))
        self.op = so * st.standard_normal(N)
        self.on = so * st.standard_normal(N)
        self.sn, self.adc = sn, adc
        self.rng = np.random.default_rng(5000 + c)

    def mm(self, X, Y):
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        sw = float(np.abs(Y).max()) or 1.0
        W = np.round(Y / sw * 127) / 127 * self.G
        FSw = np.abs(W).sum(0) + 1e-30
        sa = float(np.abs(X).max()) or 1.0
        out = np.zeros((X.shape[0], Y.shape[1]))
        for sg, off in ((1.0, self.op), (-1.0, self.on)):
            dty = np.round(np.maximum(sg * X, 0) / sa * 255) / 255
            Z = dty @ W + off * FSw
            if self.sn:
                Z = Z + self.sn * FSw * self.rng.standard_normal(Z.shape)
            FS = np.abs(Z).max(0) + 1e-30                      # auto-ranged ADC
            step = 2 * FS / 2 ** self.adc
            out += sg * np.round(Z / step) * step
        return (out * sa * sw).astype(f32)


CELLS = [("none", {}),
         ("gain 0.12%", {"sg": 0.0012}), ("gain 0.5%", {"sg": 0.005}),
         ("gain 1.7%", {"sg": 0.017}), ("gain 5%", {"sg": 0.05}),
         ("offset 1e-4", {"so": 1e-4}), ("offset 1e-3", {"so": 1e-3}), ("offset 1e-2", {"so": 1e-2}),
         ("noise 1e-4", {"sn": 1e-4}), ("noise 1e-3", {"sn": 1e-3}), ("noise 1e-2", {"sn": 1e-2}),
         ("ALL cal", {"sg": 0.0012, "so": 1e-3, "sn": 1e-3}),
         ("ALL uncal", {"sg": 0.017, "so": 1e-3, "sn": 1e-3})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=200)
    a = ap.parse_args()
    np.seterr(all="ignore"); t0 = time.time()
    Ns = [8, 24, 40]
    ksets, gammas = [(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5]
    systems = {}
    for N in Ns:
        no = N // 2
        systems[N] = [(H, A, scale.reference(H, A, no))
                      for H, A in (scale.system(N, g, ks) for ks, g in itertools.product(ksets, gammas))]
    print(f"chips={a.chips}; runs per cell = chips x 6; success = map rel. err <= {TARGET:g} within 10 OA steps")
    print(f"{'cell':>12} " + " ".join(f"{'N=' + str(N) + ' success':>17} {'st':>3}" for N in Ns))
    for label, kw in CELLS:
        row = []
        for N in Ns:
            no = N // 2; res = []
            for c in range(a.chips):
                chip = Chip(N, c, **kw)
                for H, A, ref in systems[N]:
                    P = scale.sp2(H, no, chip.mm)[0]
                    res.append(run(P, H, A, no, ref, c, 1e-10))
            ok = [k for k in res if k is not None]
            p = len(ok) / len(res); se = np.sqrt(p * (1 - p) / len(res))
            row.append(f"{p:6.3f}+-{se:.3f}{'*' if p >= 0.95 else ' '}    {int(np.median(ok)) if ok else '-':>3}")
        print(f"{label:>12} " + " ".join(row), flush=True)
    print("(* = >= 95%; st = median OA steps when converged)")
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
