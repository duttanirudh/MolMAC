"""
ripple_budget.py  --  measure the effect of MolMAC's PWM ripple on the analog front end + Ogita-Aishima refiner.

Replaces the unsourced handoff statement "ripple is 1.1-2.0e-3 of full scale and ~10x too large from 24 atoms"
(its script, ripple_check.py, could not be found).

RIPPLE MODEL: taken unchanged from MolMAC's mol_engine.py (imported, not copied):
  ripple_shape(theta, s, d), PWM_PERIOD, PWM_OFFS, and the circuit values in Cfg (rf, ron, cf, pwm_period_us,
  tile = 4). mol_engine.py describes this as a first-order model checked against LTspice; that check is not
  repeated here. Ripple is deterministic and phase-locked: it depends on the duty pattern, so it is neither the
  static offset nor the random noise swept in mc_budget.py, and the two signed passes do not cancel it.
  Per tile of 4 inputs (channel phase offsets 0/64/128/192 clocks, as in mac_ctrl.v):
      ripple = K * sum_i W_ij * ripple_shape(theta, start_(i mod 4), duty_i),   K = tile * Tp / ((rf + ron) * cf)
  summed over the N/4 tiles of a column (in the same "sum of branch products" units as mol_engine.py).
MAPPING CHOICES (mine; stated so they can be challenged):
  - weights here are signed (as in mc_budget.py); ripple is linear in the branch amplitude, so the sign carries.
  - the ADC stays mc_budget.py's single 12-bit auto-ranged converter per column (MolMAC digitizes per tile);
    every cell uses the same ADC, so differences between cells isolate the ripple.
  - no gain, offset or noise unless stated, so "ripple" rows compare directly with the "none" row.
PIPELINE, SEEDS, SYSTEMS, SUCCESS: identical to mc_budget.py (200 chips x 6 settings, N = 8, 24, 40;
  map rel. error <= 1e-5 within 10 OA steps).
Also prints the size of the ripple: median and 95th percentile of |ripple| / FS_j over all analog products.

Run:   python ripple_budget.py              (full: 200 chips; expect a few minutes)
       python ripple_budget.py --chips 2    (smoke test)

PREDICTIONS (written before any run):
  Q1  median |ripple| / FS_j lies between 1e-4 and 2e-3 at N = 8, and falls with N (branch ripples of mixed
      sign partly cancel while FS_j = sum |W_ij| grows).
  Q2  at N = 8 and N = 24, ripple at theta = 0 lowers success by less than 3 percentage points vs "none".
  Q3  the sample phase theta changes the success rate by less than 5 percentage points at every N.
  Q4  if Q2 holds, the handoff's "~10x too large from 24 atoms" is not supported in this model.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # MolMAC root, where mol_engine.py lives
from mol_engine import Cfg, PWM_OFFS, PWM_PERIOD, ripple_shape    # noqa: E402
import mc_budget                                                     # noqa: E402
import scale                                                         # noqa: E402
from s2_oa import run                                                # noqa: E402

f32 = np.float32
TARGET = 1e-5
_c = Cfg()
TILE = _c.tile
K_RIP = TILE * (_c.pwm_period_us * 1e-6) / ((_c.rf + _c.ron) * _c.cf)
STARTS = np.array([((PWM_PERIOD - o) % PWM_PERIOD) / PWM_PERIOD for o in PWM_OFFS])


class RippleChip(mc_budget.Chip):
    def __init__(self, N, c, theta=None, sn=0.0):
        super().__init__(N, c, sn=sn)
        assert N % TILE == 0
        self.theta = theta
        self.start = np.tile(STARTS, N // TILE)[None, :]
        self.stats = []

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
            if self.theta is not None:
                R = K_RIP * (ripple_shape(self.theta, self.start, dty) @ W)
                Z = Z + R
                rel = np.abs(R) / FSw
                self.stats.append((float(np.median(rel)), float(np.percentile(rel, 95))))
            if self.sn:
                Z = Z + self.sn * FSw * self.rng.standard_normal(Z.shape)
            FS = np.abs(Z).max(0) + 1e-30
            step = 2 * FS / 2 ** self.adc
            out += sg * np.round(Z / step) * step
        return (out * sa * sw).astype(f32)


CELLS = [("none", {}),
         ("ripple th=0", {"theta": 0.0}),
         ("ripple th=.25", {"theta": 0.25}),
         ("ripple th=.5", {"theta": 0.5}),
         ("rip0+n1e-4", {"theta": 0.0, "sn": 1e-4})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=200)
    a = ap.parse_args()
    np.seterr(all="ignore"); t0 = time.time()
    Ns = [8, 24, 40]
    ksets, gammas = [(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5]
    systems = {N: [(H, A, scale.reference(H, A, N // 2)) for H, A in
                   (scale.system(N, g, ks) for ks, g in itertools.product(ksets, gammas))] for N in Ns}
    print(f"ripple model from mol_engine.py: K = {K_RIP:.4f}, tile = {TILE}, channel starts = {np.round(STARTS, 3)}")
    print(f"chips={a.chips}; runs per cell = chips x 6; success = map rel. err <= {TARGET:g} within 10 OA steps")
    print(f"{'cell':>14} " + " ".join(f"{'N=' + str(N) + ' success':>17} {'st':>3}" for N in Ns)
          + "   |ripple|/FS median / p95 per N")
    for label, kw in CELLS:
        row, rip = [], []
        for N in Ns:
            res, st = [], []
            for c in range(a.chips):
                chip = RippleChip(N, c, **kw)
                for H, A, ref in systems[N]:
                    P = scale.sp2(H, N // 2, chip.mm)[0]
                    res.append(run(P, H, A, N // 2, ref, c, 1e-10))
                st += chip.stats
            ok = [k for k in res if k is not None]
            p = len(ok) / len(res); se = np.sqrt(p * (1 - p) / len(res))
            row.append(f"{p:6.3f}+-{se:.3f}{'*' if p >= 0.95 else ' '}    {int(np.median(ok)) if ok else '-':>3}")
            rip.append(f"{np.median([s[0] for s in st]):.1e}/{np.median([s[1] for s in st]):.1e}" if st else "-")
        print(f"{label:>14} " + " ".join(row) + "   " + "  ".join(rip), flush=True)
    print("(* = >= 95%; st = median OA steps when converged)")
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
