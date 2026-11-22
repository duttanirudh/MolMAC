"""
robust.py  --  robustness pass on the precision budget (200 chips; same pipeline as mc_budget.py).

(1) ADC width: error-free baseline at 12 vs 14 bits.
(2) Noise-scaling model, 14-bit ADC. Read-noise sigma per column = s_n x FS_j x f(N), FS_j = sum_i |W_ij|
    (which grows about linearly with N), with f(N) chosen so all models coincide at N = 8:
      "N"      f = 1        noise grows ~N    (the model used in mc_budget.py)
      "sqrtN"  f = sqrt(8/N)  noise grows ~sqrt(N)  (independent per-cell noise sources summed on one node)
      "fixed"  f = 8/N       noise ~constant  (noise set by the readout circuit, not the column)
    Which one is physical for a given circuit is an open hardware question; this measures how much the
    N-trend of the noise budget depends on that choice.

PREDICTIONS (written before running):
  R1  with a 14-bit ADC the error-free N = 40 baseline reaches >= 95%.
  R2  under "sqrtN", the tolerable noise at N = 40 is >= 3e-4 (vs < 1e-4 under "N").
  R3  under "fixed", the tolerable noise stays within 3x from N = 8 to N = 40: most of the N-trend in the
      noise budget comes from the modelling choice.
  R4  at N = 8 the three models give identical results (construction check).
"""
import itertools
import time
import numpy as np
import scale
import mc_budget
from s2_oa import run


class Chip2(mc_budget.Chip):
    def __init__(self, N, c, sn=0.0, adc=14, model="N"):
        super().__init__(N, c, sn=sn, adc=adc)
        self.sn = sn * {"N": 1.0, "sqrtN": np.sqrt(8 / N), "fixed": 8 / N}[model]


def main():
    np.seterr(all="ignore"); t0 = time.time()
    chips, Ns = 200, [8, 24, 40]
    ksets, gammas = [(1.0, 1.0, 1.0), (1.1, 0.9, 1.0)], [0.1, 0.3, 0.5]
    systems = {N: [(H, A, scale.reference(H, A, N // 2)) for H, A in
                   (scale.system(N, g, ks) for ks, g in itertools.product(ksets, gammas))] for N in Ns}
    cells = [("ADC12 none", dict(adc=12)), ("ADC14 none", dict(adc=14))]
    for model in ["N", "sqrtN", "fixed"]:
        for sn in [1e-4, 3e-4, 1e-3]:
            cells.append((f"{model:>5} n={sn:g}", dict(adc=14, sn=sn, model=model)))
    print(f"chips={chips}; runs per cell = {chips * 6}; * = >= 95%")
    print(f"{'cell':>18} " + " ".join(f"{'N=' + str(N):>16}" for N in Ns))
    for label, kw in cells:
        row = []
        for N in Ns:
            res = []
            for c in range(chips):
                chip = Chip2(N, c, **kw)
                for H, A, ref in systems[N]:
                    res.append(run(scale.sp2(H, N // 2, chip.mm)[0], H, A, N // 2, ref, c, 1e-10) is not None)
            p = float(np.mean(res)); se = np.sqrt(p * (1 - p) / len(res))
            row.append(f"{p:6.3f}+-{se:.3f}{'*' if p >= 0.95 else ' '}")
        print(f"{label:>18} " + " ".join(f"{r:>16}" for r in row), flush=True)
    print(f"\nruntime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
