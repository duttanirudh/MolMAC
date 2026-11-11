from mol_engine import run
import numpy as np

QO = dict(gain_sigma=0, offset_sigma=0, noise_sigma=0, adc_bits=16)

def stats(over, chips=20):
    res = [run("benzene", seed=s, **over)["result"] for s in range(chips)]
    ea = np.array([r["E_analog"] - 8.0 for r in res])
    er = np.array([r["E_refined"] - 8.0 for r in res])
    it = np.mean([r["orbitals"][0]["iters"] for r in res])
    return ea.mean(), ea.std(), er.mean(), er.std(), it

print(f"{'case':<18}{'tol':>7}{'E_an err':>10}{'std':>8}{'E_ref err':>11}{'std':>8}{'iters0':>8}")
for label, over in (("quantization only", QO), ("full imperfections", {})):
    for tol in (0.02, 0.005, 0.001):
        m1, s1, m2, s2, it = stats(dict(over, tol=tol))
        print(f"{label:<18}{tol:>7}{m1:>+10.4f}{s1:>8.4f}{m2:>+11.4f}{s2:>8.4f}{it:>8.1f}")