"""
seed_sweep.py  --  analog_mc.py part 1 rerun with a DIFFERENT range-finder seed per chip (seed = chip index),
instead of the single seed 7 used before. Same chips, modes, settings and target. The cold-start control is
averaged over 200 seeds too. Prints the same table plus steps saved vs the cold start at the 95% line.
"""
import numpy as np
from analog_mc import SETTINGS, MODES, Chip, sp2_chip, map_err, TARGET
from h1_acrolein import hamiltonian
from r123_float64 import reference, gershgorin
from refine import refine, map_from_pairs

f32 = np.float32
KS = range(7)


def err(X, H, k, ref, seed):
    try:
        _, Co, Cv, eo, ev = refine(X, H, k, seed=seed)
        return map_err(map_from_pairs(Co, Cv, eo, ev), ref)[0]
    except (np.linalg.LinAlgError, FloatingPointError):
        return np.inf


def k95(reach):
    return next((k for k in KS if np.mean(reach[k]) >= 0.95), None)


def main():
    np.seterr(all="ignore")
    Hs = [hamiltonian(g, hO, *ks) for ks, hO, g in SETTINGS]; refs = [reference(H) for H in Hs]
    chips = range(200)
    reach = {k: [] for k in KS}
    for c in chips:
        for H, ref in zip(Hs, refs):
            a, b = gershgorin(H); X = ((b * np.eye(8) - H) / (b - a)).astype(f32)
            for k in KS: reach[k].append(err(X, H, k, ref, c) <= TARGET)
    kc = k95(reach)
    print("range-finder seed = chip index; fraction reaching target at k = 0..6; k95; steps saved vs COLD")
    print(f"{'COLD':>8} " + " ".join(f"{np.mean(reach[k]):.3f}" for k in KS) + f"   k95={kc}")
    for name, args in MODES.items():
        reach = {k: [] for k in KS}
        for c in chips:
            for H, ref in zip(Hs, refs):
                X, _ = sp2_chip(H, Chip(1000 + c, *args))
                for k in KS: reach[k].append(err(X, H, k, ref, c) <= TARGET)
        kb = k95(reach)
        sv = (kc - kb) if (kc is not None and kb is not None) else None
        print(f"{name:>8} " + " ".join(f"{np.mean(reach[k]):.3f}" for k in KS) + f"   k95={kb}  saved={sv}", flush=True)


if __name__ == "__main__":
    main()
