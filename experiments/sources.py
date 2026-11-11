from mol_engine import run
import numpy as np

OFF = dict(gain_sigma=0, offset_sigma=0, noise_sigma=0, adc_bits=16)
cases = {
    "none (quantization)":  {},
    "12-bit duty only":     dict(a_bits=12),
    "gain mismatch only":   dict(gain_sigma=0.017),
    "offset only":          dict(offset_sigma=0.002),
    "noise only":           dict(noise_sigma=0.003),
    "10-bit ADC only":      dict(adc_bits=10),
    "all together":         dict(gain_sigma=0.017, offset_sigma=0.002,
                                 noise_sigma=0.003, adc_bits=10),
}
CHIPS = 40
print(f"{'case':<22}{'E_an err':>10}{'std':>8}{'E_ref err':>11}{'std':>8}{'iters0':>8}")
for label, o in cases.items():
    res = [run("benzene", seed=s, tol=0.005, **dict(OFF, **o))["result"] for s in range(CHIPS)]
    ea = np.array([r["E_analog"] - 8 for r in res])
    er = np.array([r["E_refined"] - 8 for r in res])
    it = np.mean([r["orbitals"][0]["iters"] for r in res])
    print(f"{label:<22}{ea.mean():>+10.4f}{ea.std():>8.4f}{er.mean():>+11.4f}{er.std():>8.4f}{it:>8.1f}")