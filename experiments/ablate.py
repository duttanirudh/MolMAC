from mol_engine import run
import numpy as np

cases = {
    "quantization only":  dict(gain_sigma=0, offset_sigma=0, noise_sigma=0, adc_bits=16),
    "+ ADC 10 bit":       dict(gain_sigma=0, offset_sigma=0, noise_sigma=0),
    "+ noise":            dict(gain_sigma=0, offset_sigma=0),
    "+ offset":           dict(gain_sigma=0),
    "+ gain mismatch":    dict(),
}
print(f"{'case':<20}{'top x':>8}{'E analog':>10}{'E refined':>11}")
for label, o in cases.items():
    r = run("benzene", **o)["result"]
    print(f"{label:<20}{r['orbitals'][0]['x_analog']:>8.4f}{r['E_analog']:>10.4f}{r['E_refined']:>11.4f}")

errs = [run("benzene", seed=s)["result"] for s in range(20)]
for key in ("E_analog", "E_refined"):
    e = np.array([x[key] - 8.0 for x in errs])
    print(f"20 chips, {key}: mean error {e.mean():+.4f}, std {e.std():.4f}")