import numpy as np
from test_mol_engine import sweep, opts

N = 200
CASES = [
    ("ideal (16-bit ADC)",       dict(gain_sigma=0.0, offset_sigma=0.0, noise_sigma=0.0, adc_bits=16)),
    ("ideal + 10-bit ADC",       dict(gain_sigma=0.0, offset_sigma=0.0, noise_sigma=0.0)),
    ("ideal + noise (16-bit)",   dict(gain_sigma=0.0, offset_sigma=0.0, adc_bits=16)),
]
for label, o in CASES:
    ea, er, it = sweep("naphthalene", N, opts(**o))
    print(f"{label:<26}E_an {ea.mean():+.4f} +/- {ea.std() / np.sqrt(N):.4f}  "
          f"std {ea.std():.4f}  E_ref {er.mean():+.5f}  iters {it.mean():.1f}")