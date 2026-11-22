# Catalyst-requirement pipeline under low precision

How much numerical precision does a low-precision or analog front end need, if an exact digital stage
finishes the job? Measured on a Hückel model of a Diels-Alder pair, from 8 to 64 π centres.

This branch extends MolMAC from "analog array solves for energies" to "analog array supplies a starting
guess that an exact refinement can finish", and reports the precision budget for doing so.

## Result in one line

The low-precision stage is **necessary, not optional** — a cold start fails in 95–100% of runs — and the
precision it needs **rises with molecule size**: 8 → 12 bits for digital integer, 10 → 14 bits for the
analog chain, across N = 8 → 64. Read noise is the binding error source. **The design saves no energy**:
the exact refinement alone costs about as much as diagonalizing H directly.

## Pipeline

```
diene + dienophile
  → Hückel matrix H(γ)             γ = coupling on the two forming bonds
  → SP2 purification               LOW PRECISION or analog; matrix products only
  → range finder + one Rayleigh-Ritz per block
  → Ogita-Aishima refinement       float64, 4 dense products per step
  → sensitivity map dI/dδ          per atom, first order from the refined eigenpairs
  → rule table                     coarse catalyst requirements per site
```

Observable: `I = p(C1–Cb) − p(C4–Ca)`, the forming-bond imbalance. The catalyst stand-in δ lowers one
atom's on-site energy. Units of |β|, α_C = 0.

## Headline numbers

Minimum bits for ≥ 95% success (map relative error ≤ 1e-5 within 10 refinement steps):

| π centres | HOMO–LUMO gap | Digital integer | Analog, ideal ADC range | Analog, worst-case range |
| --- | --- | --- | --- | --- |
| 8 | 0.97 | 8 | 10 | 10 |
| 24 | 0.36 | 8 | 14 | 14 |
| 40 | 0.22 | 10 | 12 | 16 |
| 64 | 0.13 | 12 | 14 | not reached by 16 |

Error budget, analog model at 8 atoms (8-bit weights and duty, 12-bit ADC, 200 simulated chips):

| Source | Tolerable at 8 atoms | At 24 atoms |
| --- | --- | --- |
| Per-cell gain error | 5% | 0.5% |
| Column offset, split per pass | 1e-3 of full scale | 1e-3 |
| Read noise | 1e-3 of full scale | ≤ 1e-4 |

MolMAC's own PWM ripple, from `mol_engine.py`'s LTspice-checked model, is 1.1–2.0e-3 of full scale and is
phase-locked, so repeated reads do not average it out. That fits the 8-atom budget and is about tenfold
too large from 24 atoms up.

## Layout

All Python files are flat and import each other by module name, so run them from this directory.
Requires `numpy`, `scipy`, `scikit-learn`, `pandas`, `biopython` (the last three only for the enzyme scripts).

**Hückel model and exact checks**

| File | What it does | Runtime |
| --- | --- | --- |
| `h0_exact.py` | Butadiene + ethylene exact checks, SP2, polarizability | < 1 s |
| `h1_acrolein.py` | Butadiene + acrolein, catalyst sign test, sensitivity map | < 1 s |
| `r123_float64.py` | Three routes to the sensitivity map vs an exact reference | < 1 s |
| `rules.py` | Rule table: levers, robustness, plain-language requirements | < 1 s |

**Precision**

| File | What it does | Runtime |
| --- | --- | --- |
| `ladder.py` | Precision ladder, float64 → int8 | ~3 s |
| `polish.py` | Int8 rescue: H-blind polish, switch point, H-aware refinement | ~2 s |
| `refine.py` | Shared H-aware refinement (range finder, aufbau check) | module |
| `analog_mc.py` | Analog Monte Carlo at 8 atoms, cold-start control; `--part 1/2/3` | ~4 min |
| `seed_sweep.py` | `analog_mc` part 1 with a range-finder seed per chip | ~3 min |
| `mc_budget.py` | Error-source precision budget, 200 chips | ~2 min |
| `robust.py` | 14-bit ADC and three noise-scaling models | ~2 min |

**Scale-up**

| File | What it does | Runtime |
| --- | --- | --- |
| `scale.py` | Size family N = 8–40, steps saved, 10 seeds | ~6 s |
| `s2_oa.py` | Ogita-Aishima refiner started from each front end | ~5 s |
| `bits.py` | Minimum bits vs N | ~5 s |
| `bits2.py` | Power-of-two auto-ranging, and N up to 64 | ~11 s |
| `no_rr.py` | Products-only starts (negative result: 0% success) | ~5 s |

**Enzyme-screening control** — each takes the dataset CSV as its first argument, default `dataset.csv`.
Run `align_ident.py` first; it writes `ident.json`, which the other two reuse.

| File | What it does | Runtime |
| --- | --- | --- |
| `align_ident.py` | BLOSUM62 local-alignment identities among the positives | ~2 s |
| `background.py` | Identity background, positives vs random negatives | ~6 s |
| `positive_control.py` | Positive control on the full dataset, with family holdout | ~7 s |
| `matched.py` | Positive control on the curation-matched cohort | ~2 s |

**Frontend** — `frontend/catalyst_pipeline.html` is the built single-file page; open it in a browser.
`engine.js` holds the numerics, `app.js` the interface, `template.html` the shell. Rebuild with:

```
python -c "t=open('template.html').read(); \
  t=t.replace('/*ENGINE*/',open('engine.js').read()).replace('/*APP*/',open('app.js').read()); \
  open('catalyst_pipeline.html','w').write(t)"
```

`node test_engine.js` checks the JavaScript engine against values exported from the Python pipeline
(`expected.json`); it agrees to about 1e-15.

## Method notes and caveats

- **Every script states its predictions in its header, written before the run.** Several were wrong; the
  misses are recorded rather than removed.
- **Rule table.** The asynchronicity lever became primary *after* the interaction-energy lever failed the
  carbonyl-oxygen control. That choice was made after seeing the control, and is flagged in `rules.py`.
  Controls it does pass: acrolein's carbonyl O wants an electron-withdrawing contact (matching Lewis-acid
  and H-bond catalysis), an all-carbon pair shows no lever, an ether O on the diene wants donation.
- **Not claimed:** analog necessity for chemistry, an energy or speed advantage, reaction mechanism,
  validated catalyst rules, or novelty beyond "not found after a limited search".
- **Model limits:** simple Hückel theory has no σ framework, sterics, solvent, geometry or Pauli
  repulsion. The analog model is behavioral; only its gain figures are sourced from measured MolMAC and
  VoltMatrix numbers, while offset and noise levels are placeholders.
- **Enzyme control:** on the full dataset the apparent signal is a curation artifact — all 22 known
  Diels-Alderases are reviewed Swiss-Prot entries against 177 of 3,597 others, so "is reviewed" alone
  scores AUC 0.975. In a curation-matched cohort the sequence features hold at AUC 0.77–0.81 under
  homolog removal, which is genuine remote-homology detection but no better than a standard BLAST or HMM
  search; the metadata features sit at chance.

## References used

- Ogita & Aishima, iterative refinement for symmetric eigendecomposition — arXiv 1909.13709 (fixed-point
  analysis, with the basic algorithm), and Terao et al., arXiv 2602.23778 (subset refinement, untested here).
- Niklasson et al., arXiv 2409.17033 — dual susceptibility, SP2, FP16-pair products.
- Dawson et al., arXiv 2407.13299 — precision requirements of the density-matrix kernel.
- Bickelhaupt group — Lewis-acid catalysis via reduced Pauli repulsion and increased asynchronicity.
