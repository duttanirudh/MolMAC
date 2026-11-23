# Technical note: catalyst-requirement pipeline under low-precision / analog arithmetic

Draft v2, 2026-11-23. Repository branch `cataline`, folder `catalyst/`.

**Status tags**

- **[V]** reproduced from a script log (or read directly from script source) in the laptop run of 2026-11-23.
- **[H]** taken from the project handoff; not re-checked.
- **[OPEN]** not yet checked, unexplained, or a known gap.

Environment of all [V] runs: home laptop, Windows, venv `cataline`; Python 3.15.2, numpy 2.5.5, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1, biopython 1.88. `bits2.py` run twice gave identical output apart from the runtime line (deterministic on one machine). Earlier workstation runs have not been compared. **[OPEN]**

---

## 1. Scope and claims

A Hückel pi-electron model of a diene/dienophile pair is solved by a low-precision or analog front end (SP2 density-matrix purification) followed by an exact float64 refinement. The pipeline reports (a) a precision budget for the front end and (b) a per-atom table of the catalyst environment that would be required. All hardware is simulated.

Claims supported by the logs:

1. A products-only refiner cannot start cold; a low-precision front end plus one Rayleigh-Ritz per block is needed. (6.1, 6.3)
2. Required precision rises with system size, up to the sizes tested. The numbers are less regular than the handoff suggested (6.2).
3. Read noise is the dominant error source, and its tolerance depends strongly on how noise is assumed to scale with N (6.4).
4. At 8 atoms, every simulated front end reaches 95% at least one refinement step before a cold start does, as a lower bound (6.1).
5. Finite-difference sensitivity (R3) is unusable on noisy analog arithmetic (6.5).

Not claimed: section 10.

## 2. Model

- Units of |beta|, alpha_C = 0, H_ii = -h_i, H_ij = -k_ij; forming bonds carry coupling gamma; the catalyst stand-in lowers one atom's on-site energy by delta (`H[k,k] -= delta`). [V from `h1_acrolein.py`, `rules.py` source]
- Systems: butadiene + ethylene; butadiene + acrolein (8 atoms, N_OCC = 4); 2-methoxybutadiene + acrolein; a polyene/polyenal family for scaling (N = 8-64; N = 8 reproduces butadiene + acrolein). Hetero-atom parameters: h_O = 1 (carbonyl, "sourced"), h_OR = 2 and C-O(ether) k = 0.8 (the latter "from memory" per `rules.py`).
- Observable: I = p(C1-Cb) - p(C4-Ca) = Tr(A D). A = |I|/(p1 + p2) only where p1 + p2 >= 0.3 (printed `nan` at gamma = 0.1). [V]
- Exact check, butadiene + ethylene [V]: E_pi 6.472136 at gamma = 0 and 8.000000 at gamma = 1; interaction energy 1.527864 at gamma = 1; charges 1 and same-sublattice bond orders 0 to ~1e-15; HOMO-LUMO gap 1.2361 to 2.0000.

## 3. Algorithms

- **SP2** (trace-correcting, Gershgorin bounds computed digitally, stall rule or fixed cap, divergence guard). Versus `eigh` for butadiene + ethylene [V]: density-matrix error <= 1.4e-15, 16-19 iterations.
- **Sensitivity routes.** R1 direct perturbation; R2 adjoint (dual) pass; R3 central finite differences. Hellmann-Feynman and polarizability agree with finite differences to ~1e-10 [V].
- **Refinement (`refine.py`).** Range finder (P times seeded Gaussian, QR), Rayleigh-Ritz per block, first-order occupied-virtual correction, aufbau swap check. At 8 x 8 this is effectively a full eigensolver, so benefit is measured against a cold-start control. The default range-finder matrix uses seed 7; per-run seeds are used where stated.
- **Ogita-Aishima (`s2_oa.py`).** Start: range finder, then one Rayleigh-Ritz per block (the only diagonalizations); then products-only steps `X <- X(I + E)` in float64. Success: map relative error <= 1e-5 within 10 steps.

## 4. Analog behavioral model [V from `analog_mc.py`, `mc_budget.py` source]

Product Y = A @ B: B programmed as weights (8-bit signed, per-matrix scale) with static per-cell gain 1 + sigma_g N(0,1); rows of A applied as 8-bit duty cycles, positive and negative parts in two passes; column sums read with a static column offset and fresh Gaussian read noise (both fractions of column full scale FS_j = sum_i |W_ij|), then a b-bit ADC.

**Parameter status (stated in the script headers):** gain 1.7% is the "measured MolMAC/VoltMatrix-scale" figure and 0.12% the "MolMAC oracle-calibrated" figure. **All offset and noise levels are placeholders, not sourced.** The budgets below are therefore requirements ("how much error can be tolerated"), not predictions about any real chip.

`analog_mc.py` modes (quantization, sigma_g, sigma_o, offset mode, sigma_n):

| Mode | Quantized | gain | offset | noise |
|---|---|---|---|---|
| IDEAL | no | 0 | 0 | 0 |
| Q8 | yes | 0 | 0 | 0 |
| G_cal / G_unc | no | 0.12% / 1.7% | 0 | 0 |
| O_same / O_split | no | 0 | 1e-2 (same in both passes / separate per pass) | 0 |
| N1 / N10 | no | 0 | 0 | 1e-3 / 1e-2 |
| ALL_cal | yes | 0.12% | 2e-3 split | 2e-3 |
| ALL_unc | yes | 1.7% | 1e-2 split | 1e-2 |

"ALL" in `mc_budget.py` is a different combination: gain 0.12% (cal) or 1.7% (uncal) with offset 1e-3 and noise 1e-3, 8-bit weights and duty, 12-bit auto-ranged ADC.

## 5. Experimental design [V from source]

- Seeds: chip c uses static seed 1000 + c (`analog_mc`, `mc_budget`), noise seed 5000 + c (`mc_budget`) or seed + 10^6 (`analog_mc`), range-finder seed = chip index (`seed_sweep`, `mc_budget`).
- `analog_mc.py`, `seed_sweep.py`: 200 chips x 18 settings = 3,600 runs per stochastic row (sign-failure denominators 28,800 = 3,600 x 8 atoms). Settings: 2 bond sets x h_O in {0.5, 1, 2} x gamma in {0.1, 0.3, 0.5}.
- `mc_budget.py`, `robust.py`: 200 chips x 6 settings = 1,200 runs per cell (h_O = 1).
- `bits.py`, `bits2.py`, `s2_oa.py`, `no_rr.py`: **60 runs per cell** (6 settings x 10 range-finder seeds), so one run = 1.7 percentage points and the standard error near 95% is about 0.03. A difference of one grid step in "minimum bits" (2 bits) is within noise for most cells.
- `scale.py`: 6 settings per N, 10 range-finder seeds; reports median [min-max] steps until all settings pass.
- Rows starred in `analog_mc` part 1 (COLD, IDEAL, Q8, O_same) are deterministic across chips: effective n = 18 settings, and they use the single default range-finder seed. They are not used for step-saving claims.

## 6. Results

### 6.1 Cold start versus front end, 8 atoms (`seed_sweep.py`) [V]

Fraction reaching target after k refinement steps; n = 3,600 per row.

| Mode | k=0 | 1 | 2 | 3 | 4 | 5 | 6 | k95 |
|---|---|---|---|---|---|---|---|---|
| COLD | .000 | .000 | .010 | .398 | .720 | .862 | .925 | none |
| IDEAL | .596 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| Q8 | .000 | .000 | .766 | .922 | .961 | .976 | .989 | 4 |
| G_cal | .000 | .964 | .999 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| G_unc | .000 | .344 | .964 | .990 | .997 | .999 | .999 | 2 |
| O_same | .536 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| O_split | .000 | .028 | .828 | .944 | .974 | .986 | .994 | 4 |
| N1 | .000 | .001 | .889 | .969 | .986 | .993 | .995 | 3 |
| N10 | .000 | .000 | .198 | .735 | .878 | .940 | .967 | 6 |
| ALL_cal | .000 | .000 | .719 | .920 | .962 | .980 | .988 | 4 |
| ALL_unc | .000 | .000 | .175 | .711 | .867 | .929 | .960 | 6 |

The cold start never reaches 95% by k = 6, so "steps saved" is undefined in the log; every front end reaching 95% by k = 6 gives "at least one step". For N10 and ALL_unc the margin at k = 6 is small (0.967 and 0.960 against 0.925).

### 6.2 Scale-up: steps saved and break-even (`scale.py`) [V]

Median [min-max over 10 seeds] steps until all 6 settings reach 1e-5 (99 = not by k = 12), and steps saved against COLD; "be" is the factor by which the front end's MAC must be cheaper than a refinement-step MAC to pay for itself.

| N (gap) | COLD | FP16PAIR (saved, be) | INT8 | AN8 | AN10 |
|---|---|---|---|---|---|
| 8 (0.970) | 4 [3-99] | 1 (+3, 2.1x) | 2 (+2, 2.5x) | 3 [2-6] (+1, 10.2x) | 2 [2-99] (+2, 5.5x) |
| 16 (0.532) | 7 [5-9] | 1 (+6, 1.2x) | 2 (+5, 1.1x) | 3 (+4, 2.7x) | 2 (+5, 2.3x) |
| 24 (0.363) | 6 [4-11] | 1 (+5, 1.5x) | 2 (+4, 1.5x) | 5 (+1, 12.4x) | 3 (+3, 4.4x) |
| 32 (0.270) | 6 [5-12] | 1 (+5, 1.6x) | 3 (+3, 2.1x) | 5 [3-99] (+1, 12.4x) | 4 (+2, 6.5x) |
| 40 (0.215) | 6 [5-9] | 1 (+5, 1.7x) | 4 (+2, 3.6x) | 6 (+0, never) | 3 (+3, 4.6x) |

(N = 12 omitted for space.) AN8 and AN10 are noise-free analog chains (8-bit weights/duty, 8- or 10-bit ADC). The MAC counts exclude the two N/2 x N/2 block diagonalizations per refinement step (stated in the log), so break-even here is not a full energy accounting.

### 6.3 Ogita-Aishima start (`s2_oa.py`, `no_rr.py`) [V]

Success over 60 runs (delta1 = 1e-10; the delta1 = 1e-3 table is identical in every cell, see section 7):

| N | COLD | FP16PAIR | INT8 | AN8 | AN10 |
|---|---|---|---|---|---|
| 8 | .05 | 1.00 | 1.00 | .90 | .93 |
| 16 | .00 | 1.00 | 1.00 | .67 | .88 |
| 24 | .00 | 1.00 | 1.00 | .33 | .70 |
| 32 | .00 | 1.00 | .87 | .18 | .57 |
| 40 | .00 | 1.00 | .77 | .03 | .43 |

Without the Rayleigh-Ritz (`no_rr.py`, up to 30 steps): NONE gives 0.00 in every case (also with cluster threshold 0.1); block power iteration (POWER, m = 20) gives 0.60 at N = 8 and ~0 at N >= 16 (FP16PAIR: 0.03 at N = 16, 0.00 at 24 and 40). One Rayleigh-Ritz per block is needed.

### 6.4 Required precision against N (`bits.py`, `bits2.py`) [V]; n = 60 per cell

Minimum bits for >= 95% success. HOMO-LUMO gap: N = 8: 0.970; 16: 0.532; 24: 0.363; 32: 0.270; 40: 0.215; 48: 0.178; 56: 0.153; 64: 0.133.

| N | Digital INTb | ideal (oracle ADC range) | worst-case range | pga (power-of-two auto-range) |
|---|---|---|---|---|
| 8 | 8 | 10 | 10 | 10 |
| 16 | 8 | 10 | 14 | 14 |
| 24 | 8 | 14 | 14 | 14 |
| 32 | 10 | 12 | 14 | 14 |
| 40 | 10 | 12 | 16 | 16 |
| 48 | 10 | 14 | not run | 16 |
| 56 | 10 | none (.80 at 14, .67 at 16) | not run | 16 |
| 64 | 12 | 14 (.97 at 14, .58 at 16) | not run | 16 |

Reading: the digital requirement rises from 8 to 12 bits; the analog requirement rises faster and irregularly. pga equals worst-case for N <= 40 (it did not change the result), so power-of-two ranging recovered nothing in this model. At N = 56 and 64 the ideal-range success rate falls when going from 14 to 16 bits (0.97 → 0.58 at N = 64: a difference far larger than the n = 60 noise); the cause is unexplained **[OPEN]**, and the N >= 48 results should not be used to state a trend. The worst-case range was run only up to N = 40 (`bits2.py` skips it above).

Variants in `bits.py` (8-bit weights/duty, ADC b bits): W8D8-worst min bits N = 8/16/24/32/40: 10/14/14/14/16; W8D8-fit: 10/10/14/12/12; ALLb-fit (all b bits): 10/12/12/12/14; INTb: 8/8/8/10/10.

### 6.5 Error budget at 8-bit weights (`mc_budget.py`, `robust.py`) [V]

Success fraction (n = 1,200 per cell; s.e. 0.002-0.014); 12-bit auto-ranged ADC; `*` = >= 0.95.

| Cell | N = 8 | N = 24 | N = 40 |
|---|---|---|---|
| none | .996* | .962* | .886 |
| gain 0.12% | .996* | .977* | .942 |
| gain 0.5% | .995* | .956* | .938 |
| gain 1.7% | .995* | .932 | .869 |
| gain 5% | .956* | .769 | .632 |
| offset 1e-4 | .994* | .962* | .943 |
| offset 1e-3 | .994* | .956* | .922 |
| offset 1e-2 | .913 | .584 | .111 |
| noise 1e-4 | .992* | .928 | .838 |
| noise 1e-3 | .939 | .600 | .210 |
| noise 1e-2 | .586 | .000 | .000 |
| ALL cal | .951* | .568 | .211 |
| ALL uncal | .950* | .547 | .233 |

Noise-scaling and ADC-width robustness (`robust.py`, 14-bit ADC unless stated; noise = s_n x FS_j x f(N), with f = 1 ("N"), sqrt(8/N) ("sqrtN"), 8/N ("fixed"), identical at N = 8):

| Cell | N = 8 | N = 24 | N = 40 |
|---|---|---|---|
| ADC12, no error | .996* | .962* | .886 |
| ADC14, no error | .996* | .998* | .959* |
| N, 1e-4 / 3e-4 / 1e-3 | .988* / .978* / .955* | .934 / .830 / .563 | .842 / .612 / .211 |
| sqrtN, 1e-4 / 3e-4 / 1e-3 | same as N | .955* / .889 / .731 | .917 / .797 / .477 |
| fixed, 1e-4 / 3e-4 / 1e-3 | same as N | .976* / .934 / .818 | .952* / .898 / .721 |

Reading:

- Gain tolerance: about 5% at 8 atoms; about 0.5% at 24 (1.7% fails).
- Noise tolerance (grid 1e-4, 3e-4, 1e-3): at N = 8 at least 1e-3 with a 14-bit ADC (0.955), but only just below the line with a 12-bit ADC (0.939). At N = 24 and 40 it is below 3e-4 under every scaling model; it is below 1e-4 under the "N" model and between 1e-4 and 3e-4 under "fixed" (N = 24 and 40) and "sqrtN" (N = 24 only). So the tolerance falls by at least 3.3x from N = 8 to N = 24 under every model, and the exact N-trend depends on the (unknown) physical scaling.
- Offset tolerance: between 1e-3 and 1e-2 at N = 8 and 24 (12-bit ADC).
- Individually acceptable errors do not compose at N = 24 (ALL: 0.568).
- At N = 40 the unperturbed 12-bit chain already fails (0.886); with a 14-bit ADC it passes (0.959). Small perturbations improve the 12-bit result (gain 0.12%: 0.942; offset 1e-4: 0.943). A plausible explanation is that random perturbation dithers a deterministic quantization error; this is a hypothesis, not tested. **[OPEN]**

### 6.6 Refined pipeline, R3 and iteration cap (`analog_mc.py`) [V]

Part 1, stochastic modes (n = 3,600): reaching target at k = 3 / k = 5, and sign failures at k = 3 (of 28,800).

| Mode | k=3 | k=5 | Sign failures |
|---|---|---|---|
| G_cal | 1.000 | 1.000 | 0 |
| G_unc | .988 | .999 | 35 |
| O_split | .928 | .984 | 355 |
| N1 | .967 | .994 | 129 |
| N10 | .635 | .917 | 1,675 |
| ALL_cal | .847 | .965 | 750 |
| ALL_unc | .617 | .912 | 1,723 |

Part 2 (R3 on the analog solver, no refinement, h = 0.03; median / 90th percentile map error): IDEAL 3.4e-4 / 5.1e-4; O_same 3.5e-4 / 4.9e-4; O_split 3.0 / 27; G_cal 5.7e-2 / 0.17; G_unc 0.33 / 2.3; N1 14 / 36.

Part 3 (SP2 cap, then k = 3): ALL_cal reaches .529 (cap 4), .426 (6), .733 (8), .868 (12), .848 (20), .847 (stall); ALL_unc .538, .436, .559, .636, .604, .617 (diverged 1.9% at cap 20). The drop from cap 4 to cap 6 is unexplained. **[OPEN]**

### 6.7 Precision ladder and sensitivity routes (`ladder.py`, `r123_float64.py`) [V]

18 settings, butadiene + acrolein; median / worst relative error of the map:

| Backend | R1 | R2 | R3 (best h) | sign failures R1 / R2 / R3 (of 144) |
|---|---|---|---|---|
| fp64 | 5.5e-15 / 5.7e-13 | 5.1e-15 / 5.7e-13 | 3.7e-9 / 5.4e-9 (1e-4) | 0 / 0 / 0 |
| fp32 | 2.0e-6 / 6.6e-5 | 2.3e-6 / 6.6e-5 | 2.1e-4 / 1.2e-3 (1e-2) | 0 / 0 / 0 |
| fp16 pair | 9.1e-6 / 1.3e-4 | 3.7e-6 / 1.3e-4 | 4.1e-4 / 9.3e-4 (3e-2) | 0 / 0 / 0 |
| int8 | 0.36 / 2.0 | 0.36 / 1.3 | 1.0 / 5.1e3 | 17 / 13 / 130 |

For int8, SP2 stalls at an idempotency floor of 5.4e-3, the relative error of I is 0.26 median (1.5 worst, at gamma = 0.1), and R3's median error is at least 1.0 at every step size tried (0.0001 to 0.3). In float64 (`r123_float64.py`): R1 and R2 match the analytic reference to <= 5.3e-15, R3 to <= 3.3e-10, |R1 - R2| <= 1.5e-16. Matrix products per full 8-atom map: R2 54-69, R1 306-391, R3 296-359.

### 6.8 Catalyst rule table (`h1_acrolein.py`, `rules.py`) [V]

Sensitivity map dI/d(lower alpha_k), h_O = 1, bond set "std", butadiene + acrolein: at gamma = 0.3: C1 +7.2e-3, C2 -4.4e-2, C3 -1.3e-3, C4 -3.8e-2, Cb +1.5e-2, Ca +1.4e-2, Cc +2.6e-2, O +2.05e-2. At h_O = 1 the oxygen is fourth by magnitude; the `O rank` column of the log gives 3, 4, 5 for h_O = 0.5, 1, 2 (I checked the h_O = 1 value against the map, not the other two).

Rule table, butadiene + acrolein, gamma = 0.3 (robust = share of parameter variants with the same sign):

| Fragment | Atom | L_A | Requirement | Robust |
|---|---|---|---|---|
| diene | C2 | -0.0438 | donating | 1.00 |
| diene | C4 | -0.0375 | donating | 1.00 |
| diene | C1 | +0.0072 | withdrawing | 1.00 |
| diene | C3 | -0.0013 | donating | 1.00 |
| dienophile | Cc | +0.0261 | withdrawing (through-space) | 1.00 |
| dienophile | O | +0.0205 | withdrawing (H-bond donor / Lewis acid), directed binding | 1.00 |
| dienophile | Cb | +0.0151 | withdrawing | 1.00 |
| dienophile | Ca | +0.0139 | withdrawing | 1.00 |

For this system the table ranks carbon sites above the carbonyl oxygen (rank 2 of 4 within the dienophile fragment). In 2-methoxybutadiene + acrolein the ether oxygen gets a donating requirement (L_A -0.0110, rank 3 in the diene) and the carbonyl O a withdrawing one (+0.0321). Butadiene + ethylene returns "no lever" (negative control, machine-precision zeros). Lowest HOMO-LUMO gap over all acrolein settings: 0.843.

The primary lever (L_A) was chosen after the interaction-energy lever gave a donating requirement for the acrolein O (L_E = -0.0011, robust). This is post-hoc and must stay disclosed.

### 6.9 MolMAC PWM ripple (`ripple_budget.py`) [V]

The ripple model is MolMAC's own (`ripple_shape`, circuit values from `Cfg` in `mol_engine.py`: K = 0.0397, tile of 4, channel start phases 0 / 0.749 / 0.498 / 0.247), imported unchanged and added to the `mc_budget.py` chip (8-bit weights and duty, 12-bit auto-ranged ADC, 200 chips x 6 settings = 1,200 runs per cell). The "none" row reproduces the `mc_budget.py` "none" row exactly (0.996 / 0.962 / 0.886), which checks the harness. theta is the ADC sample instant as a fraction of the PWM period.

| Cell | N = 8 | N = 24 | N = 40 |
|---|---|---|---|
| none | .996* | .962* | .886 |
| ripple, theta = 0 | .993* | .959* | .959* |
| ripple, theta = 0.25 | .998* | .992* | .941 |
| ripple, theta = 0.5 | 1.000* | .954* | .960* |
| ripple (theta = 0) + noise 1e-4 | .994* | .953* | .852 |

Ripple size, |ripple| / FS_j over all analog products (median / 95th percentile, theta = 0): N = 8: 1.6e-4 / 1.3e-3; N = 24: 2.8e-5 / 8.0e-4; N = 40: 9.0e-6 / 5.8e-4. The other phases differ by less than 20% in these figures.

Reading:

- In this model ripple does not reduce success at any N or phase tested: the largest drop against "none" is 0.3 percentage points (N = 8 and 24 at theta = 0), within the standard error (0.2-0.6 points).
- Ripple relative to full scale falls with N (median by about 18x from N = 8 to 40), because branch ripples of mixed sign partly cancel while FS_j grows. The statement "ripple is about 10x too large from 24 atoms" is not supported.
- The 95th-percentile ripple at N = 8 (1.3-1.4e-3) is the same order as the handoff's "1.1-2.0e-3 of full scale". Whether it is the same quantity cannot be established, since `ripple_check.py` was not found; the handoff figure is withdrawn and this table replaces it.
- At N = 40 the unperturbed 12-bit baseline fails (0.886) and ripple raises it to 0.94-0.96. Gain 0.12% and offset 1e-4 did the same in 6.5. All three are consistent with dithering of a deterministic quantization error, but that is a hypothesis, not tested here, and it means the N = 40 comparison is confounded.

Limits of this measurement: the ripple model is first-order and its LTspice check was not repeated; the ADC is one per column (MolMAC digitizes per tile of 4); matrices are signed, with the sign carried through the ripple's linearity in the branch amplitude; only three sample phases were run; no gain, offset or other noise except in the last row.

## 7. Pre-registered predictions versus outcomes

Predictions are quoted from the script headers ("written before running"); I could not verify when they were written. ✓ = held, ✗ = did not hold, ~ = partly.

| Script | Outcome |
|---|---|
| `h1_acrolein` | P1 (I > 0), P2 (dI/d-delta > 0), P3 (charge transfer > 0 and rising) ✓ in 18/18 settings. **P4 ✗ (O in top two by magnitude): fails in 18/18; O ranks 3rd-5th.** |
| `r123_float64` | Q1-Q3 ✓ (5.3e-15, 3.3e-10, 1.5e-16). Q4 ~: R2 uses fewest products; R1 is about 6x R2, not ~8x; R3 about equals R1. |
| `ladder` | L1, L2, L3, L4, L6 ✓. L5 ~: U-shape and best-h rising ✓ for fp64/fp32/fp16 pair; for int8 no step size is usable (error >= 1). |
| `rules` | C1 ✓, C4 ✓. C2 ~: O withdrawing and robust ✓, but rank 2 in its fragment (Cc is first). C3 ~: C2 and C4 donate most; C1 is withdrawing; C3 near zero. |
| `scale` | S1a ✓ (COLD 4 → 6). S1b ✗ (INT8 saves +2 to +5, more than predicted). S1c ✗ (AN8 saves +1 to +4 for N <= 32; AN10 saves +2 to +5, not 1-2). S1d ~: FP16 pair 17 → 23 iterations ✓; INT8, AN8 lower. |
| `s2_oa` | T1, T2, T3 ✓. T4 ~: AN8 below INT8 ✓; AN10 is also below INT8 at every N, not between INT8 and FP16 pair ✗. **T5 ✗: the delta1 = 1e-3 and 1e-10 tables are identical.** |
| `no_rr` | N1 ✓, N3 ✓. N2 ✗ (delta1 = 0.1 gives 0.00, no rescue). |
| `bits` | B1 ✗ (worst-case range: +6 bits from N = 8 to 40, not ~2). B2 ✗ (a fitted range saves 0 bits at N = 8 and 24). B3 ~: ✓ for fit and INTb (+2), ✗ for worst (+6) and ALLb-fit (+4). B4 ✗ (W8D8 variants do reach 95% at N = 32 and 40). |
| `bits2` | A1 ✗ (pga is 4 bits above ideal at N = 16 and 40). A2 ✗ (pga = worst for N <= 40). A3 ~: INTb +2 from N = 40 to 64; ideal is non-monotone. |
| `mc_budget` | P1 ~: ✓ at N = 8; at N = 40 the baseline fails so 0.5% is never reached. P2 ~. **P3 ✗: calibrated combination fails at N = 24 (0.568).** **P4 ✗: failure is not monotone at N = 40.** |
| `robust` | R1 ✓ (0.959). R2 ✗ (sqrtN, N = 40, 1e-4: 0.917). R3 ✗ narrowly (tolerance ratio N = 8 to 40 under "fixed" exceeds 3). R4 ✓. |
| `analog_mc` | M1 ✗ (Q8 .889, O_split .928, ALL_cal .847 below 95% at k = 3). M2 ✗ at k = 0 (0.336 vs 0.389), equal from k = 1. M3 ✗ (N10 and ALL_unc miss 36-38% at k = 3, not < 20%). H3' ✓ for O_same; magnitudes ✗ (O_split 3.0 vs 0.01-0.1 predicted; N1 14 vs <= 0.33). H4' ~. |
| `ripple_budget` | Q1 ✓ (N = 8 median 1.6e-4, falling with N), Q2 ✓ (-0.3 points at N = 8 and 24), Q3 ✓ (phase changes success by at most 3.8 points), Q4 ✓ (the "10x too large" claim is not supported). |

## 8. Cost accounting (no energy claim)

No energy was measured, and this note makes no energy claim. What can be stated:

**Digital work after the front end** [V, derived from `s2_oa.py` source and log]. Each Ogita-Aishima step is 4 dense N x N products (stated in the `s2_oa.py` header), i.e. 8 N^3 flops. With the median step counts at N = 40 (FP16 pair 1, INT8 4, AN10 5, AN8 6), the refinement costs 8, 32, 40 and 48 N^3 flops respectively, plus one Rayleigh-Ritz per block (two N/2 x N/2 symmetric eigenproblems) and the range-finder QR. A dense symmetric eigensolver with eigenvectors costs about 9 N^3 flops (Golub and Van Loan, *Matrix Computations*; figure quoted from memory, to be checked against the book before submission). So for every front end except FP16 pair, the digital refinement alone costs more than solving the problem directly, before any front-end cost is counted. This is the reason no energy saving is possible in this design, and it follows from operation counts, not from energy-per-operation figures.

**Break-even factors** [V, `scale.py`]: section 6.2. They exclude the block diagonalizations.

**Context only, not used in any claim** (literature figures, different processes and conditions, not comparable to each other without normalization):
- FP64 FMA: 26.7 pJ per instruction, measured on silicon for the FPnew unit (Mach et al., arXiv 2007.01530).
- 8b x 8b digital MAC: 0.390 pJ at 0.8 V (Taco et al., 2018, as cited in arXiv 2609.35586; primary source not checked).
- Analog in-memory macro: 25.5 TOPS/W for 8b/8b matrix computation, silicon measurement (Zheng et al., DAC 2024, doi 10.1145/3649329.3658472).

Removed from the handoff: the lower bound "1.8 TOPS/W" (no source found) and "10-42 N^3 flops" (no derivation found; replaced by the step-count derivation above).

## 9. Enzyme-screening positive control

Data: `rev_unrev.csv`, 3,619 rows, 9 columns, 22 positives (EC 5.5.1). `rev_unrev_normalized.csv` stores EC numbers as integers and cannot be used by these scripts. [V]

Full set (`positive_control.py`; EC number and catalytic activity removed): AUC (metadata / sequence / both): protocol A 0.971 / 0.909 / 0.974; B at t = 0.40: 0.969 / 0.902 / 0.971; t = 0.35: 0.968 / 0.893 / 0.970; t = 0.30: 0.966 / 0.883 / 0.968. [V]

Curation-matched cohort (`matched.py`; 199 reviewed entries, 22 positives; trivial baselines: sequence length 0.490, count of non-empty metadata fields 0.652): AUC (metadata / sequence / both): A 0.558 / 0.811 / 0.616; B t = 0.40: 0.536 / 0.793 / 0.588; t = 0.35: 0.524 / 0.784 / 0.573; t = 0.30: 0.493 / 0.769 / 0.539. [V]

Background (`background.py`; BLOSUM62 local-alignment identity): positive-vs-negative median 0.038, 95th percentile 0.098, maximum 0.499 (n = 1,320); positive-vs-positive median 0.256, maximum 0.995 (n = 462). At t = 0.30, 29.0% of positive pairs and 0.61% of background pairs exceed the threshold; the positives form families of sizes 11, 9, 1, 1. All positives are one superfamily, so the test covers sub-families, not unrelated folds. [V]

Trivial baselines on the full set [V, one-off command on 2026-11-23, not yet a committed script]: "reviewed" status alone (entry name does not start with the accession, as in `matched.py`) gives AUC 0.9754; sequence length alone gives 0.9271. These match the handoff and are the reason the full-set metadata AUC of about 0.97 is read as a curation effect. No repository script prints them; a short script should be added so they are reproducible. **[OPEN]**

## 10. What is not claimed

- That analog hardware is necessary for this problem.
- Any energy advantage. Section 8 shows from operation counts that the digital refinement alone exceeds a direct eigensolve for most front ends.
- Any statement about the real MolMAC ripple beyond the model in `mol_engine.py`: 6.9 uses that model as is. Its LTspice check was not repeated here, and the handoff's ripple figures are withdrawn.
- That the enzyme-screening method works on unrelated folds.
- Any statement about a physical chip: offsets and noise levels are placeholders; gain figures are from MolMAC/VoltMatrix-scale measurements quoted in script headers.
- The VoltMatrix noise comparison from the long document (rests on an unverified unit assumption; excluded).
- That the asynchronicity lever is the "correct" proxy: it was chosen after the positive control (post-hoc) and the Hückel model has no sigma framework, sterics, solvent, geometry or Pauli repulsion.

## 11. Threats to validity

1. Simulated hardware only; the offset and noise levels swept are placeholders (section 4).
2. Post-hoc choice of the primary catalyst lever; the pre-registered P4 (O in the top two by magnitude) failed in 18/18 settings, and the table ranks carbon sites above the carbonyl oxygen.
3. Single-superfamily enzyme data.
4. n = 60 per cell in the bits studies (s.e. up to about 0.06); large-N cells are non-monotone and N >= 48 is incomplete.
5. Noise-scaling with N is an unresolved hardware question; the tolerance trend differs by model (6.5).
6. **MolMAC ripple.** The handoff's figures (ripple 1.1-2.0e-3 of full scale; "about 10x too large from 24 atoms") are withdrawn: their script, `ripple_check.py`, was not found. The ripple model in `mol_engine.py` is deterministic, phase-locked and duty-pattern dependent, so it matches neither the offset nor the noise swept in 6.5. `ripple_budget.py` (6.9) adds that model, unchanged, to the `mc_budget.py` chip: ripple costs at most 0.3 points of success and shrinks relative to full scale with N. The measurement uses one ADC per column, a 12-bit ADC (the N = 40 baseline fails at 12 bits) and three sample phases; a 14-bit and a per-tile-ADC repeat have not been run. **[OPEN]**
7. Package versions are recorded; results from the workstation have not been compared with the laptop. **[OPEN]**
8. `polish.py` (int8 rescue by a short fp32 stage) was not run in the batch and is not reported. **[OPEN]**

## 12. Corrections log

1. Early misreading of A near zero at weak coupling: I made primary and A gated at p1 + p2 >= 0.3.
2. `refine` started from a full `eigh(P)`: replaced by a range finder.
3. R3 noise test shared one noise stream between the + and - solves: separate streams.
4. +/- shown for deterministic modes: marked `*`, effective n = 18.
5. SP2 diverged under noise in a 200-chip run: divergence guard, `safe_err`, aufbau swap, `--part` flag; a wrong-subspace lock at 8 bits fixed by the aufbau check.
6. "Analog saves 0 to +1 step" and "8-bit chain is a net cost" came from one range-finder seed (7); the 200-seed sweep reversed them. The starred rows of `analog_mc` part 1 still show the single-seed result (6.1).
7. "Worst-case ADC range costs about log2(N) bits" was overstated; the power-of-two auto-range recovered nothing.
8. fp32 refinement floor sat on the target at N >= 32: refinement moved to float64.
9. Interaction-energy lever failed the carbonyl-O positive control: asynchronicity lever made primary (post-hoc).
10. First enzyme-family clustering (3-mer cosine) merged proteins at 4-8% identity: replaced by alignment identity.
11. The original enzyme pipeline's apparent success was a curation artifact (section 9).
12. (2026-11-23) The handoff table cell "worst-case range, N = 64: not reached by 16" is not supported: `bits2.py` never runs the worst-case range above N = 40 (6.4).
13. (2026-11-23) "Noise tolerance about 1e-3 at 8 atoms, <= 1e-4 by 24 atoms" holds only with a 14-bit ADC at N = 8, and only under the "N" scaling model at N = 24; under "sqrtN" and "fixed" it is 1e-4 to 3e-4 (6.5).
14. (2026-11-23) Pre-registered predictions that failed are listed in section 7 (notably h1_acrolein P4, mc_budget P3 and P4, s2_oa T5).
15. (2026-11-23) The handoff's ripple statements ("1.1-2.0e-3 of full scale", "about 10x too large from 24 atoms") and energy figures (1.8 TOPS/W, "10-42 N^3 flops") had no source in the repository and were withdrawn. The ripple was measured directly (6.9); the energy section was rebuilt from operation counts and cited literature (section 8).

Housekeeping entries (repository and document-editing issues) are omitted.

## 13. Reproduction

Run from `catalyst/` with numpy and scipy (pandas, scikit-learn, biopython for the enzyme scripts). The enzyme scripts take the CSV path as the first argument. `analog_mc.py` and `mc_budget.py` use argparse (`--chips`, and `--part {1,2,3,all}` for the former); do not pass extra arguments to any script.

| Script | CSV | Laptop runtime |
|---|---|---|
| `h0_exact`, `h1_acrolein`, `r123_float64`, `rules` | no | under 1 s |
| `ladder` | no | about 6 s |
| `bits` / `bits2` | no | 10 s / 21 s (9 s on rerun) |
| `scale`, `s2_oa`, `no_rr` | no | 12 s, 5 s, 13 s |
| `mc_budget` | no | 109 s |
| `analog_mc` | no | 336 s |
| `robust` | no | 251 s |
| `ripple_budget` | no (imports `../mol_engine.py`) | 111 s |
| `seed_sweep` | no | about 8.5 min |
| `align_ident`, `background`, `positive_control`, `matched` | yes | 6-12 s each |

Run `align_ident.py` first: it writes `ident.json`, which `background.py` and `matched.py` read; it is not committed.

## 14. Open items

1. Add a committed script that prints the "reviewed alone" (0.9754) and "sequence length alone" (0.9271) AUCs (section 9).
2. Check the ~9 N^3 eigensolver figure against Golub and Van Loan (section 8).
3. Repeat the ripple measurement with a 14-bit ADC (where the N = 40 baseline passes) and with a per-tile ADC (6.9, 11.6).
4. Explain or re-run the N = 56 / 64 ideal-range dip and the missing worst-case N >= 48 cells with more than 60 runs per cell (6.4).
5. `s2_oa.py` delta1: check whether the cluster threshold is wired in (identical tables for 1e-3 and 1e-10).
6. Run `polish.py`; compare workstation and laptop outputs.
7. Decide on the VoltMatrix comparison, and the full-record versus judges' version of the long document.