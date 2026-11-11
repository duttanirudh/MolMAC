# MolMAC

A hybrid analog-digital multiply-accumulate (MAC) array, simulated end to end, that finds the π-electron energies of small molecules under the Hückel model. The analog array does the matrix-vector products. Digital logic handles power iteration, deflation and refinement.

Layers: a behavioral **Python** model, **LTspice** checks of settling and ripple, **Verilog** control logic verified against the Python trace, and a browser **dashboard** that replays a run.

> **Limits:** a classical simulation of a toy Hamiltonian, not quantum hardware and not chemical accuracy. Offset and noise magnitudes are placeholders; only the 1.7% gain mismatch comes from the paper. The Python model treats all rows as parallel analog nodes, while the Verilog FSM drives one tile at a time.

## Requirements
- Python 3 with `numpy`
- Vivado (XSim) for the Verilog tests, optional
- LTspice for the SPICE checks, optional

## Quick start
```powershell
python mol_engine.py benzene        # runs the model, writes dashboard.html
```
Molecules: `ethylene`, `butadiene`, `benzene`, `naphthalene`. Open `dashboard.html` in a browser. It has three tabs: the simulation replay, a guide with the run summary, and a prototype build plan. There is a dark/light toggle.

## Tests
```powershell
python test_mol_engine.py           # 17 tests; add --fast to skip the slow one
```

## Verilog (Vivado XSim)
Run from a `sim\` folder, which is git-ignored, so create it first:
```powershell
mkdir sim; cd sim
xvlog ..\rtl\mac_ctrl.v ..\tb\tb_mac_ctrl.v
xelab tb_mac_ctrl -s tb_mac_ctrl_snap
xsim tb_mac_ctrl_snap -runall
```
Use the same pattern for `tb_mac_fsm`, and for `tb_cosim` (list both `mac_ctrl.v` and `mac_fsm.v`). `-runall` is required.

## Co-simulation
Export the Python trace, then run `tb_cosim` from `sim\`:
```powershell
python export_cosim.py benzene --ideal    # writes cosim\*.hex
```
Expected: `checked 1152 channel sums (expected 1152)` then PASS. For naphthalene, export with `--calibrate oracle --iters 8 --outdir cosim_naph` and adjust the hard-coded sizes and paths in `tb_cosim.v`.

## SPICE
`spice\alt_check.py` generates the pass-to-pass settling-memory netlists. Run the `.cir` files in LTspice, then compare the logs with `python alt_check.py cmp`.

## Layout
| Folder | Contents |
|---|---|
| `rtl/` | `mac_ctrl.v` (PWM, shift register), `mac_fsm.v` (controller FSM) |
| `tb/` | testbenches |
| `spice/` | netlists and generators |
| `cosim/`, `cosim_naph/` | exported hex files |
| `experiments/` | error-budget scripts |
| root | `mol_engine.py`, tests, `export_cosim.py`, dashboard template |