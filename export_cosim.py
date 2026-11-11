"""
export_cosim.py - dump a mol_engine run as hex files for the Verilog co-simulation.

Usage:
  python export_cosim.py [molecule] [--iters N] [--ideal] [--calibrate oracle|measured]
                         [--seed S] [--outdir cosim]

Writes (all hex, one entry per line, readable with $readmemh):
  weights.hex   NROWS*NTILES lines, 32 bit, row-major (row, then tile).
                Packing {w3,w2,w1,w0}: array column 4*t+i sits at bits [8i+7:8i].
  duty.hex      NPASS*NTILES lines, 32 bit, same packing, one word per tile per pass.
                Passes come in (+, -) pairs, N evenly spaced iterations of the run.
  expect.hex    NPASS*NROWS*NTILES lines, 64 bit, order (pass, row, tile).
                Four 16 bit fields, channel i at bits [16i+15:16i], value = code * duty.
                Over one PWM period (255 clocks) the accumulated sum of
                (weight code while the channel output is high) must equal this.
  cosim_params.vh   `define constants for the testbench.

Self-checks done here, in Python, before anything is written:
  - words decode back to the engine's wcode matrix (and padded columns are 0)
  - on an --ideal chip, code*duty/255^2 equals the trace's analog product ("prod")
"""
import argparse
from pathlib import Path
import numpy as np
from mol_engine import run

TILE = 4
DMAX = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("molecule", nargs="?", default="benzene")
    ap.add_argument("--iters", type=int, default=12, help="iterations to sample (2 passes each)")
    ap.add_argument("--outdir", default="cosim")
    ap.add_argument("--ideal", action="store_true", help="ideal chip (also enables the prod check)")
    ap.add_argument("--calibrate", default="none", choices=["none", "oracle", "measured"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    over = dict(seed=a.seed, calibrate=a.calibrate)
    if a.calibrate != "none":
        over["headroom"] = 0.92
    if a.ideal:
        over.update(gain_sigma=0.0, offset_sigma=0.0, noise_sigma=0.0, adc_bits=16)
    tr = run(a.molecule, **over)
    meta, events = tr["meta"], tr["events"]
    n, nt = meta["n"], meta["nt"]
    assert meta["tile"] == TILE
    npad = nt * TILE

    # ---- weights ----
    loads = [e for e in events if e["t"] == "load"]
    assert len(loads) == n
    codes = np.zeros((n, npad), dtype=np.int64)
    for r, e in enumerate(loads):
        assert e["row"] == r and e["cycles"] == 32
        for t, w in enumerate(e["words"]):
            word = int(w, 16)
            for i in range(TILE):
                codes[r, TILE * t + i] = (word >> (8 * i)) & 0xFF
    assert np.array_equal(codes[:, :n], np.array(meta["wcode"])), "words do not decode to wcode"
    assert not codes[:, n:].any(), "padded columns must be zero"

    # ---- passes ----
    its = [e for e in events if e["t"] == "iter"]
    k = min(a.iters, len(its))
    pick = np.unique(np.round(np.linspace(0, len(its) - 1, k)).astype(int))
    duties, labels = [], []
    for ix in pick:
        for s, p in enumerate(its[ix]["passes"]):
            d = np.zeros(npad, dtype=np.int64)
            d[:n] = p["duty"]
            assert d.max() <= DMAX and d.min() >= 0
            duties.append((d, np.array(p["prod"])))
            labels.append((int(ix), p["sign"]))
    npass = len(duties)

    # ---- expected sums ----
    exp = np.zeros((npass, n, nt, TILE), dtype=np.int64)
    worst = 0.0
    for q, (d, prod) in enumerate(duties):
        full = codes * d[None, :]                                   # n x npad, <= 65025
        exp[q] = full.reshape(n, nt, TILE)
        if a.ideal:
            err = np.abs(full[:, :n] / DMAX ** 2 - prod).max()
            worst = max(worst, err)
            assert err < 6e-5, f"pass {q}: expected sums disagree with trace prod by {err:.2e}"

    # ---- write ----
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "weights.hex", "w") as f:
        for r in range(n):
            for t in range(nt):
                f.write(loads[r]["words"][t] + "\n")
    with open(out / "duty.hex", "w") as f:
        for d, _ in duties:
            for t in range(nt):
                f.write(f"{sum(int(d[TILE * t + i]) << (8 * i) for i in range(TILE)):08x}\n")
    with open(out / "expect.hex", "w") as f:
        for q in range(npass):
            for r in range(n):
                for t in range(nt):
                    f.write(f"{sum(int(exp[q, r, t, i]) << (16 * i) for i in range(TILE)):016x}\n")
    (out / "cosim_params.vh").write_text(
        f"`define COSIM_NROWS  {n}\n`define COSIM_NTILES {nt}\n`define COSIM_NPASS  {npass}\n")
    # same three numbers as hex lines, so the testbench can read them at run time ($readmemh)
    (out / "params.hex").write_text(f"{n:08x}\n{nt:08x}\n{npass:08x}\n")

    cyc = npass * n * nt * (32 + DMAX + 4)
    print(f"{a.molecule}: n={n}, tiles/row={nt}, passes={npass} "
          f"(iterations {pick.tolist()}), calibrate={a.calibrate}, ideal={a.ideal}")
    print(f"decode check OK; max |code*duty/255^2 - prod| = {worst:.2e}" if a.ideal
          else "decode check OK (prod check needs --ideal)")
    print(f"wrote {out}/ weights.hex={n * nt} duty.hex={npass * nt} expect.hex={npass * n * nt} lines")
    print(f"~{cyc:,} clock cycles in the testbench (load 32 + PWM period 255 + 4 slack per row-tile)")
    print("duty-code range per pass (min, max):",
          [(int(d.min()), int(d.max())) for d, _ in duties][:6], "...")


if __name__ == "__main__":
    main()