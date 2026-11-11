# alt_check.py  --  python alt_check.py        (write netlists)
#                   python alt_check.py cmp    (compare the three .log files)
import sys, re, math

P, TCLK, NPER, SCNT = 255, 40e-9, 199, 33      # 25 MHz clk -> 10.2 us period
OFFS = [0, 64, 128, 192]
VDD, TR = 5.0, 10e-9
VW = [0.8, 0.5, 0.3, 0.9]
RF, RON, RL, CF, TILE = 10e3, 80.0, 1e6, 100e-9, 4
PAT = {"A": [51, 153, 230, 102], "B": [230, 26, 0, 178]}   # duty codes
SEQS = ["ABABABAB", "AAAAAAAA", "BBBBBBBB"]

def intervals(seq, k):
    # high iff (cnt+off) mod P < D, with D held for the whole period
    out, off = [], OFFS[k]
    for j, ch in enumerate(seq):
        d = PAT[ch][k]
        for n in range(NPER):
            b = (j * NPER + n) * P
            if d > off:
                out.append((b, b + d - off))
            if off > 0 and d > 0:
                out.append((b + P - off, b + min(P, P - off + d)))
    return out

def merged(iv):
    m = []
    for s, e in iv:
        if m and s <= m[-1][1]:
            m[-1] = (m[-1][0], max(m[-1][1], e))
        else:
            m.append((s, e))
    return m

def write_pwl(path, iv):
    with open(path, "w") as f:
        if not iv:
            f.write("0 0\n1 0\n")                 # never-high channel
        for s, e in merged(iv):
            t0, t1 = s * TCLK, e * TCLK
            f.write(f"{t0:.9e} 0\n{t0+TR:.9e} {VDD}\n{t1:.9e} {VDD}\n{t1+TR:.9e} 0\n")

def gen(seq):
    for k in range(4):
        write_pwl(f"pwm{k+1}_{seq}.txt", intervals(seq, k))
    tend = len(seq) * NPER * P * TCLK
    L = ["* auto-generated, pass sequence " + seq,
         ".param Vdd=5 Ron=80",
         ".model SWM SW(Ron={Ron} Roff=1G Vt={Vdd/2} Vh=0.1)",
         ".subckt MULT vw pwm out",
         "Bpwmb pwmb 0 V={Vdd}-V(pwm)",
         "S1 vw  out pwm  0 SWM",
         "S2 out 0   pwmb 0 SWM",
         ".ends"]
    for k in range(4):
        L += [f"Vw{k+1} vw{k+1} 0 {VW[k]}",
              f"Vp{k+1} pwm{k+1} 0 PWL file=pwm{k+1}_{seq}.txt",
              f"X{k+1} vw{k+1} pwm{k+1} s{k+1} MULT",
              f"R{k+1} s{k+1} out 10k"]
    L += ["Cf out 0 100n", "Rl out 0 1Meg",
          ".options reltol=1e-5 vntol=1e-9 abstol=1e-12 numdgt=10",
          f".tran 0 {tend:.9e} 0 20n", ".save V(out)"]
    for j in range(len(seq)):
        ts = ((j * NPER + NPER - 1) * P + SCNT) * TCLK
        L.append(f".meas TRAN m{j} FIND V(out) AT={ts:.9e}")
    L.append(".end")
    open(f"mac4_{seq}.cir", "w").write("\n".join(L) + "\n")

def read_log(seq):
    raw = open(f"mac4_{seq}.log", "rb").read()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") \
          else raw.decode("utf-8", "ignore")
    v = {}
    for line in txt.splitlines():
        m = re.match(r"\s*m(\d+):.*?=\s*([-+0-9.eE]+)", line)
        if m:
            v[int(m.group(1))] = float(m.group(2))
    return v

def cmp():
    req = (RF + RON) / TILE
    tau = (req * RL / (req + RL)) * CF
    g = RL / (RL + req)
    vbar = {c: g * sum(VW[k] * PAT[c][k] / P for k in range(4)) / TILE for c in "AB"}
    ts = ((NPER - 1) * P + SCNT) * TCLK
    seq = "ABABABAB"
    mix = read_log(seq)
    pure = {c: read_log(c * 8) for c in "AB"}
    print(f"tau_eff = {tau*1e6:.2f} us, e^(-ts/tau) = {math.exp(-ts/tau):.3e}")
    print("pass pattern  measured_uV  predicted_uV")
    for j in range(1, len(seq) - 1):          # skip the last sample (end-of-run artifact)
        meas = (mix[j] - pure[seq[j]][j]) * 1e6
        pred = (vbar[seq[j-1]] - vbar[seq[j]]) * math.exp(-ts / tau) * 1e6
        print(f"{j:4d}  {seq[j]:>7}  {meas:11.2f}  {pred:12.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cmp":
        cmp()
    else:
        for s in SEQS:
            gen(s)
        print("wrote mac4_<seq>.cir and pwm*_<seq>.txt for", SEQS)