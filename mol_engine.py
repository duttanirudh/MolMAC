"""
mol_engine.py - Hückel pi-electron energies computed on the hybrid analog MAC model.
Runs the real computation, records every step, writes dashboard.html.
Usage:  python mol_engine.py [ethylene|butadiene|benzene|naphthalene]
"""
import json, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

PWM_PERIOD = 255                  # clocks per PWM period (matches mac_ctrl.v)
PWM_OFFS = (0, 64, 128, 192)      # channel phase offsets in clocks (matches mac_ctrl.v)


@dataclass
class Cfg:
    w_bits: int = 8
    a_bits: int = 8
    adc_bits: int = 10
    tile: int = 4                 # multipliers per summing node (matches mac4.cir / mac_ctrl.v)
    gain_sigma: float = 0.017     # PLACEHOLDER static gain mismatch
    offset_sigma: float = 0.002   # PLACEHOLDER static offset per multiplier (mac4.cir has ideal switches: no source)
    noise_sigma: float = 0.003    # PLACEHOLDER thermal noise per multiplier per read (kT/C of mac4.cir is ~1e-4 of this)
    rf: float = 10e3              # from SPICE netlist
    ron: float = 80.0             # from SPICE netlist
    rl: float = 1e6               # from SPICE netlist
    cf: float = 100e-9            # from SPICE netlist
    settle_taus: float = 8.0      # wait 8 tau before sampling: residual 3.4e-4 of a step (confirmed in LTspice)
    pwm_period_us: float = 10.0
    tol: float = 0.005            # 0.005 leaves a ~-0.005 beta Rayleigh-lag bias in E_an (naphthalene);
                                  # 0.002 removes it for ~+40% iterations. 8-bit duty quantization
                                  # sets a floor near 0.001; 0.001 also hit max_iter with noise on.
    max_iter: int = 150
    seed: int = 1                 # fixes the simulated chip and the noise
    digital_shift: bool = True    # keep s*I out of the analog array
    calibrate: str = "none"       # "none" | "measured" | "oracle" (oracle = perfect knowledge, a bound)
    cal_reads: int = 16           # reads averaged per calibration measurement
    headroom: float = 1.0         # nominal weights use this fraction of DAC full scale
    matvec_reads: int = 1         # reads averaged per analog pass during operation
    # --- non-idealities derived from the LTspice runs (all OFF by default: the regression baseline is unchanged) ---
    settle_mem: bool = False      # every pass starts from where the last pass left the node, so it keeps a residual
                                  # exp(-settle_taus) of the step. With the "alt" read order this scales y by about
                                  # (1 - 2*exp(-settle_taus)) = -0.07%, i.e. about -0.009 beta on naphthalene (K=1 or 4).
    read_order: str = "alt"       # "alt": + - + - ... (original) | "grouped": all + reads, then all - reads
                                  # (grouped cuts the settle_mem bias by K for free)
    ripple: bool = False          # deterministic PWM ripple on the node at the ADC sample instant (phase-locked,
                                  # so repeated reads do NOT average it out). First-order model checked against LTspice.
    ripple_phase: float = 0.0     # sample instant as a fraction of the PWM period (0 = period_start)

    def __post_init__(self):
        if self.read_order not in ("alt", "grouped"):
            raise ValueError(f"read_order must be 'alt' or 'grouped', got {self.read_order!r}")
        if self.ripple and self.tile != len(PWM_OFFS):
            raise ValueError("ripple model assumes the 4-channel PWM of mac_ctrl.v (tile = 4)")


def ripple_shape(theta, s, d):
    """Zero-mean PWM ripple shape of one branch (first order in Tp/RC), in units of (amplitude * Tp/RC).

    The branch is high for a fraction d of the period starting at fraction s (wrapping around the end).
    Returns  g(theta) - mean(g),  with  g(theta) = integral from 0 to theta of (high(t) - d) dt,
    theta being the sample instant as a fraction of the period. Arrays broadcast; d = 0 or 1 gives 0.
    """
    L = np.minimum(d, 1.0 - s)                 # high time before the end of the period
    W = np.maximum(0.0, s + d - 1.0)           # high time wrapped to the start of the period
    g = np.clip(theta - s, 0.0, L) + np.clip(theta, 0.0, W) - d * theta
    gbar = (L * (1.0 - s) - L ** 2 / 2) + (W - W ** 2 / 2) - d / 2
    return g - gbar


def molecule(name):
    if name == "ethylene":
        edges, xy = [(0, 1)], [(0, 0), (1, 0)]
    elif name == "butadiene":
        edges, xy = [(0, 1), (1, 2), (2, 3)], [(0, 0), (1, .5), (2, 0), (3, .5)]
    elif name == "benzene":
        edges = [(i, (i + 1) % 6) for i in range(6)]
        xy = [(float(np.cos(np.pi * i / 3)), float(np.sin(np.pi * i / 3))) for i in range(6)]
    elif name == "naphthalene":      # 10-atom outer ring plus one shared bond (0,5)
        edges = [(i, (i + 1) % 10) for i in range(10)] + [(0, 5)]
        xy = [(0, .5), (.87, 1), (1.73, .5), (1.73, -.5), (.87, -1),
              (0, -.5), (-.87, -1), (-1.73, -.5), (-1.73, .5), (-.87, 1)]
    else:
        raise SystemExit("molecule must be ethylene, butadiene, benzene or naphthalene")
    n = len(xy)
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A, edges, xy


class AnalogArray:
    """n x n array of multipliers, grouped into summing nodes of `tile` inputs."""
    def __init__(self, M, cfg, sd=0.0):
        self.cfg, self.sd = cfg, sd
        n = M.shape[0]
        self.n = n
        self.nt = -(-n // cfg.tile)
        self.npad = self.nt * cfg.tile
        self.wscale = float(M.max())
        self.wl = 2 ** cfg.w_bits - 1
        self.wl_eff = int(round(cfg.headroom * self.wl))       # code that represents wscale
        self.wunit = self.wscale * self.wl / self.wl_eff       # matrix value of a full-scale weight
        codes = np.zeros((n, self.npad), dtype=int)
        codes[:, :n] = np.rint(M / self.wscale * self.wl_eff).astype(int)
        self.nominal = codes.copy()                            # codes before any calibration
        self._load(codes)

        chip = np.random.default_rng(cfg.seed)                 # one fixed simulated chip
        self.gain = 1 + cfg.gain_sigma * chip.standard_normal((n, self.npad))
        self.offset = cfg.offset_sigma * np.sqrt(cfg.tile) * chip.standard_normal((n, self.nt))
        self.noise_rng = np.random.default_rng(cfg.seed + 1)

        self.g_load = cfg.rl / (cfg.rl + (cfg.rf + cfg.ron) / cfg.tile)   # from SPICE (confirmed in LTspice)
        tile_max = (self.nominal / self.wl).reshape(n, self.nt, cfg.tile).sum(-1).max()
        self.fs = float(1.05 * tile_max)                       # ADC range from the known nominal weights
        self.top = 2 ** (cfg.adc_bits - 1) - 1
        self.cal_passes = 0

        self.eps = float(np.exp(-cfg.settle_taus))             # settle_mem: residual of the previous pass
        self._prev = None                                      # node values at the end of the previous pass
        if cfg.ripple:
            # channel k is high while (cnt + off_k) mod 255 < D, so its window starts at cnt = (255 - off_k) mod 255
            starts = np.array([((PWM_PERIOD - o) % PWM_PERIOD) / PWM_PERIOD for o in PWM_OFFS])
            self._rip_start = np.tile(starts, self.nt)[None, :]                # 1 x npad
            # node is a SUM of tile branches here (SPICE averages them), hence the factor tile
            self._rip_k = cfg.tile * (cfg.pwm_period_us * 1e-6) / ((cfg.rf + cfg.ron) * cfg.cf)

    def _load(self, codes):
        """Load DAC codes (what the shift registers hold)."""
        c = self.cfg
        self.wcode_pad = codes
        self.wq_pad = codes / self.wl
        self.wcode = codes[:, :self.n]
        self.words = []
        for r in range(self.n):
            row = []
            for t in range(self.nt):
                cc = codes[r, t * c.tile:(t + 1) * c.tile]
                row.append(f"{sum(int(cc[i]) << (8 * i) for i in range(c.tile)):08x}")
            self.words.append(row)

    def _ripple(self, d):
        """Deterministic PWM ripple at the sample instant, in node units. d: duty fractions, length npad."""
        c = self.cfg
        a = self.wq_pad * self.gain                                            # branch amplitude (weight x gain)
        r = a * ripple_shape(c.ripple_phase, self._rip_start, d[None, :])
        return self._rip_k * r.reshape(self.n, self.nt, c.tile).sum(-1)

    def pass_(self, duty_codes):
        c, n = self.cfg, self.n
        d = np.zeros(self.npad)
        d[:n] = duty_codes / (2 ** c.a_bits - 1)
        prod = self.wq_pad * self.gain * d[None, :]                     # switch: Vw x duty
        node = prod.reshape(n, self.nt, c.tile).sum(-1) * self.g_load + self.offset
        if c.settle_mem:                                                # RC memory of the previous pass
            prev = np.zeros_like(node) if self._prev is None else self._prev
            node = node + self.eps * (prev - node)
            self._prev = node
        if c.ripple:
            node = node + self._ripple(d)
        node = node + self.noise_rng.standard_normal(node.shape) * c.noise_sigma * np.sqrt(c.tile)
        code = np.rint(np.clip(node / self.fs, -1, 1) * self.top).astype(int)   # ADC
        adc = code / self.top * self.fs
        return dict(prod=prod[:, :n], node=node, code=code, rowsum=adc.sum(1))

    def matvec(self, v):
        """y = M v using two non-negative analog passes (v+ and v-)."""
        A = 2 ** self.cfg.a_bits - 1
        vmax = float(np.abs(v).max())
        u = v / vmax
        dp = np.rint(np.clip(u, 0, 1) * A).astype(int)
        dn = np.rint(np.clip(-u, 0, 1) * A).astype(int)
        k = self.cfg.matvec_reads
        if self.cfg.read_order == "grouped":                    # all + reads, then all - reads
            p = self.pass_(dp)
            rp = p["rowsum"].copy()
            for _ in range(k - 1):
                rp += self.pass_(dp)["rowsum"]
            m = self.pass_(dn)
            rm = m["rowsum"].copy()
            for _ in range(k - 1):
                rm += self.pass_(dn)["rowsum"]
        else:                                                   # alternating + - + - ...
            p, m = self.pass_(dp), self.pass_(dn)
            rp, rm = p["rowsum"].copy(), m["rowsum"].copy()
            for _ in range(k - 1):
                rp += self.pass_(dp)["rowsum"]
                rm += self.pass_(dn)["rowsum"]
        rp /= k; rm /= k
        y = (rp - rm) / self.g_load * self.wunit * vmax + self.sd * v
        return y, p, m, dp, dn, vmax

    def calibrate(self, mode, reads):
        """Estimate every cell's gain with known inputs, then reload compensated DAC codes."""
        n, tile, nom = self.n, self.cfg.tile, self.nominal
        if mode == "oracle":
            ghat = self.gain[:, :n].copy()
            self.cal_passes = 0
        else:
            A = 2 ** self.cfg.a_bits - 1

            def read(duty):                                    # mean ADC voltage per node over `reads` reads
                acc = np.zeros((n, self.nt))
                for _ in range(reads):
                    acc += self.pass_(duty)["code"] / self.top * self.fs
                return acc / reads

            base = read(np.zeros(n, dtype=int))                # offsets (and noise) with all duties 0
            ghat = np.ones((n, n))
            for c in range(n):                                 # one column at full duty
                duty = np.zeros(n, dtype=int)
                duty[c] = A
                node = read(duty)[:, c // tile]
                wq = nom[:, c] / self.wl
                ok = wq > 0
                ghat[ok, c] = (node[ok] - base[ok, c // tile]) / (wq[ok] * self.g_load)
            ghat = np.clip(ghat, 0.5, 2.0)
            self.cal_passes = (n + 1) * reads
        need = nom[:, :n] / ghat                               # code that would cancel the gain
        new = nom.copy()
        new[:, :n] = np.clip(np.rint(need), 0, self.wl).astype(int)
        self._load(new)
        return dict(mode=mode, reads=reads, passes=self.cal_passes,
                    clipped=int((need > self.wl + 0.5).sum()))

    def gain_rms(self):
        """Residual rms gain error over active cells (needs the simulated truth)."""
        n = self.n
        act = self.nominal[:, :n] > 0
        eff = self.gain[:, :n] * self.wcode_pad[:, :n] / np.where(act, self.nominal[:, :n], 1)
        return float(np.sqrt(np.mean((eff[act] - 1) ** 2)))


def pack(p, duty, sign):
    return dict(sign=sign, duty=duty.tolist(), prod=np.round(p["prod"], 4).tolist(),
                node=np.round(p["node"], 4).tolist(), code=p["code"].tolist(),
                rowsum=np.round(p["rowsum"], 4).tolist())


def run(name, **over):
    cfg = Cfg(**over)
    A, edges, xy = molecule(name)
    n = len(A)
    s = int(A.sum(1).max())
    M = A + s * np.eye(n)
    arr = AnalogArray(A, cfg, sd=s) if cfg.digital_shift else AnalogArray(M, cfg)
    tau_us = (cfg.rf + cfg.ron) / cfg.tile * cfg.cf * 1e6
    pass_us = cfg.settle_taus * tau_us + cfg.pwm_period_us
    cal = None
    if cfg.calibrate != "none":
        cal = arr.calibrate(cfg.calibrate, cfg.cal_reads)
        cal["time_ms"] = cal["passes"] * pass_us / 1000
    exact = np.sort(np.linalg.eigvalsh(A))[::-1]
    n_occ = n // 2
    rng = np.random.default_rng(cfg.seed + 2)

    ev = [{"t": "load", "row": r, "words": arr.words[r], "cycles": 32} for r in range(n)]
    found, orbs = [], []
    t_us = arr.cal_passes * pass_us
    for k in range(n_occ):
        v = rng.standard_normal(n)
        for u in found:
            v -= (v @ u) * u
        v /= np.linalg.norm(v)
        lams = []
        for it in range(cfg.max_iter):
            y, p, m, dp, dn, vmax = arr.matvec(v)
            lam_x = float(v @ y) - s
            lams.append(lam_x)
            yo = y.copy()
            for u in found:
                yo -= (yo @ u) * u
            vn = yo / np.linalg.norm(yo)
            change = float(np.linalg.norm(vn - v))
            t_us += 2 * pass_us * cfg.matvec_reads
            ev.append(dict(t="iter", orb=k, it=it, v=np.round(v, 4).tolist(), vmax=vmax,
                           passes=[pack(p, dp, "+"), pack(m, dn, "-")],
                           y=np.round(y, 4).tolist(), vn=np.round(vn, 4).tolist(),
                           change=change, lam_x=lam_x, t_us=t_us))
            v = vn
            if change < cfg.tol and it >= 7:
                break
        found.append(v)
        x_an = float(np.mean(lams[-3:]))
        x_ref = float(v @ M @ v) - s
        orbs.append(dict(k=k, x_exact=float(exact[k]), x_analog=x_an, x_refined=x_ref, iters=it + 1))
        ev.append(dict(t="done", orb=k, x_analog=x_an, x_refined=x_ref, x_exact=float(exact[k]),
                       iters=it + 1, vec=np.round(v, 4).tolist(), t_us=t_us))

    result = dict(orbitals=orbs,
                  E_exact=float(2 * exact[:n_occ].sum()),
                  E_analog=float(2 * sum(o["x_analog"] for o in orbs)),
                  E_refined=float(2 * sum(o["x_refined"] for o in orbs)))
    meta = dict(name=name, n=n, tile=cfg.tile, nt=arr.nt, s=s, wscale=arr.wscale, fs=arr.fs,
                top=arr.top, wl=arr.wl, dmax=2 ** cfg.a_bits - 1, wcode=arr.wcode.tolist(),
                edges=[list(e) for e in edges], coords=[list(c) for c in xy],
                exact=exact.tolist(), cfg=asdict(cfg), tau_us=tau_us, pass_us=pass_us,
                gain_rms=arr.gain_rms(), cal=cal)
    return dict(meta=meta, events=ev, result=result)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "benzene"
    trace = run(name)
    m, r = trace["meta"], trace["result"]
    print(f"{name}: {m['n']} atoms, tiles/row={m['nt']}, shift s={m['s']}, ADC full scale={m['fs']:.3f}")
    print(f"{'orb':>3} {'exact x':>9} {'analog x':>9} {'refined x':>10} {'iters':>6}")
    for o in r["orbitals"]:
        print(f"{o['k']:>3} {o['x_exact']:>9.4f} {o['x_analog']:>9.4f} {o['x_refined']:>10.4f} {o['iters']:>6}")
    print(f"Total pi energy (units of beta, beyond n*alpha):  exact {r['E_exact']:.4f} | "
          f"analog {r['E_analog']:.4f} | refined {r['E_refined']:.4f}")
    if m["cal"]:
        print(f"Calibration: {m['cal']['mode']}, {m['cal']['reads']} reads, {m['cal']['passes']} passes "
              f"({m['cal'].get('time_ms', 0):.0f} ms), residual gain rms {m['gain_rms']*100:.2f}%")
    last = [e for e in trace["events"] if e["t"] == "done"][-1]
    print(f"Estimated analog hardware time: {last['t_us'] / 1000:.1f} ms")
    tpl = Path(__file__).with_name("dashboard_template.html").read_text(encoding="utf-8")
    out = Path(__file__).with_name("dashboard.html")
    out.write_text(tpl.replace("__TRACE__", json.dumps(trace)), encoding="utf-8")
    print("Wrote", out)