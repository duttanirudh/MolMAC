"""
test_mol_engine.py - regression tests for mol_engine.py.
  python test_mol_engine.py            all tests
  python test_mol_engine.py --fast     skip the slow statistical tests
Also works under pytest.
"""
import sys
from functools import lru_cache
import numpy as np
from mol_engine import Cfg, AnalogArray, molecule, run, ripple_shape

H = dict(headroom=0.92)
MOLS = ("ethylene", "butadiene", "benzene", "naphthalene")
EXACT = {"ethylene": 2.0, "butadiene": 4.4721, "benzene": 8.0, "naphthalene": 13.6832}


def opts(**kw):
    return tuple(sorted(kw.items()))


@lru_cache(maxsize=None)
def sweep(name, n_seeds, o):
    res = [run(name, seed=s, **dict(o))["result"] for s in range(n_seeds)]
    exact = res[0]["E_exact"]
    ea = np.array([r["E_analog"] - exact for r in res])
    er = np.array([r["E_refined"] - exact for r in res])
    iters = np.array([[x["iters"] for x in r["orbitals"]] for r in res])
    return ea, er, iters


CONFIGS = [
    opts(),
    opts(**H, calibrate="oracle", matvec_reads=4),
    opts(**H, calibrate="measured", cal_reads=16),
]
IDEAL = opts(gain_sigma=0.0, offset_sigma=0.0, noise_sigma=0.0, adc_bits=16)


# ---------- fast tests ----------

def test_exact_energies():
    for name, e in EXACT.items():
        got = run(name)["result"]["E_exact"]
        assert abs(got - e) < 5e-4, f"{name}: {got} vs {e}"


def test_deterministic_for_fixed_seed():
    kw = dict(H, calibrate="measured", cal_reads=4)
    a = run("benzene", seed=3, **kw)["result"]["E_analog"]
    b = run("benzene", seed=3, **kw)["result"]["E_analog"]
    assert a == b


def test_matvec_matches_digital_on_ideal_chip():
    """Checks the wunit / g_load / fs scaling: y should equal (A + sI) v."""
    cfg = Cfg(gain_sigma=0, offset_sigma=0, noise_sigma=0, adc_bits=16)
    rng = np.random.default_rng(0)
    for name in ("benzene", "naphthalene"):
        A, _, _ = molecule(name)
        n, s = len(A), int(A.sum(1).max())
        arr = AnalogArray(A, cfg, sd=s)
        for _ in range(20):
            v = rng.standard_normal(n)
            y = arr.matvec(v)[0]
            err = np.abs(y - (A + s * np.eye(n)) @ v).max() / np.abs(v).max()
            assert err < 0.01, f"{name}: relative error {err:.4f}"


def test_refined_energy_never_above_exact():
    """Ky Fan: orthonormal vectors cannot beat the true occupied subspace."""
    for name in MOLS:
        for o in CONFIGS:
            _, er, _ = sweep(name, 5, o)
            assert er.max() < 1e-9, f"{name} {dict(o)}: max E_ref err {er.max():.2e}"


def test_no_capped_orbitals():
    cap = Cfg().max_iter
    for name in ("benzene", "naphthalene"):
        for o in CONFIGS:
            _, _, it = sweep(name, 5, o)
            assert it.max() < cap, f"{name} {dict(o)}: hit {cap} iterations"


def test_ideal_chip_recovers_exact_energy():
    # tolerance 0.01 is my guess, not a measured value
    for name in MOLS:
        ea, er, _ = sweep(name, 3, IDEAL)
        assert np.abs(ea).max() < 0.01, f"{name}: E_an err {ea}"
        assert er.min() > -0.01, f"{name}: E_ref err {er}"


def _gain_rms(mode, reads=16, seeds=range(10)):
    A, _, _ = molecule("benzene")
    out = []
    for s in seeds:
        arr = AnalogArray(A, Cfg(seed=s, **H), sd=2)
        if mode != "none":
            assert arr.calibrate(mode, reads)["clipped"] == 0
        out.append(arr.gain_rms())
    return float(np.mean(out))


def test_gain_rms_bands():
    assert 0.013 < _gain_rms("none") < 0.021            # ~ gain_sigma 1.7%
    assert 0.0009 < _gain_rms("oracle") < 0.0016        # DAC step floor ~0.12%
    assert 0.0015 < _gain_rms("measured") < 0.0035      # ~0.24% at 16 reads


def test_calibration_bookkeeping():
    m = run("benzene", seed=0, calibrate="measured", cal_reads=16, **H)["meta"]
    assert m["cal"]["passes"] == (6 + 1) * 16
    assert abs(m["pass_us"] - 2026.0) < 1.0


def test_analog_time_scales_with_reads():
    for k in (1, 4):
        tr = run("benzene", seed=0, matvec_reads=k)
        n_iter = sum(1 for e in tr["events"] if e["t"] == "iter")
        t_end = [e for e in tr["events"] if e["t"] == "done"][-1]["t_us"]
        expected = 2 * tr["meta"]["pass_us"] * k
        assert abs(t_end / n_iter - expected) < 1e-6 * expected, \
            f"reads={k}: {t_end / n_iter:.1f} us/iter, expected {expected:.1f}"


# ---------- tests for the LTspice-derived options (all default to off) ----------

# E_analog / E_refined of the ORIGINAL engine (before settle_mem, read_order and ripple existed).
PINS = [
    ("benzene", dict(seed=3, calibrate="measured", cal_reads=4, **H), 7.956349400822619, 7.999571984785386),
    ("naphthalene", dict(seed=0), 13.600623209267336, 13.681058221089163),
    ("naphthalene", dict(seed=1, calibrate="oracle", matvec_reads=4, **H), 13.677889146854591, 13.683105675299782),
]


def test_defaults_reproduce_original_outputs():
    """With settle_mem, ripple off and read_order 'alt' the engine must be exactly the old one."""
    for name, kw, ea, er in PINS:
        r = run(name, **kw)["result"]
        assert abs(r["E_analog"] - ea) < 1e-9 and abs(r["E_refined"] - er) < 1e-9, (name, r["E_analog"], r["E_refined"])


def _e(name, seeds, **kw):
    return np.array([run(name, seed=s, **kw)["result"]["E_analog"] for s in seeds])


def test_read_order_validated():
    try:
        Cfg(read_order="sideways")
    except ValueError:
        return
    raise AssertionError("bad read_order was accepted")


def test_grouped_order_equals_alt_for_one_read():
    ideal = dict(IDEAL)
    for kw in (dict(), dict(settle_mem=True)):
        a = _e("benzene", range(3), read_order="alt", **ideal, **kw)
        b = _e("benzene", range(3), read_order="grouped", **ideal, **kw)
        assert np.array_equal(a, b)


def test_settle_mem_bias_ideal_chip():
    """The exp(-8) residual of the previous pass scales y by ~(1 - 2e^-8): about -0.0092 beta on naphthalene
    with alternating reads, for any K. Grouped reads cut it by ~K, a 10 tau wait by ~7x.
    Measured on 6 ideal chips: -0.00919 / -0.00938 / -0.00197 (predicted -0.00918 / -0.00918 / -0.0023)."""
    ideal, seeds = dict(IDEAL), range(6)

    def d(**kw):
        on = _e("naphthalene", seeds, settle_mem=True, **ideal, **kw)
        off = _e("naphthalene", seeds, **ideal, **kw)
        return float((on - off).mean())

    assert -0.0100 < d(matvec_reads=1) < -0.0084
    assert -0.0102 < d(matvec_reads=4) < -0.0086
    assert -0.0030 < d(matvec_reads=4, read_order="grouped") < -0.0010
    assert abs(d(matvec_reads=1, settle_taus=10.0)) < 0.25 * abs(d(matvec_reads=1))


def test_ripple_shape_matches_ltspice():
    """mac4.cir: Vw .8 .5 .3 .9, D .2 .6 .9 .4, window starts 0 .25 .5 .75 of Tp = 10 us, Rf+Ron = 10080, Cf = 100 nF.
    LTspice vpp = 2.127 mV; an exact solve of the circuit gives -0.198 mV at the start of the period."""
    Vw, D, S = np.array([.8, .5, .3, .9]), np.array([.2, .6, .9, .4]), np.array([0, .25, .5, .75])
    f = 10e-6 / (10080 * 100e-9)
    th = np.linspace(0, 1, 2001)
    r = f * sum(Vw[i] * ripple_shape(th, S[i], D[i]) for i in range(4))
    assert abs((r.max() - r.min()) - 2.127e-3) < 0.03e-3
    assert abs(r[0] - (-0.198e-3)) < 0.01e-3
    assert abs(r.mean()) < 1e-6                                   # zero mean by construction


def test_ripple_shape_zero_for_duty_0_and_1():
    th = np.linspace(0, 1, 11)
    for s in (0.0, 0.25, 0.75):
        assert np.abs(ripple_shape(th, s, 0.0)).max() < 1e-12
        assert np.abs(ripple_shape(th, s, 1.0)).max() < 1e-12


def test_ripple_bias_ideal_chip():
    """Sampling at period_start reads the node below its average, so E_an drops by about 0.0067 beta on
    naphthalene (6 ideal chips: -0.00671, range -0.0099 .. -0.0040). On the operating chip the bias moves
    between -0.0068 and +0.0042 with the sample phase (100 chips, oracle, K=4)."""
    ideal, seeds = dict(IDEAL), range(6)
    d = (_e("naphthalene", seeds, ripple=True, **ideal) - _e("naphthalene", seeds, **ideal)).mean()
    assert -0.0085 < d < -0.0050, d


# ---------- slow statistical test (naphthalene, same seeds as naph_test.py) ----------

def test_slow_naphthalene_bands():
    nocal = sweep("naphthalene", 30, opts())
    orc = sweep("naphthalene", 30, opts(**H, calibrate="oracle"))
    orc4 = sweep("naphthalene", 30, opts(**H, calibrate="oracle", matvec_reads=4))
    s0, s1, s4 = nocal[0].std(), orc[0].std(), orc4[0].std()
    assert 0.030 < s0 < 0.060, s0        # measured 0.0445
    assert 0.010 < s1 < 0.022, s1        # measured 0.0158
    assert 0.007 < s4 < 0.015, s4        # measured 0.0108
    assert s0 > 1.8 * s1                 # calibration helps (measured 2.8x)
    assert nocal[1].std() < 0.002 and orc[1].std() < 0.0005
    for _, _, it in (nocal, orc, orc4):
        assert 22 < it.mean() < 45       # measured 30 to 33


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    failed = 0
    for name, fn in sorted(list(globals().items()), key=lambda kv: kv[0]):
        if not name.startswith("test_") or (fast and name.startswith("test_slow_")):
            continue
        try:
            fn()
            print("PASS", name)
        except Exception as e:
            failed += 1
            print("FAIL", name, "-", e)
    sys.exit(1 if failed else 0)