"""
h1_acrolein.py  --  H1 test: butadiene + acrolein, catalyst as a perturbation on the carbonyl O.

Units: alpha_C = 0, |beta| = 1, beta < 0, so H = -(weighted adjacency) and alpha_X = alpha + h_X*beta -> H_XX = -h_X.
Atoms: 0..3 butadiene C1..C4 | 4 Cb (terminal CH2) | 5 Ca | 6 C(carbonyl) | 7 O
Bonds: diene 1.0; Cb=Ca k1; Ca-C k2; C=O k3.  Forming: C1-Cb (0,4), C4-Ca (3,5), both gamma.
8 pi electrons -> 4 occupied orbitals.
Catalyst stand-in: H[7,7] -= delta  (O made more electronegative; H-bond / Lewis acid on O).

Primary observable: imbalance I = p(C1-Cb) - p(C4-Ca).
A = |I|/(p1+p2) is reported only when p1+p2 >= 0.3 (it amplifies absolute error by 1/(p1+p2)).

PREDICTIONS (written before the first run):
  P1  I > 0 at delta = 0 for every parameter set and gamma (Cb bond leads; large LUMO coefficient on Cb).
  P2  dI/d(delta) > 0 for every parameter set and gamma  (H1: sign agreement with the DFT trend only).
  P3  charge transfer diene -> dienophile is > 0 and rises with delta.
  P4  (weaker) in the per-atom sensitivity map dI/d(lowering alpha_k), O ranks in the top two by magnitude.
  Falsifier of H1: P2 fails, or its sign flips between parameter sets.
"""
import itertools
import numpy as np

N_OCC = 4
FORM = [(0, 4), (3, 5)]
DIENE, DIENOPHILE = [0, 1, 2, 3], [4, 5, 6, 7]
NAMES = ["C1", "C2", "C3", "C4", "Cb", "Ca", "Cc", "O"]


def hamiltonian(gamma, hO, k1, k2, k3, delta=0.0):
    H = np.zeros((8, 8))
    for (i, j), k in [((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0),
                      ((4, 5), k1), ((5, 6), k2), ((6, 7), k3)]:
        H[i, j] = H[j, i] = -k
    for i, j in FORM:
        H[i, j] = H[j, i] = -gamma
    H[7, 7] = -hO - delta
    return H


def density(H):
    e, C = np.linalg.eigh(H)
    Co = C[:, :N_OCC]
    return 2.0 * Co @ Co.T, 2.0 * e[:N_OCC].sum(), e[N_OCC] - e[N_OCC - 1]


def obs(H):
    D, E, gap = density(H)
    p1, p2 = D[FORM[0]], D[FORM[1]]
    I = p1 - p2
    A = abs(I) / (p1 + p2) if (p1 + p2) >= 0.3 else float("nan")
    ct = 4.0 - np.trace(D[np.ix_(DIENE, DIENE)])        # electrons moved diene -> dienophile
    return dict(D=D, E=E, gap=gap, p1=p1, p2=p2, I=I, A=A, ct=ct)


def dI_dalpha_lower(gamma, hO, k1, k2, k3, h=1e-4):
    """Sensitivity of I to LOWERING each atom's on-site energy (central differences)."""
    H0 = hamiltonian(gamma, hO, k1, k2, k3)
    s = np.zeros(8)
    for k in range(8):
        Hp, Hm = H0.copy(), H0.copy()
        Hp[k, k] -= h
        Hm[k, k] += h
        s[k] = (obs(Hp)["I"] - obs(Hm)["I"]) / (2 * h)
    return s


def main():
    np.set_printoptions(precision=4, suppress=True)
    k_sets = {"std(1,1,1)": (1.0, 1.0, 1.0), "2312.02020(1.1,0.9,1)": (1.1, 0.9, 1.0)}
    hOs, gammas = [0.5, 1.0, 2.0], [0.1, 0.3, 0.5]
    rows, fails = [], {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    n = 0
    print(f"{'k-set':>22} {'hO':>4} {'gam':>4} {'p1':>7} {'p2':>7} {'I':>9} {'A':>7} "
          f"{'dI/dd':>9} {'ct':>7} {'dct/dd':>8} {'I(d=.5)':>9} {'O rank':>6}")
    for (kname, ks), hO, g in itertools.product(k_sets.items(), hOs, gammas):
        n += 1
        o = obs(hamiltonian(g, hO, *ks))
        hd = 1e-4
        op = obs(hamiltonian(g, hO, *ks, delta=hd))
        om = obs(hamiltonian(g, hO, *ks, delta=-hd))
        dI = (op["I"] - om["I"]) / (2 * hd)
        dct = (op["ct"] - om["ct"]) / (2 * hd)
        I5 = obs(hamiltonian(g, hO, *ks, delta=0.5))["I"]
        s = dI_dalpha_lower(g, hO, *ks)
        rank = int(np.argsort(-np.abs(s)).tolist().index(7)) + 1
        fails["P1"] += o["I"] <= 0
        fails["P2"] += dI <= 0
        fails["P3"] += (o["ct"] <= 0) or (dct <= 0)
        fails["P4"] += rank > 2
        rows.append((kname, hO, g, s))
        print(f"{kname:>22} {hO:4.1f} {g:4.1f} {o['p1']:7.4f} {o['p2']:7.4f} {o['I']:9.2e} "
              f"{o['A']:7.4f} {dI:9.2e} {o['ct']:7.4f} {dct:8.2e} {I5:9.2e} {rank:6d}")
    print(f"\nFailures out of {n} settings: {fails}")
    print("\nPer-atom sensitivity map dI/d(lower alpha_k), k-set std, hO=1:")
    print("      " + " ".join(f"{x:>9}" for x in NAMES))
    for kname, hO, g, s in rows:
        if kname.startswith("std") and hO == 1.0:
            print(f"g={g:.1f} " + " ".join(f"{v:9.2e}" for v in s))
    print("\nSanity: Tr D = 8 and min HOMO-LUMO gap over all settings:",
          min(obs(hamiltonian(g, hO, *ks))["gap"]
              for ks, hO, g in itertools.product(k_sets.values(), hOs, gammas)))


if __name__ == "__main__":
    main()
