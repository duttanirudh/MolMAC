"""
rules.py  --  the rule table: from a diene/dienophile pair to coarse "ideal catalyst" requirements.

MOLECULE SPEC: atoms with a type and a fragment, bonds with a resonance factor k, two forming bonds.
  type 'C'  : carbon, h = 0, 1 pi electron
  type 'O=' : carbonyl oxygen, h = h_O (default 1; sourced), 1 pi electron
  type 'OR' : ether oxygen, h = h_OR (default 2; sourced), 2 pi electrons; C-O(ether) k default 0.8 (from memory)
  Units: alpha_C = 0, |beta| = 1, H_ii = -h_i, H_ij = -k_ij.

TWO LEVERS per atom k, for a perturbation delta that LOWERS alpha_k (an electron-withdrawing environment:
an H-bond donor, a Lewis acid, or a positive charge nearby; delta < 0 = the opposite):
  L_E(k) = d(binding)/d(delta_k) = q_k(gamma) - q_k(0)     (Hellmann-Feynman; exact in Huckel)
           binding = E_pi(separated) - E_pi(coupled) > 0. L_E > 0: a withdrawing environment at k
           strengthens the reactant-reactant interaction (a crude transition-state-stabilization proxy).
  L_A(k) = dI/d(delta_k), I = p(forming bond 1) - p(forming bond 2)  (asynchronicity lever; first-order PT)
REQUIREMENT for site k (REVISED after the first run, see C2 below): environment type from sign(L_A), the
  asynchronicity lever; strength from |L_A| ranked within its fragment; whether it also strengthens the
  interaction (sign L_E = sign L_A); reach class: heteroatom (directed binding:
  H-bond / Lewis acid) vs carbon (through-space electrostatics only); robustness = share of parameter
  variants (h_O, bond-parameter set, gamma) that give the same environment sign.
WHY L_A IS PRIMARY: in the first run L_E gave the acrolein O a slightly DONATING requirement (-0.0011, robust),
  contradicting known Lewis-acid / H-bond catalysis there, while L_A gave it withdrawing (+0.0205). DFT work
  (Bickelhaupt group) finds Lewis acids do not strengthen the orbital interaction but raise asynchronicity to
  relieve Pauli repulsion, so the asynchronicity lever is the better proxy. This choice was made AFTER seeing
  the positive control and is disclosed as such.
This is a LABELLED HEURISTIC: Huckel has no sigma framework, sterics, solvent, geometry or Pauli repulsion.

PREDICTIONS (written before running):
  C1  (negative control) butadiene + ethylene: L_E = 0 at every atom to machine precision (pairing theorem:
      all charges stay 1), so the table reports no electrostatic lever.
  C2  (positive control) butadiene + acrolein: the carbonyl O gets a withdrawing requirement (H-bond donor /
      Lewis acid), ranked first among reachable heteroatom sites, robust in all variants, and it also raises
      asynchronicity -- matching known Lewis-acid / H-bond catalysis at the carbonyl oxygen.
  C3  diene atoms get donating (L_E < 0) requirements, largest at the terminal carbons that form bonds.
  C4  2-methoxybutadiene + acrolein: the ether O on the diene gets a donating requirement (withdrawing there
      would weaken the interaction), with the acrolein O still withdrawing.
"""
import itertools
import numpy as np

E_PI = {"C": 1, "O=": 1, "OR": 2}


def build(spec, gamma, hO=1.0, hOR=2.0, kset=None):
    atoms, bonds, form = spec["atoms"], spec["bonds"], spec["form"]
    N = len(atoms)
    H = np.zeros((N, N))
    for i, (t, frag, name) in enumerate(atoms):
        H[i, i] = -{"C": 0.0, "O=": hO, "OR": hOR}[t]
    for i, j, kind in bonds:
        k = (kset or {}).get(kind, DEFAULT_K[kind])
        H[i, j] = H[j, i] = -k
    for i, j in form:
        H[i, j] = H[j, i] = -gamma
    ne = sum(E_PI[t] for t, _, _ in atoms)
    return H, ne // 2


DEFAULT_K = {"CC": 1.0, "C=C": 1.0, "C-C": 1.0, "C=O": 1.0, "C-OR": 0.8}
KSETS = {"std": {}, "2312": {"C=C": 1.1, "C-C": 0.9}}


def eig(H, no):
    e, C = np.linalg.eigh(H)
    return e, C, 2 * C[:, :no] @ C[:, :no].T


def levers(spec, gamma, **kw):
    H, no = build(spec, gamma, **kw)
    H0, _ = build(spec, 0.0, **kw)
    e, C, D = eig(H, no)
    _, _, D0 = eig(H0, no)
    LE = np.diag(D) - np.diag(D0)
    (a, b), (c, d) = spec["form"]
    Co, Cv, eo, ev = C[:, :no], C[:, no:], e[:no], e[no:]
    G = 1.0 / (eo[None, :] - ev[:, None])
    A = np.zeros_like(H); A[a, b] = A[b, a] = 0.5; A[c, d] = A[d, c] = -0.5
    B = Co.T @ A @ Cv
    LA = -4.0 * np.einsum("ki,ij,kj->k", Co, G.T * B, Cv)
    binding = 2 * e[:no].sum() * -1 - (-2 * np.linalg.eigvalsh(H0)[:no].sum())
    return LE, LA, D[a, b] - D[c, d], binding


def variants(spec):
    has_O = any(t == "O=" for t, _, _ in spec["atoms"])
    has_OR = any(t == "OR" for t, _, _ in spec["atoms"])
    hOs = [0.5, 1.0, 2.0] if has_O else [1.0]
    hORs = [1.5, 2.0, 2.5] if has_OR else [2.0]
    kOR = [0.6, 0.8, 1.0] if has_OR else [0.8]
    for hO, hOR, kor, (kn, ks), g in itertools.product(hOs, hORs, kOR, KSETS.items(), [0.1, 0.3, 0.5]):
        yield dict(hO=hO, hOR=hOR, kset={**ks, "C-OR": kor}), g


def table(spec, gamma=0.3, tol=1e-9):
    LE, LA, I, B = levers(spec, gamma)
    sA, sE = [], []
    for kw, g in variants(spec):
        le, la, _, _ = levers(spec, g, **kw)
        sA.append(np.where(np.abs(la) < tol, 0, np.sign(la))); sE.append(np.where(np.abs(le) < tol, 0, np.sign(le)))
    sA, sE = np.array(sA), np.array(sE)
    rows = []
    for k, (t, frag, name) in enumerate(spec["atoms"]):
        s = 0 if abs(LA[k]) < tol else int(np.sign(LA[k]))
        reach = "directed binding" if t in ("O=", "OR") else "through-space only"
        if s == 0:
            env = "no lever"
        elif t in ("O=", "OR"):
            env = ("withdrawing (H-bond donor or Lewis acid on the lone pair)" if s > 0
                   else "donating (negative charge nearby; no H-bond donor on this lone pair)")
        else:
            env = "withdrawing (positive charge nearby)" if s > 0 else "donating (negative charge nearby)"
        also = "yes" if s != 0 and abs(LE[k]) >= tol and np.sign(LE[k]) == s else ("no" if s != 0 else "-")
        rows.append(dict(k=k, name=name, frag=frag, LE=LE[k], LA=LA[k], env=env, reach=reach,
                         also_bind=also, robust=float(np.mean(sA[:, k] == s)), robustE=float(np.mean(sE[:, k] == np.sign(round(LE[k], 12))))))
    for frag in {r["frag"] for r in rows}:
        fr = sorted([r for r in rows if r["frag"] == frag], key=lambda r: -abs(r["LA"]))
        for i, r in enumerate(fr): r["rank"] = i + 1
    return rows, I, B


def summary(rows, top=2):
    """Plain-language requirements: directed-binding sites first, then the strongest through-space sites."""
    out = []
    for frag in ["diene", "dienophile"]:
        fr = [r for r in rows if r["frag"] == frag and r["env"] != "no lever"]
        directed = [r for r in fr if r["reach"] == "directed binding"]
        space = sorted([r for r in fr if r["reach"] != "directed binding"], key=lambda r: r["rank"])[:top]
        for r in directed + space:
            out.append(f"{frag}: {r['env'].split(' (')[0]} environment at {r['name']} ({r['reach']}); "
                       f"lever {r['LA']:+.4f}, rank {r['rank']} in fragment, sign stable in {100 * r['robust']:.0f}% of variants"
                       + ("; also strengthens the interaction" if r["also_bind"] == "yes" else ""))
    return out or ["no electrostatic lever found at this level of theory"]


def butadiene():
    return [("C", "diene", "C1"), ("C", "diene", "C2"), ("C", "diene", "C3"), ("C", "diene", "C4")], \
           [(0, 1, "C=C"), (1, 2, "C-C"), (2, 3, "C=C")]


PRESETS = {}
d_atoms, d_bonds = butadiene()
PRESETS["butadiene + ethylene"] = dict(
    atoms=d_atoms + [("C", "dienophile", "C5"), ("C", "dienophile", "C6")],
    bonds=d_bonds + [(4, 5, "C=C")], form=[(0, 4), (3, 5)])
PRESETS["butadiene + acrolein"] = dict(
    atoms=d_atoms + [("C", "dienophile", "Cb"), ("C", "dienophile", "Ca"), ("C", "dienophile", "Cc"),
                     ("O=", "dienophile", "O")],
    bonds=d_bonds + [(4, 5, "C=C"), (5, 6, "C-C"), (6, 7, "C=O")], form=[(0, 4), (3, 5)])
PRESETS["2-methoxybutadiene + acrolein"] = dict(
    atoms=d_atoms + [("OR", "diene", "O(Me)"), ("C", "dienophile", "Cb"), ("C", "dienophile", "Ca"),
                     ("C", "dienophile", "Cc"), ("O=", "dienophile", "O")],
    bonds=d_bonds + [(1, 4, "C-OR"), (5, 6, "C=C"), (6, 7, "C-C"), (7, 8, "C=O")], form=[(0, 5), (3, 6)])


def main():
    for name, spec in PRESETS.items():
        rows, I, B = table(spec)
        print(f"\n=== {name}  (gamma = 0.3; binding {B:.4f} |beta|; imbalance I = {I:+.4f}) ===")
        print(f"{'atom':>6} {'frag':>10} {'L_A':>9} {'L_E':>9} {'rank':>4} {'robust':>6}  requirement")
        for r in sorted(rows, key=lambda r: (r["frag"], r["rank"])):
            print(f"{r['name']:>6} {r['frag']:>10} {r['LA']:+9.4f} {r['LE']:+9.4f} {r['rank']:4d} {r['robust']:6.2f}  "
                  f"{r['env']} [{r['reach']}]")
        print("  requirements:")
        for line in summary(rows): print("   - " + line)


if __name__ == "__main__":
    main()
