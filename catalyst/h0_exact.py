"""
h0_exact.py  --  H0 exact checks for butadiene + ethylene (float64 reference).

Units: alpha = 0, |beta| = 1, beta < 0, so H = -A_weighted (lower = more bonding).
Atoms (0-based): 0..3 = butadiene C1..C4, 4 = ethylene C5, 5 = ethylene C6.
Sublattices: A = {C1, C3, C6}, B = {C2, C4, C5}.
Forming bonds (coupling gamma): C1-C5 (0,4) and C4-C6 (3,5). gamma = 1 gives benzene.

Checks:
  1. E_pi(0) = 6.472, E_pi(1) = 8.000, dE_int(1) = 1.528  (binding energy, units of |beta|)
  2. all charges = 1 for every gamma (pairing theorem)
  3. same-sublattice bond orders = 0 (pairing theorem)
  4. forming-bond imbalance p1 - p2 = 0 (mirror symmetry C1<->C4, C5<->C6)
  5. SP2 purification (Gershgorin bounds, trace-correcting) reproduces the eigen reference
  6. Hellmann-Feynman dE/dalpha_k = q_k, and the polarizability formula
     pi_kl = 4 sum_occ sum_virt c_ik c_il c_jk c_jl / (e_i - e_j) = dq_k/dalpha_l,
     both checked against central finite differences.
"""
import numpy as np

N_OCC = 3
SUB_A = [0, 2, 5]
SUB_B = [1, 3, 4]
FORM = [(0, 4), (3, 5)]


def hamiltonian(gamma, dtype=np.float64):
    H = np.zeros((6, 6), dtype=dtype)
    for i, j in [(0, 1), (1, 2), (2, 3), (4, 5)]:
        H[i, j] = H[j, i] = -1.0
    for i, j in FORM:
        H[i, j] = H[j, i] = -gamma
    return H


def eig_reference(H, n_occ=N_OCC):
    e, C = np.linalg.eigh(H)
    Co = C[:, :n_occ]
    D = 2.0 * Co @ Co.T                       # density matrix (2 electrons/orbital)
    E = 2.0 * e[:n_occ].sum()                 # total pi energy (negative = bound)
    return e, C, D, E


def gershgorin(H):
    r = np.abs(H).sum(axis=1) - np.abs(np.diag(H))
    return float((np.diag(H) - r).min()), float((np.diag(H) + r).max())


def sp2(H, n_occ=N_OCC, max_iter=100):
    """Trace-correcting SP2. Returns projector P (D = 2P), iteration count, idempotency history.
    Stopping: simplified parameter-free rule -- stop once the idempotency error
    Tr(X - X^2) fails to decrease vs two iterations earlier (in the spirit of
    Kruchinina-Rudberg-Rubensson; not their exact criterion)."""
    emin, emax = gershgorin(H)
    n = H.shape[0]
    X = (emax * np.eye(n) - H) / (emax - emin)
    hist = []
    for it in range(1, max_iter + 1):
        X2 = X @ X                              # the ONE product per iteration
        trX, trX2 = np.trace(X), np.trace(X2)
        if abs(trX2 - n_occ) < abs(2 * trX - trX2 - n_occ):
            X = X2
        else:
            X = 2 * X - X2
        err = float(np.trace(X - X @ X))        # diagnostic only (extra product)
        hist.append(err)
        if it > 4 and err >= hist[-3]:
            break
    return X, it, hist


def observables(D, H, gamma):
    q = np.diag(D).copy()
    same = max(abs(D[i, j]) for S in (SUB_A, SUB_B) for i in S for j in S if i < j)
    p1, p2 = D[FORM[0]], D[FORM[1]]
    asym = abs(p1 - p2) / (p1 + p2) if (p1 + p2) > 1e-12 else float("nan")
    return q, same, p1, p2, asym


def polarizability(e, C, n_occ=N_OCC):
    n = len(e)
    P = np.zeros((n, n))
    for i in range(n_occ):
        for j in range(n_occ, n):
            P += 4.0 * np.outer(C[:, i] * C[:, j], C[:, i] * C[:, j]) / (e[i] - e[j])
    return P


def main():
    np.set_printoptions(precision=6, suppress=True)
    print("=== 1-4: eigen reference over gamma sweep ===")
    print(f"{'gamma':>6} {'E_bind':>10} {'dE_int':>10} {'max|q-1|':>10} "
          f"{'max|same|':>10} {'p1':>8} {'p2':>8} {'A':>10} {'gap':>7}")
    E0 = -eig_reference(hamiltonian(0.0))[3]
    for g in np.linspace(0, 1, 11):
        e, C, D, E = eig_reference(hamiltonian(g))
        q, same, p1, p2, A = observables(D, None, g)
        print(f"{g:6.2f} {-E:10.6f} {-E - E0:10.6f} {np.abs(q - 1).max():10.2e} "
              f"{same:10.2e} {p1:8.5f} {p2:8.5f} {A:10.2e} {e[N_OCC] - e[N_OCC - 1]:7.4f}")

    print("\n=== 5: SP2 vs eigen reference ===")
    print(f"{'gamma':>6} {'iters':>5} {'max|D_sp2-D_ref|':>17} {'|E_sp2-E_ref|':>14} "
          f"{'Tr P - Nocc':>12} {'||P^2-P||_F':>12}")
    for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
        H = hamiltonian(g)
        _, _, Dref, Eref = eig_reference(H)
        P, it, _ = sp2(H)
        Dsp = 2 * P
        Esp = float(np.trace(Dsp @ H))
        print(f"{g:6.2f} {it:5d} {np.abs(Dsp - Dref).max():17.2e} {abs(Esp - Eref):14.2e} "
              f"{np.trace(P) - N_OCC:12.2e} {np.linalg.norm(P @ P - P):12.2e}")

    print("\n=== 6: Hellmann-Feynman and polarizability vs finite differences (gamma=0.5) ===")
    g, h = 0.5, 1e-5
    H = hamiltonian(g)
    e, C, D, E = eig_reference(H)
    Pi = polarizability(e, C)
    hf_err, pol_err = 0.0, 0.0
    fd_pol = np.zeros((6, 6))
    for l in range(6):
        Hp, Hm = H.copy(), H.copy()
        Hp[l, l] += h
        Hm[l, l] -= h
        _, _, Dp, Ep = eig_reference(Hp)
        _, _, Dm, Em = eig_reference(Hm)
        hf_err = max(hf_err, abs((Ep - Em) / (2 * h) - D[l, l]))
        fd_pol[:, l] = (np.diag(Dp) - np.diag(Dm)) / (2 * h)
    pol_err = np.abs(fd_pol - Pi).max()
    print(f"max |dE/dalpha_k - q_k|          = {hf_err:.2e}")
    print(f"max |FD dq_k/dalpha_l - pi_kl|  = {pol_err:.2e}")
    print(f"diag(pi) (should be < 0: raising alpha_k drains atom k) = {np.diag(Pi)}")
    print(f"row sums of pi (should be 0: total charge conserved)     = {Pi.sum(axis=1)}")


if __name__ == "__main__":
    main()
