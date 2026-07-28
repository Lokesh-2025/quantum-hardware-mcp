"""
e1_entangled_lnaa.py
--------------------
E1: Entangled LNAA Transfer — the graded entangling series promised to Vadim.

Phase 5 had exactly ONE coupling pair (2 RZZ gates total). Step 4 had ZERO.
E1 builds circuits where the two-qubit gates do real work, in three grades:

  single  : Phase 5 baseline        — 1 coupling pair   (control)
  ring    : consistent-pair chain   — 5 coupling pairs
  degree3 : max 3 couplings/qubit   — 8 coupling pairs  (heavy-hex realizable)

Why the first dense attempt failed (30.1x -> 2.77x): adding couplings grows
the energy scale, so Phase 5's angles wrap the phases past pi and scramble
the interference. Fix: re-sweep (gamma, beta, p) per variant.

Coupling design rule — targets stay TRUE ground states by construction:
  Only couple pairs (i,j) whose bit relation is IDENTICAL across all targets.
    bits equal in every target  -> J = -1  (reward Z_i Z_j = +1)
    bits differ in every target -> J = +1  (reward Z_i Z_j = -1)
  Every added term is at its own minimum on every target, so adding terms
  can only WIDEN the gap — never demote a target.

Targets (3003): row 14 = 0001110, row 15 = 0001111, row 78 = 1001110
Fixed bits across all three: q1=q2=q3=1, q4=q5=0. Variable: q0, q6.
(0,6) keeps J=+0.5 to penalize row 79 — the one non-target neighbor.
"""

import os, argparse, itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import qasm2

N_QUBITS    = 7
N_SEARCH    = 128
MARKED_ROWS = [14, 15, 78]

# h sign convention (same as phase5): h>0 rewards bit=1, h<0 rewards bit=0
H_COEFFS = {0: 0.0, 1: +1.0, 2: +1.0, 3: +1.0, 4: -1.0, 5: -1.0, 6: 0.0}

# Consistent pairs derived from target bit patterns (see docstring rule)
VARIANTS = {
    "single": {(0, 6): +0.5},
    "ring": {
        (0, 6): +0.5,
        (1, 2): -1.0,   # equal (1,1) in all targets
        (2, 3): -1.0,   # equal
        (3, 4): +1.0,   # differ (1,0) in all targets
        (4, 5): -1.0,   # equal (0,0)
    },
    "degree3": {
        (0, 6): +0.5,
        (1, 2): -1.0, (2, 3): -1.0, (3, 4): +1.0, (4, 5): -1.0,
        (1, 4): +1.0,   # differ (1,0)
        (2, 5): +1.0,   # differ (1,0)
        (1, 3): -1.0,   # equal (1,1)   -> degrees: q1=3 q2=3 q3=3 q4=3 q5=2
    },
}


def ising_energy(row, j_couplings):
    spins = [(-1) ** ((row >> i) & 1) for i in range(N_QUBITS)]
    # phase5 convention: h>0 rewards bit=1 means energy term is -h*spin... but
    # phase5 code used E = sum(h_i * spin_i) with h_1..3=+1 rewarding bit=1
    # because spin(1)=-1 gives +1*-1=-1. Keep that exact convention.
    e = sum(H_COEFFS[i] * spins[i] for i in range(N_QUBITS))
    for (i, j), J in j_couplings.items():
        e += J * spins[i] * spins[j]
    return e


def check_landscape(name, j_couplings):
    energies = {r: ising_energy(r, j_couplings) for r in range(N_SEARCH)}
    worst_target = max(energies[r] for r in MARKED_ROWS)
    phantoms = [r for r in range(N_SEARCH)
                if r not in MARKED_ROWS and energies[r] <= worst_target]
    gap = min(energies[r] for r in range(N_SEARCH) if r not in MARKED_ROWS) - worst_target
    ok = "OK  ground states" if not phantoms else f"WARN phantoms at rows {phantoms}"
    print(f"  {name:8s}  worst target E={worst_target:+.1f}  "
          f"nearest non-target gap={gap:+.1f}  {ok}")
    return not phantoms


def build_circuit(j_couplings, gamma, beta, p, measure=False):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS if measure else 0)
    qc.h(range(N_QUBITS))
    for _ in range(p):
        for i, h in H_COEFFS.items():
            if abs(h) > 1e-9:
                qc.rz(2 * h * gamma, i)
        for (i, j), J in j_couplings.items():
            qc.rzz(2 * J * gamma, i, j)
        for i in range(N_QUBITS):
            qc.rx(2 * beta, i)
    if measure:
        qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def target_prob(j_couplings, gamma, beta, p):
    sv = Statevector.from_instruction(build_circuit(j_couplings, gamma, beta, p))
    probs = np.abs(sv.data) ** 2
    return sum(probs[r] for r in MARKED_ROWS)


def sweep(j_couplings, p_values=(1, 2, 3)):
    """Coarse grid, then zoom around the best point."""
    random_p = len(MARKED_ROWS) / N_SEARCH
    best = (0.0, None)  # (amp, (gamma, beta, p))
    for p in p_values:
        for gamma in np.linspace(0.05, np.pi, 30):
            for beta in np.linspace(0.05, np.pi / 2, 20):
                amp = target_prob(j_couplings, gamma, beta, p) / random_p
                if amp > best[0]:
                    best = (amp, (gamma, beta, p))
    # zoom
    amp0, (g0, b0, p0) = best
    for gamma in np.linspace(max(0.01, g0 - 0.15), g0 + 0.15, 15):
        for beta in np.linspace(max(0.01, b0 - 0.15), b0 + 0.15, 15):
            amp = target_prob(j_couplings, gamma, beta, p0) / random_p
            if amp > best[0]:
                best = (amp, (gamma, beta, p0))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help="write QASM files for IonQ submission")
    args = parser.parse_args()

    print("\nE1 — Entangled LNAA Transfer (single / ring / degree3)\n")

    print("Energy landscape check (targets must be ground states):")
    for name, j in VARIANTS.items():
        check_landscape(name, j)

    print("\nAngle sweep (exact statevector, no sampling noise):")
    results = {}
    for name, j in VARIANTS.items():
        amp, (gamma, beta, p) = sweep(j)
        two_q = len(j) * p
        results[name] = (amp, gamma, beta, p, two_q)
        print(f"  {name:8s}  amp={amp:6.2f}x   gamma={gamma:.3f}  beta={beta:.3f}  "
              f"p={p}   two-qubit gates={two_q}")

    print("\nSummary vs promises:")
    print(f"  Phase 5 on IBM was 27.78x with 2 two-qubit gates.")
    for name, (amp, *_rest, two_q) in results.items():
        print(f"  {name:8s}: {amp:6.2f}x ideal, {two_q} two-qubit gates")

    if args.export:
        outdir = os.path.join(os.path.dirname(__file__), "qasm_ionq")
        os.makedirs(outdir, exist_ok=True)
        for name, (amp, gamma, beta, p, _) in results.items():
            qc = build_circuit(VARIANTS[name], gamma, beta, p, measure=True)
            path = os.path.join(outdir, f"e1_{name}.qasm")
            qasm2.dump(qc, path)
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
