# qforge

**Qubit-efficient quantum chemistry on noisy hardware.**

Molecular ground-state calculations need more qubits and less noise than real
quantum computers have. `qforge` packages the methods that close that gap —
each one validated against an exact classical answer, with the failures
documented as carefully as the successes.

```python
from qforge import chemistry, forging, grouping, diagnostics

molecule = chemistry.build_molecule(chemistry.hydrogen_chain(4), n_electrons=4)
energy, psi = molecule.exact_ground_state()          # -2.166387 Ha

schmidt = forging.schmidt_decompose(psi, molecule.n_qubits)
terms = forging.split_pauli_terms(molecule.hamiltonian)

# How many circuits does a rank-5 hardware run actually need?
groups = grouping.build_measurement_groups(
    [a for a, _, _ in terms], molecule.n_qubits // 2
)
print(diagnostics.estimate_cost(schmidt_rank=5, n_measurement_bases=len(groups)))
# 225 circuits — not the 45 you get if you forget measurement bases
```

## Modules

| Module | What it does |
|---|---|
| `chemistry` | Geometry → integrals → Hartree-Fock → Jordan-Wigner qubit Hamiltonian |
| `forging` | **Entanglement forging** — solve N qubits using N/2 |
| `fragmentation` | Many-body expansion, molecular tailoring, **DMET** |
| `mitigation` | **ZNE**, readout correction, symmetry verification, Pauli twirling |
| `grouping` | Measure hundreds of Pauli terms with a handful of circuits |
| `diagnostics` | Bound your error and cost a run *before* paying for hardware |

## The three ideas that matter

**Entanglement forging** splits a molecule down the middle. A bipartite state
is a sum of *paired* half-states, so you prepare halves on half the qubits and
rebuild the whole from their matrix elements. An 8-qubit H4 fragment runs on
4 qubits with 11 two-qubit gates.

The cost is off-diagonal terms: no circuit prepares `<u_n|...|u_m>`, so each
Schmidt pair needs four superposition circuits. State preparations grow as
`K + 4·K(K-1)/2`.

**Grouping** is usually the largest cost lever, because hardware bills per
circuit. General commuting grouping needs a Clifford circuit per group but
packs far more terms together than the qubit-wise variant. On an H4 forged
Hamiltonian: 37 Pauli labels → **13 QWC groups, or 5 general groups**, with
diagonalising Cliffords costing 0–6 two-qubit gates. Every Clifford is verified
numerically before use.

**Diagnostics** exist because getting these numbers wrong is expensive. The
error ceiling bounds every observable from a single fidelity number
(`|Δ<P>| ≤ 2D`), and never failed across 55 configurations and 320+
measurements. The cost model encodes the pricing shape of gate-billed hardware,
including the per-circuit floor that makes shallow circuits all cost the same.

## What didn't work

These are in the library's docstrings, not hidden in a changelog:

- **Bond capping** made covalent fragmentation ~13× *worse* on a hydrogen chain
  (6.72 → 85.59 kcal/mol). Capping works when you cut a C–C bond and patch it
  with a smaller hydrogen. When the molecule is already hydrogen, the "cap" is
  chemically identical to what you removed — you insert a whole extra atom
  instead of gently satisfying a dangling bond.

- **Pauli twirling** gave no measurable benefit against depolarising noise,
  converged over 600 random twirls. Twirling converts *coherent* error into
  stochastic error; with no coherent bias present there is nothing to convert.

- **Symmetry verification in a rotated basis** silently corrupts results.
  Particle number is only conserved in the computational basis; postselecting
  after X/Y rotations discards correct shots and keeps wrong ones. It produced
  an energy ~40× worse than no mitigation. `symmetry_postselect` now refuses
  rotated bases unless explicitly overridden.

- **Single-shot DMET** beat molecular tailoring on a chain (3.40 vs 6.72
  kcal/mol, at *half* the qubits) but lost badly on a ring, and its error grew
  with fragment size — both symptoms of the missing self-consistency loop.

## Validation

Every method is checked against exact diagonalisation:

| Check | Result |
|---|---|
| H4 ground state | −2.166387 Ha, matches PySCF |
| Forged energy, rank 1 / 3 / 5 | 40.64 / 2.76 / **0.57** kcal/mol error |
| Clifford diagonalisation | `C†PC` verified diagonal for every group |
| Error ceiling | 55 configurations, 320+ measurements, zero violations |
| DMET machinery | RDMs rebuild the exact energy to ~1e-15 |

## Install

Requires `numpy`, `scipy`, `qiskit`, `qiskit-nature`. The `chemistry` module
takes an injectable integral backend, so it can be retargeted to a different
Hartree-Fock implementation without touching the rest of the library.

## License

MIT.
