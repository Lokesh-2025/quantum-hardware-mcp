"""
qforge — qubit-efficient quantum chemistry on noisy hardware.
=============================================================
A small library of methods for running molecular ground-state calculations on
quantum computers that are too small and too noisy to run them directly.

Four capabilities, all validated against exact classical answers:

* :mod:`qforge.forging` -- entanglement forging. Solve an N-qubit molecule
  using N/2 qubits, via Schmidt decomposition.
* :mod:`qforge.fragmentation` -- many-body expansion, molecular tailoring, and
  DMET. Reach molecules far larger than the qubit count allows.
* :mod:`qforge.mitigation` -- zero-noise extrapolation, readout correction,
  symmetry verification, Pauli twirling.
* :mod:`qforge.grouping` -- measure hundreds of Pauli terms with a handful of
  circuits, including verified Clifford diagonalisation.
* :mod:`qforge.diagnostics` -- bound your error and cost a run BEFORE paying
  for hardware time.

Quick start
-----------
::

    from qforge import chemistry, forging, grouping

    molecule = chemistry.build_molecule(
        chemistry.hydrogen_chain(4, spacing=1.0), n_electrons=4
    )
    energy, psi = molecule.exact_ground_state()

    schmidt = forging.schmidt_decompose(psi, molecule.n_qubits)
    terms = forging.split_pauli_terms(molecule.hamiltonian)

    # how many circuits would rank-3 forging actually need?
    groups = grouping.build_measurement_groups(
        [alpha for alpha, _, _ in terms], molecule.n_qubits // 2
    )
    preps = forging.state_preparation_count(3)
    print(f"{preps} state preps x {len(groups)} bases = {preps * len(groups)} circuits")

A note on honesty
-----------------
Docstrings in this library record what did NOT work as carefully as what did:
bond capping made fragmentation ~13x worse, Pauli twirling gave no measurable
benefit against depolarising noise, and symmetry verification silently
corrupts results if applied in a rotated measurement basis. Those notes exist
because each of them cost real debugging time to discover.
"""

from qforge import (
    chemistry,
    diagnostics,
    experiment,
    forging,
    fragmentation,
    grouping,
    integrals,
    mitigation,
)

__version__ = "0.1.0"

__all__ = [
    "chemistry",
    "integrals",
    "experiment",
    "forging",
    "fragmentation",
    "grouping",
    "mitigation",
    "diagnostics",
]
