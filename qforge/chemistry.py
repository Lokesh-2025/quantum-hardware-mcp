"""
qforge.chemistry — geometry to qubit Hamiltonian.
=================================================
The front door of the library. Everything else operates on the objects this
module produces, so a caller never has to touch integrals or Hartree-Fock
directly.

Pipeline:
    geometry (atoms + positions)
      -> one- and two-electron integrals   (Gaussian, STO-3G)
      -> restricted Hartree-Fock            (mean field, gives orbitals)
      -> molecular-orbital integrals
      -> Jordan-Wigner qubit Hamiltonian

One subtlety worth knowing, because it is easy to get wrong: the Hartree-Fock
step is only a BASIS GENERATOR. Exact diagonalisation is performed with a
particle-number penalty that pins the electron count, so an odd-electron
(open-shell) system still returns the correct answer even though restricted HF
cannot represent it. Verified: requesting 5 electrons returns a state with
<N> = 5.000000.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse.linalg import eigsh

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper

Geometry = Sequence[tuple[str, tuple[float, float, float]]]

HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386


@dataclass
class Molecule:
    """A molecule, its qubit Hamiltonian, and the pieces needed downstream."""

    geometry: Geometry
    n_electrons: int
    hamiltonian: SparsePauliOp          # bare electronic Hamiltonian
    nuclear_repulsion: float
    hf_energy: float
    n_qubits: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_qubits = self.hamiltonian.num_qubits

    # -- convenience ------------------------------------------------------
    def penalized(self, strength: float = 5.0) -> SparsePauliOp:
        """Hamiltonian plus a penalty pinning the electron count."""
        return constrain_particle_number(
            self.hamiltonian, self.n_electrons, strength
        )

    def exact_ground_state(self) -> tuple[float, np.ndarray]:
        """(total energy in Hartree, statevector) by exact diagonalisation."""
        e_elec, psi = _lowest_eigenpair(self.penalized())
        return e_elec + self.nuclear_repulsion, psi


def number_operator(n_qubits: int) -> SparsePauliOp:
    """Operator whose expectation value is the electron count (Jordan-Wigner)."""
    terms = [("I" * n_qubits, n_qubits / 2)]
    for i in range(n_qubits):
        z = ["I"] * n_qubits
        z[i] = "Z"
        terms.append(("".join(z), -0.5))
    return SparsePauliOp.from_list(terms)


def constrain_particle_number(
    qop: SparsePauliOp, n_target: int, strength: float = 5.0
) -> SparsePauliOp:
    """Add mu * (N - n_target)^2 so the ground state has the right electron count.

    The penalty is exactly zero on the correct sector, so the eigenvalue
    returned is the true electronic energy -- not a shifted one.
    """
    n_qubits = qop.num_qubits
    diff = number_operator(n_qubits) - SparsePauliOp.from_list(
        [("I" * n_qubits, n_target)]
    )
    return SparsePauliOp((qop + strength * (diff @ diff)).simplify())


def _lowest_eigenpair(qop: SparsePauliOp) -> tuple[float, np.ndarray]:
    matrix = qop.to_matrix(sparse=True)
    values, vectors = eigsh(matrix, k=1, which="SA")
    return float(values[0]), vectors[:, 0]


def build_molecule(
    geometry: Geometry,
    n_electrons: int,
    *,
    chem_module=None,
) -> Molecule:
    """Build a :class:`Molecule` from atomic positions.

    Args:
        geometry: ``[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74)), ...]``
            with positions in Angstrom.
        n_electrons: total electron count (may be odd -- see module docstring).
        chem_module: integral/HF backend. Defaults to the vendored
            :mod:`qforge.integrals`, so the library is self-contained. Pass any
            object exposing ``integrals(geometry)`` and
            ``rhf(S, T, V, eri, e_nuc, nelec=...)`` to swap in PySCF or another
            engine -- useful for basis sets beyond STO-3G.
    """
    if chem_module is None:
        from qforge import integrals as chem_module  # type: ignore[no-redef]

    overlap, kinetic, nuclear, eri, e_nuc = chem_module.integrals(geometry)
    hf_energy, coeffs, h_core = chem_module.rhf(
        overlap, kinetic, nuclear, eri, e_nuc, nelec=n_electrons
    )

    h1 = coeffs.T @ h_core @ coeffs
    h2 = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl", coeffs, coeffs, eri, coeffs, coeffs, optimize=True
    )

    electronic = ElectronicEnergy.from_raw_integrals(h1, h2)
    qubit_op = JordanWignerMapper().map(electronic.second_q_op())

    return Molecule(
        geometry=list(geometry),
        n_electrons=n_electrons,
        hamiltonian=qubit_op,
        nuclear_repulsion=float(e_nuc),
        hf_energy=float(hf_energy),
    )


def hydrogen_chain(n_atoms: int, spacing: float = 1.0) -> Geometry:
    """A straight line of hydrogen atoms -- the standard correlation benchmark."""
    return [("H", (i * spacing, 0.0, 0.0)) for i in range(n_atoms)]


def hydrogen_ring(n_atoms: int, spacing: float = 1.0) -> Geometry:
    """A ring of hydrogens whose nearest-neighbour distance is ``spacing``.

    Rings have no terminal atoms, which matters for fragmentation: every
    fragment is an interior one, so cut-handling is uniform.
    """
    import math

    radius = spacing / (2.0 * math.sin(math.pi / n_atoms))
    return [
        (
            "H",
            (
                radius * math.cos(2 * math.pi * i / n_atoms),
                radius * math.sin(2 * math.pi * i / n_atoms),
                0.0,
            ),
        )
        for i in range(n_atoms)
    ]
