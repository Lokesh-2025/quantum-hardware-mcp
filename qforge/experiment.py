"""
qforge.experiment — from a molecule to submittable circuits, and back again.
============================================================================
The other modules each do one job. This one joins them into a runnable
experiment:

    molecule -> ground state -> Schmidt decomposition -> state preparations
             -> measurement grouping -> RUNNABLE CIRCUITS (QASM)
             -> [ submit to hardware ] -> counts -> ENERGY

:func:`build_experiment` produces circuits you can hand straight to a
submission tool. :func:`reconstruct_energy` turns the returned counts back
into a molecular energy. Between those two calls the quantum computer does
its part.

The measurement subtlety that is easy to get wrong
--------------------------------------------------
After Clifford diagonalisation the circuit measures ``C^dag P C``, NOT ``P``.
The rotated operator is a Z-string carrying a sign, obtained with
``Pauli.evolve(clifford, frame='h')``. Use the wrong frame, or drop the sign,
and every energy comes out quietly wrong rather than obviously broken -- so
:func:`self_check` replays the whole pipeline on a simulator and compares
against the exact answer before anything is submitted.

Three circuit-count optimisations are applied, each measured rather than
assumed (585 circuits -> 125 for H4 at Schmidt rank 5):

* general commuting grouping, 13 measurement bases -> 5
* the real gauge, 4 phase circuits per Schmidt pair -> 2
* half-register symmetry, one register measured instead of two
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli

from qforge import chemistry, forging, grouping

HARTREE_TO_KCAL_MOL = 627.5094740631


@dataclass
class ForgedExperiment:
    """Everything needed to run a forged energy calculation and interpret it."""

    molecule: chemistry.Molecule
    exact_energy: float
    decomposition: forging.SchmidtDecomposition
    terms: list
    groups: list
    cliffords: list
    preparations: list
    beta_signs: np.ndarray
    rank: int
    residual_imaginary: float
    circuits: list[QuantumCircuit] = field(default_factory=list)
    index: list[tuple] = field(default_factory=list)

    @property
    def n_circuits(self) -> int:
        return len(self.circuits)

    @property
    def accuracy_floor_kcal_mol(self) -> float:
        """Best achievable error at this Schmidt rank, before hardware noise."""
        alpha_labels = [a for a, _, _ in self.terms]
        beta_labels = [b for _, b, _ in self.terms]
        alpha = forging.matrix_elements(
            self.decomposition.alpha_vectors, alpha_labels, self.rank
        )
        beta = forging.matrix_elements(
            self.decomposition.beta_vectors, beta_labels, self.rank
        )
        energy = forging.forged_energy(
            self.terms, self.decomposition.coefficients, alpha, beta,
            self.molecule.nuclear_repulsion, self.rank,
        )
        return abs(energy - self.exact_energy) * HARTREE_TO_KCAL_MOL


def diagonalized_pauli(label: str, clifford) -> tuple[str, float]:
    """``C^dag P C`` as ``(z_only_label, sign)``.

    ``frame='h'`` is the Heisenberg convention and the one that matches the
    circuit ``state_prep -> C^dag -> measure``. ``frame='s'`` returns a
    non-diagonal operator, which would silently produce nonsense.
    """
    evolved = Pauli(label).evolve(clifford, frame="h").to_label()
    sign = -1.0 if evolved.startswith("-") else 1.0
    stripped = evolved.lstrip("+-")
    if not set(stripped) <= {"I", "Z"}:
        raise ValueError(f"{label} did not diagonalise: got {evolved}")
    return stripped, sign


def measurement_circuit(
    state: np.ndarray, clifford, n_qubits: int, name: str
) -> QuantumCircuit:
    """Prepare the state, rotate into the measurement basis, and measure."""
    circuit = QuantumCircuit(n_qubits, n_qubits, name=name)
    circuit.compose(forging.state_preparation_circuit(state), inplace=True)
    if clifford is not None:
        circuit.compose(clifford.to_circuit().inverse(), inplace=True)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def build_experiment(
    geometry,
    n_electrons: int,
    rank: int = 5,
    *,
    optimization_level: int = 3,
) -> ForgedExperiment:
    """Build every circuit a forged energy calculation needs."""
    molecule = chemistry.build_molecule(geometry, n_electrons=n_electrons)
    exact, psi_raw = molecule.exact_ground_state()
    psi, residual = forging.real_gauge(psi_raw)

    decomposition = forging.schmidt_decompose(psi, molecule.n_qubits)
    rank = min(rank, decomposition.rank)
    terms = forging.split_pauli_terms(molecule.hamiltonian)
    half = molecule.n_qubits // 2
    alpha_labels = sorted({a for a, _, _ in terms})

    groups = grouping.build_measurement_groups(alpha_labels, half, qubit_wise=False)
    if not all(group.verified for group in groups):
        raise RuntimeError("a Clifford failed verification -- refusing to build")

    cliffords = [
        None
        if all(set(label) <= {"I", "Z"} for label in group.labels)
        else grouping.diagonalizing_clifford(group.labels, half)
        for group in groups
    ]

    # The two half-registers match only up to a per-vector sign. Reusing the
    # alpha measurements for beta without correcting those signs gave a
    # 149 kcal/mol error against a 2.76 kcal/mol floor -- wrong, but not
    # obviously wrong, which is worse.
    beta_signs = decomposition.beta_signs(rank)
    for n in range(rank):
        if not np.allclose(
            decomposition.alpha_vectors[n],
            beta_signs[n] * decomposition.beta_vectors[n],
            atol=1e-8,
        ):
            raise RuntimeError(
                f"half-registers differ by more than a sign at n={n}; measuring "
                "one register and reusing it is not valid for this molecule"
            )

    use_real_gauge = residual < 1e-10
    preparations = forging.forged_state_preparations(
        decomposition, rank, real_gauge_applied=use_real_gauge
    )

    circuits, index = [], []
    for key, state in preparations:
        for group_id, clifford in enumerate(cliffords):
            name = f"{key[0]}_{key[1]}{key[2]}_p{key[3]}_g{group_id}"
            circuit = measurement_circuit(state, clifford, half, name)
            circuits.append(
                transpile(circuit, basis_gates=["u3", "cx"],
                          optimization_level=optimization_level)
            )
            index.append((key, group_id))

    return ForgedExperiment(
        molecule=molecule, exact_energy=exact, decomposition=decomposition,
        terms=terms, groups=groups, cliffords=cliffords,
        preparations=preparations, beta_signs=beta_signs, rank=rank,
        residual_imaginary=residual, circuits=circuits, index=index,
    )


def reconstruct_energy(experiment: ForgedExperiment, counts_per_circuit) -> float:
    """Rebuild the molecular energy from measurement counts.

    ``counts_per_circuit`` must be in the same order as
    ``experiment.circuits``.
    """
    if len(counts_per_circuit) != len(experiment.index):
        raise ValueError(
            f"expected {len(experiment.index)} count dictionaries, "
            f"got {len(counts_per_circuit)}"
        )

    rank = experiment.rank
    alpha_labels = sorted({a for a, _, _ in experiment.terms})
    matrices = {label: np.zeros((rank, rank), dtype=complex) for label in alpha_labels}

    values: dict[tuple, dict[str, float]] = {}
    for (key, group_id), counts in zip(experiment.index, counts_per_circuit):
        group = experiment.groups[group_id]
        clifford = experiment.cliffords[group_id]
        bucket = values.setdefault(key, {})
        for label in group.labels:
            if clifford is None:
                bucket[label] = grouping.expectation_from_counts(counts, label)
            else:
                z_label, sign = diagonalized_pauli(label, clifford)
                bucket[label] = sign * grouping.expectation_from_counts(counts, z_label)

    has_imaginary = any(key[3] in (1, 3) for key in values)
    for label in alpha_labels:
        for n in range(rank):
            matrices[label][n, n] = values[("diag", n, n, 0)][label]
        for n in range(rank):
            for m in range(n + 1, rank):
                e0 = values[("cross", n, m, 0)][label]
                e2 = values[("cross", n, m, 2)][label]
                real = (e0 - e2) / 2
                if has_imaginary:
                    e1 = values[("cross", n, m, 1)][label]
                    e3 = values[("cross", n, m, 3)][label]
                    imag = (e3 - e1) / 2
                else:
                    imag = 0.0  # real gauge: the imaginary part is identically zero
                matrices[label][n, m] = complex(real, imag)
                matrices[label][m, n] = complex(real, -imag)

    correction = np.outer(experiment.beta_signs, experiment.beta_signs)
    beta_matrices = {b: matrices[b] * correction for _, b, _ in experiment.terms}
    return forging.forged_energy(
        experiment.terms, experiment.decomposition.coefficients,
        matrices, beta_matrices, experiment.molecule.nuclear_repulsion,
        experiment.rank,
    )


def to_qasm(circuit: QuantumCircuit, version: int = 2) -> str:
    """Serialise a circuit for submission tools that take QASM strings."""
    if version == 3:
        from qiskit import qasm3

        return qasm3.dumps(circuit)
    from qiskit import qasm2

    return qasm2.dumps(circuit)


def self_check(experiment: ForgedExperiment, shots: int = 20000,
               seed: int = 11) -> tuple[bool, float]:
    """Replay the whole pipeline on a local simulator before submitting.

    Catches sign errors, frame errors and reconstruction bugs -- each of which
    shows up on hardware only as a quietly wrong number. Returns
    ``(passed, error_kcal_mol)``.
    """
    from qiskit_aer import AerSimulator

    result = AerSimulator().run(
        experiment.circuits, shots=shots, seed_simulator=seed
    ).result()
    counts = [result.get_counts(i) for i in range(len(experiment.circuits))]
    energy = reconstruct_energy(experiment, counts)
    error = abs(energy - experiment.exact_energy) * HARTREE_TO_KCAL_MOL
    # allow the truncation floor plus room for shot noise
    return error < max(5.0, experiment.accuracy_floor_kcal_mol * 3), error
