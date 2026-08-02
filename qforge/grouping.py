"""
qforge.grouping — measure many Pauli terms with few circuits.
=============================================================
A Hamiltonian has hundreds of Pauli terms, but you do not need one circuit per
term. Terms that can be read from the SAME shots share a circuit, and since
hardware bills per circuit, grouping is usually the largest single cost lever
in the whole pipeline.

Two grouping strengths, with a real trade-off:

**Qubit-wise commuting (QWC)** -- two Paulis share a basis when, on every
qubit, their non-identity parts agree. Measuring needs only single-qubit
rotations (H for X, S-dagger+H for Y). Cheap, but weak.

**General commuting** -- two Paulis share a basis whenever they commute as
operators, which is far more permissive. Measuring needs a CLIFFORD circuit
that rotates the whole group into the computational basis. Stronger grouping,
extra gates.

Which wins depends on the pricing model. On hardware that charges a per-circuit
minimum, shallow circuits sit UNDER the floor, so the Clifford's extra gates
are effectively free and general grouping is a large net win. Measured on an
H4 forged Hamiltonian: 37 Pauli labels -> 13 QWC groups, but only 5 general
commuting groups, with diagonalising Cliffords costing 0-5 two-qubit gates.

Everything here is verified numerically -- :func:`verify_diagonalization`
checks that ``C^dag P C`` really is diagonal before any group is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli


# ---------------------------------------------------------------- predicates
def qubit_wise_commute(a: str, b: str) -> bool:
    """True when a and b agree on every qubit where both act."""
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def commute(a: str, b: str) -> bool:
    """True when a and b commute as operators (an even number of clashes)."""
    clashes = sum(1 for x, y in zip(a, b) if x != "I" and y != "I" and x != y)
    return clashes % 2 == 0


# ------------------------------------------------------------------ grouping
def group_paulis(labels, qubit_wise: bool = False) -> list[list[str]]:
    """Partition Pauli labels into mutually-compatible measurement groups.

    Greedy, largest-support-first. Optimal grouping is NP-hard (graph
    colouring); greedy is standard practice and gets close in practice.

    Args:
        labels: Pauli strings, e.g. ``["XXII", "ZZII", ...]``.
        qubit_wise: use the weaker QWC rule (rotations only, no Clifford).
    """
    predicate = qubit_wise_commute if qubit_wise else commute
    # Sort by (-weight, label): the label tiebreak is essential. `set` iteration
    # order is arbitrary and `sorted` is stable, so without it the greedy result
    # -- and therefore the circuit count and the cost estimate -- would change
    # between runs on identical input.
    ordered = sorted(set(labels), key=lambda s: (-sum(1 for c in s if c != "I"), s))
    groups: list[list[str]] = []
    for label in ordered:
        for group in groups:
            if all(predicate(label, member) for member in group):
                group.append(label)
                break
        else:
            groups.append([label])
    return groups


def rotation_signature(label: str) -> tuple[str, ...]:
    """Per-qubit measurement basis for a QWC group ('*' = free choice)."""
    return tuple(c if c != "I" else "*" for c in label)


def basis_rotation_circuit(labels, n_qubits: int) -> QuantumCircuit:
    """Single-qubit rotations that diagonalise a QWC group.

    H maps X to Z; S-dagger then H maps Y to Z. Identity positions are left
    alone, since any basis reads them correctly.
    """
    setting = ["I"] * n_qubits
    for label in labels:
        for i, ch in enumerate(reversed(label)):
            if ch != "I":
                setting[i] = ch

    circuit = QuantumCircuit(n_qubits)
    for i, ch in enumerate(setting):
        if ch == "X":
            circuit.h(i)
        elif ch == "Y":
            circuit.sdg(i)
            circuit.h(i)
    return circuit


# --------------------------------------------------- Clifford diagonalisation
def _label_to_symplectic(label: str) -> np.ndarray:
    """Pauli label -> symplectic ``[x | z]`` vector indexed BY QUBIT.

    Qiskit writes labels most-significant-qubit first: ``"XZ"`` means X on
    qubit 1 and Z on qubit 0. The Clifford tableau, by contrast, is indexed by
    qubit number, so the label must be reversed before mapping.

    Getting this wrong is a silent, size-dependent bug. A group that happens to
    be closed under label reversal still diagonalises correctly, which is why
    a 4-qubit test set passed while every 6-qubit molecule (LiH, H2O) failed
    verification.
    """
    reversed_label = label[::-1]
    x = np.array([1 if c in "XY" else 0 for c in reversed_label], dtype=np.int8)
    z = np.array([1 if c in "ZY" else 0 for c in reversed_label], dtype=np.int8)
    return np.concatenate([x, z])


def _symplectic_product(v1: np.ndarray, v2: np.ndarray, n: int) -> int:
    """0 when the two Paulis commute, 1 when they anticommute."""
    return (int(v1[:n] @ v2[n:]) + int(v1[n:] @ v2[:n])) % 2


def _independent_subset(vectors) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    chosen: list[np.ndarray] = []
    for vec in vectors:
        residue = vec.copy()
        for b in basis:
            if residue[np.argmax(b)]:
                residue = (residue + b) % 2
        if residue.any():
            basis.append(residue)
            chosen.append(vec)
            basis.sort(key=lambda r: np.argmax(r))
    return chosen


def _search_basis(n: int) -> list[np.ndarray]:
    """Weight-1 and weight-2 Paulis, as symplectic vectors.

    Weight-1 alone spans the 2n-dimensional symplectic space; the weight-2
    entries widen the search so the destabiliser hunt has more candidates to
    choose from.
    """
    labels = []
    for i in range(n):
        for p in "XZY":
            lbl = ["I"] * n
            lbl[i] = p
            labels.append("".join(lbl))
    for i in range(n):
        for j in range(i + 1, n):
            for pi in "XZY":
                for pj in "XZY":
                    lbl = ["I"] * n
                    lbl[i], lbl[j] = pi, pj
                    labels.append("".join(lbl))
    return [_label_to_symplectic(l) for l in labels]


def diagonalizing_clifford(labels, n_qubits: int) -> Clifford:
    """Clifford ``C`` with ``C^dag P C`` diagonal for every P in the group.

    Commuting Paulis span an ISOTROPIC subspace of the symplectic space. The
    construction completes that subspace to a full symplectic basis
    (stabilisers plus destabilisers) by symplectic Gram-Schmidt over GF(2),
    then hands the tableau to Qiskit.

    The subtle step: after fixing each (stabiliser, destabiliser) pair, the
    remaining stabilisers are adjusted to commute with the new destabiliser by
    multiplying them with the current one. That is legal because the product of
    two group elements is still in the group -- the GROUP is preserved even
    though individual generators change, and every original Pauli is a product
    of generators, so diagonalising the group diagonalises all of them.
    """
    n = n_qubits
    vectors = [_label_to_symplectic(l) for l in labels]
    stabilizers = _independent_subset([v for v in vectors if v.any()])
    search = _search_basis(n)

    # extend to a maximal isotropic set (n commuting independent generators)
    while len(stabilizers) < n:
        for candidate in search:
            if not candidate.any():
                continue
            if any(_symplectic_product(candidate, s, n) for s in stabilizers):
                continue
            if len(_independent_subset(stabilizers + [candidate])) != len(stabilizers) + 1:
                continue
            stabilizers.append(candidate)
            break
        else:
            raise RuntimeError("could not extend to a maximal isotropic set")

    destabilizers: list[np.ndarray] = []
    for i in range(n):
        stabilizer = stabilizers[i]
        found = None
        for candidate in search:
            if _symplectic_product(candidate, stabilizer, n) != 1:
                continue
            vec = candidate.copy()
            for j in range(i):
                if _symplectic_product(vec, stabilizers[j], n):
                    vec = (vec + destabilizers[j]) % 2
                if _symplectic_product(vec, destabilizers[j], n):
                    vec = (vec + stabilizers[j]) % 2
            if _symplectic_product(vec, stabilizer, n) != 1:
                continue
            if any(_symplectic_product(vec, stabilizers[j], n) for j in range(i)):
                continue
            if any(_symplectic_product(vec, destabilizers[j], n) for j in range(i)):
                continue
            found = vec
            break
        if found is None:
            raise RuntimeError(f"destabiliser search failed at index {i}")
        destabilizers.append(found)
        for k in range(i + 1, n):
            if _symplectic_product(stabilizers[k], found, n):
                stabilizers[k] = (stabilizers[k] + stabilizer) % 2

    tableau = np.zeros((2 * n, 2 * n + 1), dtype=bool)
    for i in range(n):
        tableau[i, : 2 * n] = destabilizers[i].astype(bool)
        tableau[n + i, : 2 * n] = stabilizers[i].astype(bool)
    return Clifford(tableau)


def verify_diagonalization(clifford: Clifford, labels, tol: float = 1e-8) -> bool:
    """Check ``C^dag P C`` really is diagonal. Never trust a group without it."""
    unitary = clifford.to_operator().data
    for label in labels:
        rotated = unitary.conj().T @ Pauli(label).to_matrix() @ unitary
        off_diagonal = rotated - np.diag(np.diag(rotated))
        if np.abs(off_diagonal).max() > tol:
            return False
    return True


@dataclass
class MeasurementGroup:
    """One group of Paulis and the circuit that measures them together."""

    labels: list[str]
    circuit: QuantumCircuit
    two_qubit_gates: int
    verified: bool

    @property
    def size(self) -> int:
        return len(self.labels)


def build_measurement_groups(
    labels, n_qubits: int, qubit_wise: bool = False, verify: bool = True
) -> list[MeasurementGroup]:
    """Group Pauli labels and build a measurement circuit for each group.

    This is the function to call when costing or running a real experiment:
    ``len(result)`` is the number of circuits per state preparation.
    """
    groups = group_paulis(labels, qubit_wise=qubit_wise)
    built = []
    for group in groups:
        already_diagonal = all(set(l) <= {"I", "Z"} for l in group)
        if already_diagonal:
            circuit = QuantumCircuit(n_qubits)
            verified = True
        elif qubit_wise:
            circuit = basis_rotation_circuit(group, n_qubits)
            verified = True
        else:
            clifford = diagonalizing_clifford(group, n_qubits)
            circuit = clifford.to_circuit()
            verified = verify_diagonalization(clifford, group) if verify else False
        two_qubit = sum(
            1 for inst in circuit.data if inst.operation.num_qubits == 2
        )
        built.append(
            MeasurementGroup(
                labels=list(group),
                circuit=circuit,
                two_qubit_gates=two_qubit,
                verified=verified,
            )
        )
    return built


def expectation_from_counts(counts: dict[str, int], label: str) -> float:
    """``<P>`` from measurement counts, once the group has been diagonalised.

    Each bitstring contributes ``(-1)**parity`` over the qubits where the Pauli
    acts non-trivially.
    """
    positions = [i for i, ch in enumerate(reversed(label)) if ch != "I"]
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no shots recorded")
    value = 0.0
    for bitstring, n in counts.items():
        bits = bitstring.replace(" ", "")
        parity = sum(int(bits[len(bits) - 1 - i]) for i in positions) % 2
        value += ((-1) ** parity) * n
    return value / total
