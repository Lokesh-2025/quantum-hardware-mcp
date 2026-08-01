"""
qforge.forging — entanglement forging: solve N qubits using N/2.
================================================================
The idea, in one line: a bipartite state can be written as a sum of PAIRED
half-states (Schmidt decomposition), so you can prepare halves on half the
qubits and rebuild the whole from their matrix elements.

    |psi> = sum_n lambda_n |u_n>_A |v_n>_B

Every Pauli string on the full register factorises exactly as
P = P_B (x) P_A, so its expectation value becomes

    <P> = sum_n lambda_n^2 A_nn B_nn
        + 2 sum_{n<m} lambda_n lambda_m Re( A_nm B_nm )

with A_nm = <u_n|P_A|u_m> and B_nm = <v_n|P_B|v_m>. The diagonal elements come
from preparing |u_n> directly. The off-diagonal ones cannot be measured
directly -- no circuit prepares "<u_n| ... |u_m>" -- so they are recovered from
four superposition circuits per pair:

    |phi_k> = ( |u_n> + i^k |u_m> ) / sqrt(2),   k = 0,1,2,3
    Re A_nm = (E_0 - E_2) / 2,      Im A_nm = (E_3 - E_1) / 2

That is where the factor-of-4 circuit cost per Schmidt pair comes from, and it
dominates the circuit count for rank K > 1:

    state preparations = K + 4 * K*(K-1)/2

Truncating to K terms leaves the state unnormalised, so the energy must be
divided by sum_{n<K} lambda_n^2. Forgetting that renormalisation is the single
most common way to get a wrong forged energy.

Reference: Eddins et al., "Doubling the size of quantum simulators by
entanglement forging", PRX Quantum 3, 010309 (2022).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp


@dataclass
class SchmidtDecomposition:
    """Result of splitting a state down the middle."""

    coefficients: np.ndarray   # lambda_n, descending
    alpha_vectors: np.ndarray  # alpha_vectors[n] = |u_n>
    beta_vectors: np.ndarray   # beta_vectors[n]  = |v_n>

    @property
    def rank(self) -> int:
        """Number of Schmidt terms with meaningful weight."""
        return int((self.coefficients > 1e-8).sum())

    def truncation_weight(self, k: int) -> float:
        """sum_{n<k} lambda_n^2 -- the norm kept by a rank-k truncation."""
        return float(sum(self.coefficients[n] ** 2 for n in range(k)))

    def is_symmetric(self, k: int | None = None, tol: float = 1e-9) -> bool:
        """True when the two halves are identical up to sign.

        When this holds, only ONE register needs to be measured on hardware --
        an exact factor-of-two saving. It is true for symmetric molecules such
        as an evenly spaced hydrogen chain.
        """
        k = k or self.rank
        return all(
            np.allclose(np.abs(self.alpha_vectors[n]), np.abs(self.beta_vectors[n]),
                        atol=tol)
            for n in range(k)
        )


def schmidt_decompose(psi: np.ndarray, n_qubits: int | None = None) -> SchmidtDecomposition:
    """Bipartite Schmidt decomposition across the middle of the register.

    Qiskit orders statevector indices as ``i = beta_index * 2**h + alpha_index``
    (qubit 0 is least significant), so reshaping to ``(2**h, 2**h)`` gives
    rows = beta, columns = alpha with no index gymnastics.

    The SVD ``mat = U S Vh`` expands as ``sum_n U[:,n] S[n] Vh[n,:]``, and the
    Schmidt form is a plain bilinear outer product (no conjugates), so
    ``u_n = Vh[n, :]`` and ``v_n = U[:, n]`` are read off directly.
    """
    dim = psi.shape[0]
    if n_qubits is None:
        n_qubits = int(np.log2(dim))
    if n_qubits % 2:
        raise ValueError(
            f"entanglement forging needs an even qubit count, got {n_qubits}"
        )
    half_dim = 2 ** (n_qubits // 2)

    matrix = psi.reshape(half_dim, half_dim)
    u_mat, singular, vh_mat = np.linalg.svd(matrix)
    return SchmidtDecomposition(
        coefficients=singular,
        alpha_vectors=vh_mat,
        beta_vectors=u_mat.T,
    )


def split_pauli_terms(
    hamiltonian: SparsePauliOp,
) -> list[tuple[str, str, complex]]:
    """Factor every Pauli string into ``(alpha_label, beta_label, coefficient)``.

    A Pauli string is a pure tensor product, so this is exact -- no
    approximation. Qiskit writes labels left-to-right as qubit(n-1)...qubit(0),
    so the FIRST half of the label acts on the high qubits (beta) and the
    second half on the low qubits (alpha), matching the ordering used by
    :func:`schmidt_decompose`.
    """
    n_qubits = hamiltonian.num_qubits
    half = n_qubits // 2
    return [
        (label[half:], label[:half], complex(coeff))
        for label, coeff in hamiltonian.to_list()
    ]


def matrix_elements(
    vectors: np.ndarray, labels, k: int
) -> dict[str, np.ndarray]:
    """Exact ``<u_n|P|u_m>`` for each label, as a k x k matrix.

    Classical reference values -- used to validate a hardware run, and to
    compute the noiseless truncation floor for a given rank.
    """
    top = vectors[:k]
    cache = {}
    for label in set(labels):
        p_matrix = Pauli(label).to_matrix()
        cache[label] = top.conj() @ p_matrix @ top.T
    return cache


def forged_energy(
    terms,
    coefficients: np.ndarray,
    alpha_matrices: dict[str, np.ndarray],
    beta_matrices: dict[str, np.ndarray],
    nuclear_repulsion: float,
    k: int,
) -> float:
    """Rebuild the total energy from half-register matrix elements.

    Works identically whether the matrices came from exact linear algebra or
    from noisy hardware measurements -- which is what makes it possible to
    compare the two on equal footing.
    """
    norm_squared = sum(coefficients[n] ** 2 for n in range(k))
    energy = 0.0
    for alpha_label, beta_label, coeff in terms:
        a_mat = alpha_matrices[alpha_label][:k, :k]
        b_mat = beta_matrices[beta_label][:k, :k]

        diagonal = sum(
            coefficients[n] ** 2 * a_mat[n, n] * b_mat[n, n] for n in range(k)
        )
        cross = sum(
            2 * coefficients[n] * coefficients[m]
            * np.real(a_mat[n, m] * b_mat[n, m])
            for n in range(k)
            for m in range(n + 1, k)
        )
        energy += coeff.real * (np.real(diagonal) + cross)

    # renormalise: a truncated Schmidt sum is not a unit vector
    return energy / norm_squared + nuclear_repulsion


def state_preparation_count(k: int) -> int:
    """Circuits needed for rank ``k``: ``k`` diagonal + 4 per Schmidt pair.

    Useful for costing a hardware run before submitting it. Multiply by the
    number of measurement bases (see :mod:`qforge.grouping`) to get the true
    circuit count -- forgetting the bases factor understates cost badly.
    """
    return k + 4 * (k * (k - 1) // 2)


def cross_term_states(
    alpha_vectors: np.ndarray, n: int, m: int
) -> list[np.ndarray]:
    """The four superposition states that recover off-diagonal elements.

    ``|phi_k> = (|u_n> + i^k |u_m>) / sqrt(2)`` for k = 0..3.
    """
    states = []
    for k in range(4):
        phi = (alpha_vectors[n] + (1j ** k) * alpha_vectors[m]) / np.sqrt(2)
        states.append(phi / np.linalg.norm(phi))
    return states


def combine_cross_measurements(expectations: list[float]) -> complex:
    """Turn the four phase-circuit results into one complex matrix element.

    ``Re = (E_0 - E_2)/2``, ``Im = (E_3 - E_1)/2``.
    """
    if len(expectations) != 4:
        raise ValueError("expected exactly four phase measurements")
    real = (expectations[0] - expectations[2]) / 2
    imag = (expectations[3] - expectations[1]) / 2
    return complex(real, imag)
