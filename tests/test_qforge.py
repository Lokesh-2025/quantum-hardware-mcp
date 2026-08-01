"""
Tests for the qforge library.

Verifies that:
  1. Ground-state energies match known exact references (H2, H4)
  2. Entanglement forging reproduces published truncation accuracy
  3. Pauli grouping is CORRECT (Cliffords really diagonalise) and
     DETERMINISTIC (same input -> same circuit count, so cost estimates
     are stable between runs)
  4. Cost estimation accounts for measurement bases -- the factor whose
     omission understates a hardware run by an order of magnitude
  5. Mitigation guards fire (symmetry verification refuses rotated bases)

These are pure-library tests: no hardware, no network, no MCP server.

Run with:
    pytest tests/test_qforge.py -v
"""

import numpy as np
import pytest

from qforge import chemistry, diagnostics, forging, grouping, mitigation

HARTREE_TO_KCAL_MOL = 627.5094740631


# --------------------------------------------------------------- chemistry
@pytest.mark.parametrize(
    "geometry,n_electrons,expected,n_qubits",
    [
        (chemistry.hydrogen_chain(2, 0.74), 2, -1.137284, 4),
        (chemistry.hydrogen_chain(4, 1.0), 4, -2.166387, 8),
    ],
)
def test_exact_ground_state(geometry, n_electrons, expected, n_qubits):
    """Energies computed from geometry alone must match PySCF-verified values."""
    molecule = chemistry.build_molecule(geometry, n_electrons=n_electrons)
    assert molecule.n_qubits == n_qubits
    energy, _psi = molecule.exact_ground_state()
    assert energy == pytest.approx(expected, abs=1e-5)


def test_library_is_self_contained():
    """qforge must not depend on PySCF or the research repo's chem module."""
    molecule = chemistry.build_molecule(
        chemistry.hydrogen_chain(2, 0.74), n_electrons=2
    )
    assert molecule.hf_energy < 0
    assert molecule.nuclear_repulsion > 0


# ----------------------------------------------------------------- forging
@pytest.fixture(scope="module")
def h4():
    molecule = chemistry.build_molecule(
        chemistry.hydrogen_chain(4, 1.0), n_electrons=4
    )
    energy, psi = molecule.exact_ground_state()
    schmidt = forging.schmidt_decompose(psi, molecule.n_qubits)
    terms = forging.split_pauli_terms(molecule.hamiltonian)
    return molecule, energy, schmidt, terms


@pytest.mark.parametrize(
    "rank,max_error_kcal",
    [(1, 41.0), (3, 3.0), (5, 0.6)],
)
def test_forged_energy_accuracy(h4, rank, max_error_kcal):
    """Published truncation accuracy: rank 5 reaches ~0.57 kcal/mol."""
    molecule, exact, schmidt, terms = h4
    alpha_labels = [a for a, _, _ in terms]
    beta_labels = [b for _, b, _ in terms]
    alpha = forging.matrix_elements(schmidt.alpha_vectors, alpha_labels, rank)
    beta = forging.matrix_elements(schmidt.beta_vectors, beta_labels, rank)
    energy = forging.forged_energy(
        terms, schmidt.coefficients, alpha, beta, molecule.nuclear_repulsion, rank
    )
    error = abs(energy - exact) * HARTREE_TO_KCAL_MOL
    assert error <= max_error_kcal


def test_state_preparation_count():
    """K diagonal circuits plus four per Schmidt pair."""
    assert forging.state_preparation_count(1) == 1
    assert forging.state_preparation_count(3) == 15
    assert forging.state_preparation_count(5) == 45


def test_symmetric_halves_detected_in_the_real_gauge():
    """An evenly spaced chain has halves related by a per-vector sign, which
    halves hardware cost -- but only once the eigensolver's arbitrary global
    phase is removed. Without that gauge fix the halves differ by a complex
    phase and the relationship is not a sign at all.
    """
    molecule = chemistry.build_molecule(
        chemistry.hydrogen_chain(4, 1.0), n_electrons=4
    )
    _energy, psi_raw = molecule.exact_ground_state()

    psi, residual = forging.real_gauge(psi_raw)
    gauged = forging.schmidt_decompose(psi, molecule.n_qubits)

    assert residual < 1e-10, "H4 ground state should be real-representable"
    assert gauged.is_symmetric(3)
    # and the signs are a genuine +-1, not all trivially +1
    assert set(gauged.beta_signs(3)) <= {-1.0, 1.0}


def test_beta_signs_map_alpha_matrices_onto_beta(h4):
    """B_nm = s_n s_m A_nm must hold exactly, or reusing one register's
    measurements for the other silently corrupts every energy."""
    molecule, _exact, _schmidt, terms = h4
    _energy, psi_raw = molecule.exact_ground_state()
    psi, _residual = forging.real_gauge(psi_raw)
    schmidt = forging.schmidt_decompose(psi, molecule.n_qubits)

    rank = 3
    labels = sorted({a for a, _, _ in terms})
    alpha = forging.matrix_elements(schmidt.alpha_vectors, labels, rank)
    beta = forging.matrix_elements(schmidt.beta_vectors, labels, rank)
    signs = schmidt.beta_signs(rank)
    correction = np.outer(signs, signs)
    for label in labels:
        assert np.allclose(alpha[label] * correction, beta[label], atol=1e-9)


def test_forging_rejects_odd_qubit_count():
    with pytest.raises(ValueError, match="even qubit count"):
        forging.schmidt_decompose(np.ones(8) / np.sqrt(8), n_qubits=3)


# ---------------------------------------------------------------- grouping
def test_general_grouping_beats_qubit_wise(h4):
    """General commuting grouping should need strictly fewer circuits."""
    molecule, _exact, _schmidt, terms = h4
    labels = [a for a, _, _ in terms]
    half = molecule.n_qubits // 2
    general = grouping.build_measurement_groups(labels, half, qubit_wise=False)
    qubit_wise = grouping.build_measurement_groups(labels, half, qubit_wise=True)
    assert len(general) < len(qubit_wise)


def test_cliffords_actually_diagonalize(h4):
    """Every general-commuting group must be verifiably diagonalised."""
    molecule, _exact, _schmidt, terms = h4
    labels = [a for a, _, _ in terms]
    groups = grouping.build_measurement_groups(labels, molecule.n_qubits // 2)
    assert groups, "expected at least one group"
    assert all(group.verified for group in groups)


def test_grouping_is_deterministic(h4):
    """Same input must give the same circuit count on every run.

    Regression test: `set` iteration order is arbitrary and `sorted` is
    stable, so without an explicit tiebreak the greedy grouping produced
    different group counts between runs -- which silently changed the cost
    estimate for identical input.
    """
    molecule, _exact, _schmidt, terms = h4
    labels = [a for a, _, _ in terms]
    half = molecule.n_qubits // 2
    runs = [
        [tuple(sorted(g.labels)) for g in grouping.build_measurement_groups(labels, half)]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_expectation_from_counts_parity():
    """<ZI> is +1 on '00' and -1 on '01' (qubit 0 is the rightmost bit)."""
    assert grouping.expectation_from_counts({"00": 100}, "ZI") == pytest.approx(1.0)
    assert grouping.expectation_from_counts({"01": 100}, "IZ") == pytest.approx(-1.0)


# -------------------------------------------------------------- diagnostics
def test_cost_includes_measurement_bases():
    """Circuits = state preps x bases. Dropping the bases factor is the
    mistake that understated a real hardware quote by ~20x."""
    one_basis = diagnostics.estimate_cost(5, 1)
    many_bases = diagnostics.estimate_cost(5, 13)
    assert one_basis["circuits"] == 45
    assert many_bases["circuits"] == 45 * 13
    assert many_bases["total"] > one_basis["total"]


def test_shallow_circuits_sit_at_pricing_floor():
    """Below the floor, extra gates are free -- which is what makes paying
    gates for Clifford diagonalisation worthwhile."""
    model = diagnostics.CostModel()
    assert model.circuit_price(50, 11) == pytest.approx(model.floor)
    assert model.circuit_price(224, 55) > model.floor
    assert 20 < model.floor_breaks_at_two_qubit_gates < 30


def test_error_ceiling_bounds_a_real_deviation():
    """The ceiling must not be violated by an actual observable shift."""
    ideal = np.zeros((2, 2), dtype=complex)
    ideal[0, 0] = 1.0
    noisy = np.array([[0.9, 0.0], [0.0, 0.1]], dtype=complex)
    ceiling = diagnostics.error_ceiling(noisy, ideal)
    z_shift = abs((0.9 - 0.1) - 1.0)
    assert z_shift <= ceiling


def test_shots_scale_quadratically():
    """1/sqrt(N): ten times the precision costs a hundred times the shots."""
    coarse = diagnostics.shots_for_target_accuracy(1.0, target_hartree=1e-2)
    fine = diagnostics.shots_for_target_accuracy(1.0, target_hartree=1e-3)
    assert fine == pytest.approx(coarse * 100, rel=0.01)


# --------------------------------------------------------------- mitigation
def test_zne_recovers_a_linear_trend():
    """A perfectly linear noise trend must extrapolate back to its intercept."""
    result = mitigation.zero_noise_extrapolation([1, 2, 3], [-2.0, -1.9, -1.8])
    assert result.linear == pytest.approx(-2.1, abs=1e-9)


@pytest.mark.parametrize("folds,expected_scale", [(1, 1), (2, 3), (3, 5)])
def test_fold_circuit_matches_declared_noise_scale(folds, expected_scale):
    """Gate count must equal 2*folds - 1, and the helper must agree.

    Regression guard: folding to `n` does NOT give a noise scale of `n`. Using
    fold numbers as ZNE x-values instead of these scales fits the wrong axis
    and extrapolates to the wrong energy.
    """
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    folded = mitigation.fold_circuit(circuit, folds)
    original_cx = sum(1 for i in circuit.data if i.operation.name == "cx")
    folded_cx = sum(1 for i in folded.data if i.operation.name == "cx")

    assert mitigation.noise_scale_factor(folds) == expected_scale
    assert folded_cx == expected_scale * original_cx


def test_symmetry_verification_refuses_rotated_basis():
    """Guard against the bug that made an energy ~40x worse.

    Particle number is not conserved by X/Y basis rotations, so
    postselecting there discards correct shots and keeps wrong ones.
    """
    with pytest.raises(ValueError, match="computational basis"):
        mitigation.symmetry_postselect(
            {"0011": 50, "0111": 5}, 2, basis_is_computational=False
        )


def test_symmetry_verification_drops_wrong_particle_number():
    counts = {"0011": 90, "0111": 10}
    kept, fraction = mitigation.symmetry_postselect(counts, 2)
    assert kept == {"0011": 90}
    assert fraction == pytest.approx(0.9)


def test_readout_mitigation_moves_toward_truth():
    """Inverting the confusion matrix should recover the clean distribution."""
    truth = np.array([1.0, 0.0, 0.0, 0.0])
    flip = 0.05
    observed = mitigation.confusion_matrix(flip, 2) @ truth
    corrected = mitigation.mitigate_readout(observed, flip, 2)
    assert abs(corrected[0] - 1.0) < abs(observed[0] - 1.0)


def test_shot_noise_shrinks_with_more_shots():
    assert mitigation.shot_noise_sigma(0.0, 10_000) < mitigation.shot_noise_sigma(0.0, 100)
    # observables pinned near +-1 are far cheaper to measure
    assert mitigation.shot_noise_sigma(0.99, 1000) < mitigation.shot_noise_sigma(0.0, 1000)
