"""
Tests for the qforge MCP tools exposed by server.py.

These verify the INTEGRATION -- that the library is actually reachable as MCP
tools and returns well-formed JSON -- as distinct from tests/test_qforge.py,
which tests the library itself.

Every tool returns a JSON string (the convention used by the other tools in
server.py), so each test parses the payload and checks its contents rather
than just checking the call did not raise.

No hardware, no network, no IBM credentials required.

Run with:
    pytest tests/test_qforge_tools.py -v
"""

import json

import pytest

import server

H2 = "H 0 0 0; H 0 0 0.74"
H4 = "H 0 0 0; H 1 0 0; H 2 0 0; H 3 0 0"


def call(tool, *args, **kwargs):
    """Invoke a tool and parse its JSON payload."""
    payload = json.loads(tool(*args, **kwargs))
    assert "error" not in payload, f"tool returned an error: {payload.get('error')}"
    return payload


# ----------------------------------------------------------- analyze_molecule
def test_analyze_molecule_h2_matches_reference():
    """H2 energy must match the PySCF-verified reference."""
    result = call(server.analyze_molecule, H2, 2)
    assert result["exact_energy_hartree"] == pytest.approx(-1.137284, abs=1e-5)
    assert result["qubits_direct"] == 4
    assert result["qubits_with_forging"] == 2


def test_analyze_molecule_h4_matches_reference():
    result = call(server.analyze_molecule, H4, 4)
    assert result["exact_energy_hartree"] == pytest.approx(-2.166387, abs=1e-5)
    assert result["qubits_direct"] == 8
    assert result["hamiltonian_terms"] == 185


def test_general_grouping_beats_qubit_wise_through_the_tool():
    """The circuit saving must actually surface in the tool output."""
    result = call(server.analyze_molecule, H4, 4)
    assert (
        result["measurement_circuits_general_commuting"]
        < result["measurement_circuits_qubit_wise"]
    )


def test_full_schmidt_rank_is_exact():
    """At full rank, forging has no truncation error -- a correctness check
    on the whole decompose/rebuild path."""
    result = call(server.analyze_molecule, H2, 2)
    floors = result["accuracy_floor_kcal_mol_by_rank"]
    assert floors[str(result["schmidt_rank"])] == pytest.approx(0.0, abs=1e-3)


def test_accuracy_improves_with_rank():
    """Keeping more Schmidt terms must never make the answer worse."""
    result = call(server.analyze_molecule, H4, 4)
    floors = result["accuracy_floor_kcal_mol_by_rank"]
    ranks = sorted(int(k) for k in floors)
    errors = [floors[str(r)] for r in ranks]
    assert errors == sorted(errors, reverse=True)


# --------------------------------------------------- plan_quantum_chemistry_run
def test_plan_recommends_the_most_accurate_affordable_option():
    result = call(server.plan_quantum_chemistry_run, H4, 4, 3000.0)
    recommended = result["recommended"]
    affordable = [o for o in result["options"] if o["fits_budget"]]
    assert recommended["fits_budget"]
    assert recommended["accuracy_floor_kcal_mol"] == min(
        o["accuracy_floor_kcal_mol"] for o in affordable
    )


def test_plan_respects_the_budget():
    small = call(server.plan_quantum_chemistry_run, H4, 4, 200.0)
    large = call(server.plan_quantum_chemistry_run, H4, 4, 10000.0)
    # a bigger budget can only buy equal or better accuracy
    assert (
        large["recommended"]["accuracy_floor_kcal_mol"]
        <= small["recommended"]["accuracy_floor_kcal_mol"]
    )


def test_plan_reports_when_nothing_fits():
    """An impossible budget must say so rather than recommend something."""
    result = json.loads(server.plan_quantum_chemistry_run(H4, 4, 1.0))
    assert result["recommended"] is None
    assert result["warning"]


def test_plan_uses_the_real_gauge_circuit_count():
    """Circuits = (rank + 2*pairs) x bases.

    The factor is TWO phase circuits per Schmidt pair, not four: in the real
    gauge the imaginary part is identically zero, so those circuits would
    measure only noise.
    """
    result = call(server.plan_quantum_chemistry_run, H4, 4, 10000.0)
    bases = result["molecule"]["measurement_bases"]
    for option in result["options"]:
        rank = option["schmidt_rank"]
        expected_preps = rank + 2 * (rank * (rank - 1) // 2)
        assert option["state_preparations"] == expected_preps
        assert option["circuits"] == expected_preps * bases


# ------------------------------------------------- recommend_error_mitigation
def test_mitigation_advice_scales_with_noise():
    """A clean shallow circuit should not be told it needs ZNE."""
    clean = call(server.recommend_error_mitigation, 2, 0.0002, 0.0, True)
    noisy = call(server.recommend_error_mitigation, 55, 0.03, 0.0, True)
    assert "zero_noise_extrapolation" not in clean["apply_in_order"]
    assert "zero_noise_extrapolation" in noisy["apply_in_order"]


def test_twirling_is_never_recommended():
    """A documented negative: no measured benefit against depolarising noise."""
    result = call(server.recommend_error_mitigation, 11, 0.002, 0.02, True)
    twirl = next(
        r for r in result["recommendations"] if r["technique"] == "pauli_twirling"
    )
    assert twirl["recommended"] is False
    assert "pauli_twirling" not in result["apply_in_order"]


def test_symmetry_verification_requires_a_conserved_quantity():
    with_symmetry = call(server.recommend_error_mitigation, 11, 0.002, 0.0, True)
    without = call(server.recommend_error_mitigation, 11, 0.002, 0.0, False)
    assert "symmetry_verification" in with_symmetry["apply_in_order"]
    assert "symmetry_verification" not in without["apply_in_order"]


def test_readout_mitigation_only_when_readout_error_present():
    none = call(server.recommend_error_mitigation, 11, 0.002, 0.0, True)
    some = call(server.recommend_error_mitigation, 11, 0.002, 0.02, True)
    assert "readout_error_mitigation" not in none["apply_in_order"]
    assert "readout_error_mitigation" in some["apply_in_order"]


# ------------------------------------------ estimate_circuit_error_ceiling
def test_ceiling_grows_with_noise_and_depth():
    low = call(server.estimate_circuit_error_ceiling, 11, 50, 0.002)
    high = call(server.estimate_circuit_error_ceiling, 11, 50, 0.03)
    deep = call(server.estimate_circuit_error_ceiling, 55, 50, 0.002)
    assert high["error_ceiling_any_observable"] > low["error_ceiling_any_observable"]
    assert deep["error_ceiling_any_observable"] > low["error_ceiling_any_observable"]


def test_ceiling_reports_fidelity_between_zero_and_one():
    result = call(server.estimate_circuit_error_ceiling, 11, 50, 0.002)
    assert 0.0 <= result["estimated_fidelity"] <= 1.0


# -------------------------------------------------------------- input handling
def test_bad_geometry_returns_a_helpful_error():
    result = json.loads(server.analyze_molecule("H 0 0", 2))
    assert "error" in result
    assert "SYMBOL X Y Z" in result["error"]


def test_oversized_system_is_refused_not_hung():
    """Exact diagonalisation is exponential; the guard must reject early."""
    chain = "; ".join(f"H {i} 0 0" for i in range(12))
    result = json.loads(server.analyze_molecule(chain, 12))
    assert "error" in result
    assert "limit" in result["error"].lower()


def test_geometry_accepts_newlines():
    newline_form = call(server.analyze_molecule, "H 0 0 0\nH 0 0 0.74", 2)
    semicolon_form = call(server.analyze_molecule, H2, 2)
    assert (
        newline_form["exact_energy_hartree"] == semicolon_form["exact_energy_hartree"]
    )
