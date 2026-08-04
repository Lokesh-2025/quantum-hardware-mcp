"""
Quantum chemistry planning tools (qforge)
=========================================
The tools in server.py answer "what hardware exists, and is my circuit valid?".
These answer the question a chemist actually starts with: "I have this
molecule -- can I run it, by what method, and what will it cost?"

Everything is computed from geometry using the qforge library in this repo.
No lookup tables of answers.

Tools:
  - analyze_molecule               : qubits, Hamiltonian terms, Schmidt rank, floors
  - plan_quantum_chemistry_run     : circuits and cost per Schmidt rank, vs a budget
  - recommend_error_mitigation     : which techniques help, from measurements not theory
  - estimate_circuit_error_ceiling : upper bound on observable error from gate counts
  - build_forged_circuits          : emit OPENQASM, self-checked on a simulator first
  - run_forged_energy              : submit those circuits to real hardware
  - collect_forged_energy          : reconstruct the energy once the jobs finish

This module is imported for its side effects: importing it registers the tools
on the shared FastMCP instance from mcp_app.
"""

import json

import numpy as np

from mcp_app import mcp

# Guard rails. Exact diagonalisation cost grows as 2^n, so a careless request
# would hang the server rather than return an error.
_QFORGE_MAX_QUBITS = 16
_QFORGE_MAX_ATOMS = 10


def _load_qforge():
    """Import qforge on demand.

    Kept lazy so the server still starts (and every other tool still works) if
    the optional chemistry dependencies are missing.
    """
    try:
        from qforge import chemistry, diagnostics, forging, grouping
        return chemistry, diagnostics, forging, grouping
    except ImportError as exc:
        raise RuntimeError(
            f"qforge unavailable ({exc}). Install with: pip install qiskit-nature"
        ) from exc


def _parse_geometry(atoms: str):
    """Parse 'H 0 0 0; H 0 0 0.74' into [("H", (0.0, 0.0, 0.0)), ...].

    Accepts ';' or newline between atoms. Positions are in Angstrom.
    """
    geometry = []
    for chunk in atoms.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if len(parts) != 4:
            raise ValueError(
                f"cannot parse atom {chunk!r} -- expected 'SYMBOL X Y Z', "
                "e.g. 'H 0 0 0.74'"
            )
        symbol, x, y, z = parts
        geometry.append((symbol, (float(x), float(y), float(z))))
    if not geometry:
        raise ValueError("no atoms given")
    if len(geometry) > _QFORGE_MAX_ATOMS:
        raise ValueError(
            f"{len(geometry)} atoms exceeds the {_QFORGE_MAX_ATOMS}-atom limit "
            "for exact solving; use fragmentation for larger systems"
        )
    return geometry


def _build_and_solve(atoms: str, n_electrons: int):
    """Shared front half: geometry -> Hamiltonian -> exact ground state."""
    chemistry, _diagnostics, forging, _grouping = _load_qforge()
    geometry = _parse_geometry(atoms)
    molecule = chemistry.build_molecule(geometry, n_electrons=n_electrons)
    if molecule.n_qubits > _QFORGE_MAX_QUBITS:
        raise ValueError(
            f"{molecule.n_qubits} qubits exceeds the {_QFORGE_MAX_QUBITS}-qubit "
            "limit for exact diagonalisation on this server"
        )
    energy, psi = molecule.exact_ground_state()

    # Remove the arbitrary global phase eigensolvers return. In the resulting
    # real gauge every matrix element is real, which halves the number of
    # phase circuits entanglement forging needs.
    pivot = int(np.argmax(np.abs(psi)))
    psi = (psi * np.exp(-1j * np.angle(psi[pivot]))).real.astype(complex)
    return molecule, energy, psi


@mcp.tool()
def analyze_molecule(atoms: str, n_electrons: int) -> str:
    """
    Work out what a molecule costs to simulate on a quantum computer.

    Computes the qubit Hamiltonian from geometry, then reports how far
    entanglement forging and Pauli grouping can cut the problem down.

    Args:
        atoms: geometry in Angstrom, e.g. "H 0 0 0; H 0 0 0.74"
               (separate atoms with ';' or newlines)
        n_electrons: total electrons, e.g. 2 for H2

    Returns JSON with qubit counts before and after forging, the Schmidt
    spectrum (which determines how much truncation is affordable), and the
    number of measurement circuits needed per state preparation.
    """
    try:
        chemistry, _diag, forging, grouping = _load_qforge()
        molecule, energy, psi = _build_and_solve(atoms, n_electrons)

        schmidt = forging.schmidt_decompose(psi, molecule.n_qubits)
        terms = forging.split_pauli_terms(molecule.hamiltonian)
        half = molecule.n_qubits // 2
        alpha_labels = sorted({a for a, _, _ in terms})

        general = grouping.build_measurement_groups(alpha_labels, half, qubit_wise=False)
        qubit_wise = grouping.build_measurement_groups(alpha_labels, half, qubit_wise=True)

        # accuracy floor at each truncation rank -- purely classical, so it
        # tells you the BEST possible result before hardware noise is added
        alpha_mats = forging.matrix_elements(schmidt.alpha_vectors, alpha_labels, schmidt.rank)
        beta_mats = forging.matrix_elements(
            schmidt.beta_vectors, [b for _, b, _ in terms], schmidt.rank
        )
        floors = {}
        for rank in range(1, min(schmidt.rank, 6) + 1):
            approx = forging.forged_energy(
                terms, schmidt.coefficients, alpha_mats, beta_mats,
                molecule.nuclear_repulsion, rank,
            )
            floors[rank] = round(abs(approx - energy) * 627.5094740631, 4)

        return json.dumps({
            "geometry": atoms,
            "n_electrons": n_electrons,
            "exact_energy_hartree": round(float(energy), 6),
            "hartree_fock_energy": round(float(molecule.hf_energy), 6),
            "correlation_energy_hartree": round(
                float(energy - molecule.hf_energy - 0.0), 6
            ),
            "qubits_direct": molecule.n_qubits,
            "qubits_with_forging": half,
            "hamiltonian_terms": len(molecule.hamiltonian),
            "unique_pauli_labels_per_register": len(alpha_labels),
            "measurement_circuits_qubit_wise": len(qubit_wise),
            "measurement_circuits_general_commuting": len(general),
            "schmidt_rank": schmidt.rank,
            "schmidt_coefficients": [round(float(c), 5) for c in schmidt.coefficients[:8]],
            "halves_symmetric": bool(schmidt.is_symmetric()),
            "accuracy_floor_kcal_mol_by_rank": floors,
            "note": (
                "accuracy_floor is the classical truncation error at each Schmidt "
                "rank -- the best achievable before any hardware noise. "
                "Chemical accuracy is 1.0 kcal/mol."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def plan_quantum_chemistry_run(atoms: str, n_electrons: int,
                               budget_usd: float,
                               price_per_circuit: float = 25.79) -> str:
    """
    Given a molecule and a budget, recommend how to actually run it.

    Answers the practical question: what is the most accurate result I can buy
    for this much money? Works through every Schmidt rank, applies the circuit
    optimisations, and reports which options fit.

    Args:
        atoms: geometry in Angstrom, e.g. "H 0 0 0; H 0 0 0.74"
        n_electrons: total electrons
        budget_usd: available credits, in dollars. Required -- there is no
            sensible default, and guessing one produces confident
            recommendations the caller cannot actually afford.
        price_per_circuit: hardware price per circuit. 25.79 is IonQ Forte's
            measured per-circuit floor for shallow circuits; check current
            rates rather than trusting this default.

    Returns JSON with a ranked plan, the recommended option, and what error
    mitigation to apply.
    """
    try:
        _chem, _diag, forging, grouping = _load_qforge()
        molecule, energy, psi = _build_and_solve(atoms, n_electrons)

        schmidt = forging.schmidt_decompose(psi, molecule.n_qubits)
        terms = forging.split_pauli_terms(molecule.hamiltonian)
        half = molecule.n_qubits // 2
        alpha_labels = sorted({a for a, _, _ in terms})
        groups = grouping.build_measurement_groups(alpha_labels, half, qubit_wise=False)
        n_bases = len(groups)

        alpha_mats = forging.matrix_elements(schmidt.alpha_vectors, alpha_labels, schmidt.rank)
        beta_mats = forging.matrix_elements(
            schmidt.beta_vectors, [b for _, b, _ in terms], schmidt.rank
        )

        options = []
        for rank in range(1, min(schmidt.rank, 8) + 1):
            # real gauge: 2 phase circuits per Schmidt pair, not 4
            preps = rank + 2 * (rank * (rank - 1) // 2)
            circuits = preps * n_bases
            cost = circuits * price_per_circuit
            approx = forging.forged_energy(
                terms, schmidt.coefficients, alpha_mats, beta_mats,
                molecule.nuclear_repulsion, rank,
            )
            options.append({
                "schmidt_rank": rank,
                "state_preparations": preps,
                "circuits": circuits,
                "estimated_cost_usd": round(cost, 2),
                "accuracy_floor_kcal_mol": round(abs(approx - energy) * 627.5094740631, 4),
                "fits_budget": bool(cost <= budget_usd),
                # NOT a prediction that a real run lands within chemical
                # accuracy. This compares the CLASSICAL truncation floor
                # against 1.0 kcal/mol and ignores hardware noise entirely.
                # Measured H4 runs on IonQ came back 125-135 kcal/mol off
                # while this floor sat at 0.57, so the two differ by more
                # than two orders of magnitude. Named for what it measures.
                "classical_floor_below_chemical_accuracy": bool(
                    abs(approx - energy) * 627.5094740631 <= 1.0
                ),
            })

        affordable = [o for o in options if o["fits_budget"]]
        best = min(affordable, key=lambda o: o["accuracy_floor_kcal_mol"]) if affordable else None

        return json.dumps({
            "molecule": {
                "geometry": atoms,
                "exact_energy_hartree": round(float(energy), 6),
                "qubits_direct": molecule.n_qubits,
                "qubits_with_forging": half,
                "measurement_bases": n_bases,
            },
            "budget_usd": budget_usd,
            "options": options,
            "recommended": best,
            "recommendation_note": (
                None if best is None else
                f"Schmidt rank {best['schmidt_rank']}: {best['circuits']} circuits, "
                f"~${best['estimated_cost_usd']:,.0f}, floor "
                f"{best['accuracy_floor_kcal_mol']} kcal/mol"
            ),
            "warning": (
                "No option fits this budget." if best is None else None
            ),
            "cost_assumptions": (
                "Assumes a flat per-circuit price, which holds only while circuits "
                "stay below the hardware's gate-count pricing threshold (~23 "
                "two-qubit gates on IonQ Forte). Zero-noise extrapolation folds "
                "circuits and typically pushes past that, costing several times "
                "more -- price ZNE separately with real folded gate counts."
            ),
            "accuracy_assumptions": (
                "accuracy_floor_kcal_mol is the CLASSICAL Schmidt-truncation "
                "error only. It is a floor, not a forecast: it assumes a "
                "noiseless device. Real hardware adds error on top, and on H4 "
                "that gap was measured at 125-135 kcal/mol against a 0.57 "
                "floor. Use recommend_error_mitigation and "
                "estimate_circuit_error_ceiling to bound the noise term before "
                "reading any of these floors as an achievable result."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def recommend_error_mitigation(two_qubit_gates: int,
                               two_qubit_error_rate: float = 0.002,
                               readout_error_rate: float = 0.0,
                               conserves_particle_number: bool = True) -> str:
    """
    Recommend which error-mitigation techniques are worth applying.

    Based on measured results from this project, including the ones that did
    NOT work -- so you skip techniques that cost circuits and return nothing.

    Args:
        two_qubit_gates: two-qubit gate count in the circuit
        two_qubit_error_rate: per-gate error, e.g. 0.002 for trapped ion,
            0.03 for typical superconducting
        readout_error_rate: measurement bit-flip probability (0 if unknown)
        conserves_particle_number: True for chemistry circuits, where symmetry
            verification is available

    Returns JSON with ranked recommendations and expected circuit overhead.
    """
    try:
        circuit_error = 1.0 - (1.0 - two_qubit_error_rate) ** two_qubit_gates
        recommendations = []

        recommendations.append({
            "technique": "readout_error_mitigation",
            "recommended": readout_error_rate > 0.001,
            "circuit_overhead": "none (classical post-processing)",
            "expected_benefit": "2.5-3.5x error reduction, measured",
            "why": (
                "Free to apply and stacks with everything else."
                if readout_error_rate > 0.001
                else "Readout error negligible or unspecified; little to gain."
            ),
        })

        recommendations.append({
            "technique": "symmetry_verification",
            "recommended": bool(conserves_particle_number),
            "circuit_overhead": "none (discards invalid shots, ~1% loss)",
            "expected_benefit": "2-4x error reduction, measured",
            "why": (
                "Shots landing on the wrong particle number are provably wrong "
                "and free to discard. ONLY valid in the computational basis -- "
                "X/Y rotations do not conserve particle number, and "
                "postselecting after them made an energy ~40x worse."
                if conserves_particle_number
                else "No conserved quantity available to check against."
            ),
        })

        needs_zne = circuit_error > 0.01
        recommendations.append({
            "technique": "zero_noise_extrapolation",
            "recommended": bool(needs_zne),
            "circuit_overhead": "3x circuits, and folded circuits are 3-5x deeper",
            "expected_benefit": "~20 kcal/mol -> 0.57 kcal/mol, measured on H4",
            "why": (
                "Circuit error is high enough that raw results will miss "
                "chemical accuracy."
                if needs_zne
                else "Circuit is shallow enough that raw results may suffice; "
                     "try unmitigated first and save the cost."
            ),
            "cost_warning": (
                "Folding multiplies noise by 2*folds-1, not folds -- 3 folds "
                "means 5x the gates. On gate-billed hardware this dominates "
                "cost, typically ~85% of a full experiment."
            ),
        })

        recommendations.append({
            "technique": "pauli_twirling",
            "recommended": False,
            "circuit_overhead": "many randomised circuit variants",
            "expected_benefit": "none measured against depolarising noise",
            "why": (
                "Twirling converts coherent error into stochastic error. Tested "
                "here against a depolarising noise model with no improvement, "
                "converged over 600 twirls. Only worth revisiting on hardware "
                "with confirmed coherent error."
            ),
        })

        return json.dumps({
            "circuit_profile": {
                "two_qubit_gates": two_qubit_gates,
                "two_qubit_error_rate": two_qubit_error_rate,
                "estimated_circuit_error": round(circuit_error, 5),
                "estimated_fidelity": round(1.0 - circuit_error, 5),
            },
            "recommendations": recommendations,
            "apply_in_order": [
                r["technique"] for r in recommendations if r["recommended"]
            ],
            "note": (
                "Recommendations come from measurements in this project, not "
                "general theory. The negative results are included deliberately "
                "so time is not spent rediscovering them."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def estimate_circuit_error_ceiling(two_qubit_gates: int, one_qubit_gates: int = 0,
                                   two_qubit_error_rate: float = 0.002,
                                   one_qubit_error_rate: float = 0.00005) -> str:
    """
    Bound the error on ANY observable before running the circuit.

    One fidelity number bounds every measurement at once, so you can tell in
    advance whether a run can possibly reach the accuracy you need -- without
    simulating each observable separately.

    Args:
        two_qubit_gates: count of two-qubit gates
        one_qubit_gates: count of one-qubit gates
        two_qubit_error_rate: per-gate error (0.002 trapped ion, 0.03 typical
            superconducting)
        one_qubit_error_rate: per-gate error

    Returns JSON with estimated fidelity and the resulting error ceiling.
    """
    try:
        _chem, diagnostics, _forging, _grouping = _load_qforge()
        fidelity = (
            (1.0 - two_qubit_error_rate) ** two_qubit_gates
            * (1.0 - one_qubit_error_rate) ** one_qubit_gates
        )
        ceiling = diagnostics.ceiling_from_fidelity(fidelity)
        chemical_accuracy = 1.0 / 627.5094740631

        return json.dumps({
            "gates": {"two_qubit": two_qubit_gates, "one_qubit": one_qubit_gates},
            "estimated_fidelity": round(float(fidelity), 6),
            "error_ceiling_any_observable": round(float(ceiling), 6),
            "chemical_accuracy_hartree": round(chemical_accuracy, 6),
            "single_observable_within_chemical_accuracy": bool(
                ceiling <= chemical_accuracy
            ),
            "interpretation": (
                "The ceiling bounds |<P>_noisy - <P>_ideal| for every Pauli "
                "observable. It is an upper bound, typically 1.4-2.2x looser "
                "than reality, so exceeding it means the run CANNOT reach the "
                "target, while satisfying it does not guarantee success. "
                "Validated across 55 configurations without a violation."
            ),
            "caveat": (
                "Derived from a gate-count fidelity estimate, which is looser "
                "than a full density-matrix calculation (measured 2-6x looser). "
                "Use for pre-flight screening, not as a final error bar."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def build_forged_circuits(atoms: str, n_electrons: int, schmidt_rank: int = 3,
                          max_circuits: int = 60) -> str:
    """
    Build the actual runnable circuits for a forged ground-state calculation.

    Turns a molecule into OpenQASM you can submit. Every circuit is a real
    entanglement-forging circuit acting on HALF the qubits the molecule would
    otherwise need.

    Use this when you want to inspect or submit the circuits yourself. Use
    run_forged_energy to build, submit and collect in one step.

    Args:
        atoms: geometry in Angstrom, e.g. "H 0 0 0; H 0 0 0.74"
        n_electrons: total electrons
        schmidt_rank: how many Schmidt terms to keep. Higher is more accurate
            but costs circuits quadratically -- check analyze_molecule for the
            accuracy floor at each rank.
        max_circuits: refuse to build more than this many (guards against
            accidentally generating hundreds of hardware jobs).

    Returns JSON with the QASM strings, circuit count, gate statistics, and a
    local simulator self-check confirming the circuits reconstruct the right
    energy before you spend any hardware time.
    """
    try:
        _chem, _diag, _forging, _grouping = _load_qforge()
        from qforge import experiment as qforge_experiment

        geometry = _parse_geometry(atoms)
        built = qforge_experiment.build_experiment(
            geometry, n_electrons, rank=schmidt_rank
        )
        if built.n_circuits > max_circuits:
            return json.dumps({
                "error": (
                    f"{built.n_circuits} circuits exceeds max_circuits="
                    f"{max_circuits}. Lower schmidt_rank or raise the limit "
                    "deliberately."
                ),
                "circuits_required": built.n_circuits,
                "schmidt_rank": built.rank,
            })

        passed, self_check_error = qforge_experiment.self_check(built)
        two_qubit = [
            sum(1 for inst in c.data if inst.operation.name == "cx")
            for c in built.circuits
        ]

        return json.dumps({
            "molecule": {
                "geometry": atoms,
                "exact_energy_hartree": round(float(built.exact_energy), 6),
                "qubits_direct": built.molecule.n_qubits,
                "qubits_per_circuit": built.molecule.n_qubits // 2,
            },
            "schmidt_rank": built.rank,
            "accuracy_floor_kcal_mol": round(built.accuracy_floor_kcal_mol, 4),
            "circuits": built.n_circuits,
            "measurement_bases": len(built.groups),
            "two_qubit_gates_per_circuit": {
                "min": min(two_qubit),
                "median": int(np.median(two_qubit)),
                "max": max(two_qubit),
            },
            "self_check": {
                "passed": bool(passed),
                "simulator_error_kcal_mol": round(float(self_check_error), 4),
                "note": (
                    "The full pipeline was replayed on a local simulator and "
                    "reconstructed the energy. A failure here means the "
                    "circuits are wrong -- do not submit them."
                ),
            },
            "qasm": [
                qforge_experiment.to_qasm(c) for c in built.circuits
            ],
            "next_step": (
                "Submit each QASM string with submit_job, keep the job_ids in "
                "order, then pass them to collect_forged_energy. Or call "
                "run_forged_energy to do all of it."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def run_forged_energy(atoms: str, n_electrons: int, device_name: str,
                      schmidt_rank: int = 2, shots: int = 1024,
                      max_circuits: int = 20) -> str:
    """
    Run a molecular ground-state calculation on real quantum hardware.

    Builds the forged circuits, submits every one to the named device, and
    returns the job IDs. Collect the energy afterwards with
    collect_forged_energy once the jobs finish.

    This submits REAL JOBS that consume queue time and, on paid hardware,
    money. It refuses to run unless the circuits pass a local simulator
    self-check first, and caps the number of jobs at max_circuits.

    Args:
        atoms: geometry in Angstrom, e.g. "H 0 0 0; H 0 0 0.74"
        n_electrons: total electrons
        device_name: target machine, e.g. "ibm_fez". Pick one with
            compare_devices or queue_status.
        schmidt_rank: Schmidt terms to keep. Rank 2 is exact for H2 and needs
            only a handful of circuits -- a sensible first hardware run.
        shots: shots per circuit (default 1024)
        max_circuits: hard cap on submitted jobs (default 20). Raise it
            deliberately, having checked the count with build_forged_circuits.

    Returns JSON with the ordered job IDs and everything
    collect_forged_energy needs.
    """
    try:
        _chem, _diag, _forging, _grouping = _load_qforge()
        from qforge import experiment as qforge_experiment

        geometry = _parse_geometry(atoms)
        built = qforge_experiment.build_experiment(
            geometry, n_electrons, rank=schmidt_rank
        )

        if built.n_circuits > max_circuits:
            return json.dumps({
                "error": (
                    f"would submit {built.n_circuits} separate jobs, above "
                    f"max_circuits={max_circuits}. Each job queues separately, "
                    "so start small: schmidt_rank=2 on H2 needs only a few."
                ),
                "circuits_required": built.n_circuits,
                "suggestion": "lower schmidt_rank, or raise max_circuits deliberately",
            })

        # Never submit circuits that cannot reconstruct the right answer in
        # simulation -- a wrong sign or frame would burn queue time for nothing.
        passed, self_check_error = qforge_experiment.self_check(built)
        if not passed:
            return json.dumps({
                "error": "circuits failed the local simulator self-check",
                "simulator_error_kcal_mol": round(float(self_check_error), 4),
                "note": "Not submitting. This indicates a bug, not device noise.",
            })

        job_ids = []
        for position, circuit in enumerate(built.circuits):
            qasm = qforge_experiment.to_qasm(circuit)
            response = json.loads(submit_job(device_name, qasm, shots=shots))
            if "error" in response:
                return json.dumps({
                    "error": f"submission failed at circuit {position}: "
                             f"{response['error']}",
                    "job_ids_submitted_so_far": job_ids,
                    "note": "Earlier jobs are already queued; cancel them if unwanted.",
                })
            job_ids.append(response["job_id"])

        return json.dumps({
            "status": "submitted",
            "device": device_name,
            "geometry": atoms,
            "n_electrons": n_electrons,
            "schmidt_rank": built.rank,
            "shots": shots,
            "circuits_submitted": len(job_ids),
            "job_ids": job_ids,
            "exact_energy_hartree": round(float(built.exact_energy), 6),
            "accuracy_floor_kcal_mol": round(built.accuracy_floor_kcal_mol, 4),
            "self_check_error_kcal_mol": round(float(self_check_error), 4),
            "next_step": (
                "Wait for all jobs to reach DONE (job_status), then call "
                "collect_forged_energy with these job_ids IN THIS ORDER."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def collect_forged_energy(atoms: str, n_electrons: int, job_ids: str,
                          schmidt_rank: int = 2) -> str:
    """
    Turn finished quantum jobs back into a molecular energy.

    Fetches the counts for each job and reconstructs the ground-state energy,
    then compares it against the exact classical answer.

    The job IDs must be in the SAME ORDER run_forged_energy returned them --
    each one corresponds to a specific state preparation and measurement
    basis, so reordering them scrambles the result.

    Args:
        atoms: same geometry used for the run
        n_electrons: same electron count
        job_ids: comma-separated job IDs, in submission order
        schmidt_rank: same rank used for the run

    Returns JSON with the measured energy, the exact reference, and the error
    in kcal/mol (chemical accuracy is 1.0).
    """
    try:
        _chem, _diag, _forging, _grouping = _load_qforge()
        from qforge import experiment as qforge_experiment

        ids = [j.strip() for j in job_ids.split(",") if j.strip()]
        geometry = _parse_geometry(atoms)
        built = qforge_experiment.build_experiment(
            geometry, n_electrons, rank=schmidt_rank
        )

        if len(ids) != built.n_circuits:
            return json.dumps({
                "error": (
                    f"got {len(ids)} job IDs but this experiment needs "
                    f"{built.n_circuits}. The geometry, electron count and "
                    "schmidt_rank must match the original run exactly."
                ),
            })

        counts_per_circuit = []
        for position, job_id in enumerate(ids):
            response = json.loads(job_results(job_id))
            if "counts" not in response:
                return json.dumps({
                    "error": (
                        f"job {job_id} (position {position}) has no counts yet: "
                        f"{response.get('status', response.get('error'))}"
                    ),
                    "note": "All jobs must be DONE before collecting.",
                })
            counts_per_circuit.append(response["counts"])

        energy = qforge_experiment.reconstruct_energy(built, counts_per_circuit)
        error_kcal = abs(energy - built.exact_energy) * 627.5094740631

        return json.dumps({
            "geometry": atoms,
            "schmidt_rank": built.rank,
            "circuits_used": built.n_circuits,
            "measured_energy_hartree": round(float(energy), 6),
            "exact_energy_hartree": round(float(built.exact_energy), 6),
            "error_kcal_mol": round(float(error_kcal), 4),
            "accuracy_floor_kcal_mol": round(built.accuracy_floor_kcal_mol, 4),
            "reached_chemical_accuracy": bool(error_kcal <= 1.0),
            "interpretation": (
                "error_kcal_mol is the total; accuracy_floor_kcal_mol is the "
                "part from Schmidt truncation alone. The difference is "
                "hardware noise. Raising schmidt_rank lowers the floor; error "
                "mitigation lowers the noise -- see recommend_error_mitigation."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


