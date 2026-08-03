"""
vqe_h2.py
----------
VQE (Variational Quantum Eigensolver) for the Hydrogen molecule (H2).

What this does:
    Finds the ground state energy of H2 — the lowest energy configuration
    of two hydrogen atoms bonded together. This is the "hello world" of
    quantum chemistry and the first step toward Jorge's receptor-ligand work.

The physics:
    H2 has a known ground state energy of -1.137 Hartree (exact classical answer).
    VQE should get close to this on a simulator, and reasonably close on real hardware.

The approach:
    1. Define the H2 Hamiltonian as a sum of Pauli operators (standard STO-3G basis)
    2. Build a parameterized ansatz circuit (hardware-efficient, 2 qubits)
    3. Use COBYLA classical optimizer to minimize ⟨ψ(θ)|H|ψ(θ)⟩
    4. Iterate until energy converges → ground state found

Usage:
    # Free — runs on local simulator
    python experiments/vqe_h2.py

    # Real IBM hardware (costs ~1-2 QPU minutes)
    python experiments/vqe_h2.py --real

    # Get results from a previously submitted real job
    python experiments/vqe_h2.py --results <job_id>
"""

import os
import sys
import argparse
import numpy as np
from scipy.optimize import minimize

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator  # local, free simulator

# ── H2 Hamiltonian (STO-3G basis, Jordan-Wigner transform, 2 qubits) ─────────
# This is the standard textbook Hamiltonian for H2 at equilibrium bond length.
# Ground state energy = -1.137 Hartree (exact).
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -1.0523732),
    ("IZ",  0.39793742),
    ("ZI", -0.39793742),
    ("ZZ", -0.01128010),
    ("XX",  0.18093119),
])

EXACT_GROUND_STATE = -1.857275  # Hartree — electronic ground state of this 2-qubit Hamiltonian
# Note: the familiar -1.137 Hartree is the *total* energy (electronic + nuclear repulsion).
# This Hamiltonian already has the nuclear repulsion subtracted out separately.

# ── Ansatz circuit ────────────────────────────────────────────────────────────

def build_ansatz(params: np.ndarray) -> QuantumCircuit:
    """
    Hardware-efficient ansatz for H2 (2 qubits, 4 parameters).

    Structure:
        RY(θ0) on qubit 0
        RY(θ1) on qubit 1
        CNOT (entangle)
        RY(θ2) on qubit 0
        RY(θ3) on qubit 1

    This is simple enough to run on noisy hardware and expressive enough
    to capture the H2 ground state.
    """
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    return qc


# ── Simulator VQE ─────────────────────────────────────────────────────────────

def run_simulator():
    """Run VQE on a free local statevector simulator."""
    print(f"\n{'='*60}")
    print("VQE for H2 Molecule — Local Simulator (free)")
    print(f"{'='*60}")
    print(f"Target ground state energy: {EXACT_GROUND_STATE} Hartree (electronic only)")
    print(f"Hamiltonian terms: {len(H2_HAMILTONIAN)} Pauli operators")
    print(f"Ansatz: 2-qubit hardware-efficient (4 parameters)\n")

    estimator = StatevectorEstimator()
    iteration_count = [0]
    energy_history = []

    def cost_fn(params):
        """Compute ⟨ψ(θ)|H|ψ(θ)⟩ — the energy expectation value."""
        qc = build_ansatz(params)
        result = estimator.run([(qc, H2_HAMILTONIAN)]).result()
        energy = result[0].data.evs.real
        energy_history.append(energy)
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]:3d}: energy = {energy:.6f} Hartree")
        return energy

    # Random starting parameters
    np.random.seed(42)
    x0 = np.random.uniform(-np.pi, np.pi, 4)
    print(f"Starting parameters: {np.round(x0, 3)}")
    print(f"Starting energy: {cost_fn(x0):.6f} Hartree\n")
    print("Optimizing...")

    result = minimize(cost_fn, x0, method="COBYLA",
                      options={"maxiter": 200, "rhobeg": 0.5})

    final_energy = result.fun
    error = abs(final_energy - EXACT_GROUND_STATE)

    print(f"\n{'='*60}")
    print(f"VQE Result")
    print(f"{'='*60}")
    print(f"Iterations:          {iteration_count[0]}")
    print(f"Converged:           {result.success}")
    print(f"VQE energy:          {final_energy:.6f} Hartree")
    print(f"Exact ground state:  {EXACT_GROUND_STATE:.6f} Hartree")
    print(f"Error:               {error:.6f} Hartree ({error*1000:.2f} mHartree)")
    print(f"Optimal parameters:  {np.round(result.x, 4)}")

    if error < 0.01:
        print(f"\n✅ Chemical accuracy achieved! (error < 10 mHartree)")
        print(f"   This means we found the H2 ground state to chemical precision.")
    elif error < 0.05:
        print(f"\n⚠️  Close but not chemical accuracy. Try more iterations.")
    else:
        print(f"\n❌ Did not converge well. The ansatz may need more layers.")

    print(f"\nFor Jorge:")
    print(f"  This same approach scales to receptor-ligand binding energies.")
    print(f"  H2 is 2 qubits. A small drug molecule needs ~10-20 qubits.")
    print(f"  IonQ's H2 processor (32 qubits, 99.9% fidelity) is ideal for this.")

    return result.x, final_energy


# ── Real IBM hardware VQE ─────────────────────────────────────────────────────

def run_real_hardware(backend_name: str = None):
    """
    Submit VQE to real IBM hardware.
    WARNING: costs ~1-2 QPU minutes from your free plan.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    from qiskit_ibm_runtime import EstimatorOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        print("ERROR: IBM_QUANTUM_TOKEN not set in .env")
        sys.exit(1)

    # First run simulator to get optimal parameters (free)
    print("Step 1: Running simulator to find optimal parameters (free)...")
    optimal_params, sim_energy = run_simulator()

    print(f"\n{'='*60}")
    print("Step 2: Running optimal circuit on real IBM hardware")
    print(f"{'='*60}")

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backends = [b for b in service.backends(operational=True)
                    if b.num_qubits >= 2]
        backend = min(backends, key=lambda b: b.status().pending_jobs)

    print(f"Backend: {backend.name}")

    # Build circuit with optimal parameters from simulator
    qc = build_ansatz(optimal_params)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    isa_hamiltonian = H2_HAMILTONIAN.apply_layout(isa_circuit.layout)

    print(f"Circuit depth (after transpile): {isa_circuit.depth()}")

    # Submit single evaluation at optimal point
    estimator = Estimator(backend)
    job = estimator.run([(isa_circuit, isa_hamiltonian)])
    job_id = job.job_id()

    print(f"\nJob submitted! ID: {job_id}")
    print(f"Simulator found energy: {sim_energy:.6f} Hartree")
    print(f"\nTo get results:")
    print(f"  python experiments/vqe_h2.py --results {job_id}")
    print(f"\nExpected: real hardware will be close to {sim_energy:.4f} Hartree")
    print(f"          (noise will push it slightly above the exact value)")


def get_results(job_id: str):
    """Retrieve real hardware VQE result."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.getenv("IBM_QUANTUM_TOKEN")
    )
    job = service.job(job_id)
    result = job.result()
    energy = result[0].data.evs.real

    print(f"\n{'='*60}")
    print(f"VQE H2 — Real Hardware Results (job {job_id})")
    print(f"{'='*60}")
    print(f"Hardware energy:     {energy:.6f} Hartree")
    print(f"Exact ground state:  {EXACT_GROUND_STATE:.6f} Hartree")
    print(f"Error:               {abs(energy - EXACT_GROUND_STATE):.6f} Hartree")

    if abs(energy - EXACT_GROUND_STATE) < 0.1:
        print(f"\n✅ Real hardware VQE succeeded!")
        print(f"   Quantum computer found H2 ground state energy on real hardware.")
    else:
        print(f"\n⚠️  Hardware noise affected the result.")
        print(f"   This is expected — noise pushes energy above the exact value.")
        print(f"   IonQ trapped ions would give a cleaner result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQE for H2 molecule")
    parser.add_argument("--real", action="store_true",
                        help="Run on real IBM hardware (costs QPU minutes)")
    parser.add_argument("--backend", help="Specific IBM backend name")
    parser.add_argument("--results", metavar="JOB_ID",
                        help="Get results from a previously submitted job")
    args = parser.parse_args()

    if args.results:
        get_results(args.results)
    elif args.real:
        run_real_hardware(args.backend)
    else:
        run_simulator()
