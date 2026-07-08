"""
phase4_grover_v2.py
-------------------
Phase 4 v2: IDEA 4 (search-space folding) applied to oracle only.

IDEA 4 — Search-space folding (Gemini safe)
  All 3003 rows in 0-127 have q4=q5=0 — classical fact.
  Remove q4, q5 checks from the oracle entirely.
  Oracle shared condition: q1=q2=q3=1 (only 2 RCCX instead of 4 RCCX + 1 CCX).
  Oracle now marks M=12 rows (instead of exact M=3).
  For M=12, N=128: optimal k=2 iterations → sin²(5×17.83°) ≈ 100% probability.

  Savings vs v1:
    Oracle:   37 CX/iter → 19 CX/iter  (RCCX×4 + CCX→RCCX×2 only)
    Diffusion: unchanged (still uses 4-ancilla RCCX chain from v1)
    Total 2-iter: 2×(19+18)=74 CX vs v1 1-iter 37 CX.
    But 2 iterations is near-perfect for M=12.

IDEA 3 — Phase-scheduled Grover (NOT implemented)
  Replacing π phase with tuned RZ(φ) would save more gates but requires
  redesigning the oracle+diffusion math together. Complex — left for v3.

Usage:
  python3 experiments/phase4_grover_v2.py              # 3-stage simulation
  python3 experiments/phase4_grover_v2.py --transpile  # gate count on hardware
  python3 experiments/phase4_grover_v2.py --smoke      # 1-iter hardware run
  python3 experiments/phase4_grover_v2.py --submit     # 2-iter hardware run
  python3 experiments/phase4_grover_v2.py --results ID
"""

import os, sys, argparse
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RCCXGate
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

# ── Constants ─────────────────────────────────────────────────────────────────

N_DATA  = 7
N_ANC   = 4            # still need 4 for the diffusion RCCX chain
N_TOTAL = N_DATA + N_ANC   # 11 qubits (same as v1, but oracle is cheaper)

MARKED_ROWS = [14, 15, 78]
MARKED_BITS = ["0001110", "0001111", "1001110"]

# IDEA 4: lossy oracle marks M=12 rows:
#   8 rows with q1=q2=q3=1 AND q6=0   (any q0/q4/q5)
#   4 rows with q1=q2=q3=1 AND q6=1 AND q0=0
# For M=12, N=128: θ=17.83°, k=2 gives sin²(5θ)≈100% → near-perfect amplification!
PHANTOM_M   = 12
N_SEARCH    = 128
N_ITER_OPT  = 2

SHOTS_SMOKE = 1024
SHOTS_FULL  = 4096
BACKEND_NAME = "ibm_marrakesh"

Q0,Q1,Q2,Q3,Q4,Q5,Q6 = range(7)
D = list(range(7))
A0,A1,A2,A3 = 7, 8, 9, 10   # all 4 ancilla (A0,A1 used by oracle; all 4 by diffusion)


# ── IDEA 4: Lossy Oracle ──────────────────────────────────────────────────────

def build_oracle(qc):
    """
    Lossy oracle — skips q4, q5 checks (saves 2 RCCX + 1 CCX per oracle).

    Shared condition = q3=q2=q1=1 (computed into A1 via 2 RCCX).
    A1 is then used as phase kickback target — same H|1>=|-> trick as v1.

    Term A: marks when A1=1 AND q6=0  →  rows {14,15,30,31,46,47,62,63}
    Term B: marks when A1=1 AND q6=1 AND q0=0  →  rows {78,94,110,126}

    NOTE: Term B uses CCX (not RCCX) because RCCX has relative phases that
    corrupt the |-> eigenstate trick. Phase kickback always requires true CCX/CX.
    """
    # Compute shared condition q1=q2=q3=1 into A1 (2 RCCX, vs 4 RCCX + CCX in v1)
    qc.append(RCCXGate(), [Q1, Q2, A0])   # A0 = q1 AND q2
    qc.append(RCCXGate(), [A0, Q3, A1])   # A1 = q1 AND q2 AND q3

    # Term A: phase -1 when A1=1 AND q6=0
    qc.x(Q6)
    qc.h(A1)
    qc.cx(Q6, A1)      # CX: A1 in |-> → phase kickback -1 when Q6=1 (orig Q6=0)
    qc.h(A1)
    qc.x(Q6)

    # Term B: phase -1 when A1=1 AND q6=1 AND q0=0  (must use CCX, not RCCX)
    qc.x(Q0)
    qc.h(A1)
    qc.ccx(Q6, Q0, A1)
    qc.h(A1)
    qc.x(Q0)

    # Uncompute shared condition (exact reverse — restores A1, A0 to |0>)
    qc.append(RCCXGate(), [A0, Q3, A1])
    qc.append(RCCXGate(), [Q1, Q2, A0])


# ── Diffusion (same as v1) ────────────────────────────────────────────────────

def build_diffusion(qc):
    """
    7-qubit diffusion — unchanged from v1.
    Uses A0,A1,A2,A3 in RCCX chain to build 6-bit condition → CCX(A3,Q5,Q6).
    Q6 is phase target (H|1>=|->). RCCX safe here (compute then uncompute cancels phases).
    CCX for the last step (actual kick).
    """
    qc.h(D); qc.x(D); qc.h(Q6)
    qc.append(RCCXGate(), [Q0, Q1, A0])
    qc.append(RCCXGate(), [A0, Q2, A1])
    qc.append(RCCXGate(), [A1, Q3, A2])
    qc.append(RCCXGate(), [A2, Q4, A3])
    qc.ccx(A3, Q5, Q6)
    qc.append(RCCXGate(), [A2, Q4, A3])
    qc.append(RCCXGate(), [A1, Q3, A2])
    qc.append(RCCXGate(), [A0, Q2, A1])
    qc.append(RCCXGate(), [Q0, Q1, A0])
    qc.h(Q6); qc.x(D); qc.h(D); qc.barrier()


# ── Full circuit ──────────────────────────────────────────────────────────────

def build_circuit(n_iter=2):
    qc = QuantumCircuit(N_TOTAL, N_DATA)
    qc.h(D)
    qc.barrier()
    for _ in range(n_iter):
        build_oracle(qc)
        build_diffusion(qc)
    qc.measure(D, range(N_DATA))
    return qc


def decompose_for_sim(qc):
    """Decompose RCCX/CCX to basis gates Aer understands."""
    return transpile(qc, basis_gates=["cx", "u", "x", "h", "z", "s", "t", "measure"])


# ── Stage A: Oracle unit test ─────────────────────────────────────────────────

def verify_oracle():
    """
    Lossy oracle marks 12 rows: 8 with q6=0 + 4 with q6=1,q0=0 (all need q1=q2=q3=1).
    Verify 3 targets are marked, row 79 (q6=1,q0=1) is NOT.
    """
    print("\n── Stage A: Oracle Verification (lossy, M=12) ──────────")
    errors = 0
    marked_count = 0

    for row in range(N_SEARCH):
        bits = format(row, f'0{N_DATA}b')
        qc_test = QuantumCircuit(N_TOTAL)
        for i, bit in enumerate(reversed(bits)):
            if bit == '1':
                qc_test.x(D[i])
        build_oracle(qc_test)
        sv = Statevector(decompose_for_sim(qc_test))
        amp = sv[row]
        phase = (amp / abs(amp)).real if abs(amp) > 1e-9 else 1.0
        did_flip = phase < -0.5

        # Lossy oracle marks: q1=q2=q3=1 AND NOT(q6=1 AND q0=1)
        q0 = (row >> 0) & 1
        q1 = (row >> 1) & 1
        q2 = (row >> 2) & 1
        q3 = (row >> 3) & 1
        q6 = (row >> 6) & 1
        lossy_should = (q1==1 and q2==1 and q3==1 and not (q6==1 and q0==1))

        if did_flip:
            marked_count += 1
        if did_flip != lossy_should:
            errors += 1
            print(f"  ❌ Row {row:3d} ({bits}): phase={phase:+.2f}  expected={lossy_should}")
        elif row in MARKED_ROWS or row == 79:
            tag = " ← 3003 target" if row in MARKED_ROWS else " ← row 79 (correctly unmarked)"
            print(f"  Row {row:3d} ({bits}): phase={phase:+.2f}  ✅{tag}")

    print(f"\n  Total marked rows: {marked_count}  (expected 12)")
    print(f"\n  ✅ Oracle correct" if errors == 0 else f"\n  ❌ {errors} errors")
    return errors == 0


# ── Stage B: 1-iteration simulation ───────────────────────────────────────────

def verify_one_iter():
    print("\n── Stage B: 1-Iteration Simulation ─────────────────────")
    qc = build_circuit(n_iter=1)
    counts = AerSimulator().run(decompose_for_sim(qc), shots=8192).result().get_counts()
    total = sum(counts.values())

    marked_total = sum(counts.get(b, 0) for b in MARKED_BITS)
    pct = round(marked_total / total * 100, 1)
    amp = round(pct / (100 / N_SEARCH * len(MARKED_ROWS)), 2)

    print(f"  Targets (3 rows): {pct}%  |  {amp}× amplification")
    for bits in MARKED_BITS:
        print(f"  Row {int(bits,2):3d}: {round(counts.get(bits,0)/total*100,1)}%")
    print(f"\n  {'✅ Signal present.' if amp > 1.5 else '❌ Weak signal.'}")
    return amp > 1.5


# ── Stage C: Full simulation (2 iterations) ───────────────────────────────────

def verify_full():
    print("\n── Stage C: Full 2-Iteration Simulation ───────────────")
    qc = build_circuit(n_iter=N_ITER_OPT)
    counts = AerSimulator().run(decompose_for_sim(qc), shots=8192).result().get_counts()
    total = sum(counts.values())

    for bits, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12]:
        row = int(bits, 2)
        pct = round(count / total * 100, 1)
        tag = " ← 3003!" if row in MARKED_ROWS else ""
        print(f"  Row {row:3d} ({bits}): {pct:5.1f}% {'█'*int(pct)}{tag}")

    marked_total = sum(counts.get(b, 0) for b in MARKED_BITS)
    pct = round(marked_total / total * 100, 1)
    random_pct = round(100 / N_SEARCH * len(MARKED_ROWS), 1)
    amp = round(pct / random_pct, 2)
    print(f"\n  Amplification: {amp}×  (random = 1.0×)")
    print(f"  (12 phantom rows also amplified — our 3 targets get ~pct/12 each)")
    print(f"\n  {'✅ Ready for hardware.' if amp > 1.5 else '❌ Investigate.'}")
    return amp > 1.5


# ── Gate count estimate ───────────────────────────────────────────────────────

def count_gates(n_iter=2):
    qc = build_circuit(n_iter=n_iter)
    dec = transpile(qc, basis_gates=["cx","u","x","h","z","s","t","measure"])
    ops = dict(dec.count_ops())
    total = sum(ops.values())
    print(f"\n── Gate count  n_iter={n_iter} ────────────────────────────")
    print(f"  Logical: {dict(qc.count_ops())}")
    print(f"  Basis gates: {ops}")
    print(f"  CX: {ops.get('cx',0)}   Total: {total}")
    print(f"  (v1 1-iter=624 total, v1 4-iter≈2687 total)")


# ── Transpile on hardware ─────────────────────────────────────────────────────

def transpile_check(backend, n_iter=2):
    print(f"\n── Transpile  backend={backend.name}  n_iter={n_iter} ───")
    qc = build_circuit(n_iter=n_iter)
    best_total, best_cz, best_isa = float("inf"), 0, None
    print(f"  {'opt':>4}  {'seed':>5}  {'CZ':>5}  {'total':>7}")
    print(f"  {'-'*28}")
    for level in [2, 3]:
        for seed in range(50):
            pm = generate_preset_pass_manager(
                backend=backend, optimization_level=level, seed_transpiler=seed)
            isa = pm.run(qc)
            ops = isa.count_ops()
            total = sum(ops.values())
            cz = ops.get("cz", 0)
            if total < best_total or seed % 10 == 0:
                tag = " ← best" if total < best_total else ""
                print(f"  {level:>4}  {seed:>5}  {cz:>5,}  {total:>7,}{tag}")
            if total < best_total:
                best_total, best_cz, best_isa = total, cz, isa
    print(f"\n  Best: {best_total} gates  {best_cz} CZ  (v1 smoke=624, v3=103)")
    return best_isa, best_total


# ── Submit ────────────────────────────────────────────────────────────────────

def submit_job(backend, isa, shots, label):
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    options = SamplerOptions()
    options.default_shots = shots
    options.dynamical_decoupling.enable = True
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True
    job = SamplerV2(backend, options=options).run([isa])
    job_id = job.job_id()
    print(f"\n  ✅ Submitted: {label}")
    print(f"  Job ID: {job_id}")
    print(f"  python3 experiments/phase4_grover_v2.py --results {job_id}")
    return job_id


# ── Results ───────────────────────────────────────────────────────────────────

def get_results(job_id):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=os.getenv("IBM_QUANTUM_TOKEN"))
    result = service.job(job_id).result()[0].data
    field = list(vars(result).keys())[0]
    counts = getattr(result, field).get_counts()
    total = sum(counts.values())

    print(f"\n{'='*60}")
    print("Phase 4 v2  |  7-qubit Grover  |  Lossy oracle  |  ibm_marrakesh")
    print(f"{'='*60}\n")
    for bits, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        row = int(bits, 2)
        pct = round(count / total * 100, 1)
        tag = " ← 3003!" if row in MARKED_ROWS else ""
        print(f"  Row {row:3d} ({bits}): {count:4d}  ({pct:5.1f}%) {'█'*int(pct)}{tag}")

    marked_pct = round(sum(counts.get(b, 0) for b in MARKED_BITS) / total * 100, 1)
    amp = round(marked_pct / (100 / N_SEARCH * len(MARKED_ROWS)), 2)
    print(f"\n  Amplification: {amp}×")
    print(f"  Phase 3v3: 4.17×  |  Phase 4 v1: 3.04×  |  Phase 4 v2: {amp}×")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transpile",   action="store_true")
    parser.add_argument("--gates",       action="store_true")
    parser.add_argument("--smoke",       action="store_true")
    parser.add_argument("--submit",      action="store_true")
    parser.add_argument("--results",     metavar="JOB_ID")
    parser.add_argument("--backend",     default=BACKEND_NAME)
    args = parser.parse_args()

    if args.results:
        get_results(args.results)
        sys.exit(0)

    print(f"\nPhase 4 v2  |  Lossy oracle (skip q4,q5)  |  M=12  |  k=2 iter")
    print(f"11 qubits  |  Oracle: 19 CX/iter  |  Diffusion: 18 CX/iter\n")

    if not verify_oracle():   sys.exit(1)
    if not verify_one_iter(): sys.exit(1)
    if not verify_full():     sys.exit(1)

    if args.gates:
        count_gates(n_iter=2)

    if args.transpile or args.smoke or args.submit:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=os.getenv("IBM_QUANTUM_TOKEN"))
        backend = service.backend(args.backend)
        n_iter = 1 if args.smoke else N_ITER_OPT
        best_isa, best_total = transpile_check(backend, n_iter=n_iter)
        if args.smoke:
            submit_job(backend, best_isa, SHOTS_SMOKE, "SMOKE v2 — 1 iter lossy")
        elif args.submit:
            submit_job(backend, best_isa, SHOTS_FULL, "FULL v2 — 2 iter lossy")
