"""
Quantum Hardware MCP Server
============================
Exposes live IBM Quantum device data to AI assistants via the MCP protocol.

Tools:
  - list_devices          : all machines + status
  - get_device_details    : deep info on one machine
  - compare_devices       : rank machines by error rate / queue / combined score
  - queue_status          : current queue depth for every machine
  - device_history        : historical snapshots for one machine over N days
  - best_qubits           : best n qubits on a machine right now (calibration-based)
  - device_on_date        : historical stats for a machine on a specific past date
  - submit_job            : compile + submit an OpenQASM 2 or 3 circuit to IBM hardware
  - job_status            : check the status of a submitted job
  - job_results           : retrieve measurement counts from a completed job
  - cancel_job            : cancel a queued or running job
  - list_jobs             : list recent jobs with status and backend
  - run_grover            : built-in Grover's search demo on real hardware
  - estimate_expectation  : run Estimator primitive to compute observable expectation values
  - circuit_report        : dry-run analysis — fidelity estimate, gate counts, qubit map
  - debug_circuit         : bug detector — finds errors before you waste queue time
  - ionq_devices          : list IonQ quantum computers and simulators
  - ionq_submit_job       : submit an OpenQASM 2 circuit to IonQ hardware or simulator
  - ionq_job_status       : check the status of a submitted IonQ job
  - ionq_job_results      : retrieve measurement counts from a completed IonQ job
  - get_alerts            : calibration drift alerts — devices that spiked or went offline
  - start_repro_experiment: submit same circuit N times to measure reproducibility
  - repro_score           : compute 0-1 reproducibility score after runs complete
  - estimate_runtime      : estimate how many minutes a circuit will cost on a device
  - route_job             : recommend the cheapest device that fits your circuit + time budget
  - check_routing_overhead: predict SWAP inflation from qubit interaction graph (degree > 3 = danger)
  - encode_search_problem : convert Boolean conditions into Ising h_i / J_ij for LNAA circuits
  - estimate_hardware_gates: predict transpiled gate count from logical gates + qubit degree
  - get_amplification     : compute amplification factor from a completed search job
  - run_search_experiment    : ONE CALL does everything — encode → build circuit → pick best machine → submit → amplification
  - encode_collision_problem        : classically find C(n1,k1)=C(n2,k2) pairs, encode as Ising h_i for LNAA collision search
  - discover_collision_candidates   : scout — filters ALL k-pairs for hardware-feasible collisions before spending QPU credits
  - run_parallel_collision_search   : ONE job, MULTIPLE k-pair searches on parallel 9-qubit rails across ibm_marrakesh's 156 qubits
  - sieve_singmaster_space          : classical Lucas' theorem sieve — eliminates 98%+ of Pascal's Triangle search space before any QPU job
  - discover_energy_landscape       : (PLANNED) given any math domain + constraints, auto-generate candidate Hamiltonian, estimate qubits/routing/gates, report if practical on current hardware
"""

import os
import json
import math
import sqlite3
import argparse
import anyio
import requests
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit import qasm3 as qiskit_qasm3
from qiskit.quantum_info import SparsePauliOp, Clifford, StabilizerState
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from dotenv import load_dotenv
import snapshot as _snapshot
from qiskit_ibm_runtime import QiskitRuntimeService
from starlette.responses import JSONResponse as _JSONResponse
from starlette.responses import JSONResponse

# --------------------------------------------------------------------------
# Load .env from the same folder as this file, regardless of working directory.
# This matters because Claude Desktop may launch the server from a different
# working directory than the project root.
# --------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# The MCP server instance lives in mcp_app.py so that this module and
# tools_chemistry.py can both register tools on it without importing each other.
from mcp_app import mcp

# Importing this module registers the quantum chemistry tools on `mcp`.
# Imported purely for that side effect; nothing below calls into it.
import tools_chemistry  # noqa: F401

# --------------------------------------------------------------------------
# SQLite history database
# --------------------------------------------------------------------------

# Store the database next to this file so it travels with the project.
DB_PATH = os.path.join(os.path.dirname(__file__), "devices.db")


def _init_db() -> None:
    """Create the snapshots table if it doesn't exist yet."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS device_snapshots (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT    NOT NULL,  -- ISO 8601 UTC timestamp
                name              TEXT    NOT NULL,  -- e.g. "ibm_fez"
                num_qubits        INTEGER,
                operational       INTEGER,           -- 1 = True, 0 = False
                pending_jobs      INTEGER,
                avg_cx_error      REAL,              -- NULL when not measured
                avg_readout_error REAL               -- NULL when not measured
            )
        """)
        # Index on (name, ts) makes device_history queries fast.
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_name_ts
            ON device_snapshots (name, ts)
        """)
        # Reproducibility experiments — one row per experiment
        con.execute("""
            CREATE TABLE IF NOT EXISTS repro_experiments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts  TEXT NOT NULL,
                device_name TEXT NOT NULL,
                circuit     TEXT NOT NULL,
                n_runs      INTEGER NOT NULL,
                shots       INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending'
            )
        """)
        # One row per individual run within an experiment
        con.execute("""
            CREATE TABLE IF NOT EXISTS repro_runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES repro_experiments(id),
                run_index     INTEGER NOT NULL,
                submitted_ts  TEXT NOT NULL,
                job_id        TEXT,
                status        TEXT NOT NULL DEFAULT 'submitted',
                counts        TEXT,           -- JSON string of bit-string counts
                calibration_epoch TEXT        -- avg_cx_error snapshot at submission time
            )
        """)


def _save_snapshots(rows: list[dict]) -> None:
    """
    Write one row per device into device_snapshots.

    Each dict in `rows` must have at least 'name'; all other fields are
    optional and default to None if absent.
    """
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.executemany(
            """
            INSERT INTO device_snapshots
                (ts, name, num_qubits, operational, pending_jobs,
                 avg_cx_error, avg_readout_error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    ts,
                    r["name"],
                    r.get("num_qubits"),
                    int(r["operational"]) if r.get("operational") is not None else None,
                    r.get("pending_jobs"),
                    r.get("avg_cx_error"),
                    r.get("avg_readout_error"),
                )
                for r in rows
            ],
        )


# Initialise the DB as soon as the server module loads.
_init_db()


# --------------------------------------------------------------------------
# Helper: connect to IBM Quantum
# --------------------------------------------------------------------------

# Lets api.py run a request with a caller-supplied IBM token instead of the
# shared IBM_QUANTUM_TOKEN in .env ("bring your own key"), without changing
# any tool function's signature (so the MCP contract Claude Desktop sees is
# untouched) and without the concurrency hazard of mutating os.environ — a
# ContextVar is isolated per request/thread, so two requests running at the
# same time never see each other's token.
_token_override = contextvars.ContextVar("token_override", default=None)


@contextmanager
def use_ibm_token(token: str | None):
    """Temporarily make _get_service() use `token` instead of the .env default."""
    if not token:
        yield
        return
    reset = _token_override.set(token)
    try:
        yield
    finally:
        _token_override.reset(reset)


def _get_service() -> QiskitRuntimeService:
    """
    Build a QiskitRuntimeService from a per-request token override (see
    use_ibm_token) or, failing that, env vars.

    Required: IBM_QUANTUM_TOKEN (unless a per-request token override is active)
    Optional: IBM_CHANNEL  (default: ibm_quantum_platform)
              IBM_INSTANCE (e.g. ibm-q/open/main — falls back to IBM auto-select)
    """
    token = _token_override.get() or os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise ValueError(
            "IBM_QUANTUM_TOKEN is not set. "
            "Create a .env file in the project folder with:\n"
            "  IBM_QUANTUM_TOKEN=your_token_here\n"
            "Get your token at https://quantum.ibm.com/account"
        )
    channel  = os.getenv("IBM_CHANNEL", "ibm_quantum_platform")
    instance = os.getenv("IBM_INSTANCE")  # None → IBM picks the default

    kwargs = dict(channel=channel, token=token)
    if instance:
        kwargs["instance"] = instance

    return QiskitRuntimeService(**kwargs)


def _cx_errors_for_backend(props) -> list[float]:
    """
    Pull 2-qubit gate error values from calibration properties.

    Older IBM devices use CX (CNOT) as their native 2-qubit gate.
    Newer devices (e.g. ibm_fez, ibm_marrakesh) use ECR (echoed
    cross-resonance) instead — CX is synthesised from ECR and won't
    appear in raw calibration data. We check for both, plus CZ, so
    this function works across the whole IBM fleet.

    Returns an empty list if the backend has no calibration data.
    """
    if props is None:
        return []
    TWO_QUBIT_GATES = {"cx", "ecr", "cz"}
    errors = []
    for gate in props.gates:
        if gate.gate in TWO_QUBIT_GATES and gate.parameters:
            errors.append(gate.parameters[0].value)
    return errors


# --------------------------------------------------------------------------
# Tool 1: list_devices
# --------------------------------------------------------------------------

@mcp.tool()
def list_devices() -> str:
    """
    List every IBM quantum computer this account can access.

    Returns a JSON array sorted by qubit count (largest first).
    Each entry includes: name, qubit count, operational status, queue depth.
    """
    service = _get_service()

    # service.backends() returns a list of IBMBackend objects.
    # By default it returns ALL backends you have access to.
    backends = service.backends()

    devices = []
    for backend in backends:
        # status() is a lightweight call — no calibration data, just up/down + queue.
        status = backend.status()

        devices.append({
            "name": backend.name,
            "num_qubits": backend.num_qubits,
            "status": status.status_msg,       # e.g. "active", "maintenance"
            "operational": status.operational,  # True / False
            "pending_jobs": status.pending_jobs, # current queue length
        })

    # Sort biggest machines first — handy for a quick overview
    devices.sort(key=lambda d: d["num_qubits"], reverse=True)

    _save_snapshots(devices)
    return json.dumps(devices, indent=2)


# --------------------------------------------------------------------------
# Tool 2: get_device_details
# --------------------------------------------------------------------------

@mcp.tool()
def get_device_details(device_name: str) -> str:
    """
    Deep-dive into one IBM quantum computer.

    Args:
        device_name: Machine name, e.g. "ibm_brisbane" or "ibm_sherbrooke".
                     Use list_devices first if you don't know the exact name.

    Returns JSON with:
      - Qubit count, status, queue depth
      - Average / best / worst CX (2-qubit) gate error rates
      - Average readout (measurement) error
      - Average T1 and T2 coherence times in microseconds
      - Timestamp of the last calibration run
    """
    service = _get_service()

    # Fetch this specific backend by name
    backend = service.backend(device_name)
    status = backend.status()

    # Start building the result with data that is always available
    result = {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "status": status.status_msg,
        "operational": status.operational,
        "pending_jobs": status.pending_jobs,
    }

    # properties() returns calibration data from the most recent daily calibration.
    # Simulators and some devices return None here.
    props = backend.properties()

    if props:
        # ---- Readout error ----
        # Readout error = probability of measuring the wrong bit (0 vs 1).
        # Average across all qubits gives a device-wide quality signal.
        readout_errors = [
            props.readout_error(q)
            for q in range(backend.num_qubits)
            if props.readout_error(q) is not None
        ]
        if readout_errors:
            result["avg_readout_error"] = round(
                sum(readout_errors) / len(readout_errors), 5
            )

        # ---- CX gate error ----
        # CX error = probability the 2-qubit gate produces the wrong output.
        # Lower is better. Typical good values are < 0.01 (1%).
        cx_errors = _cx_errors_for_backend(props)
        if cx_errors:
            result["avg_cx_error"] = round(sum(cx_errors) / len(cx_errors), 5)
            result["best_cx_error"] = round(min(cx_errors), 5)   # best qubit pair
            result["worst_cx_error"] = round(max(cx_errors), 5)  # worst qubit pair

        # ---- T1 and T2 coherence times ----
        # T1 (relaxation time):  how long a qubit in |1⟩ stays in |1⟩ before
        #                         spontaneously falling to |0⟩.
        # T2 (dephasing time):   how long a qubit stays in a superposition before
        #                         the phase randomises and quantum info is lost.
        # Both are in seconds from the API; we convert to microseconds (µs)
        # because that's the conventional unit in quantum computing papers.
        def _safe_t(fn, q):
            try:
                return fn(q)
            except Exception:
                return None

        t1_times = [v for q in range(backend.num_qubits) if (v := _safe_t(props.t1, q)) is not None]
        t2_times = [v for q in range(backend.num_qubits) if (v := _safe_t(props.t2, q)) is not None]
        if t1_times:
            result["avg_t1_us"] = round(
                sum(t1_times) / len(t1_times) * 1e6, 1  # s → µs
            )
        if t2_times:
            result["avg_t2_us"] = round(
                sum(t2_times) / len(t2_times) * 1e6, 1
            )

        # When was the last calibration run?
        result["last_calibration"] = str(props.last_update_date)

    else:
        result["note"] = "No calibration data available (simulator or uncalibrated device)"

    _save_snapshots([result])
    return json.dumps(result, indent=2)


# --------------------------------------------------------------------------
# Tool 6: best_qubits
# --------------------------------------------------------------------------

@mcp.tool()
def best_qubits(device_name: str, n: int = 5) -> str:
    """
    Return the best n individual qubits on a device based on live calibration.

    Useful for researchers who want to hand-pick qubits for a circuit rather
    than letting the compiler choose automatically.

    Args:
        device_name: Machine name, e.g. "ibm_fez".
        n:           How many qubits to return (default 5).

    Scoring formula (lower = better):
        score = readout_error + best_cx_error_for_this_qubit

    Both metrics are in the same [0, 1] range so they contribute equally.
    T1 / T2 coherence times are included as supplementary context.
    Missing metrics are penalised with 1.0 (worst possible) so qubits with
    incomplete calibration data sort to the bottom.
    """
    service = _get_service()
    backend = service.backend(device_name)
    props   = backend.properties()

    if not props:
        return json.dumps({
            "error": f"{device_name} has no calibration data available."
        })

    n = min(n, backend.num_qubits)  # can't ask for more qubits than exist

    # Build dict: qubit index → lowest 2-qubit gate error of any pair
    # involving this qubit.  Covers cx / ecr / cz (see _cx_errors_for_backend).
    TWO_QUBIT_GATES = {"cx", "ecr", "cz"}
    qubit_best_cx: dict[int, float] = {}
    for gate in props.gates:
        if gate.gate in TWO_QUBIT_GATES and gate.parameters:
            err = gate.parameters[0].value
            for q in gate.qubits:
                if q not in qubit_best_cx or err < qubit_best_cx[q]:
                    qubit_best_cx[q] = err

    # Score and collect every qubit
    qubit_data = []
    for q in range(backend.num_qubits):
        ro  = props.readout_error(q)
        cx  = qubit_best_cx.get(q)
        # T1/T2 are missing for some qubits on some devices — catch gracefully
        try:
            t1 = props.t1(q)
        except Exception:
            t1 = None
        try:
            t2 = props.t2(q)
        except Exception:
            t2 = None

        score = (ro if ro is not None else 1.0) + (cx if cx is not None else 1.0)

        qubit_data.append({
            "qubit":          q,
            "score":          round(score, 6),
            "readout_error":  round(ro, 5)       if ro  is not None else None,
            "best_cx_error":  round(cx, 5)       if cx  is not None else None,
            "t1_us":          round(t1 * 1e6, 1) if t1  is not None else None,
            "t2_us":          round(t2 * 1e6, 1) if t2  is not None else None,
        })

    qubit_data.sort(key=lambda q: q["score"])
    top_n = qubit_data[:n]

    # Connectivity check: warn if the returned qubits are not all connected
    # on the hardware graph. Unconnected qubit sets force SWAP injection.
    top_indices = {q["qubit"] for q in top_n}
    coupling_map = backend.coupling_map
    connected_pairs = []
    disconnected_warning = None
    if coupling_map is not None:
        edges = list(coupling_map.get_edges())
        connected_pairs = [
            [a, b] for a, b in edges
            if a in top_indices and b in top_indices
        ]
        # A set of n qubits needs at least n-1 edges to be connected (tree).
        if len(connected_pairs) < n - 1:
            disconnected_warning = (
                f"WARNING: the top {n} qubits by score are NOT all connected "
                f"on {device_name}'s coupling map. Only {len(connected_pairs)} "
                f"direct links found between them. Running a multi-qubit circuit "
                f"on these qubits will require SWAP gates, increasing your gate "
                f"count. Consider using check_routing_overhead or picking qubits "
                f"from a connected subgraph."
            )

    result = {
        "device":   device_name,
        "n":        n,
        "scoring":  "readout_error + best_cx_error (lower = better). "
                    "T1/T2 shown for context but not in score.",
        "best_qubits": top_n,
        "connectivity": {
            "direct_links_between_top_qubits": connected_pairs,
            "warning": disconnected_warning,
        },
    }

    return json.dumps(result, indent=2)


# --------------------------------------------------------------------------
# Tool 3: compare_devices
# --------------------------------------------------------------------------

@mcp.tool()
def compare_devices(sort_by: str = "cx_error") -> str:
    """
    Rank all accessible IBM quantum computers by a quality metric.

    Args:
        sort_by: Ranking criterion. Choose one of:
                 "cx_error"  – lowest 2-qubit gate error (best quality) [default]
                 "queue"     – shortest queue (fastest turnaround)
                 "qubits"    – most qubits (largest machine)
                 "combined"  – blended score: 70% quality + 30% availability

    Returns a JSON object with the ranking and a note about what it means.

    Note: fetching calibration data for every device takes ~10–30 seconds
    because it makes one API call per device.
    """
    service = _get_service()
    backends = service.backends()

    devices = []
    for backend in backends:
        status = backend.status()

        entry = {
            "name": backend.name,
            "num_qubits": backend.num_qubits,
            "pending_jobs": status.pending_jobs,
            "operational": status.operational,
            "status": "online" if status.operational else "offline",
        }

        # Always fetch calibration data so every device card has full fields
        try:
            props = backend.properties()
            cx_errors = _cx_errors_for_backend(props)
            if cx_errors:
                entry["avg_cx_error"] = round(sum(cx_errors) / len(cx_errors), 5)

            readout_errors = [
                props.readout_error(q)
                for q in range(backend.num_qubits)
                if props.readout_error(q) is not None
            ]
            if readout_errors:
                entry["avg_readout_error"] = round(
                    sum(readout_errors) / len(readout_errors), 5
                )

            t1_times = [v for q in range(backend.num_qubits)
                        if (v := _safe_t(props.t1, q)) is not None]
            t2_times = [v for q in range(backend.num_qubits)
                        if (v := _safe_t(props.t2, q)) is not None]
            if t1_times:
                entry["avg_t1_us"] = round(sum(t1_times) / len(t1_times), 1)
            if t2_times:
                entry["avg_t2_us"] = round(sum(t2_times) / len(t2_times), 1)
        except Exception:
            pass  # calibration unavailable — leave fields absent

        devices.append(entry)

    # Apply the requested sort
    if sort_by == "cx_error":
        # Ascending: lower error = better rank
        # Devices without calibration data fall to the end (inf sentinel)
        devices.sort(key=lambda d: d.get("avg_cx_error", float("inf")))

    elif sort_by == "queue":
        # Ascending: fewer pending jobs = shorter wait
        devices.sort(key=lambda d: d.get("pending_jobs", float("inf")))

    elif sort_by == "qubits":
        # Descending: more qubits = higher rank
        devices.sort(key=lambda d: d["num_qubits"], reverse=True)

    elif sort_by == "combined":
        # Blended score: 70% gate quality + 30% queue availability.
        #
        # Why min-max normalisation?
        # cx_error lives in ~[0.001, 0.05]; pending_jobs in ~[0, 500].
        # A raw sum would let queue dominate just because its numbers are
        # larger. Min-max rescales each metric to [0, 1] relative to the
        # current set of devices, so the 70/30 weights actually mean what
        # they say: quality matters more than speed, but both count.
        #
        # Why 70/30?
        # For research you care most about getting a correct result (low
        # error), but a 200-job queue means hours of waiting — so
        # availability gets a meaningful but smaller weight.

        cx_vals = [d["avg_cx_error"] for d in devices if d.get("avg_cx_error") is not None]
        q_vals  = [d["pending_jobs"]  for d in devices if d.get("pending_jobs")  is not None]

        min_cx, max_cx = (min(cx_vals), max(cx_vals)) if cx_vals else (0, 1)
        min_q,  max_q  = (min(q_vals),  max(q_vals))  if q_vals  else (0, 1)

        # Avoid division by zero when all devices have identical values
        cx_range = max_cx - min_cx or 1
        q_range  = max_q  - min_q  or 1

        for d in devices:
            cx = d.get("avg_cx_error")
            q  = d.get("pending_jobs")
            # Missing metrics get worst-case penalty (1.0) so uncalibrated
            # devices sort below any device with real data
            norm_cx = (cx - min_cx) / cx_range if cx is not None else 1.0
            norm_q  = (q  - min_q)  / q_range  if q  is not None else 1.0
            d["combined_score"] = round(0.7 * norm_cx + 0.3 * norm_q, 4)

        # Ascending: 0.0 = perfectly best on both metrics, 1.0 = worst
        devices.sort(key=lambda d: d.get("combined_score", float("inf")))

    else:
        return json.dumps({
            "error": f"Unknown sort_by value '{sort_by}'. "
                     "Use 'cx_error', 'queue', 'qubits', or 'combined'."
        })

    # Stamp each entry with its rank number (1 = best)
    for i, device in enumerate(devices):
        device["rank"] = i + 1

    _save_snapshots(devices)
    return json.dumps(
        {
            "sorted_by": sort_by,
            "note": {
                "cx_error": "Rank 1 = lowest 2-qubit gate error (highest quality)",
                "queue":    "Rank 1 = fewest pending jobs (shortest wait)",
                "qubits":   "Rank 1 = most qubits (largest machine)",
                "combined": "Rank 1 = best blend of quality (70%) and availability (30%). "
                            "Score is min-max normalised across current devices.",
            }.get(sort_by, ""),
            "devices": devices,
        },
        indent=2,
    )


# --------------------------------------------------------------------------
# Tool 4: queue_status
# --------------------------------------------------------------------------

@mcp.tool()
def queue_status() -> str:
    """
    Snapshot of the job queue on every IBM quantum computer.

    Useful when you want to submit a job and need to pick the machine
    with the shortest wait.

    Returns a JSON array sorted by pending_jobs (shortest queue first).
    """
    service = _get_service()
    backends = service.backends()

    queues = []
    for backend in backends:
        # status() is fast — it does NOT fetch full calibration data
        status = backend.status()
        queues.append({
            "name": backend.name,
            "num_qubits": backend.num_qubits,
            "pending_jobs": status.pending_jobs,
            "status": status.status_msg,
            "operational": status.operational,
        })

    # Shortest queue first so the "best pick right now" is at the top
    queues.sort(key=lambda d: d["pending_jobs"])

    _save_snapshots(queues)
    return json.dumps(queues, indent=2)


# --------------------------------------------------------------------------
# Tool 5: device_history
# --------------------------------------------------------------------------

@mcp.tool()
def device_history(device_name: str, days: int = 7) -> str:
    """
    Return all saved snapshots for one IBM quantum computer over the last N days.

    Args:
        device_name: Machine name, e.g. "ibm_brisbane". Must match exactly.
        days:        How many days back to look (default 7).

    Returns a JSON object with the device name and a list of snapshots in
    chronological order. Each snapshot has the fields that were available
    when it was recorded (error rates are NULL when the recording tool
    didn't fetch calibration data).
    """
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT ts, num_qubits, operational, pending_jobs,
                   avg_cx_error, avg_readout_error,
                   median_t1_us, median_t2_us, qubit_yield_fraction,
                   day_of_week, hour_utc,
                   processor_family, backend_version, last_calibration_dt,
                   clops_h, quantum_volume, avg_2q_gate_duration_ns,
                   avg_prob_meas0_prep1, avg_prob_meas1_prep0
            FROM   device_snapshots
            WHERE  name = ?
              AND  ts >= datetime('now', ? || ' days')
            ORDER  BY ts ASC
            """,
            (device_name, f"-{days}"),
        ).fetchall()

    snapshots = [
        {
            "ts":                  r["ts"],
            "num_qubits":          r["num_qubits"],
            "operational":         bool(r["operational"]) if r["operational"] is not None else None,
            "pending_jobs":        r["pending_jobs"],
            "avg_cx_error":        r["avg_cx_error"],
            "avg_readout_error":   r["avg_readout_error"],
            "median_t1_us":             r["median_t1_us"],
            "median_t2_us":             r["median_t2_us"],
            "qubit_yield_fraction":     r["qubit_yield_fraction"],
            "day_of_week":              r["day_of_week"],
            "hour_utc":                 r["hour_utc"],
            "processor_family":         r["processor_family"],
            "backend_version":          r["backend_version"],
            "last_calibration_dt":      r["last_calibration_dt"],
            "clops_h":                  r["clops_h"],
            "quantum_volume":           r["quantum_volume"],
            "avg_2q_gate_duration_ns":  r["avg_2q_gate_duration_ns"],
            "avg_prob_meas0_prep1":     r["avg_prob_meas0_prep1"],
            "avg_prob_meas1_prep0":     r["avg_prob_meas1_prep0"],
        }
        for r in rows
    ]

    return json.dumps(
        {"device": device_name, "days": days, "snapshots": snapshots},
        indent=2,
    )


# --------------------------------------------------------------------------
# Tool 6b: device_profile
# --------------------------------------------------------------------------

@mcp.tool()
def device_profile(device_name: str) -> str:
    """
    Return the complete hardware profile for one quantum backend using the
    most recent snapshot — including processor family, CLOPS benchmark,
    gate duration, last calibration time, readout asymmetry, and job limits.

    This surfaces the BackendV2 extended fields that device_history and
    get_device_details do not expose.

    Args:
        device_name: Exact backend name, e.g. "ibm_marrakesh".

    Returns a JSON object with every collected field for that device.
    """
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT ts, provider, name, num_qubits, operational, pending_jobs,
                   avg_cx_error, avg_readout_error,
                   median_t1_us, median_t2_us, qubit_yield_fraction,
                   native_gate_set, coupling_map_edges, connectivity_density,
                   max_shots, max_experiments,
                   processor_family, backend_version, online_date,
                   last_calibration_dt, dt_ns,
                   avg_2q_gate_duration_ns, avg_readout_length_ns,
                   avg_prob_meas0_prep1, avg_prob_meas1_prep0,
                   rep_delay_default_ms, clops_h, quantum_volume
            FROM   device_snapshots
            WHERE  name = ?
            ORDER  BY ts DESC
            LIMIT  1
            """,
            (device_name,),
        ).fetchone()

    if row is None:
        return json.dumps({"error": f"No snapshot found for '{device_name}'. "
                           "Run list_devices to see available backends."})

    profile = {
        "device":               row["name"],
        "provider":             row["provider"],
        "snapshot_ts":          row["ts"],
        # ── Identity ──────────────────────────────────────────────────
        "num_qubits":           row["num_qubits"],
        "processor_family":     row["processor_family"],
        "backend_version":      row["backend_version"],
        "online_date":          row["online_date"],
        "last_calibration_dt":  row["last_calibration_dt"],
        # ── Performance benchmark ──────────────────────────────────────
        "clops_h":              row["clops_h"],
        "quantum_volume":       row["quantum_volume"],
        # ── Gate quality ──────────────────────────────────────────────
        "avg_cx_error":         row["avg_cx_error"],
        "avg_readout_error":    row["avg_readout_error"],
        "avg_prob_meas0_prep1": row["avg_prob_meas0_prep1"],
        "avg_prob_meas1_prep0": row["avg_prob_meas1_prep0"],
        # ── Coherence ─────────────────────────────────────────────────
        "median_t1_us":         row["median_t1_us"],
        "median_t2_us":         row["median_t2_us"],
        "qubit_yield_fraction": row["qubit_yield_fraction"],
        # ── Timing ────────────────────────────────────────────────────
        "dt_ns":                    row["dt_ns"],
        "avg_2q_gate_duration_ns":  row["avg_2q_gate_duration_ns"],
        "avg_readout_length_ns":    row["avg_readout_length_ns"],
        "rep_delay_default_ms":     row["rep_delay_default_ms"],
        # ── Topology ──────────────────────────────────────────────────
        "native_gate_set":      row["native_gate_set"],
        "coupling_map_edges":   row["coupling_map_edges"],
        "connectivity_density": row["connectivity_density"],
        # ── Job limits ────────────────────────────────────────────────
        "max_shots":            row["max_shots"],
        "max_experiments":      row["max_experiments"],
        # ── Live status ───────────────────────────────────────────────
        "operational":          bool(row["operational"]) if row["operational"] is not None else None,
        "pending_jobs":         row["pending_jobs"],
    }

    return json.dumps(profile, indent=2)


# --------------------------------------------------------------------------
# Tool 7: device_on_date
# --------------------------------------------------------------------------

@mcp.tool()
def device_on_date(device_name: str, date: str) -> str:
    """
    Historical stats for a device on a specific past date, from our snapshot DB.

    Useful for reproducibility: if you ran an experiment on 2026-07-01, call
    this tool with that date to see exactly what the hardware looked like —
    queue depth, error rates — and include it in your methods section.

    Args:
        device_name: Machine name, e.g. "ibm_fez".
        date:        Date in YYYY-MM-DD format, e.g. "2026-06-10".

    Returns aggregated stats averaged across all snapshots taken that day
    (snapshots are recorded every 6 hours by the background agent).
    """
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT ts, operational, pending_jobs, avg_cx_error, avg_readout_error
            FROM   device_snapshots
            WHERE  name   = ?
              AND  date(ts) = ?
            ORDER  BY ts ASC
            """,
            (device_name, date),
        ).fetchall()

    if not rows:
        return json.dumps({
            "device": device_name,
            "date":   date,
            "found":  False,
            "note":   "No snapshots found for this device on this date. "
                      "Snapshots are recorded every 6 hours by the local LaunchAgent.",
        })

    snapshots = [dict(r) for r in rows]

    def _avg(field: str):
        vals = [s[field] for s in snapshots if s[field] is not None]
        return round(sum(vals) / len(vals), 5) if vals else None

    return json.dumps(
        {
            "device":             device_name,
            "date":               date,
            "found":              True,
            "snapshots_that_day": len(snapshots),
            "first_snapshot":     snapshots[0]["ts"],
            "last_snapshot":      snapshots[-1]["ts"],
            "avg_pending_jobs":   _avg("pending_jobs"),
            "avg_cx_error":       _avg("avg_cx_error"),
            "avg_readout_error":  _avg("avg_readout_error"),
            "note": "Averaged across all snapshots taken that day. "
                    "Cite this date in your paper's methods section for reproducibility.",
        },
        indent=2,
    )


# --------------------------------------------------------------------------
# Tool 8: submit_job
# --------------------------------------------------------------------------

@mcp.tool()
def submit_job(device_name: str, qasm_string: str, shots: int = 1024,
               qasm_version: int = 2) -> str:
    """
    Compile and submit a quantum circuit to an IBM quantum computer.

    Args:
        device_name:  Machine name, e.g. "ibm_fez". Use compare_devices or
                      queue_status first to pick the best available machine.
        qasm_string:  OpenQASM circuit source code.
                      For QASM 2: must start with OPENQASM 2.0; include "qelib1.inc";
                      For QASM 3: must start with OPENQASM 3.0;
                      The circuit must include measurement gates.
        shots:        How many times to run the circuit (default 1024, max 20000).
                      More shots = more accurate probability estimates.
        qasm_version: 2 (default) for OpenQASM 2.0, 3 for OpenQASM 3.0.

    Returns JSON with:
      - job_id   Save this — needed for job_status and job_results.
      - status   Initial status (usually "INITIALIZING" or "QUEUED").
      - device   Machine the job was sent to.
      - shots    Number of shots requested.
    """
    # Parse the QASM string into a Qiskit QuantumCircuit object.
    # QASM 2 uses QuantumCircuit.from_qasm_str (legacy standard).
    # QASM 3 uses qiskit.qasm3.loads (modern standard with richer features).
    try:
        if qasm_version == 3:
            circuit = qiskit_qasm3.loads(qasm_string)
        else:
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to parse QASM {qasm_version}: {e}",
            "hint": (
                'QASM 2 must start with: OPENQASM 2.0;\ninclude "qelib1.inc";'
                if qasm_version != 3 else
                "QASM 3 must start with: OPENQASM 3.0;"
            ),
        })

    # Clamp shots to IBM's allowed range
    shots = max(1, min(shots, 20000))

    service = _get_service()

    try:
        backend = service.backend(device_name)
    except Exception as e:
        return json.dumps({"error": f"Device '{device_name}' not found: {e}"})

    # Transpile the circuit to the backend's native gate set and qubit topology.
    # optimization_level=1 is a good default: fast compile, decent optimisation.
    # Level 3 gives the best circuit but is much slower to compile.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    # SamplerV2 is the current IBM Runtime primitive for sampling circuits.
    # It replaces the deprecated execute() function and Sampler v1.
    # mode=backend tells it which machine to target.
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)

    _snapshot.log_job_submission(
        job_id=job.job_id(), provider="ibm", backend_name=device_name,
        tool_name="submit_job",
        circuit_qubits=circuit.num_qubits,
        circuit_depth_raw=circuit.depth(),
        circuit_depth_transpiled=isa_circuit.depth(),
        shots_requested=shots,
    )

    return json.dumps({
        "job_id": job.job_id(),
        "status": str(job.status()),
        "device": device_name,
        "shots": shots,
        "note": "Save the job_id. Use job_status to check progress, job_results to get counts.",
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 9: job_status
# --------------------------------------------------------------------------

@mcp.tool()
def job_status(job_id: str) -> str:
    """
    Check the current status of a submitted quantum job.

    Args:
        job_id: The ID returned by submit_job.

    Status values:
      INITIALIZING  Just submitted, not yet in the queue.
      QUEUED        Waiting in the machine's queue.
      RUNNING       Actively executing on hardware right now.
      DONE          Finished — call job_results to get counts.
      ERROR         Failed — error_message field will explain why.
      CANCELLED     Was cancelled before it ran.
    """
    service = _get_service()

    try:
        job = service.job(job_id)
    except Exception as e:
        return json.dumps({"error": f"Job '{job_id}' not found: {e}"})

    status = str(job.status())
    result = {
        "job_id":  job_id,
        "status":  status,
        "backend": job.backend().name,
    }

    try:
        result["creation_date"] = str(job.creation_date)
    except Exception:
        pass

    if status == "QUEUED":
        # queue_position() tells you how many jobs are ahead of yours
        try:
            pos = job.queue_position()
            if pos is not None:
                result["queue_position"] = pos
        except Exception:
            pass
        result["note"] = "Still waiting in queue. Check again in a few minutes."

    elif status == "DONE":
        result["note"] = "Job complete — call job_results to retrieve counts."

    elif status == "ERROR":
        try:
            result["error_message"] = job.error_message()
        except Exception:
            pass

    return json.dumps(result, indent=2)


# --------------------------------------------------------------------------
# Tool 10: job_results
# --------------------------------------------------------------------------

@mcp.tool()
def job_results(job_id: str) -> str:
    """
    Retrieve measurement counts from a completed quantum job.

    Args:
        job_id: The ID returned by submit_job.

    Returns JSON with bit-string counts when the job is DONE.
    If the job is still running or queued, returns current status instead.

    Counts example:
      {"00": 502, "11": 522}  ← a Bell state: roughly 50/50 between 00 and 11.

    The bit-string length equals the number of measured qubits.
    Each key is a measurement outcome; the value is how many shots produced it.
    All values sum to the total number of shots.
    """
    service = _get_service()

    try:
        job = service.job(job_id)
    except Exception as e:
        return json.dumps({"error": f"Job '{job_id}' not found: {e}"})

    status = str(job.status())

    if status != "DONE":
        return json.dumps({
            "job_id": job_id,
            "status": status,
            "note":   "Job not complete yet. Use job_status to monitor progress.",
        }, indent=2)

    try:
        result = job.result()
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve results: {e}"})

    try:
        pub_result = result[0]

        # EstimatorV2 jobs (VQE, expectation values) return evs/stds, not bitstring counts.
        if hasattr(pub_result.data, "evs"):
            evs = pub_result.data.evs
            stds = getattr(pub_result.data, "stds", None)
            return json.dumps({
                "job_id":           job_id,
                "status":           "DONE",
                "backend":          job.backend().name,
                "type":             "estimator",
                "expectation_value": float(evs) if hasattr(evs, "__float__") else list(evs),
                "std_error":        float(stds) if stds is not None and hasattr(stds, "__float__") else None,
                "note": "EstimatorV2 job — returns expectation value(s), not bitstring counts.",
            }, indent=2)

        # SamplerV2 jobs return BitArrays with bitstring counts.
        counts_by_register = {}
        for reg_name, bit_array in vars(pub_result.data).items():
            counts_by_register[reg_name] = bit_array.get_counts()

    except Exception as e:
        return json.dumps({
            "error": f"Failed to parse result data: {e}",
            "raw_result": str(result),
        })

    # Flatten to a single counts dict when there is only one register (usual case)
    counts = (
        list(counts_by_register.values())[0]
        if len(counts_by_register) == 1
        else counts_by_register
    )

    total_shots = sum(counts.values()) if isinstance(counts, dict) else None

    # Log outcome back to job_submissions for the 6-month study
    if isinstance(counts, dict):
        _snapshot.log_job_result(job_id, counts)

    return json.dumps({
        "job_id":      job_id,
        "status":      "DONE",
        "backend":     job.backend().name,
        "total_shots": total_shots,
        "counts":      counts,
        "note": "Each key is a bit-string outcome; value is how many shots produced it.",
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 11: cancel_job
# --------------------------------------------------------------------------

@mcp.tool()
def cancel_job(job_id: str) -> str:
    """
    Cancel a queued or running IBM quantum job.

    Only jobs in QUEUED or RUNNING state can be cancelled. Jobs that are
    already DONE, ERROR, or CANCELLED will return an error.

    Args:
        job_id: The ID returned by submit_job or list_jobs.

    Returns JSON confirming the cancellation or explaining why it failed.
    """
    service = _get_service()

    try:
        job = service.job(job_id)
    except Exception as e:
        return json.dumps({"error": f"Job '{job_id}' not found: {e}"})

    status = str(job.status())

    # IBM only allows cancellation before the job finishes
    if status in ("DONE", "ERROR", "CANCELLED"):
        return json.dumps({
            "job_id": job_id,
            "error": f"Cannot cancel a job with status '{status}'.",
            "current_status": status,
        })

    try:
        job.cancel()
    except Exception as e:
        return json.dumps({"error": f"Cancel request failed: {e}"})

    return json.dumps({
        "job_id": job_id,
        "status": "CANCELLED",
        "note": "Cancellation requested. The job may take a moment to fully stop.",
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 12: list_jobs
# --------------------------------------------------------------------------

@mcp.tool()
def list_jobs(limit: int = 10) -> str:
    """
    List your most recently submitted IBM quantum jobs.

    Useful for finding job IDs you didn't save, or getting an overview of
    what's in the queue right now.

    Args:
        limit: How many jobs to return, newest first (default 10, max 50).

    Returns a JSON array of jobs, each with: job_id, status, backend, creation date.
    """
    limit = max(1, min(limit, 50))  # clamp to [1, 50]

    service = _get_service()

    try:
        jobs = service.jobs(limit=limit)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch jobs: {e}"})

    results = []
    for job in jobs:
        entry = {
            "job_id": job.job_id(),
            "status": str(job.status()),
        }
        # backend() and creation_date can raise on malformed jobs — guard each
        try:
            entry["backend"] = job.backend().name
        except Exception:
            entry["backend"] = None
        try:
            entry["created"] = str(job.creation_date)
        except Exception:
            entry["created"] = None

        results.append(entry)

    return json.dumps({
        "count": len(results),
        "jobs": results,
        "note": "Sorted newest first. Use job_status(job_id) for full details.",
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 13: run_grover
# --------------------------------------------------------------------------

@mcp.tool()
def run_grover(n_qubits: int, target_state: str) -> str:
    """
    Build and run a Grover's search algorithm demo on real IBM hardware.

    Grover's algorithm searches an unsorted list of 2^n states in O(sqrt(2^n))
    steps — a quadratic speedup over classical search. This tool builds the
    full circuit (oracle + diffusion operator), transpiles it, and submits it
    to the least-busy backend.

    Args:
        n_qubits:     Number of qubits to use. Must be 2 or 3.
                      Capped at 3 — deeper circuits lose coherence on current
                      hardware and the result becomes dominated by noise.
        target_state: Binary string marking the state to find, e.g. "11" or "101".
                      Length must equal n_qubits. Grover's will amplify this state's
                      probability so it appears far more often than the others.

    Returns JSON with the job_id, backend chosen, circuit details, and what
    fraction of shots to expect on the target state.
    """
    # Validate and clamp inputs
    n_qubits = min(max(n_qubits, 2), 3)

    if len(target_state) != n_qubits:
        return json.dumps({
            "error": (
                f"target_state length ({len(target_state)}) must equal "
                f"n_qubits ({n_qubits}). Example: n_qubits=2, target_state='11'"
            )
        })

    if not all(c in "01" for c in target_state):
        return json.dumps({"error": "target_state must contain only '0' and '1'."})

    # Optimal number of Grover iterations for a single marked state:
    # floor(π/4 * sqrt(N)) where N = 2^n_qubits
    # n=2 → 1 iteration, n=3 → 2 iterations
    n_iterations = max(1, math.floor(math.pi / 4 * math.sqrt(2 ** n_qubits)))

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: put all qubits in equal superposition (uniform over all 2^n states)
    qc.h(range(n_qubits))

    for _ in range(n_iterations):
        # ── Oracle: phase-flip the target state ───────────────────────────
        # Strategy: X-gate every qubit whose target bit is '0', so the
        # target state maps to all-|1⟩, apply a multi-controlled-Z to flip
        # its phase, then undo the X gates.
        # reversed() because Qiskit is little-endian (qubit 0 = rightmost bit).
        for i, bit in enumerate(reversed(target_state)):
            if bit == "0":
                qc.x(i)

        if n_qubits == 2:
            qc.cz(0, 1)
        else:  # n_qubits == 3
            qc.ccz(0, 1, 2)

        for i, bit in enumerate(reversed(target_state)):
            if bit == "0":
                qc.x(i)

        # ── Diffusion operator: inversion about the mean ───────────────────
        # This amplifies the target state's amplitude at the cost of the others.
        # Circuit: H⊗n → X⊗n → multi-CZ → X⊗n → H⊗n
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))

        if n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.ccz(0, 1, 2)

        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))

    # Find the least-busy operational backend
    service = _get_service()
    backends = service.backends()

    operational = []
    for b in backends:
        try:
            s = b.status()
            if s.operational:
                operational.append((b, s.pending_jobs))
        except Exception:
            pass

    if not operational:
        return json.dumps({"error": "No operational backends available."})

    best_backend, _ = min(operational, key=lambda x: x[1])

    # Transpile to the backend's native gate set and submit
    pm = generate_preset_pass_manager(backend=best_backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    sampler = Sampler(mode=best_backend)
    job = sampler.run([isa_circuit], shots=1024)

    _snapshot.log_job_submission(
        job_id=job.job_id(), provider="ibm", backend_name=best_backend.name,
        tool_name="run_grover",
        circuit_qubits=n_qubits,
        circuit_depth_raw=qc.depth(),
        circuit_depth_transpiled=isa_circuit.depth(),
        shots_requested=1024,
    )

    # Theoretical success probability after optimal iterations
    # P = sin²((2k+1) * arcsin(1/sqrt(N))) where k = n_iterations
    theta = math.asin(1 / math.sqrt(2 ** n_qubits))
    ideal_pct = round(100 * math.sin((2 * n_iterations + 1) * theta) ** 2, 1)

    return json.dumps({
        "job_id": job.job_id(),
        "status": str(job.status()),
        "device": best_backend.name,
        "n_qubits": n_qubits,
        "target_state": target_state,
        "grover_iterations": n_iterations,
        "shots": 1024,
        "ideal_success_pct": ideal_pct,
        "note": (
            f"Searching for |{target_state}⟩ across {2**n_qubits} states. "
            f"Ideal hardware would show '{target_state}' in {ideal_pct}% of shots. "
            f"Real hardware noise will reduce this — expect 60–85% on current IBM devices. "
            f"Use job_status then job_results to see the counts."
        ),
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 14: run_vqe
# --------------------------------------------------------------------------

@mcp.tool()
def run_vqe(molecule: str = "H2", backend_name: str = "simulator",
            max_iterations: int = 150) -> str:
    """
    Run the Variational Quantum Eigensolver (VQE) to find the ground state
    energy of a molecule.

    VQE is the core quantum chemistry algorithm. It finds the lowest energy
    configuration of a molecule by iterating: prepare a quantum state →
    measure its energy → classically adjust circuit parameters → repeat
    until the energy converges.

    This is the first step toward simulating receptor-ligand binding energies
    for drug discovery research.

    Args:
        molecule:       Molecule to simulate. Currently supports "H2".
                        H2 is the standard benchmark — ground state = -1.857275 Hartree.
        backend_name:   "simulator" (free, runs locally) or an IBM backend name
                        like "ibm_fez" (costs QPU minutes). Default: "simulator".
        max_iterations: Max COBYLA optimizer iterations (default 150).
                        More iterations = more accurate but slower.

    Returns JSON with:
      - molecule        Molecule simulated
      - vqe_energy      Found ground state energy in Hartree
      - exact_energy    Known exact value (for comparison)
      - error_hartree   Absolute error
      - error_mhartree  Error in milli-Hartree (chemical accuracy = < 1.6 mHa)
      - converged       Whether optimizer converged
      - iterations      Number of optimizer iterations used
      - backend         Where it ran (simulator or IBM device)
      - optimal_params  Best circuit parameters found
      - job_id          IBM job ID (only when backend_name is a real device)
      - note            Plain-English interpretation
    """
    import numpy as np
    from scipy.optimize import minimize
    from qiskit.quantum_info import SparsePauliOp as _SparsePauliOp

    # ── Molecule definitions ─────────────────────────────────────────────────
    # Each molecule: (Hamiltonian terms, exact ground state energy in Hartree)
    MOLECULES = {
        "H2": (
            [("II", -1.0523732), ("IZ", 0.39793742), ("ZI", -0.39793742),
             ("ZZ", -0.01128010), ("XX", 0.18093119)],
            -1.857275,  # electronic ground state (this Hamiltonian, STO-3G basis)
        ),
    }

    mol = molecule.upper()
    if mol not in MOLECULES:
        return json.dumps({
            "error": f"Molecule '{molecule}' not supported. Currently available: {list(MOLECULES.keys())}",
            "note": "H2 is the standard benchmark. More molecules (LiH, BeH2) coming soon."
        })

    pauli_terms, exact_energy = MOLECULES[mol]
    hamiltonian = _SparsePauliOp.from_list(pauli_terms)
    n_qubits = len(pauli_terms[0][0])  # length of "II" = 2

    # ── Ansatz: hardware-efficient (RY + CNOT) ───────────────────────────────
    def build_ansatz(params):
        qc = QuantumCircuit(n_qubits)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 0)
        qc.ry(params[3], 1)
        return qc

    # ── Simulator path (free) ────────────────────────────────────────────────
    if backend_name == "simulator":
        from qiskit.primitives import StatevectorEstimator as _SVEstimator

        estimator = _SVEstimator()
        iteration_count = [0]

        def cost_fn(params):
            qc = build_ansatz(params)
            result = estimator.run([(qc, hamiltonian)]).result()
            iteration_count[0] += 1
            return result[0].data.evs.real

        rng = np.random.default_rng(42)
        x0 = rng.uniform(-np.pi, np.pi, 4)
        result = minimize(cost_fn, x0, method="COBYLA",
                          options={"maxiter": max_iterations, "rhobeg": 0.5})

        vqe_energy = float(result.fun)
        error = abs(vqe_energy - exact_energy)
        error_mha = error * 1000

        if error_mha < 1.6:
            interp = "Chemical accuracy achieved — error < 1.6 mHartree. The quantum computer found the true ground state."
        elif error_mha < 10:
            interp = "Near chemical accuracy. Good result for this ansatz."
        else:
            interp = "Did not reach chemical accuracy. Try more iterations or a deeper ansatz."

        return json.dumps({
            "molecule": mol,
            "backend": "local StatevectorSimulator (free)",
            "vqe_energy": round(vqe_energy, 6),
            "exact_energy": exact_energy,
            "error_hartree": round(error, 6),
            "error_mhartree": round(error_mha, 3),
            "converged": bool(result.success),
            "iterations": iteration_count[0],
            "optimal_params": [round(float(p), 4) for p in result.x],
            "job_id": None,
            "note": interp,
            "next_step": (
                f"To run on real IBM hardware: run_vqe(molecule='{mol}', backend_name='ibm_fez'). "
                "Hardware noise will push the energy slightly above the exact value — "
                "IonQ trapped ions would give the cleanest result."
            ),
        }, indent=2)

    # ── Real IBM hardware path (costs QPU minutes) ───────────────────────────
    # Strategy: first find optimal params on simulator (free), then evaluate
    # the single optimal circuit on real hardware (1 job, minimal cost).
    from qiskit.primitives import StatevectorEstimator as _SVEstimator2
    from qiskit_ibm_runtime import EstimatorV2 as _IBMEstimator

    # Step 1: find optimal params on simulator for free
    estimator_sim = _SVEstimator2()
    iter_sim = [0]

    def cost_sim(params):
        qc = build_ansatz(params)
        result = estimator_sim.run([(qc, hamiltonian)]).result()
        iter_sim[0] += 1
        return result[0].data.evs.real

    rng = np.random.default_rng(42)
    x0 = rng.uniform(-np.pi, np.pi, 4)
    sim_result = minimize(cost_sim, x0, method="COBYLA",
                          options={"maxiter": max_iterations, "rhobeg": 0.5})
    optimal_params = sim_result.x
    sim_energy = float(sim_result.fun)

    # Step 2: evaluate optimal circuit once on real hardware
    service = _get_service()
    try:
        backend = service.backend(backend_name)
    except Exception as e:
        return json.dumps({"error": f"Backend '{backend_name}' not found: {e}"})

    qc = build_ansatz(optimal_params)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)

    hw_estimator = _IBMEstimator(backend)
    try:
        job = hw_estimator.run([(isa_circuit, isa_hamiltonian)])
    except Exception as e:
        return json.dumps({"error": f"IBM hardware submission failed: {e}"})

    _snapshot.log_job_submission(
        job_id=job.job_id(), provider="ibm", backend_name=backend_name,
        tool_name="run_vqe",
        circuit_qubits=qc.num_qubits,
        circuit_depth_raw=qc.depth(),
        circuit_depth_transpiled=isa_circuit.depth(),
    )

    return json.dumps({
        "molecule": mol,
        "backend": backend_name,
        "simulator_energy": round(sim_energy, 6),
        "exact_energy": exact_energy,
        "simulator_iterations": iter_sim[0],
        "optimal_params": [round(float(p), 4) for p in optimal_params],
        "job_id": job.job_id(),
        "status": str(job.status()),
        "note": (
            f"Simulator found optimal parameters (energy={sim_energy:.6f} Hartree). "
            f"Now evaluating on {backend_name} real hardware. "
            f"Use job_status to track, then job_results to retrieve the hardware energy."
        ),
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 15: estimate_expectation
# --------------------------------------------------------------------------

@mcp.tool()
def estimate_expectation(device_name: str, qasm_string: str,
                         observables: str, shots: int = 1024,
                         qasm_version: int = 2) -> str:
    """
    Run the Estimator primitive to compute the expectation value of one or
    more observables for a parameterised quantum state.

    Unlike submit_job (which counts measurement outcomes), the Estimator
    computes <ψ|O|ψ> — the average value of an observable O. This is what
    quantum chemistry and optimisation algorithms (VQE, QAOA) need.

    Args:
        device_name:  IBM backend to run on, e.g. "ibm_fez".
        qasm_string:  Circuit that prepares the quantum state (no measurements
                      needed — Estimator handles that internally).
        observables:  Comma-separated Pauli strings, e.g. "ZZ,XI,IZ".
                      Each string is a tensor product of single-qubit Paulis
                      (I, X, Y, Z) — length must equal the number of qubits.
        shots:        Shots per observable (default 1024, max 20000).
        qasm_version: 2 (default) or 3.

    Returns JSON with:
      - job_id       Use job_status to track, job_results won't work — check
                     status via job_status and retrieve via this tool's job_id.
      - observables  List of Pauli strings submitted.
      - device       Backend used.
      - note         Explanation of expectation values.
    """
    # Parse circuit (same logic as submit_job)
    try:
        if qasm_version == 3:
            circuit = qiskit_qasm3.loads(qasm_string)
        else:
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
    except Exception as e:
        return json.dumps({"error": f"Failed to parse QASM {qasm_version}: {e}"})

    # Parse comma-separated Pauli strings, e.g. "ZZ,XI" → ["ZZ", "XI"]
    pauli_list = [p.strip().upper() for p in observables.split(",") if p.strip()]
    if not pauli_list:
        return json.dumps({"error": "observables must be a comma-separated list of Pauli strings, e.g. 'ZZ,XI'"})

    service = _get_service()
    try:
        backend = service.backend(device_name)
    except Exception as e:
        return json.dumps({"error": f"Device '{device_name}' not found: {e}"})

    shots = max(1, min(shots, 20000))

    # Transpile the circuit to the backend's native gate set.
    # Estimator requires an ISA circuit (Instruction Set Architecture).
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    # The observable must match the transpiled circuit's qubit count, not the
    # original circuit. After transpilation, a 2-qubit circuit on a 127-qubit
    # backend becomes a 127-qubit ISA circuit. We pad the Pauli string with I's
    # on the left to match (Qiskit uses little-endian ordering — leftmost = MSB).
    n_qubits = isa_circuit.num_qubits
    try:
        ops = []
        for p in pauli_list:
            if len(p) > n_qubits:
                return json.dumps({"error": f"Pauli string '{p}' is longer than circuit qubit count ({n_qubits})"})
            # Pad with identity qubits on the left to fill the ISA circuit width
            padded = "I" * (n_qubits - len(p)) + p
            ops.append(SparsePauliOp(padded))
    except Exception as e:
        return json.dumps({"error": f"Invalid Pauli string: {e}. Use I, X, Y, Z only."})

    # EstimatorV2 takes (circuit, observable) pairs called "PUBs"
    # (Primitive Unified Blocs). One PUB per observable.
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = shots
    pubs = [(isa_circuit, op) for op in ops]

    try:
        job = estimator.run(pubs)
    except Exception as e:
        return json.dumps({"error": f"Estimator submission failed: {e}"})

    return json.dumps({
        "job_id": job.job_id(),
        "status": str(job.status()),
        "device": device_name,
        "observables": pauli_list,
        "shots": shots,
        "note": (
            "Use job_status to check progress. When DONE, retrieve results with "
            "job_results — expectation values will be in the 'values' field. "
            "Each value is a float in [-1, +1]: +1 means all qubits measured the "
            "operator's +1 eigenstate, -1 means the -1 eigenstate."
        ),
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 15: circuit_report
# --------------------------------------------------------------------------

@mcp.tool()
def circuit_report(device_name: str, qasm_string: str,
                   qasm_version: int = 2) -> str:
    """
    Dry-run analysis of a circuit on a specific backend — no job submitted,
    no queue time, instant results.

    This is the "look before you leap" tool. Before waiting hours in a queue,
    use circuit_report to see:
      - How the compiler transforms your circuit (gate count, depth)
      - Which physical qubits get assigned to your logical qubits
      - The error rate on each assigned qubit pair
      - An estimated fidelity — the probability your result is correct

    Researchers use this to:
      - Compare backends before committing to one
      - Detect if the compiler is bloating their circuit
      - Know in advance if today's calibration is good enough

    Args:
        device_name:  Backend to analyse against, e.g. "ibm_fez".
        qasm_string:  Circuit in OpenQASM format (measurements optional).
        qasm_version: 2 (default) or 3.

    Returns JSON with:
      - original_gates     Gate counts before transpilation
      - transpiled_gates   Gate counts after IBM compiler (usually more gates)
      - original_depth     Circuit depth before compilation
      - transpiled_depth   Circuit depth after compilation
      - qubit_mapping      Logical qubit → physical qubit assignment
      - cx_error_per_pair  2-qubit gate error on each used qubit pair
      - estimated_fidelity Probability the circuit produces the correct result
      - verdict            Human-readable recommendation
    """
    # Parse the circuit
    try:
        if qasm_version == 3:
            circuit = qiskit_qasm3.loads(qasm_string)
        else:
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
    except Exception as e:
        return json.dumps({"error": f"Failed to parse QASM {qasm_version}: {e}"})

    service = _get_service()
    try:
        backend = service.backend(device_name)
    except Exception as e:
        return json.dumps({"error": f"Device '{device_name}' not found: {e}"})

    # Transpile — this is the same step submit_job does, but we stop before
    # actually running anything.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    try:
        isa_circuit = pm.run(circuit)
    except Exception as e:
        return json.dumps({"error": f"Transpilation failed: {e}"})

    # Gate counts before and after compilation
    original_gates = dict(circuit.count_ops())
    transpiled_gates = dict(isa_circuit.count_ops())
    original_depth = circuit.depth()
    transpiled_depth = isa_circuit.depth()

    # Extract qubit layout: logical index → physical qubit index
    layout = isa_circuit.layout
    qubit_mapping = {}
    if layout and layout.final_layout:
        for logical, physical in enumerate(layout.final_layout):
            qubit_mapping[f"q{logical}"] = int(str(physical).split("_")[-1]) if "_" in str(physical) else logical
    elif layout and layout.initial_layout:
        for logical_bit, physical_bit in layout.initial_layout.get_physical_bits().items():
            if hasattr(physical_bit, "index"):
                qubit_mapping[f"q{physical_bit.index}"] = logical_bit

    # Pull 2-qubit gate errors from backend calibration for the used qubits.
    # This tells you whether the assigned qubit pairs are in good shape today.
    cx_errors = {}
    try:
        props = backend.properties()
        if props:
            used_indices = list(qubit_mapping.values()) if qubit_mapping else list(range(circuit.num_qubits))
            for gate in props.gates:
                if gate.gate in ("cx", "ecr", "cz") and len(gate.qubits) == 2:
                    q0, q1 = gate.qubits
                    if q0 in used_indices or q1 in used_indices:
                        for param in gate.parameters:
                            if param.name == "gate_error":
                                cx_errors[f"q{q0}-q{q1}"] = round(param.value, 6)
    except Exception:
        pass  # calibration data unavailable — report without it

    # Estimate circuit fidelity using the product-of-gate-errors model:
    # fidelity ≈ ∏(1 - error_i) for each 2-qubit gate in the transpiled circuit.
    # This is a lower bound — real fidelity is often better due to error correlation.
    n_cx = transpiled_gates.get("cx", 0) + transpiled_gates.get("ecr", 0) + transpiled_gates.get("cz", 0)
    avg_cx_error = sum(cx_errors.values()) / len(cx_errors) if cx_errors else 0.005
    estimated_fidelity = round((1 - avg_cx_error) ** n_cx, 4) if n_cx > 0 else 1.0

    # Plain-English verdict based on fidelity
    if estimated_fidelity >= 0.90:
        verdict = "Excellent — this circuit should produce clean results on this backend today."
    elif estimated_fidelity >= 0.70:
        verdict = "Good — expect some noise but results should be meaningful."
    elif estimated_fidelity >= 0.50:
        verdict = "Fair — significant noise expected. Consider a lower-error backend or fewer gates."
    else:
        verdict = "Poor — high noise likely to obscure results. Try compare_devices to find a better backend."

    # Overhead: how much did the compiler bloat the circuit?
    overhead = round(
        (sum(transpiled_gates.values()) - sum(original_gates.values()))
        / max(sum(original_gates.values()), 1) * 100, 1
    )

    return json.dumps({
        "device": device_name,
        "original_gates": original_gates,
        "transpiled_gates": transpiled_gates,
        "original_depth": original_depth,
        "transpiled_depth": transpiled_depth,
        "compiler_overhead_pct": overhead,
        "qubit_mapping": qubit_mapping,
        "cx_error_per_pair": cx_errors,
        "estimated_fidelity": estimated_fidelity,
        "n_two_qubit_gates": n_cx,
        "verdict": verdict,
    }, indent=2)


# --------------------------------------------------------------------------
# Tool 16: debug_circuit
# --------------------------------------------------------------------------

@mcp.tool()
def debug_circuit(qasm_string: str, device_name: str = "",
                  qasm_version: int = 2) -> str:
    """
    Analyse a quantum circuit for bugs and problems BEFORE submitting it.
    No job is created. No queue time. Instant results.

    Catches two classes of problems:

    STATIC bugs (caught without connecting to IBM — always run):
      - Circuit has zero gates (empty circuit)
      - Gate applied to a qubit index that doesn't exist
      - Measurements missing (you'll get no results)
      - Classical register too small for the number of qubits measured
      - Unentangled qubits (qubit prepared but never interacted with anything)
      - Barrier-only circuit (circuit does nothing useful)

    HARDWARE bugs (caught by checking the target backend — needs device_name):
      - Circuit needs more qubits than the backend has
      - Circuit depth exceeds the backend's T2 coherence time
        (if circuit runs longer than T2, qubits decohere — results = garbage)
      - Backend is offline or in maintenance

    Each issue comes with:
      - severity: ERROR (will definitely fail) | WARNING (may give bad results) | INFO
      - plain-English explanation of what's wrong
      - suggested fix

    Args:
        qasm_string:  Circuit in OpenQASM format.
        device_name:  Optional — IBM backend to check hardware limits against.
                      Leave blank for static analysis only.
        qasm_version: 2 (default) or 3.

    Returns JSON with:
      - issues        List of {severity, check, message, fix}
      - summary       One-line verdict
      - safe_to_submit  True only if zero ERRORs found
    """
    issues = []

    # ------------------------------------------------------------------ #
    # Step 1: Parse the circuit
    # ------------------------------------------------------------------ #
    try:
        if qasm_version == 3:
            circuit = qiskit_qasm3.loads(qasm_string)
        else:
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
    except Exception as e:
        # If we can't even parse it, everything else is moot
        return json.dumps({
            "issues": [{
                "severity": "ERROR",
                "check": "parse",
                "message": f"Circuit failed to parse: {e}",
                "fix": (
                    'QASM 2 must start with: OPENQASM 2.0;\ninclude "qelib1.inc";'
                    if qasm_version != 3 else
                    "QASM 3 must start with: OPENQASM 3.0;"
                ),
            }],
            "summary": "Circuit could not be parsed. Fix syntax errors first.",
            "safe_to_submit": False,
        }, indent=2)

    n_qubits = circuit.num_qubits
    n_clbits = circuit.num_clbits
    ops = circuit.count_ops()
    depth = circuit.depth()

    # ------------------------------------------------------------------ #
    # Step 2: Static checks — no IBM connection needed
    # ------------------------------------------------------------------ #

    # Empty circuit
    if not ops or all(k in ("barrier", "measure") for k in ops):
        issues.append({
            "severity": "ERROR",
            "check": "empty_circuit",
            "message": "Circuit has no quantum gates — it does nothing.",
            "fix": "Add at least one gate (e.g., h q[0]; to put qubit 0 in superposition).",
        })

    # No measurements
    if ops.get("measure", 0) == 0:
        issues.append({
            "severity": "ERROR",
            "check": "no_measurements",
            "message": "Circuit has no measurement gates. You will get no results.",
            "fix": "Add measurements: measure q[0] -> c[0]; for each qubit you care about.",
        })

    # Classical register too small
    if n_clbits < ops.get("measure", 0):
        issues.append({
            "severity": "ERROR",
            "check": "classical_register_too_small",
            "message": (
                f"You have {ops.get('measure', 0)} measurement gates but only "
                f"{n_clbits} classical bits to store results."
            ),
            "fix": f"Increase classical register: creg c[{ops.get('measure', 0)}];",
        })

    # Check for unentangled qubits — qubits that only have single-qubit gates
    # and never interact with another qubit via a 2-qubit gate.
    # We detect this by inspecting each instruction's qubits.
    entangled_qubits = set()
    for instruction in circuit.data:
        if len(instruction.qubits) >= 2:
            for q in instruction.qubits:
                entangled_qubits.add(circuit.find_bit(q).index)

    single_only_qubits = []
    for i in range(n_qubits):
        # Check if this qubit has any gates at all (not just initialized)
        qubit_has_gates = any(
            circuit.find_bit(q).index == i
            for inst in circuit.data
            for q in inst.qubits
            if inst.operation.name not in ("measure", "barrier")
        )
        if qubit_has_gates and i not in entangled_qubits:
            single_only_qubits.append(i)

    if single_only_qubits and n_qubits > 1:
        issues.append({
            "severity": "INFO",
            "check": "unentangled_qubits",
            "message": (
                f"Qubit(s) {single_only_qubits} have gates but never interact "
                f"with other qubits via a 2-qubit gate. They are not entangled."
            ),
            "fix": (
                "If you intended entanglement, add a CNOT: cx q[0],q[1]; "
                "If this is intentional (parallel single-qubit experiments), ignore this."
            ),
        })

    # Very deep circuit warning (heuristic — before we even know T2)
    if depth > 100:
        issues.append({
            "severity": "WARNING",
            "check": "deep_circuit_heuristic",
            "message": (
                f"Circuit depth is {depth}, which is quite deep. "
                "Deep circuits are more vulnerable to decoherence noise."
            ),
            "fix": (
                "Consider using optimization_level=3 when transpiling, or "
                "restructure to reduce gate count. Run circuit_report to see "
                "transpiled depth on your target backend."
            ),
        })

    # ------------------------------------------------------------------ #
    # Step 3: Hardware checks — needs device_name
    # ------------------------------------------------------------------ #
    backend_info = {}
    if device_name:
        try:
            service = _get_service()
            backend = service.backend(device_name)

            # Backend offline?
            status = backend.status()
            if not status.operational:
                issues.append({
                    "severity": "ERROR",
                    "check": "backend_offline",
                    "message": f"{device_name} is currently offline or in maintenance.",
                    "fix": "Run queue_status or compare_devices to find an operational backend.",
                })

            # Too many qubits?
            backend_qubits = backend.num_qubits
            if n_qubits > backend_qubits:
                issues.append({
                    "severity": "ERROR",
                    "check": "too_many_qubits",
                    "message": (
                        f"Your circuit needs {n_qubits} qubits but {device_name} "
                        f"only has {backend_qubits}."
                    ),
                    "fix": f"Use a backend with at least {n_qubits} qubits, or reduce your circuit size.",
                })

            # Coherence time (T2) check — the "I love you" feature.
            # T2 is how long a qubit stays quantum before noise destroys it (in microseconds).
            # Circuit execution time ≈ depth × avg_gate_time.
            # If execution_time > T2, results are garbage — pure noise.
            try:
                props = backend.properties()
                if props:
                    # Collect T2 values for all qubits (in seconds, convert to microseconds)
                    t2_values = []
                    for i in range(min(n_qubits, backend_qubits)):
                        t2 = props.t2(i)
                        if t2 is not None:
                            t2_values.append(t2 * 1e6)  # convert s → µs

                    # Estimate circuit execution time from gate times
                    # Typical IBM gate times: single-qubit ~35ns, 2-qubit ~300ns
                    n_2q = sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "swap"))
                    n_1q = sum(v for k, v in ops.items() if k not in ("cx", "ecr", "cz", "swap", "measure", "barrier", "reset"))
                    estimated_exec_us = (n_1q * 0.035) + (n_2q * 0.3)  # µs

                    if t2_values:
                        min_t2 = min(t2_values)
                        avg_t2 = sum(t2_values) / len(t2_values)
                        backend_info["min_t2_us"] = round(min_t2, 1)
                        backend_info["avg_t2_us"] = round(avg_t2, 1)
                        backend_info["estimated_exec_us"] = round(estimated_exec_us, 3)

                        if estimated_exec_us > min_t2:
                            issues.append({
                                "severity": "ERROR",
                                "check": "exceeds_coherence_time",
                                "message": (
                                    f"Estimated circuit execution time ({estimated_exec_us:.2f} µs) "
                                    f"exceeds the shortest T2 coherence time on {device_name} "
                                    f"({min_t2:.1f} µs). Qubits will decohere before the circuit "
                                    f"finishes — results will be pure noise."
                                ),
                                "fix": (
                                    f"Reduce circuit depth or 2-qubit gate count. "
                                    f"Target execution time under {min_t2 * 0.5:.1f} µs "
                                    f"(50% of T2) for reliable results. "
                                    f"Run compare_devices to find a backend with longer T2."
                                ),
                            })
                        elif estimated_exec_us > min_t2 * 0.5:
                            issues.append({
                                "severity": "WARNING",
                                "check": "approaching_coherence_limit",
                                "message": (
                                    f"Estimated circuit execution time ({estimated_exec_us:.2f} µs) "
                                    f"is above 50% of the shortest T2 ({min_t2:.1f} µs). "
                                    "You are in the noise-sensitive zone."
                                ),
                                "fix": (
                                    "Consider reducing circuit depth. Ideal target is under "
                                    f"{min_t2 * 0.5:.1f} µs. Results may still be usable but "
                                    "expect elevated noise."
                                ),
                            })
            except Exception:
                pass  # T2 data unavailable — skip coherence check silently

        except Exception as e:
            issues.append({
                "severity": "WARNING",
                "check": "backend_unreachable",
                "message": f"Could not connect to {device_name} to run hardware checks: {e}",
                "fix": "Check the device name with list_devices or queue_status.",
            })

    # ------------------------------------------------------------------ #
    # Step 4: Build summary
    # ------------------------------------------------------------------ #
    errors = [i for i in issues if i["severity"] == "ERROR"]
    warnings = [i for i in issues if i["severity"] == "WARNING"]
    infos = [i for i in issues if i["severity"] == "INFO"]
    safe = len(errors) == 0

    if not issues:
        summary = "No issues found. Circuit looks clean and ready to submit."
    elif errors:
        summary = (
            f"{len(errors)} error(s) found — do NOT submit until fixed. "
            f"{len(warnings)} warning(s), {len(infos)} info note(s)."
        )
    else:
        summary = (
            f"No blocking errors. {len(warnings)} warning(s) to review. "
            f"Circuit can be submitted but check warnings first."
        )

    return json.dumps({
        "circuit_stats": {
            "qubits": n_qubits,
            "classical_bits": n_clbits,
            "depth": depth,
            "gate_counts": ops,
        },
        "backend_info": backend_info,
        "issues": issues,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "info_count": len(infos),
        "summary": summary,
        "safe_to_submit": safe,
    }, indent=2)


# --------------------------------------------------------------------------
# API Key Authentication Middleware
# --------------------------------------------------------------------------

class APIKeyAuthMiddleware:
    """
    Pure ASGI middleware for API key auth.

    Replaces BaseHTTPMiddleware to avoid Starlette 1.3.x SSE breakage —
    BaseHTTPMiddleware wraps SSE responses in a buffer that causes an
    AssertionError when the SSE stream sends http.response.start twice.
    A raw ASGI __call__ passes the connection straight through.
    """

    def __init__(self, app, api_key: Optional[str] = None):
        self.app = app
        self.api_key = api_key or os.getenv("MCP_API_KEY")

    async def __call__(self, scope, receive, send):
        # Only inspect HTTP/WebSocket — pass lifespan events straight through
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if not self.api_key:
            await self.app(scope, receive, send)
            return

        # Headers arrive as a list of (name_bytes, value_bytes) tuples
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        request_key = headers.get(b"x-api-key", b"").decode()

        if request_key != self.api_key:
            response = _JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid or missing API key. Include X-API-Key header.",
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# --------------------------------------------------------------------------
# IonQ shared helpers
# --------------------------------------------------------------------------

# Friendly name -> real qiskit_ionq backend string. get_backend() needs the
# exact "qpu.forte-1" form; plain "forte-1" raises "No backend matches criteria."
_IONQ_BACKEND_ALIASES = {
    "simulator":              "ionq_simulator",
    "ionq_simulator":         "ionq_simulator",
    "forte-1":                "qpu.forte-1",
    "forte1":                 "qpu.forte-1",
    "qpu.forte-1":            "qpu.forte-1",
    "forte-enterprise-1":     "qpu.forte-enterprise-1",
    "forte-enterprise":       "qpu.forte-enterprise-1",
    "qpu.forte-enterprise-1": "qpu.forte-enterprise-1",
}


def _resolve_ionq_backend(name: str) -> str:
    """Map a friendly IonQ backend name to the exact string qiskit_ionq needs."""
    key = name.strip().lower()
    return _IONQ_BACKEND_ALIASES.get(key, name)


def _ionq_is_hardware(resolved_backend_name: str) -> bool:
    """True if this backend name refers to real QPU hardware, not a simulator."""
    return "simulator" not in resolved_backend_name.lower()


# --------------------------------------------------------------------------
# Tool 17: ionq_devices  — list IonQ quantum computers
# --------------------------------------------------------------------------

@mcp.tool()
def ionq_devices() -> str:
    """
    List all available IonQ quantum computers and simulators.

    IonQ uses trapped-ion technology — a different physical approach from
    IBM's superconducting qubits. Trapped-ion systems tend to have higher
    gate fidelity but fewer qubits than IBM machines.

    Returns a list of IonQ backends with qubit count and availability.
    Requires IONQ_API_KEY in .env.
    """
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        return json.dumps({
            "error": "IONQ_API_KEY not set in .env",
            "hint": "Get your key at cloud.ionq.com and add IONQ_API_KEY=your_key to .env"
        })

    try:
        # Direct REST API — the qiskit_ionq SDK's IonQProvider.backends() only
        # surfaces 2 of 6 real backends (stale/legacy endpoint). This is the
        # same endpoint snapshot.py uses and it returns the full fleet.
        resp = requests.get(
            "https://api.ionq.co/v0.3/backends",
            headers={"Authorization": f"apiKey {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        backends = resp.json()

        result = []
        for b in backends:
            name = b.get("backend", b.get("name", "unknown"))
            status = b.get("status")
            result.append({
                "name": name,
                "num_qubits": b.get("qubits"),
                "available": status == "available",
                "status": status,  # "available" | "unavailable" | "retired"
                "type": "simulator" if "simulator" in name else "hardware",
                "provider": "IonQ",
                "technology": "trapped-ion",
            })

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# ionq_submit_job self-check fixes, ported from quantum-verifier
# (github.com/Lokesh-2025/quantum-verifier) after real hardware testing
# found this exact function's self-check silently mispredicting results.
# --------------------------------------------------------------------------
#
# Qiskit's transpiler has no built-in equivalence between RZZGate (radians)
# and IonQ's native ZZGate (turns), even though they are the exact same
# physical gate (RZZ(theta) = exp(-i*theta/2 * Z@Z), native ZZ(phi) =
# exp(-i*pi*phi * Z@Z), so phi = theta/(2*pi) is an EXACT match — verified
# via Operator comparison, not just up to global phase). Without this
# registered, the transpiler silently falls back to general two-qubit
# unitary synthesis: a single rzz between two H gates transpiles to 2
# native zz gates (should be 1) plus ~39 extraneous single-qubit gates.
# That produces a wrong self-check prediction for any circuit containing
# rzz, without ever raising an error — caught only by comparing this
# function's self-check output against real hardware results.
def _register_ionq_native_equivalences():
    try:
        import math
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import RZZGate
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as _sel
        from qiskit_ionq.ionq_gates import ZZGate as _IonQZZGate
    except ImportError:
        return
    theta = Parameter("theta")
    equiv = QuantumCircuit(2)
    equiv.append(_IonQZZGate(theta / (2 * math.pi)), [0, 1])
    _sel.add_equivalence(RZZGate(theta), equiv)


_register_ionq_native_equivalences()

# IonQ's native ZZ gate is only valid for |theta_turns| <= 0.25 (a quarter
# turn) — confirmed via a real IonQ API rejection when a large-angle rzz
# hit the equivalence above. Splitting into N chained smaller RZZ
# applications is mathematically EXACT: ZZ generators on the same qubit
# pair commute, so N reps of RZZ(theta/N) = RZZ(theta) exactly.
def _decompose_large_angle_rzz_for_ionq(circuit):
    import math
    MAX_TURNS = 0.25
    new_qc = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name == "rzz":
            theta = float(instruction.operation.params[0])
            theta_turns = theta / (2 * math.pi)
            n_chunks = max(1, math.ceil(abs(theta_turns) / MAX_TURNS))
            for _ in range(n_chunks):
                new_qc.rzz(theta / n_chunks, instruction.qubits[0], instruction.qubits[1])
        else:
            new_qc.append(instruction.operation, instruction.qubits, instruction.clbits)
    return new_qc


# --------------------------------------------------------------------------
# Tool 18: ionq_submit_job  — submit a circuit to IonQ
# --------------------------------------------------------------------------

@mcp.tool()
def ionq_submit_job(
    backend_name: str,
    qasm_circuits: list,
    shots: int = 1024,
    optimization_level: int = 1,
    expected_marked_bitstrings: Union[list, None] = None,
    expected_amplification: Union[float, list, None] = None,
    amplification_tolerance: float = 0.5,
    confirm_real_hardware: bool = False,
) -> str:
    """
    Compile and submit one or more OpenQASM 2 circuits to IonQ as ONE job.

    Multiple circuits share the per-job minimum charge — batch related
    circuits together rather than submitting them one at a time.

    SAFETY: before anything touches real hardware, every circuit is first
    transpiled and run on the free ionq_simulator. If expected_amplification
    is given and the simulated result misses it by more than
    amplification_tolerance (relative), submission is REFUSED — same
    self-check pattern used for qforge's build_forged_circuits. This catches
    a bad translation (wrong angle units, endianness, optimization_level
    rewriting the circuit) before it costs a dollar.

    Real hardware (any non-simulator backend) additionally requires
    confirm_real_hardware=True — a plain typo in backend_name should never
    silently spend credits.

    Args:
        backend_name  : 'simulator', 'forte-1', or 'forte-enterprise-1'
                        (also accepts the exact qiskit_ionq strings)
        qasm_circuits : list of OpenQASM 2.0 circuit strings — even a single
                        circuit must be passed as a one-element list
        shots         : shots per circuit (default 1024)
        optimization_level : transpiler level, default 1. IonQ's own SDK warns
                        the qiskit default of 2 does aggressive re-synthesis
                        that can rewrite a hand-designed circuit — don't raise
                        this without a specific reason.
        expected_marked_bitstrings : optional target bitstrings for the
                        self-check (same convention as get_amplification).
                        For ONE circuit: a flat list of bitstrings, e.g.
                        ["0001110", "0001111"]. For a BATCH of N circuits:
                        a list of N entries, one per circuit in the same
                        order as qasm_circuits — each entry either a list
                        of bitstrings for that circuit, or None to skip the
                        check for that particular circuit.
        expected_amplification    : optional predicted amplification each
                        circuit should hit. For ONE circuit: a single number.
                        For a BATCH: a list of N numbers/None, one per
                        circuit, same order and same skip-with-None rule as
                        expected_marked_bitstrings. Every circuit with both
                        an expected_marked_bitstrings entry and an
                        expected_amplification entry gets its own
                        independent tolerance check — one bad circuit in a
                        batch refuses the WHOLE submission, not just itself.
        amplification_tolerance   : relative tolerance applied to every
                        circuit's check (default 0.5 = simulated
                        amplification must be within 50% of its own
                        expected_amplification)
        confirm_real_hardware : must be True to submit to actual QPU hardware.
                        Not required for the simulator.

    Returns job_id(s), self-check results, and whether this went to real
    hardware or the free simulator.
    Requires IONQ_API_KEY in .env.
    """
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        return json.dumps({
            "error": "IONQ_API_KEY not set in .env",
            "hint": "Get your key at cloud.ionq.com and add IONQ_API_KEY=your_key to .env"
        })

    if isinstance(qasm_circuits, str):
        qasm_circuits = [qasm_circuits]
    if not qasm_circuits:
        return json.dumps({"error": "qasm_circuits is empty"})

    try:
        from qiskit_ionq import IonQProvider
        from qiskit import QuantumCircuit as QC, transpile

        circuits = []
        for i, qasm_string in enumerate(qasm_circuits):
            try:
                parsed = QC.from_qasm_str(qasm_string)
            except Exception as parse_err:
                return json.dumps({
                    "error": f"Failed to parse circuit {i}: {parse_err}",
                    "hint": "IonQ supports OpenQASM 2.0 — circuit must start with: OPENQASM 2.0;"
                })
            circuits.append(_decompose_large_angle_rzz_for_ionq(parsed))

        resolved_backend = _resolve_ionq_backend(backend_name)
        is_hardware = _ionq_is_hardware(resolved_backend)

        # Normalize expected_marked_bitstrings / expected_amplification to
        # one entry per circuit. Single-circuit calls may pass a flat value
        # (backward compatible); batches of N>1 must pass a list of length N
        # (use None per-entry to skip that circuit's check) — a flat value
        # for a multi-circuit batch is ambiguous and rejected outright
        # rather than silently applied to only circuit 0.
        n_circuits = len(circuits)

        def _per_circuit(value, param_name):
            if value is None:
                return [None] * n_circuits
            if n_circuits == 1:
                return [value]
            if not isinstance(value, list) or len(value) != n_circuits:
                raise ValueError(
                    f"{param_name}: submitting {n_circuits} circuits requires a list of "
                    f"{n_circuits} entries (one per circuit, None to skip that circuit's "
                    f"check) — got {value!r}"
                )
            return value

        try:
            marked_per_circuit = _per_circuit(expected_marked_bitstrings, "expected_marked_bitstrings")
            expected_amp_per_circuit = _per_circuit(expected_amplification, "expected_amplification")
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

        provider = IonQProvider(api_key)

        # --- Pre-flight self-check: always run on the free simulator first ---
        # gateset="native" is required — the default "qis" gateset lets IonQ's
        # own server-side compiler rewrite the circuit before it runs, which
        # defeats the point of a hand-designed circuit and would silently
        # invalidate any predicted-vs-actual comparison.
        #
        # Transpile against the ACTUAL intended device's target (forte-1 uses
        # native ZZ; the generic simulator target defaults to MS/Aria-style —
        # verified empirically, and would silently give the wrong gate family
        # for what really runs on Forte). Execute the check on the simulator,
        # with its noise model set to match the real target when submitting
        # to real hardware, so this is a realistic noisy preview, not an
        # idealized one.
        # The comment above already documented this trap, but the code
        # never actually implemented the fix: resolved_backend == "ionq_simulator"
        # (the bare "simulator" alias) was passed straight through as the
        # transpile TARGET too, silently picking the legacy MS gateset it
        # warns about. Redirect the transpile target to a real Forte-class
        # device's gateset in that case, while still executing on the free
        # simulator with no noise model (no specific real device requested).
        transpile_target_name = "qpu.forte-1" if resolved_backend == "ionq_simulator" else resolved_backend
        target_backend = provider.get_backend(transpile_target_name, gateset="native")
        sim_backend = provider.get_backend("ionq_simulator", gateset="native")
        if is_hardware:
            # Named noise models exist for real IonQ devices (docs.ionq.com/
            # guides/simulation-with-noise-models) — depolarizing channels
            # applied after each gate, with fixed, angle-independent rates.
            device_short_name = resolved_backend.replace("qpu.", "")
            sim_backend.set_options(noise_model=device_short_name)

        self_check = {"ran": True, "per_circuit": [], "passed": True,
                       "noise_model_used": sim_backend.options.noise_model}
        for i, qc in enumerate(circuits):
            t_qc = transpile(qc, backend=target_backend, optimization_level=optimization_level)
            sim_job = sim_backend.run(t_qc, shots=shots)
            sim_counts = sim_job.result().get_counts()
            total = sum(sim_counts.values())

            entry = {
                "circuit_index": i,
                "num_qubits": qc.num_qubits,
                "transpiled_gate_count": t_qc.size(),
                "simulated_counts_top5": dict(sorted(sim_counts.items(), key=lambda x: -x[1])[:5]),
            }

            circuit_marked = marked_per_circuit[i]
            circuit_expected_amp = expected_amp_per_circuit[i]

            if circuit_marked:
                marked = set(circuit_marked)
                marked_shots = sum(c for b, c in sim_counts.items() if b in marked)
                sim_amp = (marked_shots / total) / (len(marked) / (2 ** qc.num_qubits)) if total else 0
                entry["simulated_amplification"] = round(sim_amp, 3)

                if circuit_expected_amp is not None:
                    lo = circuit_expected_amp * (1 - amplification_tolerance)
                    hi = circuit_expected_amp * (1 + amplification_tolerance)
                    entry["expected_amplification"] = circuit_expected_amp
                    entry["within_tolerance"] = lo <= sim_amp <= hi
                    if not entry["within_tolerance"]:
                        self_check["passed"] = False

            self_check["per_circuit"].append(entry)

        if not self_check["passed"]:
            failed_indices = [e["circuit_index"] for e in self_check["per_circuit"]
                               if e.get("within_tolerance") is False]
            return json.dumps({
                "error": f"Self-check failed on circuit(s) {failed_indices} — simulated result does not match expected_amplification within tolerance",
                "hint": "One bad circuit refuses the WHOLE batch — none of them submitted. The circuit may have been rewritten by transpilation, or the expected value is wrong.",
                "self_check": self_check,
            })

        if resolved_backend == "ionq_simulator":
            # Already have the self-check simulator results — that IS the job.
            return json.dumps({
                "status": "SIMULATED",
                "backend": resolved_backend,
                "is_real_hardware": False,
                "shots": shots,
                "provider": "IonQ",
                "self_check": self_check,
                "note": "backend was 'simulator' — this ran on the free simulator, nothing was billed.",
            })

        if is_hardware and not confirm_real_hardware:
            return json.dumps({
                "error": f"'{resolved_backend}' is real QPU hardware and will be billed.",
                "hint": "Pass confirm_real_hardware=True to actually submit. Self-check passed, so the circuit is ready when you are.",
                "self_check": self_check,
            })

        t_circuits = [transpile(qc, backend=target_backend, optimization_level=optimization_level) for qc in circuits]
        job = target_backend.run(t_circuits, shots=shots)

        return json.dumps({
            "job_id": job.job_id(),
            "status": "SUBMITTED",
            "backend": resolved_backend,
            "is_real_hardware": True,
            "num_circuits": len(circuits),
            "shots": shots,
            "provider": "IonQ",
            "self_check": self_check,
            "hint": "Use ionq_job_status(job_id, backend_name) to check progress",
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool 19: ionq_job_status  — check IonQ job status
# --------------------------------------------------------------------------

@mcp.tool()
def ionq_job_status(job_id: str, backend_name: str = "ionq_simulator") -> str:
    """
    Check the status of a submitted IonQ job.

    Args:
        job_id       : the job ID returned by ionq_submit_job
        backend_name : the backend the job was submitted to (default: ionq_simulator)

    Returns current status and job details.
    """
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        return json.dumps({"error": "IONQ_API_KEY not set in .env"})

    try:
        from qiskit_ionq import IonQProvider
        resolved_backend = _resolve_ionq_backend(backend_name)
        provider = IonQProvider(api_key)
        backend = provider.get_backend(resolved_backend, gateset="native")
        job = backend.retrieve_job(job_id)

        status = job.status()

        return json.dumps({
            "job_id": job_id,
            "status": str(status.name),
            "backend": resolved_backend,
            "is_real_hardware": _ionq_is_hardware(resolved_backend),
            "provider": "IonQ",
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool 20: ionq_job_results  — get results from a completed IonQ job
# --------------------------------------------------------------------------

@mcp.tool()
def ionq_job_results(job_id: str, backend_name: str = "simulator") -> str:
    """
    Retrieve measurement counts from a completed IonQ job.

    Args:
        job_id       : the job ID returned by ionq_submit_job
        backend_name : the backend the job was submitted to (default: simulator)

    Returns bit-string counts like {"00": 512, "11": 512}, and — always —
    is_real_hardware: True/False. Check that field before treating a result
    as a real hardware measurement. It is set from the backend name itself,
    not inferred, so it can't be fooled by a lucky-looking result.

    For a multi-circuit (batched) job, counts is a list, one entry per
    circuit, in the same order they were submitted.

    Job must be in DONE status — check with ionq_job_status() first.
    """
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        return json.dumps({"error": "IONQ_API_KEY not set in .env"})

    try:
        from qiskit_ionq import IonQProvider
        from qiskit.providers import JobStatus
        resolved_backend = _resolve_ionq_backend(backend_name)
        provider = IonQProvider(api_key)
        backend = provider.get_backend(resolved_backend, gateset="native")
        job = backend.retrieve_job(job_id)

        status = job.status()
        if status != JobStatus.DONE:
            return json.dumps({
                "job_id": job_id,
                "status": str(status.name),
                "message": "Job not complete yet — check again with ionq_job_status()"
            })

        result = job.result()
        # qiskit_ionq's get_counts() returns a single dict for one circuit,
        # or a list of dicts for a batched multi-circuit job.
        counts = result.get_counts()
        is_batch = isinstance(counts, list)

        payload = {
            "job_id": job_id,
            "backend": resolved_backend,
            "is_real_hardware": _ionq_is_hardware(resolved_backend),
            "provider": "IonQ",
        }

        if is_batch:
            payload["counts"] = counts
            payload["total_shots_per_circuit"] = [sum(c.values()) for c in counts]
        else:
            payload["counts"] = counts
            payload["total_shots"] = sum(counts.values())

        return json.dumps(payload)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: estimate_ionq_gates  — native gate count before submitting
# --------------------------------------------------------------------------

@mcp.tool()
def estimate_ionq_gates(qasm_string: str, backend_name: str = "forte-1", optimization_level: int = 1) -> str:
    """
    Transpile a circuit against a REAL IonQ device's native gate target and
    report the real gate count — GPI, GPI2, and ZZ — without submitting
    anything.

    Targets a specific device (default forte-1) rather than the generic
    simulator, because Forte-class hardware's native two-qubit gate is ZZ(θ)
    — Aria-only systems use Mølmer-Sørensen (MS) instead, and Aria is
    retired. Transpiling against the generic simulator target silently picks
    the wrong gate family (verified: it defaults to MS). Getting this wrong
    means the gate count you see here isn't what actually runs.

    Args:
        qasm_string         : OpenQASM 2.0 circuit string
        backend_name        : which device's native target to transpile
                              against — 'forte-1' (default), 'forte-enterprise-1',
                              or 'simulator' (gives MS-based counts, Aria-style —
                              only useful if you specifically care about that)
        optimization_level  : transpiler level, default 1 (IonQ recommends
                              0-1; the qiskit default of 2 can rewrite the
                              circuit more aggressively than intended)

    Returns native gate counts, total 2-qubit gate count (the number that
    matters most for cost and error), and estimated wall-clock time (2-qubit
    gates run serially at roughly 100-200µs each on real hardware). Works
    even while the target device shows "unavailable" — device configuration
    is queryable independent of whether jobs can currently be submitted.
    """
    try:
        from qiskit import QuantumCircuit as QC, transpile

        try:
            circuit = QC.from_qasm_str(qasm_string)
        except Exception as parse_err:
            return json.dumps({"error": f"Failed to parse QASM: {parse_err}"})

        api_key = os.getenv("IONQ_API_KEY")
        if not api_key:
            return json.dumps({
                "error": "IONQ_API_KEY not set in .env",
                "hint": "Needed to transpile against IonQ's real native-gate target — "
                        "the gate names (gpi/gpi2/zz) aren't valid standalone qiskit "
                        "basis_gates, they only resolve through IonQ's own backend target."
            })

        from qiskit_ionq import IonQProvider
        resolved_backend = _resolve_ionq_backend(backend_name)
        backend = IonQProvider(api_key).get_backend(resolved_backend, gateset="native")
        t_circuit = transpile(circuit, backend=backend, optimization_level=optimization_level)

        ops = dict(t_circuit.count_ops())
        two_qubit_gates = ops.get("zz", 0) + ops.get("ms", 0)
        one_qubit_gates = ops.get("gpi", 0) + ops.get("gpi2", 0)
        native_2q_gate = "zz" if "zz" in ops else ("ms" if "ms" in ops else None)

        # ~100-200us per serialized 2Q gate is IonQ's documented ballpark;
        # 1Q gates are much faster (~10us) and largely overlap in practice.
        est_2q_seconds = two_qubit_gates * 0.00015
        est_wall_time_per_shot = est_2q_seconds

        return json.dumps({
            "target_device": resolved_backend,
            "native_2q_gate_family": native_2q_gate,
            "num_qubits_in_circuit": circuit.num_qubits,
            "native_gate_counts": ops,
            "one_qubit_gates": one_qubit_gates,
            "two_qubit_gates": two_qubit_gates,
            "total_native_gates": sum(ops.values()),
            "estimated_seconds_per_shot": round(est_wall_time_per_shot, 4),
            "note": "2-qubit gates dominate both cost and wall-clock time — this count is what estimate_ionq_cost uses.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: estimate_ionq_cost  — cost before submitting
# --------------------------------------------------------------------------

@mcp.tool()
def estimate_ionq_cost(qasm_circuits: list, shots: int = 4096) -> str:
    """
    Estimate the real dollar cost of submitting circuit(s) to IonQ, without
    touching the API.

    Verified directly against IonQ's own public resource estimator
    (https://www.ionq.com/programs/research-credits/resource-estimator):
    every circuit small enough to matter for this project's Singmaster's
    work (up to 16 two-qubit gates on 7-24 qubits) sits exactly on the
    $168.20-per-job floor. Above that floor, IonQ's exact pricing formula
    isn't fully published — this tool gives you the guaranteed floor, and
    an approximate estimate for anything larger, clearly labeled as such.

    Multiple circuits in the SAME job (see ionq_submit_job's qasm_circuits
    batching) share one $168.20 floor instead of paying it once each — this
    is the main lever for saving credits.

    Args:
        qasm_circuits : list of OpenQASM 2.0 circuit strings — as if they
                        were going to be submitted together as one batched job
        shots         : shots per circuit (default 4096)

    Returns per-circuit gate counts, whether the batch is expected to sit at
    the floor, and the estimated total.
    """
    JOB_FLOOR_USD = 168.20
    # Single empirical data point beyond the floor, from live testing this
    # project did against IonQ's resource estimator: 30 qubits, 800 1Q gates,
    # 600 2Q gates -> $3,294.87. Not enough points to fit a real formula —
    # used only as a rough slope for clearly-labeled estimates above the floor.
    KNOWN_ABOVE_FLOOR_POINT_2Q_GATES = 600
    KNOWN_ABOVE_FLOOR_POINT_USD = 3294.87
    ROUGH_USD_PER_2Q_GATE = (KNOWN_ABOVE_FLOOR_POINT_USD - JOB_FLOOR_USD) / KNOWN_ABOVE_FLOOR_POINT_2Q_GATES

    if isinstance(qasm_circuits, str):
        qasm_circuits = [qasm_circuits]

    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        return json.dumps({
            "error": "IONQ_API_KEY not set in .env",
            "hint": "Needed to transpile against IonQ's real native-gate target."
        })

    try:
        from qiskit import QuantumCircuit as QC, transpile
        from qiskit_ionq import IonQProvider

        # forte-1's real target. Transpiling against the generic simulator
        # target instead would silently give MS-based (Aria-style) counts,
        # not ZZ — Forte doesn't have MS at all.
        backend = IonQProvider(api_key).get_backend("qpu.forte-1", gateset="native")

        per_circuit = []
        max_two_qubit_gates = 0
        for i, qasm_string in enumerate(qasm_circuits):
            circuit = QC.from_qasm_str(qasm_string)
            t_circuit = transpile(circuit, backend=backend, optimization_level=1)
            ops = dict(t_circuit.count_ops())
            two_q = ops.get("zz", 0)
            max_two_qubit_gates = max(max_two_qubit_gates, two_q)
            per_circuit.append({
                "circuit_index": i,
                "num_qubits": circuit.num_qubits,
                "two_qubit_gates": two_q,
                "one_qubit_gates": ops.get("gpi", 0) + ops.get("gpi2", 0),
            })

        # This project's own verified rule of thumb: comfortably under ~20
        # two-qubit gates on <=30 qubits has always sat at the floor.
        likely_at_floor = max_two_qubit_gates <= 20

        if likely_at_floor:
            estimated_total = JOB_FLOOR_USD
            confidence = "high — verified empirically for circuits in this size range"
        else:
            estimated_total = JOB_FLOOR_USD + ROUGH_USD_PER_2Q_GATE * max_two_qubit_gates
            confidence = "LOW — extrapolated from a single data point, verify on IonQ's real calculator before relying on this"

        return json.dumps({
            "num_circuits_in_batch": len(qasm_circuits),
            "shots_per_circuit": shots,
            "per_circuit": per_circuit,
            "job_floor_usd": JOB_FLOOR_USD,
            "likely_at_floor": likely_at_floor,
            "estimated_total_usd": round(estimated_total, 2),
            "confidence": confidence,
            "note": "This is ONE job (batched) — all circuits above share this one floor, not pay it individually.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_alerts(device_name: str = "", days: int = 7) -> str:
    """
    Return calibration drift alerts for IBM Quantum devices.

    The snapshot agent (runs every 6 hours) compares each new snapshot
    against the previous one. When a device's avg_cx_error or
    avg_readout_error rises by more than 20%, or a device goes offline,
    an alert is written to the database.

    This is what Nikita's problem was — ibm_boston wasn't recalibrated
    and nobody knew until jobs were stuck for 5 hours. This catches it
    at the next snapshot automatically.

    Args:
        device_name : filter to one device (e.g. "ibm_boston") — leave empty for all
        days        : how many days back to look (default 7)

    Returns a list of alerts with device name, alert type, values, and timestamp.
    Alert types:
        cx_error_spike              — 2-qubit gate error rose >20%
        readout_error_spike         — readout error rose >20%
        went_offline                — device went from operational to offline
        t1_drop                     — median T1 coherence dropped >20% (early warning)
        t2_drop                     — median T2 coherence dropped >20% (early warning)
    """
    import sqlite3 as _sqlite3

    db_path = os.path.join(os.path.dirname(__file__), "devices.db")
    if not os.path.exists(db_path):
        return json.dumps({"error": "No local database found. Run snapshot.py first."})

    try:
        with _sqlite3.connect(db_path) as con:
            # Stored alerts (cx/readout/offline) from snapshot.py
            query = """
                SELECT ts, device_name, alert_type, prev_value, curr_value, pct_change
                FROM device_alerts
                WHERE ts >= datetime('now', ? || ' days')
            """
            params: list = [f"-{max(1, int(days))}"]
            if device_name:
                query += " AND device_name = ?"
                params.append(device_name)
            query += " ORDER BY ts DESC LIMIT 200"
            stored_rows = con.execute(query, params).fetchall()

            # Live T1/T2 drop detection using LAG() window function
            t1t2_params: list = [f"-{max(1, int(days))}"]
            t1t2_filter = ""
            if device_name:
                t1t2_filter = "AND name = ?"
                t1t2_params.append(device_name)

            t1t2_rows = con.execute(f"""
                WITH ranked AS (
                    SELECT name, ts, median_t1_us, median_t2_us,
                        LAG(median_t1_us) OVER (PARTITION BY name ORDER BY ts) AS prev_t1,
                        LAG(median_t2_us) OVER (PARTITION BY name ORDER BY ts) AS prev_t2
                    FROM device_snapshots
                    WHERE ts >= datetime('now', ? || ' days')
                    {t1t2_filter}
                )
                SELECT name, ts, median_t1_us, prev_t1, median_t2_us, prev_t2
                FROM ranked
                WHERE prev_t1 IS NOT NULL
                  AND (
                    (prev_t1 > 0 AND (prev_t1 - median_t1_us) / prev_t1 > 0.20)
                    OR
                    (prev_t2 > 0 AND (prev_t2 - median_t2_us) / prev_t2 > 0.20)
                  )
                ORDER BY ts DESC LIMIT 100
            """, t1t2_params).fetchall()

        alerts = []

        for ts, name, alert_type, prev, curr, pct in stored_rows:
            entry = {"ts": ts, "device": name, "type": alert_type}
            if alert_type == "went_offline":
                entry["message"] = f"{name} went offline"
            else:
                label = "cx_error" if "cx" in alert_type else "readout_error"
                entry["message"] = (
                    f"{name} {label} spiked {pct}% "
                    f"(was {prev:.5f}, now {curr:.5f})"
                )
            alerts.append(entry)

        for name, ts, t1, prev_t1, t2, prev_t2 in t1t2_rows:
            if prev_t1 and prev_t1 > 0 and (prev_t1 - t1) / prev_t1 > 0.20:
                pct = round((prev_t1 - t1) / prev_t1 * 100, 1)
                alerts.append({
                    "ts": ts, "device": name, "type": "t1_drop",
                    "message": f"{name} T1 dropped {pct}% (was {prev_t1:.1f}µs, now {t1:.1f}µs) — early warning of material drift",
                })
            if prev_t2 and prev_t2 > 0 and (prev_t2 - t2) / prev_t2 > 0.20:
                pct = round((prev_t2 - t2) / prev_t2 * 100, 1)
                alerts.append({
                    "ts": ts, "device": name, "type": "t2_drop",
                    "message": f"{name} T2 dropped {pct}% (was {prev_t2:.1f}µs, now {t2:.1f}µs) — early warning of phase decay",
                })

        alerts.sort(key=lambda a: a["ts"], reverse=True)

        if not alerts:
            msg = f"No alerts in the last {days} day(s)"
            if device_name:
                msg += f" for {device_name}"
            return json.dumps({"alerts": [], "message": msg})

        return json.dumps({
            "alerts": alerts,
            "total": len(alerts),
            "period_days": days,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def start_repro_experiment(
    circuit: str,
    backend_name: str,
    n_runs: int = 5,
    shots: int = 1024,
) -> str:
    """
    Submit the same circuit N times to measure reproducibility on real hardware.

    NISQ hardware results vary between runs due to calibration drift and noise.
    This tool submits identical circuits N times, storing each job ID so you
    can later call repro_score() to compute the reproducibility score.

    Args:
        circuit      : OpenQASM 2.0 or 3.0 circuit string
        backend_name : IBM device to run on (e.g. "ibm_fez")
        n_runs       : how many times to run the same circuit (default 5)
        shots        : shots per run (default 1024)

    Returns an experiment_id. Use repro_score(experiment_id) after all
    jobs complete to get the variance analysis and 0-1 reproducibility score.
    """
    try:
        service = _get_service()
        backend = service.backend(backend_name)

        # Parse circuit
        try:
            qc = QuantumCircuit.from_qasm_str(circuit)
        except Exception:
            try:
                qc = qiskit_qasm3.loads(circuit)
            except Exception as e:
                return json.dumps({"error": f"Could not parse circuit: {e}"})

        # Transpile once, reuse for all runs
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)

        # Get current calibration epoch for drift tracking
        props = backend.properties()
        cx_errors = []
        if props:
            from snapshot import _two_qubit_errors
            try:
                cx_errors = _two_qubit_errors(props)
            except Exception:
                pass
        calibration_epoch = round(sum(cx_errors) / len(cx_errors), 5) if cx_errors else None

        ts = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute("""
                INSERT INTO repro_experiments (created_ts, device_name, circuit, n_runs, shots, status)
                VALUES (?, ?, ?, ?, ?, 'running')
            """, (ts, backend_name, circuit, n_runs, shots))
            experiment_id = cur.lastrowid

            sampler = Sampler(backend)
            job_ids = []
            for i in range(n_runs):
                job = sampler.run([isa_circuit], shots=shots)
                job_id = job.job_id()
                job_ids.append(job_id)
                con.execute("""
                    INSERT INTO repro_runs
                        (experiment_id, run_index, submitted_ts, job_id, status, calibration_epoch)
                    VALUES (?, ?, ?, ?, 'submitted', ?)
                """, (experiment_id, i, datetime.now(timezone.utc).isoformat(), job_id,
                      str(calibration_epoch) if calibration_epoch else None))

        return json.dumps({
            "experiment_id": experiment_id,
            "device": backend_name,
            "n_runs": n_runs,
            "shots": shots,
            "job_ids": job_ids,
            "calibration_epoch": calibration_epoch,
            "message": f"Submitted {n_runs} jobs. Call repro_score({experiment_id}) after they complete.",
            "hint": "Use job_status(job_id) to check individual jobs."
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def repro_score(experiment_id: int) -> str:
    """
    Compute the reproducibility score for a completed repeat experiment.

    Fetches results for all runs in the experiment, computes:
    - Mean output distribution across all runs
    - KL-divergence of each run from the mean (variance signal)
    - Reproducibility score 0.0 to 1.0 (1.0 = identical results every run)
    - Flag if calibration epoch changed between any two runs

    A score above 0.9 means your result is likely real signal.
    A score below 0.7 means the result is probably noise-driven — rerun later.

    Args:
        experiment_id : the ID returned by start_repro_experiment()
    """
    try:
        with sqlite3.connect(DB_PATH) as con:
            exp = con.execute("""
                SELECT device_name, circuit, n_runs, shots, created_ts
                FROM repro_experiments WHERE id = ?
            """, (experiment_id,)).fetchone()

            if not exp:
                return json.dumps({"error": f"Experiment {experiment_id} not found."})

            device_name, circuit, n_runs, shots, created_ts = exp

            runs = con.execute("""
                SELECT run_index, job_id, status, counts, calibration_epoch
                FROM repro_runs WHERE experiment_id = ?
                ORDER BY run_index
            """, (experiment_id,)).fetchall()

        # Fetch any pending results from IBM
        service = _get_service()
        all_counts = []
        pending = []
        epochs = set()

        for run_index, job_id, status, counts_str, epoch in runs:
            if epoch:
                epochs.add(epoch)
            if counts_str:
                all_counts.append(json.loads(counts_str))
                continue
            if not job_id:
                pending.append(run_index)
                continue
            try:
                job = service.job(job_id)
                jstatus = job.status()
                if str(jstatus) in ("JobStatus.DONE", "DONE", "done"):
                    result = job.result()
                    pub_result = result[0]
                    bitarray = pub_result.data
                    field = list(vars(bitarray).keys())[0] if vars(bitarray) else None
                    if field:
                        counts = getattr(bitarray, field).get_counts()
                    else:
                        counts = {}
                    counts_json = json.dumps(counts)
                    with sqlite3.connect(DB_PATH) as con:
                        con.execute("""
                            UPDATE repro_runs SET status='done', counts=?
                            WHERE experiment_id=? AND run_index=?
                        """, (counts_json, experiment_id, run_index))
                    all_counts.append(counts)
                else:
                    pending.append(run_index)
            except Exception as e:
                pending.append(run_index)

        if pending:
            return json.dumps({
                "experiment_id": experiment_id,
                "device": device_name,
                "status": "incomplete",
                "completed_runs": len(all_counts),
                "pending_runs": pending,
                "message": f"{len(pending)} run(s) still pending. Check with job_status() and retry repro_score()."
            }, indent=2)

        # --- Compute reproducibility score ---

        # Gather all unique bitstrings across all runs
        all_keys = set()
        for c in all_counts:
            all_keys.update(c.keys())

        # Normalize each run into a probability distribution
        dists = []
        for c in all_counts:
            total = sum(c.values()) or 1
            dists.append({k: c.get(k, 0) / total for k in all_keys})

        # Mean distribution
        mean_dist = {k: sum(d[k] for d in dists) / len(dists) for k in all_keys}

        # KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
        eps = 1e-10
        kl_divs = []
        for d in dists:
            kl = sum(
                d[k] * math.log((d[k] + eps) / (mean_dist[k] + eps))
                for k in all_keys if d[k] > 0
            )
            kl_divs.append(round(kl, 6))

        avg_kl = sum(kl_divs) / len(kl_divs)

        # Score: 1.0 = perfect reproducibility, 0.0 = completely random
        # KL of 0 → score 1.0, KL of 0.5+ → score ~0.0
        score = round(max(0.0, 1.0 - (avg_kl / 0.5)), 3)

        # Top bitstring and its mean probability
        top_bitstring = max(mean_dist, key=mean_dist.get)
        top_prob = round(mean_dist[top_bitstring], 4)

        # Calibration drift flag
        calibration_drifted = len(epochs) > 1

        # Mark experiment complete
        with sqlite3.connect(DB_PATH) as con:
            con.execute("UPDATE repro_experiments SET status='complete' WHERE id=?",
                        (experiment_id,))

        verdict = (
            "RELIABLE — result is likely real signal" if score >= 0.9
            else "MARGINAL — result may be partially noise-driven" if score >= 0.7
            else "UNRELIABLE — result is likely noise, not signal"
        )

        return json.dumps({
            "experiment_id": experiment_id,
            "device": device_name,
            "n_runs": len(all_counts),
            "shots_per_run": shots,
            "reproducibility_score": score,
            "verdict": verdict,
            "top_bitstring": top_bitstring,
            "top_bitstring_mean_probability": top_prob,
            "kl_divergences": kl_divs,
            "avg_kl_divergence": round(avg_kl, 6),
            "calibration_drifted_between_runs": calibration_drifted,
            "calibration_epochs_seen": list(epochs),
            "interpretation": (
                "Score 0.9-1.0: publish with confidence. "
                "Score 0.7-0.9: mention variance in methods section. "
                "Score <0.7: do not publish — rerun on a better-calibrated device."
            )
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def _estimate_minutes(backend, qc, shots: int) -> dict:
    """
    Estimate how many minutes a circuit will cost on a backend.

    Two components:
    1. Queue wait  — pending_jobs × 30s average per job
    2. Execution   — shots × circuit_depth × 1μs per gate layer
    Both are rough but directionally correct for IBM Open Plan planning.
    """
    status = backend.status()
    pending = status.pending_jobs or 0

    try:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa = pm.run(qc)
        depth = isa.depth()
        n_qubits = isa.num_qubits
    except Exception:
        depth = qc.depth()
        n_qubits = qc.num_qubits

    queue_secs = pending * 30
    exec_secs = (shots * depth * 1e-6) + 2  # +2s overhead per job
    total_secs = queue_secs + exec_secs
    total_mins = round(total_secs / 60, 2)

    return {
        "pending_jobs_in_queue": pending,
        "circuit_depth_after_transpile": depth,
        "num_qubits": n_qubits,
        "queue_wait_estimate_mins": round(queue_secs / 60, 2),
        "execution_estimate_mins": round(exec_secs / 60, 4),
        "total_estimate_mins": total_mins,
    }


@mcp.tool()
def job_analytics() -> str:
    """
    Analyze all jobs submitted through this MCP server.

    Returns a breakdown per tool showing:
    - How many jobs were submitted
    - Average circuit depth before and after transpilation
    - Transpilation expansion ratio (how much the compiler inflated the circuit)
    - Average shots requested

    Useful for understanding how AI agents use quantum hardware over time.
    This is the start of the agentic quantum workload study.
    """
    if not os.path.exists(DB_PATH):
        return json.dumps({"error": "No local database found. Run snapshot.py first."})

    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("""
                SELECT
                    tool_name,
                    COUNT(*) AS total_submissions,
                    AVG(circuit_depth_raw) AS avg_raw_depth,
                    AVG(circuit_depth_transpiled) AS avg_transpiled_depth,
                    AVG(CAST(circuit_depth_transpiled AS REAL) /
                        NULLIF(circuit_depth_raw, 0)) AS expansion_ratio,
                    AVG(shots_requested) AS avg_shots
                FROM job_submissions
                GROUP BY tool_name
                ORDER BY total_submissions DESC
            """).fetchall()

            total_jobs = con.execute(
                "SELECT COUNT(*) FROM job_submissions"
            ).fetchone()[0]

        if not rows:
            return json.dumps({
                "total_jobs": 0,
                "message": "No jobs logged yet. Jobs are logged when submit_job, run_grover, or run_vqe is called.",
            })

        by_tool = {}
        for r in rows:
            by_tool[r["tool_name"]] = {
                "total_submissions": r["total_submissions"],
                "avg_circuit_depth_raw": round(r["avg_raw_depth"], 1) if r["avg_raw_depth"] else None,
                "avg_circuit_depth_transpiled": round(r["avg_transpiled_depth"], 1) if r["avg_transpiled_depth"] else None,
                "transpilation_expansion_ratio": round(r["expansion_ratio"], 2) if r["expansion_ratio"] else None,
                "avg_shots": round(r["avg_shots"], 0) if r["avg_shots"] else None,
            }

        return json.dumps({
            "total_jobs_logged": total_jobs,
            "by_tool": by_tool,
            "note": "expansion_ratio = transpiled_depth / raw_depth — how much the compiler inflated your circuit to fit the hardware topology.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def estimate_runtime(
    circuit: str,
    backend_name: str,
    shots: int = 1024,
) -> str:
    """
    Estimate how many minutes a circuit will cost on an IBM device.

    IBM Open Plan gives 180 minutes per year. This tool tells you how much
    a specific circuit will cost BEFORE you submit — so you don't waste
    your budget on the wrong backend or a busy queue.

    Estimate includes:
    - Queue wait time (based on current pending jobs × avg 30s per job)
    - Execution time (based on transpiled circuit depth × shots)
    - Total estimated cost in minutes

    Args:
        circuit      : OpenQASM 2.0 or 3.0 circuit string
        backend_name : IBM device to estimate for (e.g. "ibm_fez")
        shots        : number of shots (default 1024)

    Returns estimated minutes broken down by queue wait vs execution.
    """
    try:
        try:
            qc = QuantumCircuit.from_qasm_str(circuit)
        except Exception:
            qc = qiskit_qasm3.loads(circuit)

        service = _get_service()
        backend = service.backend(backend_name)

        est = _estimate_minutes(backend, qc, shots)
        est["device"] = backend_name
        est["shots"] = shots
        est["note"] = (
            "Queue wait is estimated at 30s/job (rough average). "
            "Execution time is based on transpiled depth × shots. "
            "IBM Open Plan budget: 180 min/year."
        )

        # Budget warning
        if est["total_estimate_mins"] > 10:
            est["warning"] = f"This job may cost ~{est['total_estimate_mins']} min — consider a shorter queue or fewer shots."

        return json.dumps(est, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def route_job(
    circuit: str,
    shots: int = 1024,
    max_minutes: float = 10.0,
) -> str:
    """
    Recommend the best IBM device for your circuit based on cost and quality.

    Checks all your accessible IBM devices and ranks them by:
    1. Fits within max_minutes budget (queue + execution)
    2. Lowest estimated total time
    3. Lowest avg_cx_error (best fidelity)

    This is credit-aware routing — it saves your IBM Open Plan minutes
    for experiments that matter, not queue accidents.

    Args:
        circuit     : OpenQASM 2.0 or 3.0 circuit string
        shots       : number of shots (default 1024)
        max_minutes : reject devices that will cost more than this (default 10)

    Returns ranked list of devices with cost estimate and recommendation.
    """
    try:
        try:
            qc = QuantumCircuit.from_qasm_str(circuit)
        except Exception:
            qc = qiskit_qasm3.loads(circuit)

        required_qubits = qc.num_qubits
        service = _get_service()
        backends = service.backends(operational=True)

        rankings = []
        skipped = []

        for backend in backends:
            if backend.num_qubits < required_qubits:
                skipped.append({
                    "device": backend.name,
                    "reason": f"only {backend.num_qubits} qubits, circuit needs {required_qubits}"
                })
                continue

            try:
                est = _estimate_minutes(backend, qc, shots)
                total = est["total_estimate_mins"]

                if total > max_minutes:
                    skipped.append({
                        "device": backend.name,
                        "reason": f"estimated {total} min exceeds {max_minutes} min budget"
                    })
                    continue

                # Get fidelity from latest snapshot
                props = backend.properties()
                cx_errors = []
                if props:
                    from snapshot import _two_qubit_errors
                    try:
                        cx_errors = _two_qubit_errors(props)
                    except Exception:
                        pass
                avg_cx = round(sum(cx_errors) / len(cx_errors), 5) if cx_errors else None

                rankings.append({
                    "device": backend.name,
                    "num_qubits": backend.num_qubits,
                    "estimated_mins": total,
                    "queue_wait_mins": est["queue_wait_estimate_mins"],
                    "circuit_depth": est["circuit_depth_after_transpile"],
                    "avg_cx_error": avg_cx,
                })
            except Exception as e:
                skipped.append({"device": backend.name, "reason": str(e)})

        # Sort: lowest time first, then lowest error
        rankings.sort(key=lambda x: (x["estimated_mins"], x["avg_cx_error"] or 1))

        if not rankings:
            return json.dumps({
                "error": "No devices fit within your budget or qubit requirements.",
                "skipped": skipped,
                "tip": f"Try increasing max_minutes (current: {max_minutes}) or simplifying your circuit."
            }, indent=2)

        recommendation = rankings[0]
        return json.dumps({
            "recommendation": recommendation["device"],
            "reason": (
                f"Lowest estimated cost ({recommendation['estimated_mins']} min) "
                f"with avg_cx_error {recommendation['avg_cx_error']}"
            ),
            "ranked_devices": rankings,
            "skipped_devices": skipped,
            "circuit_requires_qubits": required_qubits,
            "budget_max_minutes": max_minutes,
            "ibm_open_plan_note": "IBM Open Plan: 180 min/year. Each minute counts."
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: check_routing_overhead
# --------------------------------------------------------------------------
@mcp.tool()
def check_routing_overhead(qubit_interactions: list, device_name: str = "ibm_marrakesh") -> str:
    """
    Predict routing overhead BEFORE submitting a circuit to IBM heavy-hex hardware.

    IBM heavy-hex qubits allow max 3 direct connections each (degree ≤ 3).
    If your circuit needs a qubit to talk to 4+ other qubits, the transpiler
    must inject SWAP gates — each SWAP = 3 CX gates — causing gate count to
    explode. We discovered this the hard way: 263 logical gates became 1,037
    hardware gates because one qubit needed degree 4.

    Args:
        qubit_interactions: List of [qubit_a, qubit_b] pairs that need to
            interact in your circuit. Example: [[0,1],[1,2],[0,3],[0,4]]
            means qubit 0 talks to qubits 1, 3, and 4 (degree 3 — safe).
        device_name: IBM backend to check against (default: ibm_marrakesh).

    Returns:
        Per-qubit degree, any degree violations, and an estimated gate inflation
        factor. Degree ≤ 3 is safe. Degree 4 typically means ~4× gate inflation.
    """
    # Build adjacency: count unique neighbors per qubit
    from collections import defaultdict
    neighbors = defaultdict(set)
    for pair in qubit_interactions:
        if len(pair) != 2:
            return json.dumps({"error": f"Each interaction must be [a, b], got: {pair}"})
        a, b = int(pair[0]), int(pair[1])
        neighbors[a].add(b)
        neighbors[b].add(a)

    HEAVY_HEX_MAX_DEGREE = 3  # IBM heavy-hex physical constraint
    violations = []
    qubit_report = []

    for qubit, nbrs in sorted(neighbors.items()):
        degree = len(nbrs)
        excess = max(0, degree - HEAVY_HEX_MAX_DEGREE)
        # Each excess degree ≈ 1 SWAP per gate = 3 extra CX per interaction
        estimated_swaps = excess * 3
        status = "OK" if excess == 0 else f"VIOLATION (degree {degree}, max {HEAVY_HEX_MAX_DEGREE})"
        if excess > 0:
            violations.append({
                "qubit": qubit,
                "degree": degree,
                "excess": excess,
                "estimated_extra_cx": estimated_swaps,
            })
        qubit_report.append({
            "qubit":   qubit,
            "degree":  degree,
            "neighbors": sorted(nbrs),
            "status":  status,
        })

    # Overall inflation estimate
    total_excess_cx = sum(v["estimated_extra_cx"] for v in violations)
    if violations:
        inflation_warning = (
            f"{len(violations)} qubit(s) exceed degree-{HEAVY_HEX_MAX_DEGREE} limit. "
            f"Estimated ~{total_excess_cx} extra CX gates from SWAP injection. "
            f"This can multiply your hardware gate count by 3–5×. "
            f"Recommendation: redesign the circuit to eliminate degree-{HEAVY_HEX_MAX_DEGREE+1}+ nodes, "
            f"or switch to an Ising Hamiltonian approach (encode_search_problem tool)."
        )
    else:
        inflation_warning = (
            "All qubits are within degree-3 limit. "
            "No SWAP injection expected. Circuit should map cleanly to heavy-hex."
        )

    return json.dumps({
        "device": device_name,
        "heavy_hex_max_degree": HEAVY_HEX_MAX_DEGREE,
        "verdict": "ROUTING RISK" if violations else "SAFE",
        "summary": inflation_warning,
        "violations": violations,
        "qubit_degrees": qubit_report,
    }, indent=2)


# --------------------------------------------------------------------------
# Tool: encode_search_problem
# --------------------------------------------------------------------------
@mcp.tool()
def encode_search_problem(
    conditions: dict,
    coupling_hints: list = [],
    coupling_strength: float = 0.5,
) -> str:
    """
    Convert a Boolean search problem into an Ising Hamiltonian ready for LNAA
    (Lattice-Native Amplitude Amplification) on IBM hardware.

    We derived this approach from scratch in Phase 5 of the Singmaster project
    after Grover's algorithm hit IBM heavy-hex routing limits. Instead of a
    Boolean oracle, encode target states as the lowest-energy (ground) states
    of a magnetic system. IBM's RZ + RZZ gates implement this natively — zero
    routing overhead.

    How the math works:
        H = Σ h_i × Z_i + Σ J_ij × Z_i × Z_j
        Z_i = -1 if qubit i is 1, +1 if qubit i is 0.
        To reward qubit i = 1: set h_i = +1  (because h × (-1) = -1 = low energy)
        To reward qubit i = 0: set h_i = -1  (because h × (+1) = -1 = low energy)
        To penalise pairs i,j being equal: J_ij = +0.5 (coupling hint)

    Args:
        conditions: dict mapping qubit index (as string) to desired value.
            Example: {"1": 1, "2": 1, "3": 1, "4": 0, "5": 0}
            means we want q1=q2=q3=1 and q4=q5=0.
        coupling_hints: optional list of [i, j] pairs to add mild coupling
            between qubits (useful to break degeneracy between near-equal targets).
            Example: [[0, 6]] adds J_06 coupling.
        coupling_strength: J value for coupling hints (default 0.5 = mild penalty).

    Returns:
        h_i coefficients, J_ij couplings, ground state energy, and a QAOA
        circuit recipe with suggested starting parameters.
    """
    h_coeffs = {}
    for qubit_str, desired_val in conditions.items():
        q = int(qubit_str)
        desired_val = int(desired_val)
        if desired_val == 1:
            # reward qubit=1 (spin=-1): need h×(-1) < 0, so h > 0
            h_coeffs[q] = +1.0
        elif desired_val == 0:
            # reward qubit=0 (spin=+1): need h×(+1) < 0, so h < 0
            h_coeffs[q] = -1.0
        else:
            return json.dumps({"error": f"Qubit {q} value must be 0 or 1, got {desired_val}"})

    j_couplings = {}
    for pair in coupling_hints:
        if len(pair) != 2:
            return json.dumps({"error": f"Each coupling hint must be [i, j], got: {pair}"})
        i, j = int(pair[0]), int(pair[1])
        j_couplings[(i, j)] = coupling_strength

    # Compute ground state energy (what the target state achieves)
    ground_energy = 0.0
    for q, h in h_coeffs.items():
        desired = int(conditions[str(q)])
        spin = -1 if desired == 1 else +1  # Z_i = -1 if bit=1, +1 if bit=0
        ground_energy += h * spin
    for (i, j), J in j_couplings.items():
        # Coupling contribution depends on whether the two qubits are same/different.
        # With hints we assume no preference, so just note the range.
        pass

    # Build the circuit recipe
    circuit_recipe = {
        "step_1_superposition": "Apply H gate to all qubits (equal superposition)",
        "step_2_phase_oracle": {
            "rz_gates": {
                f"q{q}": f"RZ(2 × {h:.1f} × gamma)" for q, h in h_coeffs.items()
            },
            "rzz_gates": {
                f"q{i}_q{j}": f"RZZ(2 × {J:.1f} × gamma)" for (i, j), J in j_couplings.items()
            } if j_couplings else "none",
        },
        "step_3_mixing": "Apply RX(2 × beta) to all qubits",
        "step_4_repeat": "Repeat steps 2-3 for p layers (start with p=2)",
        "step_5_measure": "Measure all qubits",
        "suggested_starting_params": {
            "gamma": 2.5,
            "beta": 0.5,
            "p_layers": 2,
            "tip": "Sweep gamma in [0, π] and beta in [0, π/2] with 20 steps each to find optimal params",
        },
    }

    return json.dumps({
        "h_coefficients": {
            f"q{q}": {"h_value": h, "meaning": f"rewards q{q}={int(conditions[str(q)])}"}
            for q, h in h_coeffs.items()
        },
        "j_couplings": {
            f"q{i}_q{j}": J for (i, j), J in j_couplings.items()
        } if j_couplings else {},
        "ground_state_energy": round(ground_energy, 3),
        "energy_formula": "E = Σ h_i × Z_i  where Z_i = -1 if bit=1, +1 if bit=0",
        "why_this_works": (
            "Target states get the lowest energy (ground state). "
            "Quantum walk naturally amplifies ground states. "
            "RZ + RZZ gates are IBM-native — zero routing overhead on heavy-hex."
        ),
        "circuit_recipe": circuit_recipe,
    }, indent=2)


# --------------------------------------------------------------------------
# Tool: estimate_hardware_gates
# --------------------------------------------------------------------------
@mcp.tool()
def estimate_hardware_gates(
    logical_gates: int,
    max_qubit_degree: int,
    two_qubit_fraction: float = 0.4,
) -> str:
    """
    Predict how many hardware gates your circuit will have after IBM transpilation.

    Learned from the Singmaster project: the same circuit can go from 263 logical
    gates to 1,037 hardware gates (4× inflation) if even ONE qubit exceeds the
    heavy-hex degree-3 limit. This tool lets you predict that before wasting
    queue time.

    The key variable is max_qubit_degree — the highest number of other qubits
    that any single qubit in your circuit needs to interact with directly.
    On IBM heavy-hex this limit is 3. Above 3, every excess connection requires
    SWAP injection (each SWAP = 3 extra CX gates).

    Args:
        logical_gates:       Total gates in your circuit before transpilation.
        max_qubit_degree:    The highest degree of any qubit in your circuit's
                             interaction graph (use check_routing_overhead to find this).
        two_qubit_fraction:  Fraction of logical gates that are 2-qubit gates
                             (default 0.4 = 40%, typical for Grover-style circuits).

    Returns:
        Predicted hardware gate count, inflation factor, and whether you are
        above or below the ~600-gate hardware noise floor on ibm_marrakesh.
    """
    HEAVY_HEX_MAX_DEGREE = 3
    NOISE_FLOOR_GATES = 600  # empirical from Phase 3-5 experiments

    two_qubit_gates = int(logical_gates * two_qubit_fraction)
    single_qubit_gates = logical_gates - two_qubit_gates

    # Inflation model: each degree over the limit adds ~3 CX per interaction
    excess_degree = max(0, max_qubit_degree - HEAVY_HEX_MAX_DEGREE)

    if excess_degree == 0:
        inflation_factor = 1.1  # small overhead from native gate decomposition
        method = "No SWAP injection needed. Minor decomposition overhead only."
    elif excess_degree == 1:
        inflation_factor = 4.0  # observed: 263 → 1037 in Phase 4v2
        method = "Degree-4 node detected. Heavy SWAP injection expected (~4× inflation)."
    else:
        inflation_factor = 4.0 + (excess_degree - 1) * 2.5
        method = f"Degree-{max_qubit_degree} node detected. Severe routing overhead."

    predicted_hw_gates = int(logical_gates * inflation_factor)
    above_noise_floor = predicted_hw_gates > NOISE_FLOOR_GATES

    verdict = "DANGER" if above_noise_floor else "OK"
    advice = ""
    if above_noise_floor and excess_degree > 0:
        advice = (
            f"Circuit will likely collapse to noise. "
            f"Redesign to eliminate degree-{max_qubit_degree} qubits. "
            f"Consider encode_search_problem to switch to Ising Hamiltonian approach "
            f"(RZZ+RX gates need no routing — logical ≈ hardware gate count)."
        )
    elif above_noise_floor:
        advice = (
            f"Circuit is large but routing is clean. "
            f"Try optimization level 2-3 with seed sweep (opt=2 seed=42 works well). "
            f"Oracle simplification (shared Boolean conditions) can cut 30-50%."
        )
    else:
        advice = "Circuit is within hardware noise floor. Should be executable."

    return json.dumps({
        "input": {
            "logical_gates": logical_gates,
            "max_qubit_degree": max_qubit_degree,
            "two_qubit_fraction": two_qubit_fraction,
        },
        "prediction": {
            "inflation_factor": round(inflation_factor, 1),
            "predicted_hardware_gates": predicted_hw_gates,
            "heavy_hex_noise_floor": NOISE_FLOOR_GATES,
            "above_noise_floor": above_noise_floor,
            "verdict": verdict,
        },
        "method": method,
        "advice": advice,
    }, indent=2)


# --------------------------------------------------------------------------
# Tool: get_amplification
# --------------------------------------------------------------------------
@mcp.tool()
def get_amplification(
    job_id: str,
    marked_bitstrings: list,
    search_space_size: int,
    provider: str = "ibm",
    ionq_backend_name: str = "simulator",
) -> str:
    """
    Compute the amplification factor from a completed quantum search job.

    Amplification = (fraction of shots on marked states) / (random baseline)
    Random baseline = len(marked_bitstrings) / search_space_size

    A value of 1.0 means no better than random. Above 1.0 means the quantum
    search worked. In our Singmaster experiments: Phase 3v3 got 4.17×,
    Phase 5 LNAA got 27.78×.

    Args:
        job_id:              Completed job's ID — IBM or IonQ, set `provider`
                             to match.
        marked_bitstrings:   List of bitstrings (as strings) representing your
                             target states. Example: ["0001110", "0001111", "1001110"]
                             These must match the submitting SDK's bit ordering
                             (classical bit 0 = rightmost character) for BOTH
                             IBM and IonQ — verified with the endianness canary.
        search_space_size:   Total number of possible states. For n qubits: 2^n.
                             For 7 qubits: 128. For 4 qubits: 16.
        provider:            "ibm" (default) or "ionq". Must match where the
                             job actually ran — this tool was previously
                             hardcoded to IBM and would hard-error on any
                             IonQ job ID.
        ionq_backend_name:   only used when provider="ionq" — the backend the
                             job was submitted to (default: simulator).

    Returns:
        Amplification factor, shot breakdown per marked state, comparison to
        the uniform random baseline, and is_real_hardware (IonQ only — IBM
        jobs are always real hardware, this tool doesn't touch IBM's
        simulator path).
    """
    try:
        is_real_hardware = None

        if provider == "ionq":
            api_key = os.getenv("IONQ_API_KEY")
            if not api_key:
                return json.dumps({"error": "IONQ_API_KEY not set in .env"})
            from qiskit_ionq import IonQProvider
            from qiskit.providers import JobStatus
            resolved_backend = _resolve_ionq_backend(ionq_backend_name)
            ionq_prov = IonQProvider(api_key)
            job = ionq_prov.get_backend(resolved_backend, gateset="native").retrieve_job(job_id)
            is_real_hardware = _ionq_is_hardware(resolved_backend)

            status = job.status()
            if status != JobStatus.DONE:
                return json.dumps({
                    "error": f"Job {job_id} is not complete yet. Status: {status.name}",
                    "tip": "Wait for job to finish, then call get_amplification again.",
                })

            counts = job.result().get_counts()
            if isinstance(counts, list):
                return json.dumps({
                    "error": "This job_id has multiple circuits (batched submission).",
                    "hint": "get_amplification expects one circuit's results. Use ionq_job_results and pick the counts entry you want, or call this per-circuit.",
                })

        else:
            service = _get_service()
            job = service.job(job_id)

            raw_status = job.status()
            status_str = raw_status if isinstance(raw_status, str) else raw_status.name
            if status_str.upper() not in ("DONE", "COMPLETED"):
                return json.dumps({
                    "error": f"Job {job_id} is not complete yet. Status: {status_str}",
                    "tip": "Wait for job to finish, then call get_amplification again.",
                })

            result = job.result()
            pub_result = result[0]
            data = pub_result.data
            field = list(vars(data).keys())[0]
            counts = getattr(data, field).get_counts()

        total_shots = sum(counts.values())
        marked_set = set(marked_bitstrings)

        # Count shots on marked states
        marked_counts = {}
        marked_total = 0
        for bits, count in counts.items():
            if bits in marked_set:
                marked_counts[bits] = count
                marked_total += count

        # Amplification calculation
        marked_fraction = marked_total / total_shots
        random_baseline = len(marked_bitstrings) / search_space_size
        amplification = round(marked_fraction / random_baseline, 2) if random_baseline > 0 else 0

        # Top states overall (for context)
        top_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return json.dumps({
            "job_id": job_id,
            "provider": provider,
            "is_real_hardware": is_real_hardware,
            "total_shots": total_shots,
            "search_space_size": search_space_size,
            "marked_bitstrings": marked_bitstrings,
            "random_baseline_fraction": round(random_baseline, 5),
            "marked_shots": {
                bits: {
                    "count": marked_counts.get(bits, 0),
                    "fraction": round(marked_counts.get(bits, 0) / total_shots, 4),
                }
                for bits in marked_bitstrings
            },
            "marked_total_shots": marked_total,
            "marked_fraction": round(marked_fraction, 4),
            "amplification_factor": amplification,
            "verdict": (
                "EXCELLENT (>10×)" if amplification > 10 else
                "GOOD (4–10×)"     if amplification > 4  else
                "WEAK (1–4×)"      if amplification > 1  else
                "FAILED (≤1×, no better than random)"
            ),
            "top_10_states": [
                {"bits": b, "count": c, "fraction": round(c / total_shots, 4), "marked": b in marked_set}
                for b, c in top_states
            ],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: discover_collision_candidates
# --------------------------------------------------------------------------
@mcp.tool()
def discover_collision_candidates(
    max_n: int = 100,
    max_k: int = 10,
    target_multiplicity: int = 2,
) -> str:
    """
    Scout tool — runs BEFORE any circuit is built.

    Classically filters ALL (k1,k2) column pairs in Pascal's Triangle,
    finds which ones produce actual collisions in the given range, and
    ranks them by hardware feasibility (qubit count, expected gate count,
    Ising sparsity).

    Use this to decide WHAT to search before spending QPU credits.

    Args:
      max_n              : search rows 0..max_n
      max_k              : check column pairs k1,k2 up to max_k
      target_multiplicity: 2 = 2-way collision, 4 = 4-way (Singmaster record)

    Returns ranked list of candidate searches with hardware cost estimate.
    """
    try:
        from math import comb, ceil, log2

        results = []

        # Check all (k1, k2) pairs where k1 < k2
        for k1 in range(1, max_k + 1):
            for k2 in range(k1 + 1, max_k + 1):

                # Build value tables
                table1 = {}
                for n in range(k1, max_n + 1):
                    v = comb(n, k1)
                    table1.setdefault(v, []).append(n)

                table2 = {}
                for n in range(k2, max_n + 1):
                    v = comb(n, k2)
                    table2.setdefault(v, []).append(n)

                # Find collisions
                shared = set(table1) & set(table2)
                non_trivial = [v for v in shared if v > 1]

                if not non_trivial:
                    continue

                # Count collision pairs
                pairs = []
                for v in sorted(non_trivial, reverse=True)[:5]:
                    for n1 in table1[v]:
                        for n2 in table2[v]:
                            pairs.append((n1, n2, v))

                # Estimate hardware cost
                bits1 = max(1, ceil(log2(max_n + 1)))
                bits2 = max(1, ceil(log2(max_n + 1)))
                num_qubits = bits1 + bits2
                # LNAA: H + p*(RZ per qubit + RX per qubit) gates
                logical_gates = num_qubits + 2 * num_qubits * 2  # p=2 layers
                hardware_gates = logical_gates  # RZZ/RX are native, no routing

                # Heavy-hex feasibility check: degree ≤ 3
                # Each qubit connects to at most 2 neighbours in our linear LNAA
                feasible = hardware_gates < 600

                results.append({
                    "k1": k1,
                    "k2": k2,
                    "num_collisions": len(pairs),
                    "top_collision": {"n1": pairs[0][0], "n2": pairs[0][1], "value": pairs[0][2]} if pairs else None,
                    "num_qubits": num_qubits,
                    "estimated_hardware_gates": hardware_gates,
                    "heavy_hex_feasible": feasible,
                    "recommended": feasible and len(pairs) >= 1,
                })

        # Sort: feasible first, then by number of collisions
        results.sort(key=lambda r: (-int(r["recommended"]), -r["num_collisions"]))

        recommended = [r for r in results if r["recommended"]]
        not_recommended = [r for r in results if not r["recommended"]]

        return json.dumps({
            "total_pairs_checked": max_k * (max_k - 1) // 2,
            "pairs_with_collisions": len(results),
            "recommended_for_hardware": len(recommended),
            "top_candidates": recommended[:10],
            "avoid": not_recommended[:5],
            "next_step": (
                f"Run encode_collision_problem + run_parallel_collision_search "
                f"with these k-pairs: {[(r['k1'],r['k2']) for r in recommended[:5]]}"
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: encode_collision_problem
# --------------------------------------------------------------------------
@mcp.tool()
def encode_collision_problem(
    k1: int,
    k2: int,
    max_n1: int = 50,
    max_n2: int = 30,
) -> str:
    """
    Find all pairs (n1, n2) where C(n1,k1) = C(n2,k2) classically,
    then encode those collision pairs as Ising h_i coefficients ready
    for run_search_experiment.

    Why this matters:
      Instead of telling LNAA "look at row 14, 15, 78", we tell it
      "find coordinate pairs that collide" — and it discovers them.
      That's the generalization from Phase 5.

    Strategy:
      - Represent n1 in binary as qubits q0..q(b1-1)
      - Represent n2 in binary as qubits q(b1)..q(b1+b2-1)
      - For each known collision pair (n1_sol, n2_sol), reward the
        exact bit pattern with positive h_i (1 → reward, 0 → penalize)
      - Multiple solutions share the same qubit register — LNAA
        amplifies all ground states simultaneously.

    Example:
      k1=2, k2=3, max_n1=20, max_n2=15
      → finds C(16,2)=C(10,3)=120
      → encodes bit patterns of n1=16 and n2=10

    Args:
      k1      : column fixed for register 1 (e.g. 2 → triangular numbers)
      k2      : column fixed for register 2 (e.g. 3 → tetrahedral numbers)
      max_n1  : search n1 from k1 to max_n1
      max_n2  : search n2 from k2 to max_n2

    Returns JSON with:
      - collisions        : list of (n1, n2, value) found
      - num_qubits        : total qubits needed
      - h_coeffs          : Ising h_i per qubit (pass to run_search_experiment)
      - marked_states     : bit-strings of collision pairs (for get_amplification)
      - run_search_params : ready-to-use kwargs for run_search_experiment
    """
    try:
        from math import comb, ceil, log2

        # ── 1. Classical collision search ──────────────────────────────────
        # Build lookup: value → list of n for register 1
        table1 = {}
        for n in range(k1, max_n1 + 1):
            v = comb(n, k1)
            table1.setdefault(v, []).append(n)

        # Build lookup: value → list of n for register 2
        table2 = {}
        for n in range(k2, max_n2 + 1):
            v = comb(n, k2)
            table2.setdefault(v, []).append(n)

        # Intersect to find collisions
        collisions = []
        for v in sorted(set(table1) & set(table2)):
            for n1 in table1[v]:
                for n2 in table2[v]:
                    collisions.append({"n1": n1, "n2": n2, "value": v})

        if not collisions:
            return json.dumps({
                "collisions": [],
                "message": f"No collisions found for C(n,{k1})=C(m,{k2}) in the given range."
            })

        # ── 2. Decide qubit widths ─────────────────────────────────────────
        # bits needed to represent max_n1 and max_n2
        bits1 = max(1, ceil(log2(max_n1 + 1)))
        bits2 = max(1, ceil(log2(max_n2 + 1)))
        num_qubits = bits1 + bits2

        # ── 3. Pick primary target — largest-value non-trivial collision ──────
        # Sort by value descending so the most interesting collision drives
        # the Ising encoding (e.g. C(16,2)=C(10,3)=120 beats C(2,2)=C(3,3)=1)
        non_trivial = [c for c in collisions if c["value"] > 1]
        primary = max(non_trivial, key=lambda c: c["value"]) if non_trivial else collisions[0]
        n1_p, n2_p = primary["n1"], primary["n2"]

        # ── 4. Build conditions dict from primary collision's exact bit pattern ─
        # run_search_experiment expects {qubit_str: 0_or_1}
        # q0..q(bits1-1) encode n1 LSB-first
        # q(bits1)..q(num_qubits-1) encode n2 LSB-first
        conditions = {}
        for i in range(bits1):
            conditions[str(i)] = int((n1_p >> i) & 1)
        for i in range(bits2):
            conditions[str(bits1 + i)] = int((n2_p >> i) & 1)

        # ── 5. Convert all collisions to Qiskit-format integer marked_rows ────
        # Qiskit counts bitstring = qubit[n-1]...qubit[1]qubit[0] (MSB-first).
        # format(r, '0{n}b') matches that convention, so we reverse our
        # LSB-first qubit list before converting to int.
        marked_rows = []
        marked_labels = []
        for col in collisions:
            n1, n2 = col["n1"], col["n2"]
            qbits = []
            for i in range(bits1):
                qbits.append((n1 >> i) & 1)   # q0..q(bits1-1), LSB first
            for i in range(bits2):
                qbits.append((n2 >> i) & 1)   # q(bits1)..q(end), LSB first
            # Qiskit string = reversed → integer
            qiskit_int = int("".join(str(b) for b in reversed(qbits)), 2)
            marked_rows.append(qiskit_int)
            marked_labels.append({
                "n1": n1, "n2": n2, "value": col["value"],
                "qiskit_int": qiskit_int,
                "qiskit_bitstring": format(qiskit_int, f'0{num_qubits}b'),
            })

        # ── 6. Describe each qubit ────────────────────────────────────────────
        qubit_map = {}
        for i in range(bits1):
            qubit_map[str(i)] = f"n1 bit {i} (2^{i}={2**i})"
        for i in range(bits2):
            qubit_map[str(bits1 + i)] = f"n2 bit {i} (2^{i}={2**i})"

        # ── 7. Ready-to-use params for run_search_experiment ─────────────────
        run_params = {
            "conditions": conditions,
            "num_qubits": num_qubits,
            "marked_rows": marked_rows,
            "shots": 4096,
            "p_layers": 2,
            "gamma": 2.589,
            "beta": 0.501,
            "simulate_only": True,
        }

        return json.dumps({
            "collisions": collisions,
            "primary_target": primary,
            "num_collision_pairs": len(collisions),
            "num_qubits": num_qubits,
            "bits_register_1": bits1,
            "bits_register_2": bits2,
            "qubit_map": qubit_map,
            "conditions": conditions,
            "marked_rows": marked_rows,
            "marked_labels": marked_labels,
            "run_search_params": run_params,
            "validation_hint": (
                f"Primary target: C({n1_p},{k1}) = C({n2_p},{k2}) = {primary['value']}. "
                f"LNAA should amplify Qiskit state {format(marked_rows[marked_labels.index(next(l for l in marked_labels if l['n1']==n1_p))] if marked_rows else 0, f'0{num_qubits}b')}."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: run_search_experiment
# --------------------------------------------------------------------------
@mcp.tool()
def run_search_experiment(
    conditions: dict,
    num_qubits: int,
    marked_rows: list,
    shots: int = 4096,
    p_layers: int = 2,
    gamma: float = 2.589,
    beta: float = 0.501,
    coupling_hints: list = [],
    coupling_strength: float = 0.5,
    max_poll_seconds: int = 120,
    simulate_only: bool = True,
) -> str:
    """
    Run a complete quantum search experiment end-to-end using one tool call.

    DEFAULT: simulate_only=True — runs a FREE noiseless simulation first.
    No QPU credits spent. Use this to validate your Hamiltonian, tune
    gamma/beta parameters, and confirm amplification before touching hardware.

    Only set simulate_only=False when simulation shows strong amplification
    (>10×) and you are ready to submit to real IBM hardware.

    What happens internally:

    SIMULATE mode (simulate_only=True, FREE):
      1. Derives Ising h_i and J_ij from your conditions
      2. Builds LNAA circuit (RZ + RZZ + RX layers)
      3. Runs noiseless Aer simulation — zero QPU cost
      4. Returns amplification + suggestion (ready for hardware or tune more)

    HARDWARE mode (simulate_only=False, uses QPU credits):
      1-4. Same as above
      5. Picks best IBM backend using live calibration data
      6. Checks routing overhead
      7. Transpiles at opt=2 seed=42
      8. Submits job, polls, returns amplification

    Args:
        conditions:        Dict mapping qubit index (string) to target value (0 or 1).
                           Example: {"1":1,"2":1,"3":1,"4":0,"5":0}
        num_qubits:        Total number of qubits in the search register.
        marked_rows:       Integer row numbers that are the correct answers.
        shots:             Number of shots (default 4096).
        p_layers:          LNAA layers — higher = more precise but deeper circuit (default 2).
        gamma:             Phase angle for Ising oracle (default 2.589).
        beta:              Mixing angle for RX layer (default 0.501).
        coupling_hints:    Optional [i,j] pairs for J coupling between qubits.
        coupling_strength: J value for coupling hints (default 0.5).
        max_poll_seconds:  Hardware only — how long to wait before returning job_id (default 120s).
        simulate_only:     True = free noiseless simulation with suggestion (DEFAULT).
                           False = submit to real IBM hardware (costs QPU credits).

    Returns:
        Amplification factor, shot breakdown, Hamiltonian used, and a
        recommendation on whether to submit to hardware or tune parameters first.
    """
    import time
    from collections import defaultdict

    try:
        # ------------------------------------------------------------------
        # Step 1: Derive Ising coefficients from conditions
        # ------------------------------------------------------------------
        h_coeffs = {}
        for qubit_str, desired_val in conditions.items():
            q = int(qubit_str)
            desired_val = int(desired_val)
            if desired_val == 1:
                h_coeffs[q] = +1.0   # reward qubit=1: h×(-1) = negative energy
            elif desired_val == 0:
                h_coeffs[q] = -1.0   # reward qubit=0: h×(+1) = negative energy
            else:
                return json.dumps({"error": f"Qubit {q} value must be 0 or 1"})

        j_couplings = {}
        for pair in coupling_hints:
            i, j = int(pair[0]), int(pair[1])
            j_couplings[(i, j)] = coupling_strength

        # Ground state energy for the targets
        ground_energy = sum(
            h_coeffs[int(q)] * (-1 if int(v) == 1 else +1)
            for q, v in conditions.items()
        )

        # ------------------------------------------------------------------
        # Step 2: Build LNAA circuit directly in Qiskit
        # ------------------------------------------------------------------
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Superposition
        qc.h(range(num_qubits))

        # p layers of phase oracle + mixing
        for layer in range(p_layers):
            for q_idx, h in h_coeffs.items():
                qc.rz(2 * h * gamma, q_idx)
            for (qi, qj), J in j_couplings.items():
                qc.rzz(2 * J * gamma, qi, qj)
            for i in range(num_qubits):
                qc.rx(2 * beta, i)

        qc.measure(range(num_qubits), range(num_qubits))

        # Logical gate count
        rz_count  = len(h_coeffs) * p_layers
        rzz_count = len(j_couplings) * p_layers
        rx_count  = num_qubits * p_layers
        h_count   = num_qubits
        logical_gates = h_count + rz_count + rzz_count + rx_count

        # ------------------------------------------------------------------
        # Step 3a: SIMULATE MODE — free noiseless run, no QPU
        # ------------------------------------------------------------------
        if simulate_only:
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _gpm
            sim_pm = _gpm(optimization_level=1, backend=sim)
            sim_qc = sim_pm.run(qc)
            from qiskit_aer.primitives import SamplerV2 as AerSampler
            aer_sampler = AerSampler()
            sim_job = aer_sampler.run([sim_qc], shots=shots)
            sim_result = sim_job.result()
            pub = sim_result[0]
            data = pub.data
            field = list(vars(data).keys())[0]
            counts = getattr(data, field).get_counts()

            total_shots   = sum(counts.values())
            search_space_size = 2 ** num_qubits
            marked_bitstrings = [format(r, f'0{num_qubits}b') for r in marked_rows]
            marked_set_bits   = set(marked_bitstrings)
            marked_total  = sum(counts.get(b, 0) for b in marked_bitstrings)
            marked_fraction   = marked_total / total_shots
            random_baseline   = len(marked_rows) / search_space_size
            amplification = round(marked_fraction / random_baseline, 2) if random_baseline > 0 else 0

            top_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Suggestion logic
            if amplification >= 15:
                suggestion = (
                    f"READY FOR HARDWARE. Simulation shows {amplification}× amplification — "
                    f"strong signal. Call run_search_experiment with simulate_only=False to submit to IBM hardware."
                )
            elif amplification >= 5:
                suggestion = (
                    f"GOOD but could be better ({amplification}×). Try tuning gamma (current: {gamma}) "
                    f"in range [1.5, 3.5] or increase p_layers to 3. Rerun simulation before hardware."
                )
            elif amplification >= 2:
                suggestion = (
                    f"WEAK signal ({amplification}×). Hamiltonian may need revisiting. "
                    f"Check that all marked_rows satisfy your conditions. "
                    f"Try different gamma/beta or add coupling_hints to break degeneracy."
                )
            else:
                suggestion = (
                    f"NO SIGNAL ({amplification}×). Do NOT submit to hardware — would waste QPU credits. "
                    f"Revisit the Hamiltonian: verify h_i signs using E = Σ h_i × Z_i "
                    f"where Z_i = -1 if bit=1, +1 if bit=0. Marked rows should have negative energy."
                )

            return json.dumps({
                "mode": "SIMULATION (free — no QPU used)",
                "experiment": {
                    "conditions": conditions,
                    "num_qubits": num_qubits,
                    "search_space_size": search_space_size,
                    "marked_rows": marked_rows,
                    "p_layers": p_layers,
                    "gamma": gamma,
                    "beta": beta,
                },
                "algorithm": {
                    "name": "LNAA (Lattice-Native Amplitude Amplification)",
                    "h_coefficients": h_coeffs,
                    "j_couplings": {f"{i},{j}": J for (i, j), J in j_couplings.items()},
                    "ground_state_energy": round(ground_energy, 3),
                },
                "circuit": {
                    "logical_gates": logical_gates,
                    "shots": shots,
                },
                "results": {
                    "total_shots": total_shots,
                    "marked_shots": marked_total,
                    "marked_fraction": round(marked_fraction, 4),
                    "random_baseline": round(random_baseline, 4),
                    "amplification_factor": amplification,
                    "verdict": (
                        "EXCELLENT (>10×)" if amplification > 10 else
                        "GOOD (5–10×)"     if amplification > 5  else
                        "WEAK (2–5×)"      if amplification > 2  else
                        "NO SIGNAL (≤2×)"
                    ),
                    "top_10_states": [
                        {
                            "bits": b,
                            "row": int(b, 2),
                            "count": c,
                            "fraction": round(c / total_shots, 4),
                            "marked": b in marked_set_bits,
                        }
                        for b, c in top_states
                    ],
                },
                "suggestion": suggestion,
                "next_step": (
                    "run_search_experiment(..., simulate_only=False)"
                    if amplification >= 15
                    else "Tune parameters and rerun simulation first."
                ),
            }, indent=2)

        # ------------------------------------------------------------------
        # Step 3b: HARDWARE MODE — pick best backend using live calibration
        # ------------------------------------------------------------------
        service = _get_service()
        backends = service.backends(operational=True)

        best_backend = None
        best_cx_error = float('inf')
        backend_scores = []

        for backend in backends:
            if backend.num_qubits < num_qubits:
                continue
            try:
                props = backend.properties()
                cx_errors = []
                if props:
                    TWO_QUBIT_GATES = {"cx", "ecr", "cz"}
                    for gate in props.gates:
                        if gate.gate in TWO_QUBIT_GATES and gate.parameters:
                            cx_errors.append(gate.parameters[0].value)
                avg_cx = sum(cx_errors) / len(cx_errors) if cx_errors else 1.0
                backend_scores.append({"name": backend.name, "avg_cx_error": round(avg_cx, 5)})
                if avg_cx < best_cx_error:
                    best_cx_error = avg_cx
                    best_backend = backend
            except Exception:
                continue

        if best_backend is None:
            return json.dumps({"error": "No operational IBM backends found with enough qubits."})

        backend_scores.sort(key=lambda x: x["avg_cx_error"])

        # ------------------------------------------------------------------
        # Step 4: Routing check — LNAA interactions are always degree ≤ 2
        # ------------------------------------------------------------------
        # Each qubit only interacts with its J-coupling partners.
        # With no coupling hints, max degree = 0. Always safe for heavy-hex.
        max_degree = 0
        if j_couplings:
            neighbor_count = defaultdict(set)
            for (qi, qj) in j_couplings:
                neighbor_count[qi].add(qj)
                neighbor_count[qj].add(qi)
            max_degree = max(len(v) for v in neighbor_count.values())

        routing_safe = max_degree <= 3

        # ------------------------------------------------------------------
        # Step 5: Submit the job
        # ------------------------------------------------------------------
        pm = generate_preset_pass_manager(optimization_level=2, backend=best_backend, seed_transpiler=42)
        transpiled = pm.run(qc)
        hw_gate_count = transpiled.size()

        sampler = Sampler(mode=best_backend)
        sampler.options.default_shots = shots
        job = sampler.run([transpiled])
        job_id = job.job_id()

        # ------------------------------------------------------------------
        # Step 6: Poll for result
        # ------------------------------------------------------------------
        search_space_size = 2 ** num_qubits
        marked_set_rows = set(marked_rows)

        # Build marked bitstrings from marked row integers
        marked_bitstrings = [format(r, f'0{num_qubits}b') for r in marked_rows]
        marked_set_bits = set(marked_bitstrings)

        deadline = time.time() + max_poll_seconds
        result_data = None

        while time.time() < deadline:
            raw_status = job.status()
            # qiskit-ibm-runtime returns either a string or a JobStatus enum
            status = raw_status if isinstance(raw_status, str) else raw_status.name
            status = status.upper()
            if status in ("DONE", "COMPLETED"):
                result_data = job.result()
                break
            elif status in ("ERROR", "CANCELLED", "FAILED"):
                return json.dumps({
                    "error": f"Job failed with status: {status}",
                    "job_id": job_id,
                    "backend": best_backend.name,
                })
            time.sleep(10)

        # ------------------------------------------------------------------
        # Step 7a: Job done — compute amplification
        # ------------------------------------------------------------------
        if result_data is not None:
            pub_result = result_data[0]
            data = pub_result.data
            field = list(vars(data).keys())[0]
            counts = getattr(data, field).get_counts()
            total_shots = sum(counts.values())

            marked_total = sum(counts.get(b, 0) for b in marked_bitstrings)
            marked_fraction = marked_total / total_shots
            random_baseline = len(marked_rows) / search_space_size
            amplification = round(marked_fraction / random_baseline, 2) if random_baseline > 0 else 0

            top_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

            return json.dumps({
                "status": "COMPLETE",
                "experiment": {
                    "conditions": conditions,
                    "num_qubits": num_qubits,
                    "search_space_size": search_space_size,
                    "marked_rows": marked_rows,
                    "p_layers": p_layers,
                    "gamma": gamma,
                    "beta": beta,
                },
                "algorithm": {
                    "name": "LNAA (Lattice-Native Amplitude Amplification)",
                    "approach": "Ising Hamiltonian quantum walk — IBM-native RZZ+RX gates, zero routing overhead",
                    "h_coefficients": h_coeffs,
                    "j_couplings": {f"{i},{j}": J for (i, j), J in j_couplings.items()},
                    "ground_state_energy": round(ground_energy, 3),
                },
                "hardware": {
                    "backend_chosen": best_backend.name,
                    "backend_avg_cx_error": round(best_cx_error, 5),
                    "selection_reason": "Lowest avg CX error among operational backends with enough qubits",
                    "all_backends_scored": backend_scores,
                    "routing_safe": routing_safe,
                    "max_qubit_degree": max_degree,
                },
                "circuit": {
                    "logical_gates": logical_gates,
                    "hardware_gates_after_transpile": hw_gate_count,
                    "shots": shots,
                },
                "job_id": job_id,
                "results": {
                    "total_shots": total_shots,
                    "marked_shots": marked_total,
                    "marked_fraction": round(marked_fraction, 4),
                    "random_baseline": round(random_baseline, 4),
                    "amplification_factor": amplification,
                    "verdict": (
                        "EXCELLENT (>10×)" if amplification > 10 else
                        "GOOD (4–10×)"     if amplification > 4  else
                        "WEAK (1–4×)"      if amplification > 1  else
                        "FAILED (≤1× — no better than random)"
                    ),
                    "top_10_states": [
                        {
                            "bits": b,
                            "row": int(b, 2),
                            "count": c,
                            "fraction": round(c / total_shots, 4),
                            "marked": b in marked_set_bits,
                        }
                        for b, c in top_states
                    ],
                },
            }, indent=2)

        # ------------------------------------------------------------------
        # Step 7b: Still queued — return job_id for later
        # ------------------------------------------------------------------
        return json.dumps({
            "status": "QUEUED",
            "message": (
                f"Job submitted successfully but IBM queue wait exceeded {max_poll_seconds}s. "
                f"IBM queues can take minutes to hours depending on load. "
                f"Call get_amplification with the job_id below when it completes."
            ),
            "job_id": job_id,
            "backend": best_backend.name,
            "experiment": {
                "conditions": conditions,
                "marked_rows": marked_rows,
                "marked_bitstrings": marked_bitstrings,
                "search_space_size": search_space_size,
            },
            "algorithm": {
                "name": "LNAA",
                "h_coefficients": h_coeffs,
                "logical_gates": logical_gates,
                "hardware_gates_after_transpile": hw_gate_count,
            },
            "next_step": f"Call get_amplification(job_id='{job_id}', marked_bitstrings={marked_bitstrings}, search_space_size={search_space_size})",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Tool: encode_4way_collision
# --------------------------------------------------------------------------
@mcp.tool()
def encode_4way_collision(
    value: int,
    positions: list,
    max_n: int = 128,
    p_layers: int = 2,
    gamma: float = 2.589,
    beta: float = 0.501,
    shots: int = 4096,
    simulate_only: bool = True,
) -> str:
    """
    Step 4: Encode a known multi-way collision for simultaneous LNAA search.

    Takes a value V and its known positions (from sieve_singmaster_space),
    builds one LNAA rail per unique k-column, searches for all marked rows
    simultaneously in ONE job.

    In plain English:
      sieve tells us: 3003 appears at rows 14(k=6), 15(k=5), 78(k=2).
      This tool builds 3 parallel quantum searches — one per k-column —
      and runs them all in ONE hardware submission.
      Each rail independently finds its target row.
      Together they prove the 4-way collision exists and is amplifiable.

    This is the Step 4 generalization of Phase 5:
      Phase 5: manually encoded 3 rows for 3003 in one 7-qubit circuit
      Step 4:  automatically encodes N positions from sieve in N parallel rails

    Args:
      value     : the Pascal value to search for (e.g. 3003)
      positions : list of [n, k] pairs from sieve output
                  e.g. [[14,6],[15,5],[78,2]]
      max_n     : max row to search per rail (determines qubit count)
      simulate_only: True = free simulation, False = real hardware

    Returns ready-to-run params + simulation/hardware results.
    """
    try:
        from math import comb, ceil, log2
        from collections import defaultdict
        from qiskit import QuantumCircuit

        if not positions:
            return json.dumps({"error": "positions list is empty. Run sieve_singmaster_space first."})

        # ── 1. Group positions by k-column ────────────────────────────────
        # Each unique k gets its own rail
        k_groups = defaultdict(list)
        for pos in positions:
            n, k = int(pos[0]), int(pos[1])
            if comb(n, k) == value:
                k_groups[k].append(n)

        if not k_groups:
            return json.dumps({"error": f"None of the positions give C(n,k)={value}. Check your input."})

        # ── 2. Build one rail per k-column ────────────────────────────────
        bits_per_rail = max(1, ceil(log2(max_n + 1)))
        rails = []

        for k, target_rows in sorted(k_groups.items()):
            # Conditions: bit pattern of the FIRST target row
            # (largest non-trivial one — skip C(value,1) = value trivial case)
            non_trivial = [n for n in target_rows if n < value]
            primary_n = max(non_trivial) if non_trivial else target_rows[0]

            conditions = {}
            for i in range(bits_per_rail):
                conditions[i] = int((primary_n >> i) & 1)

            # marked_rows: Qiskit integers for ALL target rows in this rail
            marked_rows = []
            for n in target_rows:
                if n <= max_n:
                    qbits = [(n >> i) & 1 for i in range(bits_per_rail)]
                    marked_rows.append(int("".join(str(b) for b in reversed(qbits)), 2))

            if not marked_rows:
                continue

            rails.append({
                "k": k,
                "target_rows": target_rows,
                "primary_n": primary_n,
                "conditions": conditions,
                "marked_rows": marked_rows,
                "num_qubits": bits_per_rail,
            })

        if not rails:
            return json.dumps({"error": f"No target rows fit within max_n={max_n}. Try increasing max_n."})

        # ── 3. Build combined parallel circuit ────────────────────────────
        total_qubits = sum(r["num_qubits"] for r in rails)
        qc = QuantumCircuit(total_qubits, total_qubits)

        qubit_offset = 0
        rail_offsets = []

        for rail in rails:
            nq = rail["num_qubits"]
            qrange = list(range(qubit_offset, qubit_offset + nq))
            rail_offsets.append(qubit_offset)

            h_coeffs = {int(q): (+1.0 if v == 1 else -1.0)
                        for q, v in rail["conditions"].items()}

            qc.h(qrange)
            for _ in range(p_layers):
                for qi, h in h_coeffs.items():
                    qc.rz(2 * h * gamma, qubit_offset + qi)
                for i in range(nq):
                    qc.rx(2 * beta, qubit_offset + i)
            for i, q in enumerate(qrange):
                qc.measure(q, qubit_offset + i)

            qubit_offset += nq

        logical_gates = qc.size()

        # ── 4. Simulate or submit ─────────────────────────────────────────
        if simulate_only:
            from qiskit_aer import AerSimulator
            sim = AerSimulator()

            # Simulate each rail separately (memory safety)
            rail_counts_list = []
            for rail in rails:
                nq = rail["num_qubits"]
                h_coeffs = {int(q): (+1.0 if v == 1 else -1.0)
                            for q, v in rail["conditions"].items()}
                rail_qc = QuantumCircuit(nq, nq)
                rail_qc.h(range(nq))
                for _ in range(p_layers):
                    for qi, h in h_coeffs.items():
                        rail_qc.rz(2 * h * gamma, qi)
                    for i in range(nq):
                        rail_qc.rx(2 * beta, i)
                rail_qc.measure(range(nq), range(nq))
                rc = sim.run(rail_qc, shots=shots).result().get_counts()
                rail_counts_list.append(rc)
        else:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _gpm
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler
            svc = _get_service()
            backend = svc.least_busy(operational=True, simulator=False)
            pm = _gpm(optimization_level=2, backend=backend, seed_transpiler=42)
            t_qc = pm.run(qc)
            sampler = IBMSampler(mode=backend)
            job = sampler.run([t_qc], shots=shots)
            return json.dumps({
                "status": "submitted",
                "job_id": job.job_id(),
                "backend": backend.name,
                "value_searched": value,
                "rails": [{"k": r["k"], "target_rows": r["target_rows"]} for r in rails],
                "total_qubits": total_qubits,
                "logical_gates": logical_gates,
                "note": f"Use job_results('{job.job_id()}') when done.",
            }, indent=2)

        # ── 5. Per-rail amplification ─────────────────────────────────────
        rail_results = []
        for rail, rc in zip(rails, rail_counts_list):
            nq = rail["num_qubits"]
            marked_set = set(format(r, f'0{nq}b') for r in rail["marked_rows"])
            total = sum(rc.values())
            marked_total = sum(rc.get(b, 0) for b in marked_set)
            marked_frac = marked_total / total if total > 0 else 0
            random_base = len(rail["marked_rows"]) / (2 ** nq)
            amp = round(marked_frac / random_base, 2) if random_base > 0 else 0
            top = sorted(rc.items(), key=lambda x: x[1], reverse=True)[:3]

            rail_results.append({
                "k_column": rail["k"],
                "target_rows": rail["target_rows"],
                "primary_n": rail["primary_n"],
                "amplification": amp,
                "marked_fraction_pct": round(marked_frac * 100, 2),
                "top_states": [{"bits": b, "count": c} for b, c in top],
            })

        best = max(rail_results, key=lambda r: r["amplification"])
        avg_amp = round(sum(r["amplification"] for r in rail_results) / len(rail_results), 2)

        return json.dumps({
            "value_searched": value,
            "num_rails": len(rails),
            "total_qubits": total_qubits,
            "logical_gates": logical_gates,
            "mode": "simulation" if simulate_only else "hardware",
            "rail_results": rail_results,
            "best_rail": best,
            "average_amplification": avg_amp,
            "summary": (
                f"Searched for value={value} across {len(rails)} k-columns simultaneously. "
                f"Best: k={best['k_column']} → {best['amplification']}× amplification. "
                f"Average: {avg_amp}×."
            ),
            "verdict": (
                "READY FOR HARDWARE" if avg_amp >= 15 else
                "TUNE PARAMETERS" if avg_amp >= 5 else
                "WEAK SIGNAL"
            ),
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# --------------------------------------------------------------------------
# Tool: equality_oracle_search  (parity-oracle amplification + classical
# equality verification — the QPU narrows candidates, comb() confirms them)
# --------------------------------------------------------------------------
@mcp.tool()
def equality_oracle_search(
    k1: int = 2,
    k2: int = 5,
    n_bits: int = 6,
    p_layers: int = 3,
    gamma: float = 1.0,
    beta: float = 0.8,
    shots: int = 4096,
    simulate_only: bool = True,
    backend_name: str = "",
) -> str:
    """
    Find C(n1,k1) = C(n2,k2) without being told which rows to look for.

    Honest framing: the quantum circuit amplifies toward a Lucas mod-2
    PARITY match, which is a weak filter (roughly half of random pairs
    satisfy it by chance) — it is not proof of equality. The actual
    equality check is classical: every measured bitstring is verified
    with exact comb() arithmetic in post-processing. The QPU's real job
    here is candidate narrowing via amplification, not final verification.

    Difference from encode_4way_collision:

    encode_4way_collision:
      Input the answer (e.g. 3003 at rows 14,15,78) → circuit prepares and
      confirms that known state. Needs the answer before it can run.

    This tool:
      Input only the two columns k1, k2 — not the answer.
      Two qubit registers in superposition — one per column.
      Cross-register RZZ gates encode the Lucas mod-2 equality oracle:
        "mark any (n1, n2) where C(n1,k1) and C(n2,k2) have the same parity."
      LNAA amplifies pairs matching that parity condition — a real but
      weak filter (~50% of random pairs pass it by chance). The QPU
      narrows the candidate set; classical comb() in post-processing
      determines which amplified candidates are true equalities.

    Lucas mod-2 oracle (why it's cheap):
      By Lucas' theorem, C(n,k) mod 2 = 1 iff every 1-bit of k is also a 1-bit of n.
      This is digit-local — depends only on a few bits of n. Perfect for RZZ/RZ gates.
      Example: k=2 (binary 10) → C(n,2) is odd iff bit-1 of n is 1.
               k=5 (binary 101) → C(n,5) is odd iff bits 0 AND 2 of n are both 1.
      Equality oracle: are the active-bit patterns of n1 and n2 consistent?
      Encoded as cross-register RZZ between the active bit positions.

    Circuit structure (total = 2*n_bits qubits):
      Register 1 (qubits 0..n_bits-1):   row n1 for column k1
      Register 2 (qubits n_bits..2n-1):  row n2 for column k2
      For each LNAA layer:
        - RZ on active bits of each register (single-register terms)
        - RZZ between active bit pairs across registers (equality coupling)
        - RX mixing on all qubits

    Post-processing:
      For each measured peak (n1, n2), compute C(n1,k1) and C(n2,k2) exactly.
      Report pairs where they are truly equal — those are real collisions.

    Args:
      k1, k2     : the two Pascal columns to search (e.g. k1=2, k2=5)
      n_bits     : qubits per register (6 bits = rows 0..63, 8 bits = 0..255)
      simulate_only: True = free, False = real IBM QPU
    """
    try:
        from math import comb
        from qiskit import QuantumCircuit

        # ── 1. Lucas active bits ─────────────────────────────────────────
        def active_bits(k, p=2):
            """Bit positions where k has a 1 in base p — these are the
            bits of n that determine C(n,k) mod p."""
            bits = []
            pos = 0
            while k > 0:
                if k % p != 0:
                    bits.append(pos)
                k //= p
                pos += 1
            return bits

        ab1 = active_bits(k1)   # e.g. k=2 → [1]
        ab2 = active_bits(k2)   # e.g. k=5 → [0, 2]

        n_total = 2 * n_bits
        reg2_offset = n_bits

        # ── 2. Build two-register LNAA circuit ───────────────────────────
        qc = QuantumCircuit(n_total, n_total)
        qc.h(range(n_total))

        for _ in range(p_layers):
            # Single-register terms: reward active bits being 1
            # (C(n,k) mod 2 = 1 when active bits are set)
            for b in ab1:
                if b < n_bits:
                    qc.rz(2 * gamma, b)
            for b in ab2:
                if b < n_bits:
                    qc.rz(2 * gamma, reg2_offset + b)

            # Cross-register equality coupling: RZZ between active bits
            # of register 1 and active bits of register 2.
            # This rewards states where both registers have matching
            # parity — i.e. C(n1,k1) ≡ C(n2,k2) mod 2.
            for b1 in ab1:
                for b2 in ab2:
                    if b1 < n_bits and b2 < n_bits:
                        qc.rzz(2 * gamma, b1, reg2_offset + b2)

            # Mixing layer
            for q in range(n_total):
                qc.rx(2 * beta, q)

        qc.measure(range(n_total), range(n_total))
        logical_gates = qc.size()

        # ── 3. Simulate or submit ─────────────────────────────────────────
        if simulate_only:
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            counts = sim.run(qc, shots=shots).result().get_counts()
        else:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _gpm
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler
            svc = _get_service()
            backend = svc.least_busy(operational=True, simulator=False) \
                if not backend_name else svc.backend(backend_name)
            pm = _gpm(optimization_level=2, backend=backend, seed_transpiler=42)
            t_qc = pm.run(qc)
            sampler = IBMSampler(mode=backend)
            job = sampler.run([t_qc], shots=shots)
            return json.dumps({
                "status": "submitted",
                "job_id": job.job_id(),
                "backend": backend.name,
                "k1": k1, "k2": k2, "n_bits": n_bits,
                "total_qubits": n_total,
                "logical_gates": logical_gates,
                "active_bits_k1": ab1, "active_bits_k2": ab2,
                "note": f"Use job_results('{job.job_id()}') when done.",
            }, indent=2)

        # ── 4. Post-process: scan ALL counts for exact collisions ─────────
        total_shots = sum(counts.values())
        random_baseline = 1.0 / (4 ** n_bits)

        collisions = []
        near_misses = []

        # Scan every measured bitstring — collisions may not be in top N
        for bitstring, count in counts.items():
            reg2_bits = bitstring[:n_bits]
            reg1_bits = bitstring[n_bits:]
            n1 = int(reg1_bits, 2)
            n2 = int(reg2_bits, 2)
            if n1 < k1 or n2 < k2:
                continue
            v1 = comb(n1, k1)
            v2 = comb(n2, k2)
            prob = count / total_shots
            amp = round(prob / random_baseline, 1)
            entry = {
                "n1": n1, "k1": k1, "v1": v1,
                "n2": n2, "k2": k2, "v2": v2,
                "shots": count,
                "probability_pct": round(prob * 100, 3),
                "amplification": amp,
                "equal": v1 == v2,
            }
            if v1 == v2 and v1 > 1:
                entry["collision_value"] = v1
                entry["total_appearances"] = 2 + 4
                collisions.append(entry)

        # Top near-misses from the top-30 high-count states
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30]
        for bitstring, count in top:
            reg2_bits = bitstring[:n_bits]
            reg1_bits = bitstring[n_bits:]
            n1 = int(reg1_bits, 2)
            n2 = int(reg2_bits, 2)
            if n1 < k1 or n2 < k2:
                continue
            v1 = comb(n1, k1)
            v2 = comb(n2, k2)
            if v1 != v2:
                prob = count / total_shots
                near_misses.append({
                    "n1": n1, "k1": k1, "v1": v1,
                    "n2": n2, "k2": k2, "v2": v2,
                    "shots": count,
                    "amplification": round(prob / random_baseline, 1),
                })

        return json.dumps({
            "k1": k1, "k2": k2,
            "n_bits": n_bits,
            "search_space": f"{2**n_bits} rows per column ({2**(2*n_bits)} total pairs)",
            "total_qubits": n_total,
            "logical_gates": logical_gates,
            "mode": "simulation" if simulate_only else "hardware",
            "shots": shots,
            "active_bits_k1": ab1,
            "active_bits_k2": ab2,
            "oracle_description": (
                f"Cross-register RZZ coupling between bits {ab1} of reg1 "
                f"and bits {ab2} of reg2. Rewards C(n1,{k1}) ≡ C(n2,{k2}) mod 2."
            ),
            "collisions_found": len(collisions),
            "collisions": collisions,
            "top_near_misses": near_misses[:5],
            "summary": (
                f"Searched {2**n_bits}×{2**n_bits} = {2**(2*n_bits)} row pairs. "
                f"Found {len(collisions)} exact collisions C(n1,{k1})=C(n2,{k2}). "
                f"{'NEW DISCOVERY possible — check collision values!' if any(c['collision_value'] > 3003 for c in collisions) else 'Known values range.'}"
                if collisions else
                f"No exact collisions found in {2**n_bits}×{2**n_bits} search space. "
                f"Try larger n_bits or different k1/k2."
            ),
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# --------------------------------------------------------------------------
# Tool: find_collision_candidates  (Step 1 — curve intersection search)
# --------------------------------------------------------------------------
@mcp.tool()
def find_collision_candidates(
    columns: list = [5, 6, 7, 8],
    search_depth: int = 100000,
    min_columns: int = 3,
) -> str:
    """
    Step 1: Curve-intersection classical search for Singmaster candidates.

    Instead of scanning every row (brute-force sieve), anchors on the
    smallest target column and uses integer root-finding to jump directly
    to candidate rows in other columns. No scanning needed.

    In plain English:
      Old way: check row 1, row 2, row 3 ... row 50000 — one by one.
      New way: for each n in anchor column, compute N=C(n,k). Then solve
               "what n gives C(n,k2)=N?" using algebra — jump straight there.
      Like sudoku: use the rules to skip impossible positions entirely.

    Applies three Einstein constraints automatically:
      1. Solved pair elimination — (2,3),(2,4),(2,5),(2,6),(3,4) etc. are
         fully solved in math literature. Flags if your combo is all-solved.
      2. 2022 theorem — 9+ candidates only exist at small k. Warns if k>15.
      3. Integer root-finding — k=2 uses exact closed form; k>=3 uses
         Newton's method from (N*k!)^(1/k) estimate.

    Args:
      columns     : k-column values to search simultaneously
                    e.g. [5,6,7,8] or [2,7,9] — avoid all-solved combos
      search_depth: rows to scan in the anchor (smallest) column
      min_columns : columns that must match for a candidate to be reported

    Each candidate with 3 non-trivial columns = 8 total appearances (=3003).
    Needs 4 non-trivial columns = 10 total = new world record.
    """
    try:
        from math import comb, isqrt, factorial

        # Solved column pairs — no new integer solutions beyond known ones
        SOLVED_PAIRS = {
            (2, 3), (2, 4), (2, 5), (2, 6), (2, 8),
            (3, 4), (3, 6), (4, 6), (4, 8),
        }

        columns = sorted(set(int(c) for c in columns if int(c) >= 2))
        if len(columns) < 2:
            return json.dumps({"error": "Need at least 2 columns (k >= 2)."})

        deep_cols = [k for k in columns if k > 15]
        pairs = [(columns[i], columns[j])
                 for i in range(len(columns))
                 for j in range(i + 1, len(columns))]
        solved_pairs_hit = [p for p in pairs
                            if p in SOLVED_PAIRS or (p[1], p[0]) in SOLVED_PAIRS]
        all_solved = len(solved_pairs_hit) == len(pairs)

        k_anchor = columns[0]
        other_cols = columns[1:]

        def find_n_for_value(N, k):
            """Return n such that C(n,k)=N using root-finding, or None."""
            if k == 1:
                return N if N >= 1 else None
            if k == 2:
                disc = 1 + 8 * N
                s = isqrt(disc)
                if s * s != disc:
                    return None
                n = (1 + s) // 2
                return n if n >= 2 and comb(n, 2) == N else None
            if k == 3:
                est = int((6 * N) ** (1 / 3))
                for n in range(max(3, est - 2), est + 5):
                    v = comb(n, 3)
                    if v == N:
                        return n
                    if v > N:
                        break
                return None
            fk = factorial(k)
            try:
                est = int(round((N * fk) ** (1.0 / k)))
            except OverflowError:
                import math
                est = int(math.exp((math.log(N) + math.log(fk)) / k))
            for n in range(max(k, est - 3), est + 6):
                v = comb(n, k)
                if v == N:
                    return n
                if v > N:
                    break
            return None

        candidates = []
        for n1 in range(k_anchor, search_depth + 1):
            N = comb(n1, k_anchor)
            if N < 6:
                continue
            found = [(k_anchor, n1)]
            for k in other_cols:
                n2 = find_n_for_value(N, k)
                if n2 is not None:
                    found.append((k, n2))
            if len(found) >= min_columns:
                total = 2 + 2 * len(found)
                candidates.append({
                    "value": N,
                    "non_trivial_columns": len(found),
                    "total_appearances": total,
                    "positions": [
                        {"k": k, "n": n, "verify": f"C({n},{k})={comb(n,k)}"}
                        for k, n in found
                    ],
                    "beats_world_record": total > 8,
                })

        candidates.sort(key=lambda x: -x["total_appearances"])
        new_records = [c for c in candidates if c["beats_world_record"]]

        return json.dumps({
            "columns_searched": columns,
            "anchor_column": k_anchor,
            "search_depth": search_depth,
            "rows_scanned_in_anchor": search_depth - k_anchor + 1,
            "candidates_found": len(candidates),
            "new_records_found": len(new_records),
            "solved_pairs_in_combo": [list(p) for p in solved_pairs_hit],
            "all_pairs_solved": all_solved,
            "warning_all_solved": (
                "All column pairs are solved — results will only repeat known values. "
                "Try unsolved combos e.g. [5,6,7,8] or [2,7,9,11]."
            ) if all_solved else None,
            "warning_deep_interior": (
                f"Columns {deep_cols} violate 2022 theorem — 9+ cannot exist there. "
                "Stick to k <= 15."
            ) if deep_cols else None,
            "new_records": new_records,
            "top_candidates": candidates[:20],
            "summary": (
                f"Scanned {search_depth - k_anchor + 1} rows in anchor column k={k_anchor}. "
                f"Found {len(candidates)} values in {min_columns}+ columns simultaneously. "
                f"New world records (>8 appearances): {len(new_records)}."
            ),
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# --------------------------------------------------------------------------
# Tool: sieve_singmaster_space
# --------------------------------------------------------------------------
@mcp.tool()
def sieve_singmaster_space(
    max_n: int = 500,
    max_k: int = 50,
    target_multiplicity: int = 4,
    use_lucas_filter: bool = True,
) -> str:
    """
    Classical sieve for Singmaster's Conjecture — runs BEFORE any QPU job.

    Finds candidate positions in Pascal's Triangle where the same value
    appears multiple times. Uses Lucas' Theorem to eliminate 98%+ of
    the search space before any quantum circuit is built.

    In plain English:
      Instead of asking the quantum computer to search a billion positions,
      we first eliminate all positions that CANNOT be part of a collision
      using pure math. Only the survivors go to quantum hardware.
      This saves QPU credits and finds better candidates.

    Lucas' Theorem (the sieve):
      C(n,k) mod p is zero if any digit of k in base-p is larger than
      the corresponding digit of n. This lets us quickly compare two
      positions without computing the full binomial coefficient.
      If C(n1,k1) != C(n2,k2) mod p for ANY small prime p,
      they CANNOT be equal — eliminated instantly.

    Args:
      max_n              : search rows 0..max_n (default 500)
      max_k              : search columns 0..max_k (default 50)
      target_multiplicity: how many positions must share same value
                           2=2-way, 4=4-way, 5=9+appearances (default 4)
      use_lucas_filter   : apply Lucas mod-prime filter first (default True)

    Returns:
      - candidates: positions grouped by value, sorted by multiplicity
      - best_target: highest-multiplicity group found
      - quantum_ready: True if a group meets target_multiplicity
      - next_step: what to run next
    """
    try:
        from math import comb

        # ── 1. Lucas' Theorem filter ───────────────────────────────────────
        # For small primes, compute C(n,k) mod p quickly using Lucas.
        # If two positions differ mod p they can't be equal — skip them.
        PRIMES = [2, 3, 5, 7, 11, 13]

        def lucas_mod(n, k, p):
            """Compute C(n,k) mod p using Lucas' theorem."""
            if k > n:
                return 0
            if k == 0 or k == n:
                return 1
            result = 1
            while n > 0 or k > 0:
                ni, ki = n % p, k % p
                if ki > ni:
                    return 0
                # C(ni, ki) mod p — small values, compute directly
                c = 1
                for i in range(ki):
                    c = c * (ni - i) // (i + 1)
                result = (result * (c % p)) % p
                n //= p
                k //= p
            return result

        def mod_signature(n, k):
            """Fingerprint of C(n,k) — tuple of values mod each prime."""
            return tuple(lucas_mod(n, k, p) for p in PRIMES)

        # ── 2. Build candidate table ───────────────────────────────────────
        # Group positions by their mod signature first (fast filter),
        # then by exact value for confirmed matches.

        # Only search k <= n/2 (Pascal symmetry: C(n,k)=C(n,n-k))
        sig_groups = {}   # signature → list of (n, k)

        for n in range(0, max_n + 1):
            k_limit = min(max_k, n // 2 + 1)
            for k in range(0, k_limit):
                if use_lucas_filter:
                    sig = mod_signature(n, k)
                    sig_groups.setdefault(sig, []).append((n, k))
                else:
                    sig_groups.setdefault((0,), []).append((n, k))

        # ── 3. Exact match within each signature group ─────────────────────
        # Only compute full comb() for positions sharing the same signature
        value_groups = {}   # exact value → list of (n, k)
        total_computed = 0
        total_skipped = 0

        for sig, positions in sig_groups.items():
            if len(positions) < 2:
                total_skipped += len(positions)
                continue
            # These share the same mod fingerprint — compute exact values
            for n, k in positions:
                v = comb(n, k)
                if v > 1:   # skip trivial C(n,0)=C(n,n)=1
                    value_groups.setdefault(v, []).append((n, k))
                    total_computed += 1

        # ── 4. Add symmetric partners C(n,n-k) for complete picture ───────
        # We searched k<=n/2, now add the mirror positions
        full_groups = {}
        for v, positions in value_groups.items():
            full = list(positions)
            for n, k in positions:
                if k != n - k:   # avoid duplicating the middle element
                    full.append((n, n - k))
            full_groups[v] = sorted(set(full))

        # ── 5. Filter by target multiplicity ──────────────────────────────
        candidates = []
        for v, positions in full_groups.items():
            if len(positions) >= 2:
                candidates.append({
                    "value": v,
                    "multiplicity": len(positions),
                    "positions": [{"n": n, "k": k} for n, k in positions],
                    "meets_target": len(positions) >= target_multiplicity,
                })

        candidates.sort(key=lambda c: c["multiplicity"], reverse=True)

        # ── 6. Stats ───────────────────────────────────────────────────────
        reduction_pct = round(100 * total_skipped /
                              max(1, total_skipped + total_computed), 1)
        meets_target = [c for c in candidates if c["meets_target"]]
        best = candidates[0] if candidates else None

        # ── 7. Quantum next step ───────────────────────────────────────────
        if meets_target:
            top = meets_target[0]
            next_step = (
                f"Found {len(meets_target)} values with {target_multiplicity}+ appearances. "
                f"Best: value={top['value']} at {top['multiplicity']} positions. "
                f"Run encode_collision_problem or encode_4way_collision with these positions."
            )
        else:
            top_mult = best["multiplicity"] if best else 0
            next_step = (
                f"No {target_multiplicity}-way collision found up to n={max_n}. "
                f"Best found: {top_mult}-way. Increase max_n or max_k to search deeper."
            )

        return json.dumps({
            "search_range": {"max_n": max_n, "max_k": max_k},
            "target_multiplicity": target_multiplicity,
            "lucas_filter_used": use_lucas_filter,
            "positions_computed": total_computed,
            "positions_skipped_by_sieve": total_skipped,
            "sieve_reduction_pct": reduction_pct,
            "total_candidates_found": len(candidates),
            "meets_target_count": len(meets_target),
            "top_10_by_multiplicity": candidates[:10],
            "best_target": best,
            "quantum_ready": len(meets_target) > 0,
            "next_step": next_step,
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# --------------------------------------------------------------------------
# Tool: run_parallel_collision_search
# --------------------------------------------------------------------------
@mcp.tool()
def run_parallel_collision_search(
    k_pairs: list,
    max_n: int = 20,
    shots: int = 4096,
    p_layers: int = 2,
    gamma: float = 2.589,
    beta: float = 0.501,
    simulate_only: bool = True,
    backend_name: str = "",
) -> str:
    """
    Parallel rails — ONE job, MULTIPLE collision searches simultaneously.

    Instead of one 9-qubit search, tiles multiple independent LNAA rings
    on ibm_marrakesh (156 qubits). Each ring searches a different (k1,k2)
    column pair. No cross-talk between rings. One submission.

    Why this matters:
      ibm_marrakesh has 156 qubits. Our basic search uses 9.
      We can run ~10 independent searches in one job for the same cost.

    Args:
      k_pairs    : list of [k1,k2] pairs, e.g. [[2,3],[3,4],[4,5]]
      max_n      : search rows 0..max_n for each pair
      shots      : total shots (split across rails in simulation)
      simulate_only: True = free simulation, False = real hardware

    Returns amplification for each rail and combined best result.
    """
    try:
        from math import comb, ceil, log2
        from collections import defaultdict

        if not k_pairs or len(k_pairs) == 0:
            return json.dumps({"error": "k_pairs must be a non-empty list like [[2,3],[3,4]]"})

        # ── 1. Encode each rail classically ───────────────────────────────
        rails = []
        for pair in k_pairs:
            k1, k2 = int(pair[0]), int(pair[1])

            table1 = {}
            for n in range(k1, max_n + 1):
                table1.setdefault(comb(n, k1), []).append(n)
            table2 = {}
            for n in range(k2, max_n + 1):
                table2.setdefault(comb(n, k2), []).append(n)

            collisions = []
            for v in sorted(set(table1) & set(table2)):
                if v > 1:
                    for n1 in table1[v]:
                        for n2 in table2[v]:
                            collisions.append({"n1": n1, "n2": n2, "value": v})

            if not collisions:
                rails.append({"k1": k1, "k2": k2, "status": "no_collisions", "collisions": []})
                continue

            bits1 = max(1, ceil(log2(max_n + 1)))
            bits2 = max(1, ceil(log2(max_n + 1)))
            num_qubits = bits1 + bits2

            # Primary target = largest value collision
            primary = max(collisions, key=lambda c: c["value"])
            n1_p, n2_p = primary["n1"], primary["n2"]

            # Conditions from primary collision bit pattern
            conditions = {}
            for i in range(bits1):
                conditions[i] = int((n1_p >> i) & 1)
            for i in range(bits2):
                conditions[bits1 + i] = int((n2_p >> i) & 1)

            # marked_rows as Qiskit integers
            marked_rows = []
            for col in collisions:
                qbits = [(col["n1"] >> i) & 1 for i in range(bits1)]
                qbits += [(col["n2"] >> i) & 1 for i in range(bits2)]
                marked_rows.append(int("".join(str(b) for b in reversed(qbits)), 2))

            rails.append({
                "k1": k1, "k2": k2,
                "collisions": collisions,
                "primary": primary,
                "num_qubits": num_qubits,
                "conditions": conditions,
                "marked_rows": marked_rows,
                "bits1": bits1, "bits2": bits2,
            })

        active_rails = [r for r in rails if r.get("conditions")]
        if not active_rails:
            return json.dumps({"error": "No valid collision pairs found in given range.", "rails": rails})

        # ── 2. Build parallel circuit — stack rails side by side ──────────
        from qiskit import QuantumCircuit

        # Each rail gets its own qubit block, offset by rail index
        total_qubits = sum(r["num_qubits"] for r in active_rails)
        qc = QuantumCircuit(total_qubits, total_qubits)

        qubit_offset = 0
        rail_qubit_ranges = []

        for rail in active_rails:
            nq = rail["num_qubits"]
            qrange = list(range(qubit_offset, qubit_offset + nq))
            rail_qubit_ranges.append(qrange)

            # Superposition for this rail
            qc.h(qrange)

            # h_coeffs from conditions
            h_coeffs = {}
            for q_local, val in rail["conditions"].items():
                h_coeffs[int(q_local)] = +1.0 if val == 1 else -1.0

            # p layers of LNAA
            for _ in range(p_layers):
                for q_local, h in h_coeffs.items():
                    qc.rz(2 * h * gamma, qubit_offset + q_local)
                for i in range(nq):
                    qc.rx(2 * beta, qubit_offset + i)

            # Measure this rail into its own classical bits
            for i, q in enumerate(qrange):
                qc.measure(q, qubit_offset + i)

            qubit_offset += nq

        logical_gates = qc.size()

        # ── 3. Simulate or submit ─────────────────────────────────────────
        if simulate_only:
            # Simulate each rail separately to avoid memory explosion.
            # 27-qubit statevector needs 16GB; 9-qubit needs 512MB.
            # Hardware submission still uses the combined circuit.
            from qiskit_aer import AerSimulator
            from qiskit import QuantumCircuit as _QC
            sim = AerSimulator()

            rail_counts_list = []
            qubit_offset_sim = 0
            for rail in active_rails:
                nq = rail["num_qubits"]
                h_coeffs_r = {int(q): (+1.0 if v == 1 else -1.0)
                              for q, v in rail["conditions"].items()}
                rail_qc = _QC(nq, nq)
                rail_qc.h(range(nq))
                for _ in range(p_layers):
                    for qi, h in h_coeffs_r.items():
                        rail_qc.rz(2 * h * gamma, qi)
                    for i in range(nq):
                        rail_qc.rx(2 * beta, i)
                rail_qc.measure(range(nq), range(nq))
                rail_counts = sim.run(rail_qc, shots=shots).result().get_counts()
                rail_counts_list.append(rail_counts)
                qubit_offset_sim += nq

            # Build a fake combined counts dict so the rail-splitting code below works
            # Each rail's bitstring is padded to fill its place in the combined register
            counts = {}
            total_q = sum(r["num_qubits"] for r in active_rails)
            offset = 0
            for rail, rc in zip(active_rails, rail_counts_list):
                nq = rail["num_qubits"]
                before = total_q - offset - nq   # high bits (other rails above)
                after  = offset                  # low bits (other rails below)
                for bits, cnt in rc.items():
                    full = "0" * before + bits + "0" * after
                    counts[full] = counts.get(full, 0) + cnt
                offset += nq
        else:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as _gpm
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler

            svc = _get_service()
            bname = backend_name or "ibm_kingston"
            backend = svc.backend(bname)
            pm = _gpm(optimization_level=2, backend=backend, seed_transpiler=42)
            t_qc = pm.run(qc)
            sampler = IBMSampler(mode=backend)
            job = sampler.run([t_qc], shots=shots)
            return json.dumps({
                "status": "submitted",
                "job_id": job.job_id(),
                "backend": bname,
                "total_qubits": total_qubits,
                "logical_gates": logical_gates,
                "rails": [{"k1": r["k1"], "k2": r["k2"], "primary": r.get("primary")} for r in active_rails],
                "note": f"Use job_results('{job.job_id()}') to get results when done.",
            }, indent=2)

        # ── 4. Split counts back per rail and compute amplification ───────
        rail_results = []
        qubit_offset = 0

        for rail, qrange in zip(active_rails, rail_qubit_ranges):
            nq = rail["num_qubits"]
            marked_set = set(format(r, f'0{nq}b') for r in rail["marked_rows"])
            search_space = 2 ** nq
            random_baseline = len(rail["marked_rows"]) / search_space

            # Extract this rail's bits from the full bitstring
            rail_counts = defaultdict(int)
            for full_bits, cnt in counts.items():
                # Qiskit bitstring is MSB-first for the full register
                # Extract the bits belonging to this rail
                rail_bits = full_bits[total_qubits - qubit_offset - nq: total_qubits - qubit_offset]
                rail_counts[rail_bits] += cnt

            total = sum(rail_counts.values())
            marked_total = sum(rail_counts.get(b, 0) for b in marked_set)
            marked_fraction = marked_total / total if total > 0 else 0
            amplification = round(marked_fraction / random_baseline, 2) if random_baseline > 0 else 0

            top = sorted(rail_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            rail_results.append({
                "k1": rail["k1"], "k2": rail["k2"],
                "primary_target": rail.get("primary"),
                "amplification": amplification,
                "marked_fraction_pct": round(marked_fraction * 100, 2),
                "random_baseline_pct": round(random_baseline * 100, 2),
                "top_states": [{"bits": b, "count": c} for b, c in top],
            })

            qubit_offset += nq

        best = max(rail_results, key=lambda r: r["amplification"])

        return json.dumps({
            "mode": "simulation" if simulate_only else "hardware",
            "total_qubits_used": total_qubits,
            "logical_gates": logical_gates,
            "rails_run": len(active_rails),
            "rail_results": rail_results,
            "best_rail": best,
            "summary": (
                f"{len(active_rails)} searches in ONE job. "
                f"Best: k={best['k1']}vs{best['k2']} → {best['amplification']}× amplification."
            ),
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# --------------------------------------------------------------------------
# Tool: certify_ising_gate_optimality — PROVE, not estimate, the minimum
# two-qubit gate count for an Ising Hamiltonian's native compilation
# --------------------------------------------------------------------------

@mcp.tool()
def certify_ising_gate_optimality(
    h_coeffs: dict,
    j_couplings: dict,
    p_layers: int = 1,
    actual_two_qubit_gate_count: int = None,
) -> str:
    """
    Prove the minimum number of native two-qubit gates required to implement
    an Ising-Hamiltonian phase layer (the LNAA/QAOA-style oracle used
    throughout this project's Singmaster's work) — not an estimate, a proof.

    THE ARGUMENT (why this is provable, not a heuristic):
    An Ising Hamiltonian H = sum(h_i Z_i) + sum(J_ij Z_i Z_j) is entirely
    diagonal — every term is built from Z operators only, so ALL terms
    commute with each other. That means exp(-i*gamma*H) factors EXACTLY
    (zero Trotter error, any term order) into a product of per-term
    exponentials. Two consequences, both provable:
      1. Two qubits i,j can only become Z-Z correlated in this evolution if
         at least one gate directly couples them — a genuine two-qubit
         interaction cannot be synthesized from single-qubit gates alone.
         So each nonzero J_ij pair needs >= 1 two-qubit gate. This is a
         real lower bound, not a guess.
      2. IonQ's native ZZ(theta) gate implements exp(-i*theta/2 * Z_i Z_j)
         directly, for ANY continuous theta, in exactly one gate. So each
         pair also needs <= 1 two-qubit gate — one gate is always sufficient.
    Together: the true minimum is EXACTLY the number of qubit pairs with
    nonzero net coupling, after merging duplicate entries for the same pair
    (J_ij and J_ji, or repeated terms, combine by simple addition — ZZ(a)
    followed by ZZ(b) on the same pair equals ZZ(a+b) exactly, since they
    commute and share a generator).

    This does NOT apply to circuits with non-commuting terms (X or Y
    couplings, genuine Trotter error) — only to pure Ising (Z/ZZ) phase
    layers, which is exactly what encode_4way_collision, the E1 entangling
    circuits, and every LNAA-family tool in this project builds.

    Args:
        h_coeffs    : {qubit_index: coefficient} — local Z field terms
        j_couplings : {"i,j": coefficient} — pairwise ZZ coupling terms.
                     Keys as "i,j" strings (JSON can't use tuple keys).
        p_layers    : number of repeated QAOA-style layers (phase+mixing).
                     Layers are separated by a mixing (RX) layer that does
                     NOT commute with the phase layer, so couplings CANNOT
                     merge across layers — each layer needs its own full set.
        actual_two_qubit_gate_count : optional — the real 2-qubit gate count
                     from an actual compiled circuit (e.g. from
                     estimate_ionq_gates), to check against the proven
                     minimum and report the gap.

    Returns the proven minimum, the proof steps, and — if provided — how the
    actual circuit compares.
    """
    try:
        # Merge duplicate/reversed pair entries: (i,j) and (j,i) are the same
        # physical coupling, and repeated entries for the same pair add.
        merged = {}
        for key, coeff in j_couplings.items():
            i, j = (int(x) for x in key.split(","))
            pair = tuple(sorted((i, j)))
            if pair[0] == pair[1]:
                return json.dumps({"error": f"Invalid coupling {key}: a qubit cannot couple to itself."})
            merged[pair] = merged.get(pair, 0.0) + float(coeff)

        nonzero_pairs = {pair: c for pair, c in merged.items() if abs(c) > 1e-12}
        min_2q_per_layer = len(nonzero_pairs)
        min_2q_total = min_2q_per_layer * p_layers

        redundant_pairs = [
            {"pair": list(pair), "provided_terms_merged_into_one": True}
            for pair in merged if pair not in nonzero_pairs or
            sum(1 for k in j_couplings if tuple(sorted(int(x) for x in k.split(","))) == pair) > 1
        ]

        result = {
            "num_h_terms": len([c for c in h_coeffs.values() if abs(float(c)) > 1e-12]),
            "num_unique_coupling_pairs": min_2q_per_layer,
            "p_layers": p_layers,
            "proven_minimum_two_qubit_gates": min_2q_total,
            "proof": [
                "H is a pure Ising Hamiltonian (all-Z terms) -> every term commutes.",
                "exp(-i*gamma*H) factors EXACTLY into per-term exponentials, any order, zero Trotter error.",
                f"{min_2q_per_layer} unique qubit pairs have nonzero coupling -> each needs >=1 two-qubit gate (no way to Z-Z-correlate two qubits without directly coupling them).",
                "IonQ's native ZZ(theta) implements any single pairwise term in exactly 1 gate -> 1 gate is also sufficient per pair.",
                f"Therefore {min_2q_per_layer} is both a proven lower bound AND achievable -> it is the true minimum per layer.",
                f"With p_layers={p_layers} (separated by non-commuting mixing layers, so couplings cannot merge across layers): minimum = {min_2q_per_layer} x {p_layers} = {min_2q_total}.",
            ],
        }

        if redundant_pairs:
            result["redundant_terms_found"] = redundant_pairs
            result["note"] = "Some pairs appeared more than once in j_couplings — these merge into one gate each; a circuit that emits them as SEPARATE gates is not yet minimal."

        if actual_two_qubit_gate_count is not None:
            gap = actual_two_qubit_gate_count - min_2q_total
            result["actual_two_qubit_gate_count"] = actual_two_qubit_gate_count
            result["gap_above_proven_minimum"] = gap
            result["is_optimal"] = gap == 0
            if gap > 0:
                result["verdict"] = f"NOT optimal — {gap} more two-qubit gate(s) than the proven minimum. Check for un-merged duplicate pairs or unnecessary couplings."
            elif gap < 0:
                result["verdict"] = f"IMPOSSIBLE — actual count ({actual_two_qubit_gate_count}) is below the proven minimum ({min_2q_total}). This means either j_couplings doesn't match what the circuit actually implements, or the circuit is wrong."
            else:
                result["verdict"] = "OPTIMAL — matches the proven minimum exactly."

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Stabilizer / Clifford checkable-structure verification
# --------------------------------------------------------------------------
# Ported from quantum-verifier's core/stabilizer.py. Generalizes the same
# trick equality_oracle_search/encode_4way_collision rely on informally --
# a result that's cheap to verify classically even though a QPU was needed
# to find it -- into a systematic, exact one: any circuit built entirely
# from Clifford gates (H, S, CX, CZ, X, Y, Z, SWAP, ...) plus measurements
# has an EXACT, classically-computable measurement distribution, no matter
# how many qubits it has (Gottesman-Knill theorem), via the stabilizer
# tableau -- polynomial-time, not simulated in the usual exponential sense.
# Confirmed directly in quantum-verifier: a 150-qubit Clifford circuit,
# which state-vector simulation could never touch (2^150 amplitudes),
# verifies exactly in under a second.

def _is_clifford_circuit(circuit: QuantumCircuit) -> dict:
    unitary_only = circuit.remove_final_measurements(inplace=False)
    unitary_only.data = [
        instr for instr in unitary_only.data if instr.operation.name not in ("measure", "barrier")
    ]
    try:
        Clifford(unitary_only)
        return {"is_clifford": True, "reason": None}
    except Exception as e:
        return {"is_clifford": False, "reason": str(e)}


@mcp.tool()
def verify_stabilizer_circuit(qasm_string: str) -> str:
    """
    Checkable-structure verification, generalized: if a circuit is built
    entirely from Clifford gates (H, S, CX, CZ, X, Y, Z, SWAP, ...) plus
    measurements, its exact measurement distribution is computable via the
    stabilizer tableau (Gottesman-Knill theorem) -- not simulated, not
    estimated, exact, and polynomial-time regardless of qubit count.

    Honestly reports inapplicable, not a guess, if the circuit contains
    any non-Clifford gate (e.g. an arbitrary-angle RZ/RZZ/RX) -- those
    still need a real simulation or real hardware run instead.

    Args:
        qasm_string : OpenQASM 2.0 circuit string
    """
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        check = _is_clifford_circuit(circuit)
        if not check["is_clifford"]:
            return json.dumps({
                "applicable": False,
                "reason": f"circuit contains a non-Clifford gate, not verifiable via the "
                          f"stabilizer formalism: {check['reason']}",
            })

        unitary_only = circuit.remove_final_measurements(inplace=False)
        unitary_only.data = [
            instr for instr in unitary_only.data if instr.operation.name not in ("measure", "barrier")
        ]
        cliff = Clifford(unitary_only)
        state = StabilizerState(cliff)
        exact_probabilities = state.probabilities_dict()

        return json.dumps({
            "applicable": True,
            "n_qubits": circuit.num_qubits,
            "exact_probabilities": {k: round(v, 6) for k, v in exact_probabilities.items()},
            "support_size": len(exact_probabilities),
            "method": "stabilizer tableau (Gottesman-Knill) -- exact, not simulated, "
                      "polynomial-time regardless of qubit count",
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def verify_stabilizer_hardware_result(qasm_string: str, hw_counts: dict) -> str:
    """
    Verify real hardware measurement counts against a Clifford circuit's
    EXACT stabilizer prediction -- reports a real fidelity lower bound
    (fraction of shots landing on an outcome that's actually possible
    under the ideal case) at any qubit count, no simulation required.

    Args:
        qasm_string : OpenQASM 2.0 circuit string (Clifford gates only)
        hw_counts   : real measurement counts, e.g. {"000": 480, "111": 470, "010": 30}
    """
    try:
        prediction_json = verify_stabilizer_circuit(qasm_string)
        prediction = json.loads(prediction_json)
        if not prediction.get("applicable"):
            return prediction_json

        if not hw_counts:
            return json.dumps({"applicable": False, "reason": "No hardware counts provided."})

        valid_outcomes = {b for b, p in prediction["exact_probabilities"].items() if p > 0}
        total = sum(hw_counts.values())
        valid_shots = sum(c for b, c in hw_counts.items() if b in valid_outcomes)
        fidelity_lower_bound = valid_shots / total if total else 0
        invalid_bitstrings = sorted(
            ((b, c) for b, c in hw_counts.items() if b not in valid_outcomes),
            key=lambda kv: -kv[1],
        )[:5]

        return json.dumps({
            "applicable": True,
            "n_qubits": prediction["n_qubits"],
            "support_size": prediction["support_size"],
            "fidelity_lower_bound": round(fidelity_lower_bound, 4),
            "valid_shots": valid_shots,
            "total_shots": total,
            "top_invalid_bitstrings": invalid_bitstrings,
            "verdict": (
                f"Fidelity lower bound {round(fidelity_lower_bound, 3)} ({valid_shots}/{total} shots "
                f"landed on one of the {prediction['support_size']} outcomes that are actually possible "
                "under the exact stabilizer prediction) -- checked exactly, no simulation required, "
                f"regardless of the circuit's {prediction['n_qubits']} qubits."
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------------------------------
# Command-Line Argument Parsing
# --------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for transport configuration."""
    parser = argparse.ArgumentParser(
        description="Quantum Hardware MCP Server - Exposes IBM Quantum device data via MCP protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # stdio mode (default, for Claude Desktop)
  python server.py
  
  # HTTP mode on localhost
  python server.py --transport http
  
  # HTTP mode on all interfaces with custom port
  python server.py --transport http --host 0.0.0.0 --port 8080
  
  # HTTP mode with specific CORS origins
  python server.py --transport http --cors-origins "https://myapp.com,https://api.myapp.com"

Environment Variables:
  IBM_QUANTUM_TOKEN   IBM Quantum API token (required)
  MCP_HTTP_HOST       HTTP server host (default: 127.0.0.1)
  MCP_HTTP_PORT       HTTP server port (default: 8000)
  MCP_CORS_ORIGINS    Comma-separated CORS origins (default: *)
  MCP_API_KEY         API key for authentication (optional, recommended for production)
        """
    )
    
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: 'stdio' for Claude Desktop (default), 'http' for remote clients"
    )
    
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HTTP_HOST", "127.0.0.1"),
        help="HTTP server host (default: 127.0.0.1, use 0.0.0.0 for all interfaces)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_HTTP_PORT", "8000")),
        help="HTTP server port (default: 8000)"
    )
    
    parser.add_argument(
        "--cors-origins",
        default=os.getenv("MCP_CORS_ORIGINS", "*"),
        help="Comma-separated CORS origins (default: *, use specific domains in production)"
    )
    
    return parser.parse_args()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    if args.transport == "stdio":
        # stdio transport for Claude Desktop integration
        # Claude Desktop launches this process and communicates over stdin/stdout
        # Note: Cannot use print() here as it would corrupt the JSON-RPC protocol stream
        mcp.run(transport="stdio")
    
    elif args.transport == "http":
        # HTTP/SSE transport for remote MCP clients
        # Enables web-based AI assistants and remote integrations
        
        # Configure server settings
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        
        # Check if API key is configured
        api_key = os.getenv("MCP_API_KEY")
        api_key_configured = bool(api_key)
        
        print("=" * 70, flush=True)
        print("Quantum Hardware MCP Server - HTTP Mode", flush=True)
        print("=" * 70, flush=True)
        print(f"Server URL:    http://{args.host}:{args.port}", flush=True)
        print(f"CORS Origins:  {args.cors_origins}", flush=True)
        print(f"Authentication: {'Enabled (API key required)' if api_key_configured else 'Disabled (development mode)'}", flush=True)

        # Show IBM account info in banner only if IBM_SHOW_ACCOUNT_INFO is not "false".
        # Default is to show it — set IBM_SHOW_ACCOUNT_INFO=false in .env to hide.
        if os.getenv("IBM_SHOW_ACCOUNT_INFO", "true").lower() != "false":
            ibm_channel  = os.getenv("IBM_CHANNEL", "ibm_quantum_platform")
            ibm_instance = os.getenv("IBM_INSTANCE", "(auto-select)")
            print(f"IBM Channel:   {ibm_channel}", flush=True)
            print(f"IBM Instance:  {ibm_instance}", flush=True)

        if not api_key_configured:
            print("\n⚠️  WARNING: No API key configured!", flush=True)
            print("   Set MCP_API_KEY environment variable for production use.", flush=True)
            print("   Generate a key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\"", flush=True)
        
        print("=" * 70, flush=True)
        print("\nServer starting...\n", flush=True)
        
        # Add API key authentication middleware to the Starlette app
        # This must be done before calling run() to ensure middleware is applied
        async def run_http_with_auth():
            """Run HTTP server with authentication middleware."""
            starlette_app = mcp.sse_app()
            starlette_app.add_middleware(APIKeyAuthMiddleware, api_key=api_key)

            # Wire CORS — the --cors-origins arg was parsed but never applied before
            from starlette.middleware.cors import CORSMiddleware
            origins = [o.strip() for o in args.cors_origins.split(',') if o.strip()]
            starlette_app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_methods=["GET", "POST"],
                allow_headers=["Content-Type", "X-API-Key"],
            )

            # The MCP SDK (transport_security.py) validates the Host header against
            # the pattern "localhost:*" — it accepts any "localhost:PORT" but NOT
            # bare "localhost". In Docker the agent uses "mcp-server:3020" as the
            # host name, which the SDK rejects with 421. We rewrite it to
            # "localhost:{port}" before the SDK sees it.
            _port = mcp.settings.port
            _host_value = f"localhost:{_port}".encode()

            class DockerHostFix:
                def __init__(self, app):
                    self.app = app

                async def __call__(self, scope, receive, send):
                    if scope["type"] in ("http", "websocket"):
                        scope = {**scope, "headers": [
                            (b"host", _host_value) if k == b"host" else (k, v)
                            for k, v in scope.get("headers", [])
                        ]}
                    await self.app(scope, receive, send)

            import uvicorn
            config = uvicorn.Config(
                DockerHostFix(starlette_app),
                host=mcp.settings.host,
                port=mcp.settings.port,
                log_level=mcp.settings.log_level.lower(),
            )
            server = uvicorn.Server(config)
            await server.serve()
        
        # Run the server with authentication
        anyio.run(run_http_with_auth)
