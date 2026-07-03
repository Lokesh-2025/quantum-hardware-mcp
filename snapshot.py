"""
snapshot.py
-----------
Fetches live calibration stats for every accessible quantum device across
IBM Quantum, IonQ, and AWS Braket, then persists them.

Where it writes depends on the environment:
  - Locally (LaunchAgent):  SQLite devices.db  → feeds device_history + report.py
  - GitHub Actions (CI):    CSV data/snapshots.csv → committed to the repo as history

Run manually:
    .venv/bin/python snapshot.py

Or let the LaunchAgent / GitHub Actions call it automatically every 6 hours.
"""

import os
import sys
import csv
import json
import sqlite3
import statistics
import requests
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

# Load .env from the same directory as this file.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

BASE_DIR  = os.path.dirname(__file__)
DB_PATH   = os.path.join(BASE_DIR, "devices.db")
CSV_PATH  = os.path.join(BASE_DIR, "data", "snapshots.csv")

CSV_FIELDS = ["ts", "provider", "name", "num_qubits", "operational",
              "pending_jobs", "avg_cx_error", "avg_readout_error",
              "median_t1_us", "median_t2_us", "native_gate_set",
              "coupling_map_edges", "connectivity_density",
              "qubit_yield_fraction", "max_shots"]


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS device_snapshots (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                   TEXT    NOT NULL,
                provider             TEXT    NOT NULL DEFAULT 'ibm',
                name                 TEXT    NOT NULL,
                num_qubits           INTEGER,
                operational          INTEGER,
                pending_jobs         INTEGER,
                avg_cx_error         REAL,
                avg_readout_error    REAL,
                median_t1_us         REAL,
                median_t2_us         REAL,
                native_gate_set      TEXT,
                coupling_map_edges   INTEGER,
                connectivity_density REAL,
                qubit_yield_fraction REAL,
                max_shots            INTEGER
            )
        """)
        # Add columns to existing DBs that don't have them yet
        for col, typedef in [
            ("provider",             "TEXT NOT NULL DEFAULT 'ibm'"),
            ("median_t1_us",         "REAL"),
            ("median_t2_us",         "REAL"),
            ("native_gate_set",      "TEXT"),
            ("coupling_map_edges",   "INTEGER"),
            ("connectivity_density", "REAL"),
            ("qubit_yield_fraction", "REAL"),
            ("max_shots",            "INTEGER"),
        ]:
            try:
                con.execute(f"ALTER TABLE device_snapshots ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # job_submissions — tracks every job sent through the MCP server
        con.execute("""
            CREATE TABLE IF NOT EXISTS job_submissions (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                       TEXT NOT NULL,
                job_id                   TEXT,
                provider                 TEXT,
                backend_name             TEXT,
                tool_name                TEXT,
                circuit_qubits           INTEGER,
                circuit_depth_raw        INTEGER,
                circuit_depth_transpiled INTEGER,
                shots_requested          INTEGER,
                agent_loop_iteration     INTEGER,
                was_preflight_checked    INTEGER DEFAULT 0,
                was_ai_corrected         INTEGER DEFAULT 0
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_submissions_ts
            ON job_submissions (ts)
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_name_ts
            ON device_snapshots (name, ts)
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS device_alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                device_name TEXT NOT NULL,
                alert_type  TEXT NOT NULL,
                prev_value  REAL,
                curr_value  REAL,
                pct_change  REAL
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_device
            ON device_alerts (device_name, ts)
        """)


def _save_snapshots(rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.executemany(
            """
            INSERT INTO device_snapshots
                (ts, provider, name, num_qubits, operational, pending_jobs,
                 avg_cx_error, avg_readout_error,
                 median_t1_us, median_t2_us, native_gate_set,
                 coupling_map_edges, connectivity_density,
                 qubit_yield_fraction, max_shots)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    ts,
                    r.get("provider", "ibm"),
                    r["name"],
                    r.get("num_qubits"),
                    int(r["operational"]) if r.get("operational") is not None else None,
                    r.get("pending_jobs"),
                    r.get("avg_cx_error"),
                    r.get("avg_readout_error"),
                    r.get("median_t1_us"),
                    r.get("median_t2_us"),
                    r.get("native_gate_set"),
                    r.get("coupling_map_edges"),
                    r.get("connectivity_density"),
                    r.get("qubit_yield_fraction"),
                    r.get("max_shots"),
                )
                for r in rows
            ],
        )


def _write_csv(rows: list[dict]) -> None:
    """
    Append snapshot rows to data/snapshots.csv.
    Creates the file with a header row on the first call.
    Used by GitHub Actions so the history is committed as plain text.
    """
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    new_file = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "ts":                   ts,
                "provider":             r.get("provider", "ibm"),
                "name":                 r["name"],
                "num_qubits":           r.get("num_qubits"),
                "operational":          r.get("operational"),
                "pending_jobs":         r.get("pending_jobs"),
                "avg_cx_error":         r.get("avg_cx_error"),
                "avg_readout_error":    r.get("avg_readout_error"),
                "median_t1_us":         r.get("median_t1_us"),
                "median_t2_us":         r.get("median_t2_us"),
                "native_gate_set":      r.get("native_gate_set"),
                "coupling_map_edges":   r.get("coupling_map_edges"),
                "connectivity_density": r.get("connectivity_density"),
                "qubit_yield_fraction": r.get("qubit_yield_fraction"),
                "max_shots":            r.get("max_shots"),
            })


DRIFT_THRESHOLD = 0.20  # alert when a metric rises by more than 20%

def _check_and_write_alerts(rows: list[dict], ts: str) -> int:
    """Compare current snapshot against the previous one.
    Write a device_alerts row whenever a metric spikes or a device goes offline.
    Returns the number of alerts written.
    """
    alerts = []
    with sqlite3.connect(DB_PATH) as con:
        for row in rows:
            name = row["name"]

            # Fetch the most recent previous snapshot for this device
            prev = con.execute("""
                SELECT avg_cx_error, avg_readout_error, operational
                FROM device_snapshots
                WHERE name = ?
                ORDER BY ts DESC
                LIMIT 1
            """, (name,)).fetchone()

            if prev is None:
                continue  # first ever snapshot for this device — nothing to compare

            prev_cx, prev_readout, prev_op = prev

            # Check avg_cx_error spike
            curr_cx = row.get("avg_cx_error")
            if prev_cx and curr_cx and prev_cx > 0:
                pct = (curr_cx - prev_cx) / prev_cx
                if pct > DRIFT_THRESHOLD:
                    alerts.append((ts, name, "cx_error_spike", prev_cx, curr_cx, round(pct * 100, 1)))

            # Check avg_readout_error spike
            curr_ro = row.get("avg_readout_error")
            if prev_readout and curr_ro and prev_readout > 0:
                pct = (curr_ro - prev_readout) / prev_readout
                if pct > DRIFT_THRESHOLD:
                    alerts.append((ts, name, "readout_error_spike", prev_readout, curr_ro, round(pct * 100, 1)))

            # Check device went offline
            curr_op = int(row["operational"]) if row.get("operational") is not None else None
            if prev_op == 1 and curr_op == 0:
                alerts.append((ts, name, "went_offline", 1.0, 0.0, None))

        if alerts:
            con.executemany("""
                INSERT INTO device_alerts (ts, device_name, alert_type, prev_value, curr_value, pct_change)
                VALUES (?, ?, ?, ?, ?, ?)
            """, alerts)

    return len(alerts)


def _two_qubit_errors(props) -> list[float]:
    """Return error values for all 2-qubit gates.
    Eagle-class devices use ECR, not CX — filtering by gate name misses them.
    """
    if props is None:
        return []
    return [
        g.parameters[0].value
        for g in props.gates
        if len(g.qubits) == 2 and g.parameters
    ]


def collect_ionq() -> list[dict]:
    """Fetch live calibration data from IonQ REST API. Free — no job credits used."""
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        print("  [IonQ] IONQ_API_KEY not set — skipping", file=sys.stderr)
        return []

    try:
        resp = requests.get(
            "https://api.ionq.co/v0.3/backends",
            headers={"Authorization": f"apiKey {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        backends = resp.json()
    except Exception as e:
        print(f"  [IonQ] Failed to fetch backends: {e}", file=sys.stderr)
        return []

    rows = []
    for b in backends:
        # IonQ returns fidelity data per backend
        fidelity = b.get("characterization", {}) or {}
        row = {
            "provider":   "ionq",
            "name":       b.get("backend", b.get("name", "unknown")),
            "num_qubits": b.get("qubits"),
            "operational": 1 if b.get("status") == "available" else 0,
            "pending_jobs": None,
            # IonQ reports 1q/2q gate fidelity — convert to error rate (1 - fidelity)
            "avg_cx_error": round(1 - fidelity["2q"]["mean"], 5)
                if fidelity.get("2q", {}).get("mean") else None,
            "avg_readout_error": round(1 - fidelity.get("1q", {}).get("mean", 1), 5)
                if fidelity.get("1q", {}).get("mean") else None,
        }
        rows.append(row)

    print(f"  [IonQ] Collected {len(rows)} backends")
    return rows


def collect_braket() -> list[dict]:
    """Fetch live device status from AWS Braket. Free — no QPU credits used."""
    key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret  = os.getenv("AWS_SECRET_ACCESS_KEY")
    region  = os.getenv("AWS_REGION", "us-east-1")

    if not key_id or not secret:
        print("  [Braket] AWS credentials not set — skipping", file=sys.stderr)
        return []

    try:
        client = boto3.client(
            "braket",
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name=region,
        )
        # Search for all QPU devices (not simulators)
        response = client.search_devices(filters=[])
        devices = response.get("devices", [])
    except Exception as e:
        print(f"  [Braket] Failed to fetch devices: {e}", file=sys.stderr)
        return []

    rows = []
    for d in devices:
        arn = d.get("deviceArn", "")
        name = d.get("deviceName", arn.split("/")[-1])
        status = d.get("deviceStatus", "")

        # Try to get detailed calibration data
        avg_cx_error = None
        avg_readout_error = None
        num_qubits = None
        try:
            detail = client.get_device(deviceArn=arn)
            caps = json.loads(detail.get("deviceCapabilities", "{}"))
            # Qubit count from paradigm
            paradigm = caps.get("paradigm", {})
            num_qubits = paradigm.get("qubitCount")
            # Rigetti-style fidelity data
            specs = caps.get("specs", {})
            two_q = specs.get("2Q", {})
            if two_q:
                fidelities = [v.get("fCZ") or v.get("fXY") or v.get("f")
                              for v in two_q.values() if isinstance(v, dict)]
                fidelities = [f for f in fidelities if f is not None]
                if fidelities:
                    avg_cx_error = round(1 - sum(fidelities) / len(fidelities), 5)
            one_q = specs.get("1Q", {})
            if one_q:
                ro = [v.get("fRO") for v in one_q.values()
                      if isinstance(v, dict) and v.get("fRO")]
                if ro:
                    avg_readout_error = round(1 - sum(ro) / len(ro), 5)
        except Exception:
            pass  # calibration data optional — status alone is useful

        rows.append({
            "provider":         "braket/" + d.get("providerName", "unknown").lower(),
            "name":             name,
            "num_qubits":       num_qubits,
            "operational":      1 if status == "ONLINE" else 0,
            "pending_jobs":     None,
            "avg_cx_error":     avg_cx_error,
            "avg_readout_error": avg_readout_error,
        })

    print(f"  [Braket] Collected {len(rows)} QPU devices")
    return rows


def collect_ibm() -> list[dict]:
    """Fetch live calibration data from IBM Quantum."""
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        print("  [IBM] IBM_QUANTUM_TOKEN not set — skipping", file=sys.stderr)
        return []

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backends = service.backends()
    rows = []
    for backend in backends:
        status = backend.status()
        props = backend.properties()
        row = {
            "provider":   "ibm",
            "name":       backend.name,
            "num_qubits": backend.num_qubits,
            "operational": status.operational,
            "pending_jobs": status.pending_jobs,
        }
        if props:
            cx = _two_qubit_errors(props)
            if cx:
                row["avg_cx_error"] = round(sum(cx) / len(cx), 5)
            readout = [
                props.readout_error(q)
                for q in range(backend.num_qubits)
                if props.readout_error(q) is not None
            ]
            if readout:
                row["avg_readout_error"] = round(sum(readout) / len(readout), 5)

            # T1/T2 medians (in microseconds) — some qubits may lack data
            t1_vals, t2_vals = [], []
            for q in range(backend.num_qubits):
                try:
                    v = props.t1(q)
                    if v is not None:
                        t1_vals.append(v * 1e6)
                except Exception:
                    pass
                try:
                    v = props.t2(q)
                    if v is not None:
                        t2_vals.append(v * 1e6)
                except Exception:
                    pass
            if t1_vals:
                row["median_t1_us"] = round(statistics.median(t1_vals), 1)
            if t2_vals:
                row["median_t2_us"] = round(statistics.median(t2_vals), 1)

            # Qubit yield — fraction with valid T1 calibration
            row["qubit_yield_fraction"] = round(len(t1_vals) / backend.num_qubits, 3)

        # Native gate set
        try:
            row["native_gate_set"] = ",".join(sorted(backend.operation_names))
        except Exception:
            pass

        # Coupling map topology
        try:
            cm = backend.coupling_map
            if cm is not None:
                edges = len(list(cm.get_edges()))
                n = backend.num_qubits
                row["coupling_map_edges"] = edges
                row["connectivity_density"] = round(
                    edges / (n * (n - 1)), 4) if n > 1 else 0
        except Exception:
            pass

        # Max shots per job
        try:
            row["max_shots"] = backend.max_shots
        except Exception:
            try:
                row["max_shots"] = backend.configuration().max_shots
            except Exception:
                pass

        rows.append(row)

    print(f"  [IBM] Collected {len(rows)} backends")
    return rows


def log_job_submission(job_id: str, provider: str, backend_name: str,
                       tool_name: str, circuit_qubits: int = None,
                       circuit_depth_raw: int = None,
                       circuit_depth_transpiled: int = None,
                       shots_requested: int = None,
                       agent_loop_iteration: int = None,
                       was_preflight_checked: bool = False,
                       was_ai_corrected: bool = False) -> None:
    """Log every job submission for the agentic workload study."""
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""
                INSERT INTO job_submissions
                    (ts, job_id, provider, backend_name, tool_name,
                     circuit_qubits, circuit_depth_raw, circuit_depth_transpiled,
                     shots_requested, agent_loop_iteration,
                     was_preflight_checked, was_ai_corrected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, job_id, provider, backend_name, tool_name,
                  circuit_qubits, circuit_depth_raw, circuit_depth_transpiled,
                  shots_requested, agent_loop_iteration,
                  int(was_preflight_checked), int(was_ai_corrected)))
    except Exception:
        pass  # never crash the main flow over logging


def collect() -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting multi-provider snapshot...")

    all_rows = []
    all_rows += collect_ibm()
    all_rows += collect_ionq()
    all_rows += collect_braket()

    rows = all_rows
    if not rows:
        print("ERROR: No data collected from any provider.", file=sys.stderr)
        sys.exit(1)

    if os.getenv("GITHUB_ACTIONS"):
        _write_csv(rows)
        print(f"[{datetime.now(timezone.utc).isoformat()}] "
              f"Wrote {len(rows)} rows to {CSV_PATH}")
    else:
        ts = datetime.now(timezone.utc).isoformat()
        n_alerts = _check_and_write_alerts(rows, ts)
        _save_snapshots(rows)
        print(f"[{datetime.now(timezone.utc).isoformat()}] "
              f"Saved {len(rows)} snapshots ({len([r for r in rows if r.get('provider','ibm')=='ibm'])} IBM, "
              f"{len([r for r in rows if r.get('provider','')=='ionq'])} IonQ, "
              f"{len([r for r in rows if 'braket' in r.get('provider','')])} Braket)"
              + (f" | {n_alerts} drift alert(s)" if n_alerts else ""))


if __name__ == "__main__":
    if not os.getenv("GITHUB_ACTIONS"):
        _init_db()
    collect()
