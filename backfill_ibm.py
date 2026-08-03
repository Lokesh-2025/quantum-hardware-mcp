"""
backfill_ibm.py
---------------
Pulls IBM's historical calibration database for all 3 accessible backends
and inserts the data into devices.db day by day.

IBM keeps calibration history from when each device came online.
ibm_marrakesh came online 2024-08-07 — API returns data from ~Oct 2024.

Run once:
    .venv/bin/python backfill_ibm.py

Takes ~30 minutes. Safe to re-run — skips dates already in the DB.
"""

import os, sys, sqlite3, statistics, time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DB_PATH  = os.path.join(os.path.dirname(__file__), "devices.db")
BACKENDS = ["ibm_marrakesh", "ibm_fez", "ibm_kingston"]

# Pull from this date forward (Sept 2024 returns 404 — Oct is safe)
START_DATE = datetime(2024, 10, 1, tzinfo=timezone.utc)
END_DATE   = datetime.now(timezone.utc).replace(hour=0, minute=0,
                                                second=0, microsecond=0)


def _two_qubit_errors(props):
    if props is None:
        return []
    return [g.parameters[0].value for g in props.gates
            if len(g.qubits) == 2 and g.parameters]


def already_have(con, name, date_str):
    """Return True if we already have a snapshot for this device on this date."""
    row = con.execute("""
        SELECT 1 FROM device_snapshots
        WHERE name = ? AND ts LIKE ?
        LIMIT 1
    """, (name, f"{date_str}%")).fetchone()
    return row is not None


def extract_row(backend, props, ts, day_of_week, hour_utc):
    """Build a snapshot dict from a BackendV2 + BackendProperties pair."""
    row = {
        "ts":          ts,
        "provider":    "ibm",
        "name":        backend.name,
        "num_qubits":  backend.num_qubits,
        "operational": 1,       # historical — assume operational (no status API for past)
        "pending_jobs": None,   # not available historically
        "day_of_week": day_of_week,
        "hour_utc":    hour_utc,
    }

    if props is None:
        return row

    # CX / 2Q gate errors
    cx = _two_qubit_errors(props)
    if cx:
        row["avg_cx_error"] = round(sum(cx) / len(cx), 5)

    # Readout error
    readout = [props.readout_error(q) for q in range(backend.num_qubits)
               if props.readout_error(q) is not None]
    if readout:
        row["avg_readout_error"] = round(sum(readout) / len(readout), 5)

    # T1 / T2 medians
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
    row["qubit_yield_fraction"] = round(len(t1_vals) / backend.num_qubits, 3)

    # Last calibration timestamp
    try:
        row["last_calibration_dt"] = props.last_update_date.isoformat()
    except Exception:
        pass

    # Asymmetric readout errors + readout length from qubit items
    try:
        readout_lengths, prob01, prob10 = [], [], []
        for q in range(backend.num_qubits):
            for item in props.qubits[q]:
                if item.name == "readout_length" and item.value is not None:
                    readout_lengths.append(item.value)
                elif item.name == "prob_meas0_prep1" and item.value is not None:
                    prob01.append(item.value)
                elif item.name == "prob_meas1_prep0" and item.value is not None:
                    prob10.append(item.value)
        if readout_lengths:
            row["avg_readout_length_ns"] = round(
                sum(readout_lengths) / len(readout_lengths), 1)
        if prob01:
            row["avg_prob_meas0_prep1"] = round(sum(prob01) / len(prob01), 6)
        if prob10:
            row["avg_prob_meas1_prep0"] = round(sum(prob10) / len(prob10), 6)
    except Exception:
        pass

    # Static fields — same for all historical rows of this device
    try:
        row["native_gate_set"] = ",".join(sorted(backend.operation_names))
    except Exception:
        pass
    try:
        cm = backend.coupling_map
        if cm is not None:
            edges = len(list(cm.get_edges()))
            n = backend.num_qubits
            row["coupling_map_edges"] = edges
            row["connectivity_density"] = round(edges / (n * (n - 1)), 4) if n > 1 else 0
    except Exception:
        pass
    try:
        pt = backend.processor_type
        row["processor_family"] = f"{pt['family']} r{pt.get('revision', '')}"
    except Exception:
        pass
    try:
        row["backend_version"] = backend.backend_version
    except Exception:
        pass
    try:
        od = backend.online_date
        row["online_date"] = od.isoformat() if od else None
    except Exception:
        pass
    try:
        row["dt_ns"] = round(backend.dt * 1e9, 3)
    except Exception:
        pass
    try:
        d = backend._data
        row["max_shots"]       = d.get("max_shots")
        row["clops_h"]         = d.get("clops_h")
        row["max_experiments"] = d.get("max_experiments")
        row["quantum_volume"]  = d.get("quantum_volume")
    except Exception:
        pass

    return row


def insert_row(con, r):
    con.execute("""
        INSERT INTO device_snapshots
            (ts, provider, name, num_qubits, operational, pending_jobs,
             avg_cx_error, avg_readout_error, median_t1_us, median_t2_us,
             native_gate_set, coupling_map_edges, connectivity_density,
             qubit_yield_fraction, max_shots, day_of_week, hour_utc,
             processor_family, backend_version, online_date,
             last_calibration_dt, dt_ns, avg_readout_length_ns,
             avg_prob_meas0_prep1, avg_prob_meas1_prep0,
             clops_h, max_experiments, quantum_volume)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r.get("ts"), r.get("provider"), r.get("name"), r.get("num_qubits"),
        r.get("operational"), r.get("pending_jobs"),
        r.get("avg_cx_error"), r.get("avg_readout_error"),
        r.get("median_t1_us"), r.get("median_t2_us"),
        r.get("native_gate_set"), r.get("coupling_map_edges"),
        r.get("connectivity_density"), r.get("qubit_yield_fraction"),
        r.get("max_shots"), r.get("day_of_week"), r.get("hour_utc"),
        r.get("processor_family"), r.get("backend_version"),
        r.get("online_date"), r.get("last_calibration_dt"),
        r.get("dt_ns"), r.get("avg_readout_length_ns"),
        r.get("avg_prob_meas0_prep1"), r.get("avg_prob_meas1_prep0"),
        r.get("clops_h"), r.get("max_experiments"), r.get("quantum_volume"),
    ))


def main():
    import snapshot as snap
    snap._init_db()

    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=os.getenv("IBM_QUANTUM_TOKEN"))

    # Pre-load all 3 backend objects (static metadata)
    print("Loading backends...")
    backends = {name: service.backend(name) for name in BACKENDS}

    total_days = (END_DATE - START_DATE).days
    inserted = 0
    skipped  = 0
    errors   = 0

    with sqlite3.connect(DB_PATH) as con:
        current = START_DATE
        while current < END_DATE:
            date_str = current.strftime("%Y-%m-%d")
            dow      = current.weekday()   # 0=Monday
            hour     = 12                  # noon UTC as canonical daily snapshot

            for bname, backend in backends.items():
                if already_have(con, bname, date_str):
                    skipped += 1
                    continue

                try:
                    props = backend.properties(datetime=current)
                    ts    = current.replace(hour=hour).isoformat()
                    row   = extract_row(backend, props, ts, dow, hour)
                    insert_row(con, row)
                    con.commit()
                    inserted += 1
                    print(f"  ✓ {date_str} {bname}  cx={row.get('avg_cx_error','?')}  "
                          f"t1={row.get('median_t1_us','?')}µs")
                except Exception as e:
                    errors += 1
                    print(f"  ✗ {date_str} {bname}  {e}", file=sys.stderr)

                time.sleep(0.5)  # be polite to IBM's API

            days_done = (current - START_DATE).days + 1
            pct = round(days_done / total_days * 100, 1)
            print(f"[{pct}%] {date_str} complete — "
                  f"{inserted} inserted, {skipped} skipped, {errors} errors")

            current += timedelta(days=1)

    print(f"\nDone. {inserted} rows inserted, {skipped} already existed, {errors} errors.")


if __name__ == "__main__":
    main()
