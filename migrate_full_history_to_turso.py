"""
migrate_full_history_to_turso.py
----------------------------------
Widens the shared Turso database to hold the FULL data both tools collect,
not just the narrower 10-field subset quantum-verifier originally used.
Run once, 2026-08-31, after core/turso.py's device_snapshots table was
ALTERed to add the 20 extra fields this repo's local devices.db already
had (native_gate_set, CLOPS, quantum_volume, calibration timestamps, etc.)

Rebuilds device_snapshots FRESH (wipes then re-inserts) from this repo's
local devices.db, rather than trying to merge/upsert against what was
already there -- that earlier partial migration is a strict subset of
what this one covers, so a clean rebuild is simpler and avoids duplicate
rows for the same (ts, name) with different columns filled in.

Also migrates the REAL depth of qubit_snapshots/pair_snapshots -- this
repo's local archive has 686,309 / 754,480 rows (vs. the ~24k/14k
quantum-verifier's own smaller, newer archive had when first migrated),
going back to each device's actual online_date. INSERT OR IGNORE on the
existing UNIQUE constraints, so this is safe to re-run.

Run manually, once:
    .venv/bin/python migrate_full_history_to_turso.py
"""
import os
import sqlite3
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from turso_db import is_configured, execute, execute_batch

DB_PATH = os.path.join(os.path.dirname(__file__), "devices.db")
BATCH = 500

DEVICE_FIELDS = [
    "ts", "provider", "name", "num_qubits", "operational", "pending_jobs",
    "avg_cx_error", "avg_readout_error", "median_t1_us", "median_t2_us",
    "native_gate_set", "coupling_map_edges", "connectivity_density",
    "qubit_yield_fraction", "max_shots", "day_of_week", "hour_utc",
    "processor_family", "backend_version", "online_date",
    "last_calibration_dt", "dt_ns", "avg_2q_gate_duration_ns",
    "avg_readout_length_ns", "avg_prob_meas0_prep1", "avg_prob_meas1_prep0",
    "rep_delay_default_ms", "clops_h", "max_experiments", "quantum_volume",
]


def migrate_devices(con) -> None:
    print("Wiping and rebuilding device_snapshots in Turso (full field set)...")
    execute("DELETE FROM device_snapshots")

    cols_sql = ", ".join(DEVICE_FIELDS)
    rows = con.execute(
        f"SELECT {cols_sql} FROM device_snapshots WHERE provider IN ('ibm','ionq') AND name != 'simulator'"
    ).fetchall()
    print(f"Local devices.db has {len(rows)} real ibm/ionq device rows (full fields).")

    placeholders = ", ".join("?" for _ in DEVICE_FIELDS)
    statements = [(f"INSERT INTO device_snapshots ({cols_sql}) VALUES ({placeholders})", row)
                  for row in rows]
    for i in range(0, len(statements), BATCH):
        execute_batch(statements[i:i + BATCH])
        print(f"  device_snapshots: {min(i + BATCH, len(statements))}/{len(statements)}")

    total = execute("SELECT COUNT(*) FROM device_snapshots")[0][0]
    earliest, latest = execute("SELECT MIN(ts), MAX(ts) FROM device_snapshots")[0]
    print(f"device_snapshots in Turso now: {total} rows, {earliest} -> {latest}")


def migrate_qubit_pair(con) -> None:
    for table, cols in [
        ("qubit_snapshots",
         "device_name, qubit_index, property_name, value, unit, vendor_measured_at, polled_at"),
        ("pair_snapshots",
         "device_name, qubit1, qubit2, gate_name, property_name, value, unit, vendor_measured_at, polled_at"),
    ]:
        rows = con.execute(f"SELECT {cols} FROM {table}").fetchall()
        print(f"Migrating {len(rows)} {table} rows from local devices.db (the real deep archive)...")
        placeholders = ", ".join("?" for _ in cols.split(", "))
        statements = [(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})", row)
                      for row in rows]
        for i in range(0, len(statements), BATCH):
            execute_batch(statements[i:i + BATCH])
            if i % (BATCH * 20) == 0:
                print(f"  {table}: {min(i + BATCH, len(statements))}/{len(statements)}")
        total = execute(f"SELECT COUNT(*) FROM {table}")[0][0]
        print(f"{table} in Turso now: {total} rows")


def main() -> None:
    if not is_configured():
        print("TURSO_DATABASE_URL/TURSO_AUTH_TOKEN not set")
        return
    con = sqlite3.connect(DB_PATH)
    migrate_devices(con)
    migrate_qubit_pair(con)
    con.close()


if __name__ == "__main__":
    main()
