"""
Backfill real per-qubit and per-pair calibration history for IBM devices,
using backend.properties(datetime=...), confirmed live to work back to each
device's real online_date with no cutoff (verified 2026-08-24 against
ibm_fez: works back to 2024-05-14, matching its online_date exactly).

This is a ONE-TIME historical catch-up. Going forward, snapshot.py's
regular collection cycle (every 2h, local LaunchAgent) keeps this archive
current on its own via save_qubit_and_pair_snapshot() — see snapshot.py.
Re-run this script only if backfilling a new device or a gap.

Schema and row-writing logic are shared with snapshot.py (single source
of truth, so the live collector and this backfill can't drift out of
sync) — this script just drives the same functions across many historical
dates instead of one live snapshot.
"""
import gzip
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from qiskit_ibm_runtime import QiskitRuntimeService

import snapshot as _snapshot

DB_PATH = _snapshot.DB_PATH
DEVICES = ["ibm_fez"]


def backfill_device(con, service, device_name, days_back_start=830):
    backend = service.backend(device_name)
    now = datetime.now(timezone.utc)
    cur = con.cursor()

    n_days = 0
    n_qubit_rows = 0
    n_pair_rows = 0
    n_raw_rows = 0
    n_errors = 0

    for days_back in range(days_back_start, -1, -1):
        target = now - timedelta(days=days_back)
        try:
            props = backend.properties(datetime=target)
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  [{device_name}] {target.date()}: query failed: {e}", file=sys.stderr)
            continue
        if props is None:
            continue

        polled_at = datetime.now(timezone.utc).isoformat()
        vendor_measured_at = str(props.last_update_date)

        raw = _snapshot.properties_to_raw_dict(props)
        blob = gzip.compress(json.dumps(raw).encode("utf-8"))
        cur.execute(
            "INSERT OR IGNORE INTO raw_properties_archive "
            "(device_name, provider, vendor_measured_at, raw_json_gzip, polled_at) "
            "VALUES (?, 'ibm', ?, ?, ?)",
            (device_name, vendor_measured_at, blob, polled_at),
        )
        if cur.rowcount:
            n_raw_rows += 1

        for qi, qubit_params in enumerate(props.qubits):
            for nduv in qubit_params:
                cur.execute(
                    "INSERT OR IGNORE INTO qubit_snapshots "
                    "(device_name, provider, qubit_index, property_name, value, unit, "
                    " vendor_measured_at, polled_at) VALUES (?, 'ibm', ?, ?, ?, ?, ?, ?)",
                    (device_name, qi, nduv.name, nduv.value, nduv.unit,
                     str(nduv.date), polled_at),
                )
                if cur.rowcount:
                    n_qubit_rows += 1

        for g in props.gates:
            if len(g.qubits) != 2:
                continue
            q1, q2 = g.qubits
            for p in g.parameters:
                cur.execute(
                    "INSERT OR IGNORE INTO pair_snapshots "
                    "(device_name, provider, qubit1, qubit2, gate_name, property_name, "
                    " value, unit, vendor_measured_at, polled_at) "
                    "VALUES (?, 'ibm', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (device_name, q1, q2, g.gate, p.name, p.value, p.unit,
                     str(p.date), polled_at),
                )
                if cur.rowcount:
                    n_pair_rows += 1

        n_days += 1
        if n_days % 100 == 0:
            con.commit()
            print(f"  [{device_name}] processed {n_days} days "
                  f"({target.date()}), {n_qubit_rows} qubit rows, "
                  f"{n_pair_rows} pair rows so far")

    con.commit()
    print(f"[{device_name}] DONE: {n_days} days queried, {n_errors} errors, "
          f"{n_qubit_rows} new qubit rows, {n_pair_rows} new pair rows, "
          f"{n_raw_rows} new raw archive rows")


def main():
    _snapshot._init_db()
    con = sqlite3.connect(DB_PATH)

    service = QiskitRuntimeService()

    for device_name in DEVICES:
        print(f"Backfilling {device_name}...")
        backfill_device(con, service, device_name)

    con.close()


if __name__ == "__main__":
    main()
