"""
resume_pair_migration.py
--------------------------
One-time resume script, 2026-08-31: the original full migration
(migrate_full_history_to_turso.py) was killed partway through by a
session interruption, not a real failure -- device_snapshots and
qubit_snapshots had already finished (confirmed directly: Turso's counts
matched or exceeded the local source), only pair_snapshots was partial
(404,304 of 754,636 local rows).

Rather than re-run the whole script (which would safely but wastefully
re-send ~1.1M already-inserted rows via INSERT OR IGNORE), this skips
straight to pair_snapshots and resumes from an offset near where it
stopped -- a little overlap at the boundary is fine, INSERT OR IGNORE
handles it for free.

Run manually, once:
    .venv/bin/python resume_pair_migration.py
"""
import sqlite3
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from turso_db import is_configured, execute, execute_batch

DB_PATH = os.path.join(os.path.dirname(__file__), "devices.db")
BATCH = 500

# Small safety margin before the last confirmed Turso count, so a little
# re-send overlap happens (harmless, INSERT OR IGNORE) rather than risking
# a gap if insertion order wasn't perfectly stable.
RESUME_OFFSET = max(0, 404_304 - 5_000)


def main() -> None:
    if not is_configured():
        print("TURSO_DATABASE_URL/TURSO_AUTH_TOKEN not set")
        return

    con = sqlite3.connect(DB_PATH)
    cols = "device_name, qubit1, qubit2, gate_name, property_name, value, unit, vendor_measured_at, polled_at"
    rows = con.execute(
        f"SELECT {cols} FROM pair_snapshots ORDER BY id LIMIT -1 OFFSET {RESUME_OFFSET}"
    ).fetchall()
    con.close()
    print(f"Resuming pair_snapshots from offset {RESUME_OFFSET}: {len(rows)} rows to send.")

    placeholders = ", ".join("?" for _ in cols.split(", "))
    statements = [(f"INSERT OR IGNORE INTO pair_snapshots ({cols}) VALUES ({placeholders})", row)
                  for row in rows]
    for i in range(0, len(statements), BATCH):
        execute_batch(statements[i:i + BATCH])
        if i % (BATCH * 20) == 0:
            print(f"  pair_snapshots: {min(i + BATCH, len(statements))}/{len(statements)}")

    total = execute("SELECT COUNT(*) FROM pair_snapshots")[0][0]
    print(f"pair_snapshots in Turso now: {total} rows")


if __name__ == "__main__":
    main()
