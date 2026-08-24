"""
Tests for check_chip_identity and its helper _qubit_fingerprint_vector,
added 2026-08-24 as the first real feature built on the new per-qubit/
per-pair calibration archive (schema in backfill_qubit_history.py).

Uses a temp SQLite DB seeded with synthetic-but-realistic per-qubit rows
so these never touch the real devices.db or real hardware. The threshold
logic (GAP_BASELINE) is calibrated against ibm_fez's real 831-day history
(see server.py comments) — these tests check the *logic* behaves sanely
on constructed cases, not that the specific baseline numbers are exactly
right for other devices.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as s


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_devices.db")
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE qubit_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_name TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'ibm',
            qubit_index INTEGER NOT NULL, property_name TEXT NOT NULL,
            value REAL, unit TEXT, vendor_measured_at TEXT NOT NULL, polled_at TEXT NOT NULL,
            UNIQUE(device_name, qubit_index, property_name, vendor_measured_at)
        );
        CREATE TABLE pair_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_name TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'ibm',
            qubit1 INTEGER NOT NULL, qubit2 INTEGER NOT NULL, gate_name TEXT NOT NULL,
            property_name TEXT NOT NULL, value REAL, unit TEXT,
            vendor_measured_at TEXT NOT NULL, polled_at TEXT NOT NULL,
            UNIQUE(device_name, qubit1, qubit2, gate_name, property_name, vendor_measured_at)
        );
    """)
    con.commit()
    con.close()
    monkeypatch.setattr(s, "DB_PATH", db_path)
    return db_path


def _seed_qubit_snapshot(db_path, device, qubit, prop, value, measured_at):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT OR IGNORE INTO qubit_snapshots "
        "(device_name, qubit_index, property_name, value, unit, vendor_measured_at, polled_at) "
        "VALUES (?, ?, ?, ?, 'us', ?, ?)",
        (device, qubit, prop, value, measured_at, measured_at),
    )
    con.commit()
    con.close()


def _seed_pair_snapshot(db_path, device, q1, q2, value, measured_at):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT OR IGNORE INTO pair_snapshots "
        "(device_name, qubit1, qubit2, gate_name, property_name, value, unit, "
        " vendor_measured_at, polled_at) VALUES (?, ?, ?, 'cz', 'gate_error', ?, '', ?, ?)",
        (device, q1, q2, value, measured_at, measured_at),
    )
    con.commit()
    con.close()


def _iso(dt):
    return dt.isoformat()


def _seed_identical_snapshots(db_path, device, n_qubits, now, days_ago_list):
    """Seed N qubits with a fixed, deterministic-but-varied pattern at
    every date in days_ago_list — simulates an unchanged chip observed
    at multiple points in time (values differ per-qubit, so there's a
    real rank pattern to correlate, but each qubit's own value is
    identical across all seeded dates, which should read as highly
    consistent)."""
    for days_ago in days_ago_list:
        measured_at = _iso(now - timedelta(days=days_ago))
        for qi in range(n_qubits):
            _seed_qubit_snapshot(db_path, device, qi, "T1", 30.0 + qi * 0.5, measured_at)
            _seed_qubit_snapshot(db_path, device, qi, "T2", 25.0 + qi * 0.4, measured_at)
            _seed_qubit_snapshot(db_path, device, qi, "readout_error", 0.01 + qi * 0.0001, measured_at)
        for qi in range(n_qubits - 1):
            _seed_pair_snapshot(db_path, device, qi, qi + 1, 0.005 + qi * 0.00005, measured_at)


# ---------------------------------------------------------------------------
# _qubit_fingerprint_vector
# ---------------------------------------------------------------------------

def test_fingerprint_vector_returns_seeded_values(temp_db):
    now = datetime.now(timezone.utc)
    measured_at = _iso(now - timedelta(days=1))
    _seed_qubit_snapshot(temp_db, "dev_a", 0, "T1", 42.0, measured_at)
    _seed_qubit_snapshot(temp_db, "dev_a", 0, "T2", 38.0, measured_at)

    vec = s._qubit_fingerprint_vector("dev_a")
    assert vec[0]["t1"] == 42.0
    assert vec[0]["t2"] == 38.0


def test_fingerprint_vector_respects_as_of_cutoff(temp_db):
    now = datetime.now(timezone.utc)
    old_date = _iso(now - timedelta(days=30))
    new_date = _iso(now - timedelta(days=1))
    _seed_qubit_snapshot(temp_db, "dev_a", 0, "T1", 20.0, old_date)
    _seed_qubit_snapshot(temp_db, "dev_a", 0, "T1", 99.0, new_date)

    vec_recent = s._qubit_fingerprint_vector("dev_a")
    assert vec_recent[0]["t1"] == 99.0

    vec_old = s._qubit_fingerprint_vector("dev_a", as_of=_iso(now - timedelta(days=10)))
    assert vec_old[0]["t1"] == 20.0


def test_avg_gate_error_averages_across_incident_pairs(temp_db):
    now = datetime.now(timezone.utc)
    measured_at = _iso(now - timedelta(days=1))
    _seed_pair_snapshot(temp_db, "dev_a", 0, 1, 0.01, measured_at)
    _seed_pair_snapshot(temp_db, "dev_a", 1, 2, 0.03, measured_at)

    vec = s._qubit_fingerprint_vector("dev_a")
    assert vec[1]["avg_gate_error"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# check_chip_identity — no live hardware call (online_date lookup mocked out)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_live_online_date_lookup(monkeypatch):
    """_get_service() would try a real IBM connection — make it fail cleanly
    so the bring-up guard is skipped (best-effort by design) in every test
    here, keeping these fully offline."""
    def _boom():
        raise RuntimeError("no live service in tests")
    monkeypatch.setattr(s, "_get_service", _boom)


def test_no_history_returns_clean_error(temp_db):
    result = json.loads(s.check_chip_identity("nonexistent_device"))
    assert "error" in result
    assert "No per-qubit history" in result["error"]


def test_unchanged_chip_reads_as_consistent(temp_db):
    now = datetime.now(timezone.utc)
    _seed_identical_snapshots(temp_db, "dev_a", n_qubits=40, now=now,
                               days_ago_list=[0, 2, 5, 7, 9, 12])
    result = json.loads(s.check_chip_identity("dev_a", compare_days_back=7))
    assert result["verdict"] == "consistent"
    assert result["avg_raw_correlation"] > 0.9


def test_scrambled_qubit_order_is_not_consistent(temp_db):
    """Simulates a qubit relabeling / identity change: the SAME set of
    values exists at both dates, but reassigned to different qubit
    indices — correlation against the real (unscrambled) qubit indices
    should collapse."""
    now = datetime.now(timezone.utc)
    n = 40
    old_date = _iso(now - timedelta(days=7))
    new_date = _iso(now)

    for qi in range(n):
        _seed_qubit_snapshot(temp_db, "dev_a", qi, "T1", 30.0 + qi * 0.5, old_date)
        _seed_qubit_snapshot(temp_db, "dev_a", qi, "T2", 25.0 + qi * 0.4, old_date)
        _seed_qubit_snapshot(temp_db, "dev_a", qi, "readout_error", 0.01 + qi * 0.0001, old_date)
    # New date: same value pool, but reversed qubit assignment
    for qi in range(n):
        source = n - 1 - qi
        _seed_qubit_snapshot(temp_db, "dev_a", qi, "T1", 30.0 + source * 0.5, new_date)
        _seed_qubit_snapshot(temp_db, "dev_a", qi, "T2", 25.0 + source * 0.4, new_date)
        _seed_qubit_snapshot(temp_db, "dev_a", qi, "readout_error", 0.01 + source * 0.0001, new_date)
    for qi in range(n - 1):
        _seed_pair_snapshot(temp_db, "dev_a", qi, qi + 1, 0.005, old_date)
        _seed_pair_snapshot(temp_db, "dev_a", qi, qi + 1, 0.005, new_date)

    result = json.loads(s.check_chip_identity("dev_a", compare_days_back=7))
    assert result["verdict"] != "consistent"
    assert result["avg_raw_correlation"] < 0.3


def test_not_enough_qubits_returns_error(temp_db):
    now = datetime.now(timezone.utc)
    _seed_identical_snapshots(temp_db, "dev_a", n_qubits=3, now=now, days_ago_list=[0, 7])
    result = json.loads(s.check_chip_identity("dev_a", compare_days_back=7))
    assert "error" in result


def test_gap_calibrated_baseline_is_included_in_response(temp_db):
    now = datetime.now(timezone.utc)
    _seed_identical_snapshots(temp_db, "dev_a", n_qubits=40, now=now,
                               days_ago_list=[0, 2, 5, 7, 9, 12])
    result = json.loads(s.check_chip_identity("dev_a", compare_days_back=7))
    assert "expected_for_this_gap" in result
    assert "median" in result["expected_for_this_gap"]
    assert "p10" in result["expected_for_this_gap"]
