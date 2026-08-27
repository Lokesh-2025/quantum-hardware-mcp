"""
Tests for the automatic pre-submission drift gate added 2026-08-22:
submit_job (IBM) and ionq_submit_job (IonQ real hardware) now check
_recent_drift_alert() before submitting, refusing by default if this
exact device had a real calibration alert (error spike, T1/T2 drop, or
went offline) in the last 24 hours. confirm_despite_drift_alert=True
overrides it, same shape as the existing confirm_real_hardware gate.

_recent_drift_alert() itself is tested directly against an isolated
temp database (never the real devices.db). The submit_job/ionq_submit_job
gate tests monkeypatch _recent_drift_alert so they never touch the real
database or real hardware/API — they only confirm the gate fires and
short-circuits before any real submission call.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as s


BELL_QASM2 = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
""".strip()

IONQ_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
""".strip()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_devices.db")
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE device_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, name TEXT NOT NULL,
            median_t1_us REAL, median_t2_us REAL
        )
    """)
    con.execute("""
        CREATE TABLE device_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, device_name TEXT NOT NULL, alert_type TEXT NOT NULL,
            prev_value REAL, curr_value REAL, pct_change REAL
        )
    """)
    con.commit()
    con.close()
    monkeypatch.setattr(s, "DB_PATH", db_path)
    return db_path


def _insert_alert(db_path, device_name, alert_type, ts, prev=0.01, curr=0.02, pct=100.0):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO device_alerts (ts, device_name, alert_type, prev_value, curr_value, pct_change) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, device_name, alert_type, prev, curr, pct),
    )
    con.commit()
    con.close()


def _insert_t1t2(db_path, device_name, ts, t1, t2):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO device_snapshots (ts, name, median_t1_us, median_t2_us) VALUES (?, ?, ?, ?)",
        (ts, device_name, t1, t2),
    )
    con.commit()
    con.close()


def _iso(dt):
    return dt.isoformat()


# ---------------------------------------------------------------------------
# _recent_drift_alert direct tests
# ---------------------------------------------------------------------------

def test_no_alert_returns_none(temp_db):
    assert s._recent_drift_alert("ibm_fez") is None


def test_fresh_cx_error_spike_is_detected(temp_db):
    now = datetime.now(timezone.utc)
    _insert_alert(temp_db, "ibm_fez", "cx_error_spike", _iso(now - timedelta(hours=2)))
    result = s._recent_drift_alert("ibm_fez")
    assert result is not None
    assert result["type"] == "cx_error_spike"


def test_stale_alert_outside_window_is_ignored(temp_db):
    now = datetime.now(timezone.utc)
    _insert_alert(temp_db, "ibm_fez", "cx_error_spike", _iso(now - timedelta(hours=48)))
    assert s._recent_drift_alert("ibm_fez", hours=24) is None


def test_alert_on_different_device_is_ignored(temp_db):
    now = datetime.now(timezone.utc)
    _insert_alert(temp_db, "qpu.forte-1", "went_offline", _iso(now - timedelta(hours=1)))
    assert s._recent_drift_alert("ibm_fez") is None


def test_fresh_t1_drop_is_detected_without_stored_alert_row(temp_db):
    now = datetime.now(timezone.utc)
    _insert_t1t2(temp_db, "qpu.forte-enterprise-1", _iso(now - timedelta(hours=10)), t1=188_000_000, t2=950_000)
    _insert_t1t2(temp_db, "qpu.forte-enterprise-1", _iso(now - timedelta(hours=2)), t1=100_000_000, t2=950_000)
    result = s._recent_drift_alert("qpu.forte-enterprise-1")
    assert result is not None
    assert result["type"] == "t1_drop"


def test_small_t1_change_under_threshold_is_not_flagged(temp_db):
    now = datetime.now(timezone.utc)
    _insert_t1t2(temp_db, "qpu.forte-enterprise-1", _iso(now - timedelta(hours=10)), t1=188_000_000, t2=950_000)
    _insert_t1t2(temp_db, "qpu.forte-enterprise-1", _iso(now - timedelta(hours=2)), t1=180_000_000, t2=950_000)
    assert s._recent_drift_alert("qpu.forte-enterprise-1") is None


# ---------------------------------------------------------------------------
# submit_job (IBM) gate — monkeypatch _recent_drift_alert directly so this
# never touches the real database or real hardware.
# ---------------------------------------------------------------------------

def test_submit_job_blocked_by_fresh_drift_alert(monkeypatch):
    fake_alert = {"ts": "2026-08-22T00:00:00+00:00", "type": "cx_error_spike",
                  "prev_value": 0.01, "curr_value": 0.05, "pct_change": 400.0}
    monkeypatch.setattr(s, "_recent_drift_alert", lambda name, hours=24: fake_alert)

    def _boom(*a, **k):
        raise AssertionError("submit_job must not reach _get_service() when blocked by drift alert")
    monkeypatch.setattr(s, "_get_service", _boom)

    result = s.submit_job("ibm_fez", BELL_QASM2, shots=128)
    data = json.loads(result)
    assert "error" in data
    assert "drift_alert" in data
    assert data["drift_alert"]["type"] == "cx_error_spike"


def test_submit_job_override_bypasses_the_gate(monkeypatch):
    fake_alert = {"ts": "2026-08-22T00:00:00+00:00", "type": "cx_error_spike",
                  "prev_value": 0.01, "curr_value": 0.05, "pct_change": 400.0}
    monkeypatch.setattr(s, "_recent_drift_alert", lambda name, hours=24: fake_alert)

    def _sentinel(*a, **k):
        raise RuntimeError("reached real submission path")
    monkeypatch.setattr(s, "_get_service", _sentinel)

    with pytest.raises(RuntimeError, match="reached real submission path"):
        s.submit_job("ibm_fez", BELL_QASM2, shots=128, confirm_despite_drift_alert=True)


def test_submit_job_no_alert_does_not_block(monkeypatch):
    monkeypatch.setattr(s, "_recent_drift_alert", lambda name, hours=24: None)

    def _sentinel(*a, **k):
        raise RuntimeError("reached real submission path")
    monkeypatch.setattr(s, "_get_service", _sentinel)

    with pytest.raises(RuntimeError, match="reached real submission path"):
        s.submit_job("ibm_fez", BELL_QASM2, shots=128)


# ---------------------------------------------------------------------------
# ionq_submit_job gate — real IonQ API key required (self-check always runs
# on the free simulator first, exactly like existing IonQ tests), but the
# drift block returns before target_backend.run() is ever reached for real
# hardware, so this never spends real money.
# ---------------------------------------------------------------------------

IONQ_KEY_PRESENT = bool(os.getenv("IONQ_API_KEY"))
pytestmark_ionq = pytest.mark.skipif(
    not IONQ_KEY_PRESENT, reason="IONQ_API_KEY not set — skipping live IonQ drift-gate test"
)


@pytestmark_ionq
def test_ionq_submit_job_blocked_by_fresh_drift_alert(monkeypatch):
    fake_alert = {"ts": "2026-08-22T00:00:00+00:00", "type": "t1_drop",
                  "prev_value": 188_000_000, "curr_value": 100_000_000, "pct_change": 46.8}
    monkeypatch.setattr(s, "_recent_drift_alert", lambda name, hours=24: fake_alert)

    result = s.ionq_submit_job(
        "forte-1", [IONQ_QASM], shots=128, confirm_real_hardware=True,
    )
    data = json.loads(result)
    assert "error" in data
    assert "drift_alert" in data
    assert data["drift_alert"]["type"] == "t1_drop"


@pytestmark_ionq
def test_ionq_submit_job_simulator_skips_drift_check(monkeypatch):
    fake_alert = {"ts": "2026-08-22T00:00:00+00:00", "type": "t1_drop",
                  "prev_value": 188_000_000, "curr_value": 100_000_000, "pct_change": 46.8}
    monkeypatch.setattr(s, "_recent_drift_alert", lambda name, hours=24: fake_alert)

    result = s.ionq_submit_job("simulator", [IONQ_QASM], shots=128)
    data = json.loads(result)
    assert "error" not in data
    assert data["status"] == "SIMULATED"
