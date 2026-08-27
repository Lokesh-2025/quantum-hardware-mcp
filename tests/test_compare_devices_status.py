"""
Tests for compare_devices' status/availability fix, added 2026-08-25 —
a real user (quantum-chemistry-vqe) reported a top-ranked device sitting
queued indefinitely because it was actually in a non-"active" state that
compare_devices couldn't see, since it collapsed IBM's real status message
down to just "online"/"offline" based on `operational` alone. Confirmed
by reading the code directly: `operational` can stay True in states that
aren't really usable the same way "active" is.

Mocks _get_service()/backend objects — no real IBM API calls, and no live
device is currently in a non-active state to test against directly.
"""
import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as s


class _FakeStatus:
    def __init__(self, status_msg, operational, pending_jobs=0):
        self.status_msg = status_msg
        self.operational = operational
        self.pending_jobs = pending_jobs


class _FakeGate:
    def __init__(self, gate, qubits, error):
        self.gate = gate
        self.qubits = qubits
        self.parameters = [MagicMock(value=error)]


class _FakeProps:
    def __init__(self, n_qubits, cx_error):
        self.gates = [_FakeGate("cx", [i, i + 1], cx_error) for i in range(n_qubits - 1)]
        self._n = n_qubits

    def readout_error(self, q):
        return 0.01

    def t1(self, q):
        return 100e-6

    def t2(self, q):
        return 80e-6


class _FakeBackend:
    def __init__(self, name, num_qubits, status_msg, operational, cx_error=0.01, pending_jobs=0):
        self.name = name
        self.num_qubits = num_qubits
        self._status = _FakeStatus(status_msg, operational, pending_jobs)
        self._props = _FakeProps(num_qubits, cx_error)

    def status(self):
        return self._status

    def properties(self):
        return self._props


def _mock_service(monkeypatch, backends):
    fake_service = MagicMock()
    fake_service.backends.return_value = backends
    monkeypatch.setattr(s, "_get_service", lambda: fake_service)
    monkeypatch.setattr(s, "_save_snapshots", lambda rows: None)  # skip real db writes


def test_active_operational_devices_get_ranked(monkeypatch):
    backends = [
        _FakeBackend("dev_a", 27, "active", True, cx_error=0.01),
        _FakeBackend("dev_b", 27, "active", True, cx_error=0.02),
    ]
    _mock_service(monkeypatch, backends)
    result = json.loads(s.compare_devices(sort_by="cx_error"))
    names = [d["name"] for d in result["devices"]]
    assert names == ["dev_a", "dev_b"]
    assert result["unavailable_devices"] == []


def test_non_active_status_excluded_from_ranking_even_if_operational(monkeypatch):
    """The real bug: a device can report operational=True while its real
    status message isn't 'active' -- must not be ranked, must be reported
    separately instead."""
    backends = [
        _FakeBackend("dev_good", 27, "active", True, cx_error=0.005),  # best score
        _FakeBackend("dev_maintenance", 27, "internal", True, cx_error=0.001),  # even better score, but not active
    ]
    _mock_service(monkeypatch, backends)
    result = json.loads(s.compare_devices(sort_by="cx_error"))
    ranked_names = [d["name"] for d in result["devices"]]
    assert "dev_maintenance" not in ranked_names, "a non-active device must never be ranked, regardless of score"
    assert ranked_names == ["dev_good"]
    assert len(result["unavailable_devices"]) == 1
    assert result["unavailable_devices"][0]["name"] == "dev_maintenance"
    assert result["unavailable_devices"][0]["status"] == "internal"


def test_non_operational_device_excluded_even_with_active_status_string(monkeypatch):
    backends = [
        _FakeBackend("dev_a", 27, "active", True),
        _FakeBackend("dev_b", 27, "active", False),  # operational=False
    ]
    _mock_service(monkeypatch, backends)
    result = json.loads(s.compare_devices(sort_by="cx_error"))
    ranked_names = [d["name"] for d in result["devices"]]
    assert ranked_names == ["dev_a"]
    assert result["unavailable_devices"][0]["name"] == "dev_b"


def test_unavailable_note_is_none_when_nothing_excluded(monkeypatch):
    backends = [_FakeBackend("dev_a", 27, "active", True)]
    _mock_service(monkeypatch, backends)
    result = json.loads(s.compare_devices(sort_by="cx_error"))
    assert result["unavailable_note"] is None


def test_unavailable_note_present_when_something_excluded(monkeypatch):
    backends = [
        _FakeBackend("dev_a", 27, "active", True),
        _FakeBackend("dev_b", 27, "maintenance", True),
    ]
    _mock_service(monkeypatch, backends)
    result = json.loads(s.compare_devices(sort_by="cx_error"))
    assert result["unavailable_note"] is not None
    assert "active" in result["unavailable_note"].lower()


def test_status_field_preserves_real_message_not_collapsed_online_offline(monkeypatch):
    backends = [_FakeBackend("dev_a", 27, "active", True)]
    _mock_service(monkeypatch, backends)
    result = json.loads(s.compare_devices(sort_by="cx_error"))
    assert result["devices"][0]["status"] == "active"  # not "online"
