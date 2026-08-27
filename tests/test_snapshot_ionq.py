"""
Test for collect_ionq() (snapshot.py) -- confirms the real fix for a real
bug found 2026-08-22: IonQ's /v0.3/backends list response does not include
fidelity data inline, so the collector was silently saving null calibration
fields for every snapshot since it was written (354 rows, checked directly
against the live database). The fix fetches each backend's separate
characterization_url, mirroring the working pattern already used in
quantum-verifier's providers/ionq.py::ionq_compare_devices.

Live IonQ API call -- skips automatically if IONQ_API_KEY isn't configured.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from dotenv import load_dotenv

load_dotenv()
IONQ_KEY_PRESENT = bool(os.getenv("IONQ_API_KEY"))
pytestmark = pytest.mark.skipif(
    not IONQ_KEY_PRESENT, reason="IONQ_API_KEY not set — skipping live IonQ snapshot test"
)

from snapshot import collect_ionq


def test_collect_ionq_returns_real_non_null_calibration_for_active_qpus():
    rows = collect_ionq()
    assert rows, "expected at least one backend row"

    active_qpus = [r for r in rows if r["name"].startswith("qpu.") and r["operational"] == 1]
    assert active_qpus, "expected at least one operational real QPU (e.g. forte-1)"

    for row in active_qpus:
        assert row["avg_cx_error"] is not None, f"{row['name']}: avg_cx_error is still null"
        assert row["avg_readout_error"] is not None, f"{row['name']}: avg_readout_error is still null"
        assert 0 <= row["avg_cx_error"] < 1
        assert 0 <= row["avg_readout_error"] < 1


def test_collect_ionq_provider_field_is_always_ionq():
    rows = collect_ionq()
    assert all(r["provider"] == "ionq" for r in rows)
