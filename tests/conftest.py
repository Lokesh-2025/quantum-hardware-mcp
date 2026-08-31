"""
Session-wide test safety for the shared Turso database integration added
2026-08-30 (see turso_db.py, and _run_query() in server.py).

server.py calls load_dotenv() at MODULE level (line ~77), so simply
`import server as s` -- which most test files here already do -- loads
the REAL TURSO_DATABASE_URL/TURSO_AUTH_TOKEN from .env into this process's
environment. Without this fixture, test_drift_gate.py's direct,
unmocked calls to _recent_drift_alert() against an isolated temp_db
(monkeypatch.setattr(s, "DB_PATH", ...)) would silently hit the real,
live, shared Turso database instead of the isolated test db, since
_run_query() tries Turso first regardless of what DB_PATH is set to.

Patches _get_turso_client itself (not just the env vars) -- robust
regardless of WHEN load_dotenv() fires relative to this fixture. Mirrors
the identical fix already made on the quantum-verifier side.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True, scope="session")
def _block_real_turso_during_tests():
    import server as s
    original = s._get_turso_client
    s._get_turso_client = lambda: None
    yield
    s._get_turso_client = original
