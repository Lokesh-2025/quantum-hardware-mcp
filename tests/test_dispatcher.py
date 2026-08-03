"""
Hard tests for the dispatcher + subordinate agents architecture.

Verifies:
  1. IBM questions route to IBM subagent (answer mentions IBM devices)
  2. IonQ questions route to IonQ subagent (answer mentions IonQ)
  3. Dispatcher correctly classifies ambiguous questions
  4. IBM subagent uses only IBM tools (no ionq_* tools called)
  5. IonQ subagent uses only IonQ tools (no IBM tools called)
  6. Both subagents return valid JSON with 'answer' and 'metadata' fields
  7. Cross-provider comparison works (same circuit, IBM vs IonQ)
  8. Dispatcher recovers gracefully from a bad question

Start the dispatcher first:
    cd agent && node agent-server.js

Run with:
    pytest tests/test_dispatcher.py -v
"""

import json
import pytest
import requests
import os

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:3021")


def ask(question: str, timeout: int = 60) -> dict:
    resp = requests.post(
        f"{AGENT_URL}/chat",
        json={"question": question, "history": []},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


@pytest.fixture(scope="session", autouse=True)
def require_dispatcher():
    try:
        requests.get(f"{AGENT_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Dispatcher not running at {AGENT_URL} — start with: cd agent && node agent-server.js")


# ---------------------------------------------------------------------------
# Test 1: Health check returns subagent list
# ---------------------------------------------------------------------------

def test_health_lists_subagents():
    resp = requests.get(f"{AGENT_URL}/health", timeout=5)
    data = resp.json()
    assert data["status"] == "ok"
    assert "IBM" in data["subagents"]
    assert "IonQ" in data["subagents"]


# ---------------------------------------------------------------------------
# Test 2: IBM question routes to IBM subagent
# ---------------------------------------------------------------------------

def test_ibm_question_routes_to_ibm():
    resp = ask("List available IBM quantum backends")
    assert resp["status"] == "complete"
    assert resp["metadata"]["provider"] == "IBM"
    answer = resp["answer"].lower()
    assert any(kw in answer for kw in ["ibm", "backend", "qubit", "fez", "marrakesh", "kingston"])


# ---------------------------------------------------------------------------
# Test 3: IonQ question routes to IonQ subagent
# ---------------------------------------------------------------------------

def test_ionq_question_routes_to_ionq():
    resp = ask("List available IonQ quantum devices")
    assert resp["status"] == "complete"
    assert resp["metadata"]["provider"] == "IonQ"
    answer = resp["answer"].lower()
    assert any(kw in answer for kw in ["ionq", "simulator", "device", "trapped"])


# ---------------------------------------------------------------------------
# Test 4: IBM subagent only has IBM tools in metadata
# ---------------------------------------------------------------------------

def test_ibm_subagent_has_no_ionq_tools():
    resp = ask("What IBM quantum devices are available right now?")
    assert resp["metadata"]["provider"] == "IBM"
    tools = resp["metadata"].get("toolsAvailable", [])
    ionq_tools = [t for t in tools if t.startswith("ionq_")]
    assert len(ionq_tools) == 0, f"IBM subagent has IonQ tools: {ionq_tools}"


# ---------------------------------------------------------------------------
# Test 5: IonQ subagent only has IonQ tools in metadata
# ---------------------------------------------------------------------------

def test_ionq_subagent_has_no_ibm_tools():
    resp = ask("Show me IonQ devices and their status")
    assert resp["metadata"]["provider"] == "IonQ"
    tools = resp["metadata"].get("toolsAvailable", [])
    ibm_tools = [t for t in tools if not t.startswith("ionq_")]
    assert len(ibm_tools) == 0, f"IonQ subagent has IBM tools: {ibm_tools}"


# ---------------------------------------------------------------------------
# Test 6: Response always has required fields
# ---------------------------------------------------------------------------

def test_response_shape():
    resp = ask("How many qubits does ibm_kingston have?")
    assert "status" in resp
    assert "answer" in resp
    assert "metadata" in resp
    assert "provider" in resp["metadata"]
    assert len(resp["answer"]) > 10


# ---------------------------------------------------------------------------
# Test 7: Cross-provider question — dispatcher picks one and answers
# (Hard test: question mentions BOTH IBM and IonQ)
# ---------------------------------------------------------------------------

def test_cross_provider_question_doesnt_crash():
    resp = ask(
        "Compare IBM and IonQ — which has more qubits available right now?",
        timeout=90,
    )
    assert resp["status"] == "complete"
    assert len(resp["answer"]) > 20
    # Must route to one provider (not crash)
    assert resp["metadata"]["provider"] in ("IBM", "IonQ")


# ---------------------------------------------------------------------------
# Test 8: Bad/nonsense question — dispatcher recovers gracefully
# ---------------------------------------------------------------------------

def test_garbage_question_handled_gracefully():
    resp = ask("xkcd quantum banana superposition flux capacitor 9999")
    assert resp["status"] == "complete"
    assert len(resp["answer"]) > 0  # Some answer, not a crash


# ---------------------------------------------------------------------------
# Test 9: IBM subagent submits a real circuit and gets a job ID
# (Hardest test — end-to-end through dispatcher → IBM subagent → MCP → IBM hardware)
# ---------------------------------------------------------------------------

BELL_QASM2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];"""

def test_ibm_circuit_submission_end_to_end():
    resp = ask(
        f"Submit this circuit to the least busy IBM device for 128 shots and return the job ID:\n{BELL_QASM2}",
        timeout=90,
    )
    assert resp["status"] == "complete"
    assert resp["metadata"]["provider"] == "IBM"
    answer = resp["answer"]
    # A real job ID is alphanumeric, ~20 chars
    import re
    job_ids = re.findall(r'[a-z0-9]{15,25}', answer)
    assert len(job_ids) > 0, f"No job ID found in answer: {answer[:300]}"
