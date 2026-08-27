"""
Tests for _find_connected_qubit_subset (server.py), added 2026-08-25 —
fixes a real bug reported by a real user (quantum-chemistry-vqe): best_qubits
used to pick the top-n qubits by individual score alone, only warning after
the fact if they happened not to be connected. Confirmed live and
reproducible against real ibm_fez before this fix: best_qubits('ibm_fez',
n=8) returned 8 qubits with ZERO real connections between them.

Uses qiskit's real CouplingMap class with small, hand-built synthetic
topologies — no live IBM API calls, pure algorithm tests.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit.transpiler import CouplingMap

import server as s


def test_finds_connected_subset_on_a_line():
    """0-1-2-3-4 line topology. Best individual scores are scattered
    (0, 2, 4), but a connected subset of 3 must exist somewhere on the line."""
    cmap = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])
    scores = {0: 0.01, 1: 0.5, 2: 0.02, 3: 0.5, 4: 0.03}
    result = s._find_connected_qubit_subset(scores, cmap, n=3)
    assert result is not None
    assert len(result) == 3
    # must be an actually-connected subgraph
    edges = set(map(tuple, cmap.get_edges()))
    edges |= {(b, a) for a, b in edges}
    for q in result:
        assert any((q, other) in edges for other in result if other != q)


def test_reproduces_the_real_reported_bug_scenario():
    """The exact real bug: 8 best-individually-scored qubits scattered
    across a chip with no connections between any of them, while a
    genuinely connected (if slightly worse-scored) group of 8 exists
    elsewhere. The fix must prefer the connected group."""
    # A tight, fully-connected cluster: qubits 100-107 (8 qubits, decent scores)
    cluster_edges = [[100 + i, 100 + i + 1] for i in range(7)]
    # 8 scattered, individually "better"-scored qubits, isolated from
    # everything (each only connects to some irrelevant qubit far away)
    scattered = [1, 20, 41, 62, 83, 90, 95, 99]
    isolating_edges = [[q, 500 + i] for i, q in enumerate(scattered)]  # dead-end stubs
    cmap = CouplingMap(cluster_edges + isolating_edges)

    scores = {q: 0.001 * (i + 1) for i, q in enumerate(scattered)}  # scattered qubits score BEST
    for i, q in enumerate(range(100, 108)):
        scores[q] = 0.01 + 0.001 * i  # cluster qubits score slightly worse, but are connected

    result = s._find_connected_qubit_subset(scores, cmap, n=8)
    assert result is not None
    assert len(result) == 8
    assert result == set(range(100, 108)), "must find the real connected cluster, not the scattered set"


def test_returns_none_when_no_connected_subset_of_that_size_exists():
    cmap = CouplingMap([[0, 1]])  # only 2 qubits, only ever connected as a pair
    scores = {0: 0.01, 1: 0.02, 2: 0.03}  # qubit 2 isn't even in the coupling map
    result = s._find_connected_qubit_subset(scores, cmap, n=3)
    assert result is None


def test_returns_none_with_no_coupling_map():
    result = s._find_connected_qubit_subset({0: 0.1, 1: 0.2}, None, n=2)
    assert result is None


def test_single_qubit_request_always_trivially_connected():
    cmap = CouplingMap([[0, 1], [1, 2]])
    scores = {0: 0.5, 1: 0.1, 2: 0.3}
    result = s._find_connected_qubit_subset(scores, cmap, n=1)
    assert result == {1}  # best individual qubit, trivially "connected" alone


def test_prefers_best_total_score_among_multiple_valid_connected_subsets():
    """Two separate connected clusters both large enough for n=2 -- the
    cheaper one (lower total score) should win."""
    cmap = CouplingMap([[0, 1], [10, 11]])
    scores = {0: 0.01, 1: 0.01, 10: 0.5, 11: 0.5}
    result = s._find_connected_qubit_subset(scores, cmap, n=2, num_seeds=4)
    assert result == {0, 1}
