# Quantum Hardware MCP Server

A production MCP server that gives AI assistants programmatic access to live quantum hardware across IBM Quantum, IonQ, and AWS Braket. Natural language in. Real quantum results out. No dashboards. No manual API calls.

Built in collaboration with [Jack Woehr](https://github.com/jwoehr) — IBM Quantum veteran, Qiskit contributor.

Listed on [Glama](https://glama.ai), [mcp.so](https://mcp.so), and [PulseMCP](https://pulsemcp.com).

---

## Why this exists

Quantum researchers lose hours to operational overhead:

- Manually checking which device has the lowest error rate today
- Submitting the same circuit to IBM, then separately to IonQ, then comparing by hand
- Losing reproducibility context between runs — "what was the CX error when I ran Figure 3?"
- No pre-flight — wasting queue time on circuits that fail at transpile
- No cross-provider queue visibility — IBM backlogged for 3 days, IonQ open, no way to know without checking each dashboard manually
- Discovering routing failures only after wasting QPU credits — a degree-4 qubit on heavy-hex silently causes 4× gate inflation

This server eliminates that overhead. Your AI assistant handles device selection, circuit validation, routing overhead prediction, job submission, result retrieval, and amplification analysis through a single interface.

---

## What we discovered running real experiments

We have been using this server to run real quantum experiments on IBM ibm_marrakesh — not just as a demo, but as active research infrastructure. The results changed how we built the server.

**The routing failure discovery (Phase 4):**
We built a 7-qubit Grover circuit to search Pascal's Triangle for rows where 3003 appears. The circuit had 263 logical gates. After transpilation: 1,037 hardware gates. The signal collapsed.

The root cause was not the transpiler. It was **graph embedding**: one ancilla qubit needed 4 direct connections in the circuit interaction graph. IBM heavy-hex topology allows max 3 connections per qubit. The transpiler had no choice but to inject ~300 SWAP gates (3 CX each) to route around the constraint.

This is now baked into the server as `check_routing_overhead` — it detects degree-4 violations before you submit.

**The LNAA breakthrough (Phase 5):**
After discovering that Grover's oracle structure creates an unfixable degree-4 node on heavy-hex, we scrapped Grover entirely. We derived an Ising Hamiltonian from scratch where the target rows (14, 15, 78) are the ground states of a magnetic system. IBM's RZZ and RX gates implement this natively — no routing, no SWAP, no ancilla.

Result: **27.78× amplification with 135 hardware gates** on ibm_marrakesh.

Previous best: 4.17× with 103 gates (Phase 3, Grover).

This is the first time Lattice-Native Amplitude Amplification has been applied to Singmaster's Conjecture on real quantum hardware. The insight — encode targets as ground states, not Boolean conditions — is now the `encode_search_problem` tool.

---

## Fleet coverage

19 backends across three providers:

| Provider | Backends | Access |
|----------|----------|--------|
| IBM Quantum | 3 QPUs (ibm_torino 133q, ibm_marrakesh 156q, ibm_fez 156q) | API token |
| IonQ | 6 (Aria, Forte, Harmony + simulators) | API key |
| AWS Braket | 10 (QuEra Aquila 256q, IonQ via Braket, Rigetti via Braket, simulators) | IAM credentials |

All 19 are polled every 2 hours. The dataset grows continuously — ML routing recommendations are planned once 60+ days of data accumulate.

---

## System architecture

```mermaid
graph TD
    User["User / AI Assistant"]

    subgraph Control Plane
        Dispatcher["Dispatcher\nagent-server.js\nRoutes IBM vs IonQ vs Braket"]
        IBMAgent["IBM Subagent\nibm-subagent.js"]
        IonQAgent["IonQ Subagent\nionq-subagent.js"]
    end

    subgraph Execution Plane
        MCP["MCP Server\nserver.py\n34 tools"]
        IBMAPI["IBM Quantum API\nQiskit Runtime"]
        IonQAPI["IonQ REST API"]
        BraketAPI["AWS Braket API"]
    end

    subgraph Observability Plane
        Snapshot["snapshot.py\nRuns every 2h"]
        DB["devices.db\nSQLite — local history"]
        CSV["data/snapshots.csv\nPublic — GitHub Actions CI"]
        Jobs["job_submissions\nAgentic workload log"]
        Report["report.py\nDaily fleet report"]
        Alerts["Calibration drift alerts\nCX / readout / T1 / T2"]
    end

    User --> Dispatcher
    Dispatcher --> IBMAgent
    Dispatcher --> IonQAgent
    IBMAgent --> MCP
    IonQAgent --> MCP
    MCP --> IBMAPI
    MCP --> IonQAPI
    MCP --> BraketAPI
    Snapshot --> DB
    Snapshot --> CSV
    Snapshot --> Alerts
    MCP --> Jobs
    DB --> MCP
    Jobs --> MCP
    Report --> DB
```

---

## How it works

**Step 1 — Request classification**
The dispatcher reads your message and classifies it: IBM job, IonQ job, or cross-provider comparison. Each subagent sees only the tools for its provider — no accidental cross-wiring.

**Step 2 — Pre-flight validation**
Before touching the queue, `debug_circuit` catches missing measurements, decoherence bound violations, and qubit count mismatches. `circuit_report` does a full dry-run transpile — gate counts, qubit mapping, per-pair CX error, estimated fidelity — all without submitting. `check_routing_overhead` detects degree-4 qubit violations that would cause SWAP flooding.

**Step 3 — Credit-aware routing**
`estimate_runtime` computes QPU minutes before submission. `route_job` ranks backends by cost × error rate and picks the cheapest option that meets your fidelity requirement.

**Step 4 — Execution**
`submit_job` compiles to the backend's native gate set (OpenQASM 2.0 or 3.0), submits, and returns a `job_id`. `job_status` and `job_results` close the loop.

**Step 5 — Analysis**
`get_amplification` computes the amplification factor directly from a job ID and your marked bitstrings — no manual result parsing.

**Step 6 — Observability**
Every 2 hours, `snapshot.py` records calibration state across all 19 backends. Drift alerts fire when CX error, readout error, T1, or T2 spikes >20%. `repro_score` runs KL-divergence across N identical runs to quantify hardware reliability. Every job submission is logged for longitudinal workload analysis.

---

## Tools (34 total)

### Device intelligence

| Tool | What it does |
|------|-------------|
| `list_devices` | All accessible IBM backends with live operational status |
| `get_device_details` | Per-qubit T1/T2, readout error, gate error, queue depth |
| `compare_devices` | Rank by CX error, queue depth, qubit count, or combined score |
| `queue_status` | Current queue snapshot across all backends |
| `best_qubits` | Score and rank qubits by calibration quality — warns if top qubits aren't physically connected on the coupling map |
| `device_history` | Calibration snapshots over the last N days |
| `device_on_date` | Exact calibration state on any past date — for paper reproducibility |

### Job lifecycle

| Tool | What it does |
|------|-------------|
| `submit_job` | Transpile and submit OpenQASM 2.0 or 3.0 — returns `job_id` |
| `job_status` | QUEUED / RUNNING / DONE / ERROR |
| `job_results` | Bit-string measurement counts from a completed job |
| `cancel_job` | Cancel a queued or running job |
| `list_jobs` | Recent jobs with status, backend, and timestamps |

### Pre-flight and cost control

| Tool | What it does |
|------|-------------|
| `debug_circuit` | Pre-submission check: missing measurements, decoherence violations, qubit mismatches |
| `circuit_report` | Full dry-run: gate counts, qubit mapping, per-pair CX errors, estimated fidelity |
| `estimate_runtime` | QPU minutes + queue wait estimate before you submit |
| `route_job` | Credit-aware routing — cheapest backend that meets your error threshold |

### Circuit intelligence (derived from real experiments)

| Tool | What it does |
|------|-------------|
| `check_routing_overhead` | Input: qubit interaction pairs → detects degree>3 nodes → predicts SWAP flood and gate inflation before it happens. Learned from Phase 4: degree-4 node caused 263→1,037 gate explosion. |
| `encode_search_problem` | Input: Boolean conditions like `{"1":1, "4":0}` → derives Ising h_i and J_ij coefficients with full sign derivation and QAOA circuit recipe. The math behind Phase 5's 27.78× result. |
| `estimate_hardware_gates` | Predicts transpiled gate count from logical gates + max qubit degree. Knows the empirical ~600-gate noise floor on ibm_marrakesh. |
| `get_amplification` | Input: job ID + marked bitstrings → amplification factor, per-state shot breakdown, verdict (EXCELLENT/GOOD/WEAK/FAILED). |

### Algorithms and chemistry

| Tool | What it does |
|------|-------------|
| `run_grover` | Full Grover's search — builds oracle + diffusion operator, picks least-busy backend, submits |
| `run_vqe` | Variational Quantum Eigensolver — H2 ground state to chemical accuracy |
| `estimate_expectation` | Estimator primitive: computes ⟨ψ\|O\|ψ⟩ for Pauli observables |

### Observability

| Tool | What it does |
|------|-------------|
| `get_alerts` | Calibration drift alerts — spikes >20% in CX error, readout error, T1, or T2 |
| `start_repro_experiment` | Run the same circuit N times, record variance across runs |
| `repro_score` | KL-divergence reproducibility score (0 = identical, 1 = maximally different) |
| `job_analytics` | Aggregate stats across all logged jobs — transpilation expansion ratios, per-tool breakdown |

### IonQ

| Tool | What it does |
|------|-------------|
| `ionq_devices` | All IonQ backends and simulators with live status |
| `ionq_submit_job` | Submit OpenQASM 2.0/3.0 to IonQ hardware or simulator |
| `ionq_job_status` | Job status on IonQ |
| `ionq_job_results` | Measurement counts from a completed IonQ job |

---

## Real experiments: Singmaster's Conjecture on IBM hardware

Singmaster's Conjecture asks whether any integer appears 9+ times in Pascal's Triangle. We are using this server to run quantum search experiments on ibm_marrakesh, attempting to amplify rows containing 3003 above the hardware noise floor.

All job IDs are real. All results are reproducible.

| Phase | Approach | Gates | Amplification | Finding |
|-------|----------|-------|---------------|---------|
| Phase 1 | Grover, 4 qubits, target=6 | 611 | **4.11×** | Signal clear |
| Phase 2 | Grover, 4 qubits, target=3003, unoptimized | 16,271 | **1.04×** | Noise floor hit — brackets coherence limit |
| Phase 3 v3 | Grover, 4 qubits, opt=2 seed=42, ancilla diffusion | 103 | **4.17×** | 99.4% gate reduction from Phase 2 |
| Phase 4 v1 | Grover, 7 qubits, rows 14+15+78 | 624 | **3.04×** | Row 78 found on hardware for the first time |
| Phase 4 v2 | Grover, 7 qubits, lossy oracle, 2 iterations | 1,037 | **1.92×** | Routing failure — degree-4 node caused 4× gate inflation |
| Phase 5 | LNAA, 7 qubits, Ising Hamiltonian quantum walk | **135** | **27.78×** | Record — 6.7× better than previous best |

**Phase 5 detail (ibm_marrakesh, job d971ok52su3c739hhmpg, 4096 shots):**

```
Row  78 (1001110): 1157 shots (28.2%) ← 3003  C(78,2)=3003
Row  15 (0001111): 1056 shots (25.8%) ← 3003  C(15,5)=3003
Row  14 (0001110):  455 shots (11.1%) ← 3003  C(14,6)=3003
Combined: 65.1% of shots on 3 target rows. Random baseline: 2.34%.
Amplification: 27.78×
Hardware gates: 135 (vs 1,037 for Phase 4 v2)
```

**Key insight:** IBM heavy-hex is an Ising lattice. RZZ + RX gates are native — zero routing overhead. Encoding search targets as ground states of a Hamiltonian outperforms Boolean oracle + diffusion when the hardware topology constrains qubit connectivity to degree ≤ 3.

Full experiment history and code: [singmasters-conjecture](https://github.com/Lokesh-2025/singmasters-conjecture)

---

### Observability plane — calibration history

`snapshot.py` runs every 2 hours via GitHub Actions:

| Field | Why it matters |
|-------|---------------|
| `avg_cx_error` | Primary gate quality metric |
| `avg_readout_error` | State-preparation and measurement overhead |
| `median_t1_us` | Median coherence time — robust to outlier qubits |
| `median_t2_us` | Dephasing time — degrades faster than T1 under noise |
| `qubit_yield_fraction` | Fraction of qubits with usable T1/T2 |
| `connectivity_density` | Edges / max-possible-edges — IBM heavy-hex ~0.015 vs IonQ all-to-all = 1.0 |
| `gate_set_size` | Number of native gates — affects transpilation depth |
| `max_circuit_depth` | Hard limit before decoherence kills the result |
| `native_2q_gate` | CX vs ECR vs ZZ — matters for circuit rewriting |
| `day_of_week` | 0=Monday … 6=Sunday — for weekly seasonality modeling |
| `hour_utc` | 0–23 — for time-of-day queue pattern detection |

**Job submissions table** — every call to `submit_job`, `run_grover`, or `run_vqe` writes a row:

```
job_id · provider · backend · tool · circuit_qubits · circuit_depth_raw
circuit_depth_transpiled · shots · agent_loop_iteration
was_preflight_checked · was_ai_corrected · day_of_week · hour_utc
```

---

## Test suite

```bash
python tests/test_all_tools.py
```

28 checks across all tools. Read-only tools hit the real IBM and IonQ APIs. Write tools are tested against validation paths only — zero QPU credits spent.

---

## Project structure

```
quantum-hardware-mcp/
├── server.py                      # MCP server — 34 tools
├── snapshot.py                    # Multi-provider calibration snapshot (every 2h)
├── report.py                      # Daily fleet report
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── agent/
│   ├── agent-server.js            # Dispatcher — control plane router
│   ├── chat.js                    # Terminal interface
│   └── subagents/
│       ├── base-subagent.js       # Shared ReAct loop
│       ├── ibm-subagent.js        # IBM specialist
│       └── ionq-subagent.js       # IonQ specialist
├── experiments/
│   ├── singmasters_grover.py      # Phase 1 — Grover, target=6
│   ├── singmasters_3003.py        # Phase 2 — Grover, target=3003, coherence limit
│   ├── phase3_grover_v3.py        # Phase 3 v3 — 103 gates, 4.17×
│   ├── phase4_grover_7q.py        # Phase 4 v1 — 7 qubits, row 78 found
│   ├── phase4_grover_v2.py        # Phase 4 v2 — lossy oracle, routing failure discovered
│   ├── phase5_lnaa.py             # Phase 5 — LNAA, 27.78× amplification (RECORD)
│   └── vqe_h2.py                  # VQE for H2 molecule ground state
├── tests/
│   └── test_all_tools.py          # Smoke test suite
├── data/
│   └── snapshots.csv              # Public calibration history (updated by CI every 2h)
└── .github/workflows/
    └── snapshot.yml               # GitHub Actions: snapshot every 2h
```

---

## Quick start

**Prerequisites:** Python 3.10+, Node.js 18+, IBM Quantum account (free), LLM API key.

```bash
git clone https://github.com/Lokesh-2025/quantum-hardware-mcp.git
cd quantum-hardware-mcp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd agent && npm install && cd ..
cp .env.example .env        # add IBM token + LLM key
docker compose up --build   # starts MCP server + agent
node agent/chat.js          # open terminal chat
```

---

## Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "quantum-hardware": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/quantum-hardware-mcp/server.py"]
    }
  }
}
```

Restart Claude Desktop. All 34 tools appear under the hammer icon.

---

## LLM provider support

| Provider | Cost | Env var |
|----------|------|---------|
| Anthropic Claude | Paid | `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` |
| Google Gemini | Free tier | `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` |
| OpenAI | Paid | `LLM_PROVIDER=openai` + `OPENAI_API_KEY` |
| Ollama | Free, local | `LLM_PROVIDER=ollama` + `OLLAMA_MODEL` |
| vLLM | Self-hosted | `LLM_PROVIDER=vllm` + `VLLM_BASE_URL` |

---

## Roadmap

**Completed**
- [x] IBM Quantum tools — device intelligence, job lifecycle, pre-flight, routing
- [x] IonQ support — devices, submit, status, results
- [x] AWS Braket integration — 10 backends in snapshot pipeline
- [x] Multi-agent control plane — dispatcher + IBM/IonQ specialist subagents
- [x] Calibration drift alerts — CX error, readout error, T1, T2
- [x] Reproducibility scoring — KL-divergence across N runs
- [x] Credit-aware routing — QPU cost estimation before submit
- [x] Singmaster Phase 1 — Grover 4.11× (depth 611)
- [x] Singmaster Phase 2 — coherence limit bracketed at depth 16,271
- [x] Singmaster Phase 3 v3 — 4.17× at 103 gates (99.4% reduction from Phase 2)
- [x] Singmaster Phase 4 v1 — 7 qubits, row 78 found, 3.04×
- [x] Singmaster Phase 4 v2 — routing failure diagnosed as graph embedding problem
- [x] Singmaster Phase 5 LNAA — **27.78× amplification, 135 gates (record)**
- [x] `check_routing_overhead` — degree>3 detection before SWAP flood
- [x] `encode_search_problem` — Boolean conditions → Ising Hamiltonian coefficients
- [x] `estimate_hardware_gates` — predicts transpiled gate count + noise floor warning
- [x] `get_amplification` — amplification factor from job ID + marked bitstrings
- [x] `best_qubits` connectivity check — warns when top qubits aren't physically linked
- [x] Temporal indexing — day_of_week + hour_utc on all snapshots and jobs
- [x] Job submissions table — transpilation expansion ratio tracking
- [x] Listed on Glama, mcp.so, PulseMCP

**Next**
- [ ] `generate_walk_hamiltonian` — encode_search_problem output → QASM directly
- [ ] `build_grover_circuit` — problem description → QASM, no Python needed
- [ ] `algorithm_selector` — decides Grover vs LNAA based on circuit + hardware analysis
- [ ] `run_search_experiment` — full pipeline in one call: encode → build → submit → amplification
- [ ] `mutate_circuit_topology` — rewrites algorithm math to fit heavy-hex natively (routing dilation = 1.0)
- [ ] `inject_hardware_layout` — bypasses stochastic transpiler, maps directly to high-coherence qubits
- [ ] VQE on real IBM hardware — H2 hardware result
- [ ] Quantum Rush Hour detection — weekly queue seasonality
- [ ] Smart routing brain — cross-provider ML recommendations
- [ ] Publication package generator — job ID → figures + BibTeX + methods section

---

## License

MIT — see [LICENSE](LICENSE).
