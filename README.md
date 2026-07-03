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

This server eliminates that overhead. Your AI assistant handles device selection, circuit validation, job submission, result retrieval, and cross-provider comparison through a single interface.

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
        MCP["MCP Server\nserver.py\n27 tools"]
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
Before touching the queue, `debug_circuit` catches missing measurements, decoherence bound violations, and qubit count mismatches. `circuit_report` does a full dry-run transpile — gate counts, qubit mapping, per-pair CX error, estimated fidelity — all without submitting.

**Step 3 — Credit-aware routing**
`estimate_runtime` computes QPU minutes before submission. `route_job` ranks backends by cost × error rate and picks the cheapest option that meets your fidelity requirement.

**Step 4 — Execution**
`submit_job` compiles to the backend's native gate set (OpenQASM 2.0 or 3.0), submits, and returns a `job_id`. `job_status` and `job_results` close the loop.

**Step 5 — Observability**
Every 2 hours, `snapshot.py` records calibration state across all 19 backends. Drift alerts fire when CX error, readout error, T1, or T2 spikes >20%. `repro_score` runs KL-divergence across N identical runs to quantify hardware reliability. Every job submission is logged for longitudinal workload analysis.

---

## Planes

### Control plane — routing and coordination

| Component | Role |
|-----------|------|
| `agent-server.js` | Dispatcher: classifies request, spawns the right subagent |
| `ibm-subagent.js` | IBM specialist — only exposes IBM tools to the LLM |
| `ionq-subagent.js` | IonQ specialist — only exposes IonQ tools to the LLM |
| `base-subagent.js` | Shared ReAct loop (observe → think → act) used by both |

### Execution plane — quantum hardware interface

27 tools across IBM Quantum (22), IonQ (4), and analytics (1). All live in `server.py`.

**IBM Quantum — device intelligence**

| Tool | What it does |
|------|-------------|
| `list_devices` | All accessible IBM backends with live operational status |
| `get_device_details` | Per-qubit T1/T2, readout error, gate error, queue depth |
| `compare_devices` | Rank by CX error, queue depth, qubit count, or combined score — includes avg readout error, avg T1/T2, connectivity |
| `queue_status` | Current queue snapshot across all backends |
| `best_qubits` | Score and rank qubits on a device by calibration quality |
| `device_history` | Calibration snapshots for a device over the last N days — includes median T1/T2, qubit yield, day/hour |
| `device_on_date` | Exact calibration state on any past date — for paper reproducibility |

**IBM Quantum — job lifecycle**

| Tool | What it does |
|------|-------------|
| `submit_job` | Transpile and submit OpenQASM 2.0 or 3.0 — returns `job_id` |
| `job_status` | QUEUED / RUNNING / DONE / ERROR |
| `job_results` | Bit-string measurement counts (Sampler) or expectation values (Estimator) from a completed job |
| `cancel_job` | Cancel a queued or running job |
| `list_jobs` | Recent jobs with status, backend, and timestamps |

**IBM Quantum — pre-flight and cost control**

| Tool | What it does |
|------|-------------|
| `debug_circuit` | Pre-submission check: missing measurements, decoherence violations, qubit mismatches |
| `circuit_report` | Full dry-run: gate counts, qubit mapping, per-pair CX errors, estimated fidelity |
| `estimate_runtime` | QPU minutes + queue wait estimate before you submit |
| `route_job` | Credit-aware routing — cheapest backend that meets your error threshold |

**IBM Quantum — algorithms and chemistry**

| Tool | What it does |
|------|-------------|
| `run_grover` | Full Grover's search — builds oracle + diffusion operator, picks least-busy backend, submits |
| `run_vqe` | Variational Quantum Eigensolver — H2 ground state to chemical accuracy (0.0 mHartree) in ~60 iterations |
| `estimate_expectation` | Estimator primitive: computes ⟨ψ\|O\|ψ⟩ for Pauli observables (VQE, QAOA, quantum chemistry) |

**IBM Quantum — observability**

| Tool | What it does |
|------|-------------|
| `get_alerts` | Calibration drift alerts — spikes >20% in CX error, readout error, T1, or T2 (T1/T2 use SQL LAG window function) |
| `start_repro_experiment` | Run the same circuit N times, record variance across runs |
| `repro_score` | KL-divergence reproducibility score (0 = identical, 1 = maximally different) |
| `job_analytics` | Aggregate stats across all logged jobs — transpilation expansion ratios, average shots, per-tool breakdown |

**IonQ**

| Tool | What it does |
|------|-------------|
| `ionq_devices` | All IonQ backends and simulators with live status |
| `ionq_submit_job` | Submit OpenQASM 2.0/3.0 to IonQ hardware or simulator |
| `ionq_job_status` | Job status on IonQ |
| `ionq_job_results` | Measurement counts from a completed IonQ job |

### Observability plane — calibration history

`snapshot.py` runs every 2 hours via GitHub Actions (was 6h — upgraded for higher-resolution Rush Hour detection):

**Per-snapshot fields collected:**

| Field | Why it matters |
|-------|---------------|
| `avg_cx_error` | Primary gate quality metric |
| `avg_readout_error` | State-preparation and measurement overhead |
| `median_t1_us` | Median coherence time — median is robust to one outlier qubit skewing the average |
| `median_t2_us` | Dephasing time — degrades faster than T1 under environmental noise |
| `qubit_yield_fraction` | Fraction of qubits with usable T1/T2 — a device with 90/133 working qubits is different from 133/133 |
| `connectivity_density` | Edges / max-possible-edges — IBM heavy-hex ~0.015 vs IonQ all-to-all = 1.0 |
| `gate_set_size` | Number of native gates — affects transpilation depth |
| `max_circuit_depth` | Hard limit before decoherence kills the result |
| `native_2q_gate` | CX vs ECR vs ZZ — matters for circuit rewriting |
| `day_of_week` | 0=Monday … 6=Sunday — for weekly seasonality modeling |
| `hour_utc` | 0–23 — for time-of-day queue pattern detection |

**Job submissions table** — every call to `submit_job`, `run_grover`, or `run_vqe` writes a row to `job_submissions`:

```
job_id · provider · backend · tool · circuit_qubits · circuit_depth_raw
circuit_depth_transpiled · shots · agent_loop_iteration
was_preflight_checked · was_ai_corrected · day_of_week · hour_utc
```

This is the foundation for a first-of-its-kind agentic quantum workload study — once the dataset matures we can answer: which backends see the most AI-driven traffic, what is the typical transpilation expansion ratio by provider, how does shot count correlate with circuit depth.

**Storage:**
- `devices.db` (SQLite, local) — feeds `device_history`, `device_on_date`, drift alerts, `job_analytics`
- `data/snapshots.csv` (public, committed by CI) — permanent append-only calibration record

`report.py` generates a daily fleet summary at 8am — error trends, device rankings, alert history.

---

## Real experiments on quantum hardware

### Pascal's Triangle encoding — IBM ibm\_kingston

Binary-encode Pascal's Triangle values as quantum states, measure preparation fidelity on real superconducting hardware.

| Circuit | IBM ibm_kingston | IonQ simulator |
|---------|-----------------|----------------|
| C(6,3) = 20 → `\|10100⟩` | 942/1000 — **94.2%** | 1000/1000 — 100% |
| C(10,5) = 252 → `\|11111100⟩` | 903/1024 — **88.1%** | — |
| C(15,5) = 3003 → 12-bit state | 837/1024 — **81.7%** | — |

### Singmaster's Conjecture — Grover's search — IBM ibm\_marrakesh

Singmaster's Conjecture asks whether any integer appears 9+ times in Pascal's Triangle. We use Grover's search to amplify rows containing a target value above the noise floor.

| Experiment | Target | Marked rows | Raw depth | Transpiled depth | Amplification | Interpretation |
|------------|--------|-------------|-----------|-----------------|---------------|----------------|
| Phase 1 | 6 | rows 4, 6 | — | 611 | **4.11×** | Signal clear above noise |
| Phase 2 | 3003 | rows 14, 15 | 101 | 16,271 (161× expansion) | **1.04×** | Noise floor — hardware coherence limit reached |

Phase 2 is not a failure — it **brackets the coherence limit of ibm_marrakesh** between 611 and 16,271 transpiled gates. The 161× transpilation expansion ratio (101 raw gates → 16,271 after routing and basis translation) is itself a hardware characterization result: IBM heavy-hex topology forces gate decompositions that blow up shallow circuits. This number belongs in the paper.

Full code and raw results: [singmasters-conjecture](https://github.com/Lokesh-2025/singmasters-conjecture) (collaboration with [Jack Woehr](https://github.com/jwoehr)).

### VQE — H2 molecule ground state energy

Variational Quantum Eigensolver on a 2-qubit hardware-efficient ansatz (RY + CNOT). COBYLA optimizer.

| Backend | Iterations | VQE energy | Exact energy | Error |
|---------|------------|-----------|--------------|-------|
| Local simulator | 60 | −1.857275 Ha | −1.857275 Ha | **0.0 mHa** |
| IBM real hardware | pending | — | −1.857275 Ha | — |

Chemical accuracy threshold: < 1.6 mHa. Simulator achieves exact convergence. Real hardware run pending — IonQ trapped ions expected to outperform IBM superconducting due to lower gate error.

This is a stepping stone toward receptor-ligand binding energy simulations for drug discovery research.

---

## Test suite

```bash
python tests/test_all_tools.py
```

28 checks across all 27 tools. Read-only tools hit the real IBM and IonQ APIs. Write tools are tested against validation paths only — zero QPU credits spent.

```
Total: 28 checks | ✅ 28 passed | ❌ 0 failed | ⏭️  0 skipped
All tools operational!
```

---

## Project structure

```
quantum-hardware-mcp/
├── server.py                      # MCP server — all 27 IBM + IonQ + analytics tools
├── snapshot.py                    # Multi-provider calibration snapshot agent (every 2h)
├── report.py                      # Daily fleet report
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── agent/
│   ├── agent-server.js            # Dispatcher — control plane router
│   ├── chat.js                    # Terminal interface
│   ├── subagents/
│   │   ├── base-subagent.js       # Shared ReAct loop
│   │   ├── ibm-subagent.js        # IBM specialist
│   │   └── ionq-subagent.js       # IonQ specialist
├── experiments/
│   ├── singmasters_grover.py      # Grover's search — Phase 1 (target: 6)
│   ├── singmasters_3003.py        # Grover's search — Phase 2 (target: 3003, coherence limit)
│   └── vqe_h2.py                  # VQE for H2 molecule ground state
├── tests/
│   └── test_all_tools.py          # 28-check smoke test suite
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

Restart Claude Desktop. All 27 tools appear under the hammer icon.

---

## LLM provider support

| Provider | Cost | Env var |
|----------|------|---------|
| Anthropic Claude | Paid | `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` |
| Google Gemini | Free tier | `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` |
| OpenAI | Paid | `LLM_PROVIDER=openai` + `OPENAI_API_KEY` |
| Ollama | Free, local | `LLM_PROVIDER=ollama` + `OLLAMA_MODEL` |
| vLLM | Self-hosted | `LLM_PROVIDER=vllm` + `VLLM_BASE_URL` |

For sensitive research (pharmaceutical, unpublished academic work): run Ollama locally. The LLM never leaves your machine. The MCP server only contacts IBM/IonQ/Braket when you explicitly submit a job.

---

## Roadmap

- [x] IBM Quantum tools — device intelligence, job lifecycle, pre-flight, routing
- [x] IonQ support — devices, submit, status, results
- [x] AWS Braket integration — 10 backends added to snapshot pipeline
- [x] Multi-agent control plane — dispatcher + IBM/IonQ specialist subagents
- [x] Calibration drift alerts — CX error, readout error, T1, T2 (LAG window detection)
- [x] Reproducibility scoring — KL-divergence across N runs
- [x] Credit-aware routing — QPU cost estimation before submit
- [x] Pascal's Triangle on real IBM hardware — 94.2% fidelity at C(6,3)
- [x] Singmaster's Conjecture Phase 1 — Grover's search — 4.11× amplification (depth 611)
- [x] Singmaster's Conjecture Phase 2 — coherence limit bracketed at depth 16,271 on ibm_marrakesh
- [x] VQE for H2 — chemical accuracy (0.0 mHartree) on simulator
- [x] Multi-provider snapshot pipeline — IBM + IonQ + AWS Braket, every 2h (upgraded from 6h)
- [x] Extended calibration fields — 11 fields per snapshot including median T1/T2, qubit yield, connectivity density
- [x] Temporal indexing — day_of_week + hour_utc on every snapshot and job submission
- [x] Job submissions table — agentic workload tracking with transpilation expansion ratios
- [x] job_analytics tool — aggregate workload stats across all submitted jobs
- [x] Full smoke test suite — 28/28 passing, zero QPU credits spent
- [x] Listed on Glama, mcp.so, PulseMCP
- [ ] IonQ real hardware experiments (QPU access pending)
- [ ] VQE on real IBM hardware — H2 hardware result
- [ ] Circuit fingerprinting — cache results, skip resubmitting identical circuits
- [ ] Quantum Rush Hour detection — weekly queue seasonality via STL decomposition (needs 60+ days of data)
- [ ] Smart routing brain — cross-provider ML recommendations (needs 60+ days of data)
- [ ] Autonomous daily report agent
- [ ] Circuit image understanding — accept a circuit diagram image as input, interpret and submit

---

## License

MIT — see [LICENSE](LICENSE).
