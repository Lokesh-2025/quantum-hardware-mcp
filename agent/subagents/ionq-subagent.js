/**
 * IonQ quantum hardware subagent.
 * Handles IonQ-specific tools (trapped-ion hardware).
 * Spawned by the dispatcher (agent-server.js) via stdio.
 *
 * Usage (by dispatcher only — not called directly):
 *   echo '{"question":"...","history":[]}' | node ionq-subagent.js
 */

const { runSubagent } = require('./base-subagent');

// Only IonQ tools
const toolFilter = tool => tool.name.startsWith('ionq_');

const systemPrompt = `You are an IonQ quantum hardware specialist agent.
You have access to IonQ tools for listing trapped-ion devices, submitting circuits,
checking job status, and retrieving results.
IonQ uses trapped-ion qubits which have different characteristics than IBM's
superconducting qubits — higher fidelity but fewer qubits currently.
Answer only IonQ quantum hardware questions. Be concise and precise.`;

// No Qiskit model for IonQ — Qiskit model is IBM-focused
runSubagent(toolFilter, systemPrompt, qiskitEnabled = false).catch(err => {
    process.stdout.write(JSON.stringify({ answer: `IonQ subagent fatal: ${err.message}`, metadata: {} }));
    process.exit(1);
});
