/**
 * IBM quantum hardware subagent.
 * Handles all IBM-specific tools and Qiskit code questions.
 * Spawned by the dispatcher (agent-server.js) via stdio.
 *
 * Usage (by dispatcher only — not called directly):
 *   echo '{"question":"...","history":[]}' | node ibm-subagent.js
 */

const { runSubagent } = require('./base-subagent');

// Only IBM tools (everything except ionq_*)
const toolFilter = tool => !tool.name.startsWith('ionq_');

const systemPrompt = `You are an IBM quantum hardware specialist agent.
You have access to IBM Quantum tools for listing devices, submitting circuits,
checking job status, comparing backends, and analyzing circuits.
You also have a Qiskit specialist model for writing and debugging quantum code.
Answer only IBM quantum hardware questions. Be concise and precise.`;

// Enable Qiskit model routing — it's IBM's own model tuned for Qiskit
runSubagent(toolFilter, systemPrompt, qiskitEnabled = true).catch(err => {
    process.stdout.write(JSON.stringify({ answer: `IBM subagent fatal: ${err.message}`, metadata: {} }));
    process.exit(1);
});
