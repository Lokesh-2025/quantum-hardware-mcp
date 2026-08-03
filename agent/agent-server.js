/**
 * Dispatcher agent — routes user questions to IBM or IonQ subagents via stdio.
 *
 * Architecture (Jack's design):
 *   chat.js → HTTP → dispatcher (this file)
 *                       ├── child_process.spawn → ibm-subagent.js  (stdio)
 *                       └── child_process.spawn → ionq-subagent.js (stdio)
 *
 * The dispatcher asks the LLM to classify the question (IBM / IonQ / both),
 * then spawns the appropriate subagent, passes the question via stdin,
 * and returns the subagent's stdout answer back to the user.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const ProviderFactory = require('./shared/providers/provider-factory');
const ProviderConfig = require('./shared/config/provider-config');
const { requestLoggerMiddleware, createLogger } = require('./lib/request-logger');
const { getLLMLimiter } = require('./shared/concurrency/limiters');
require('dotenv').config();

// Validate config at startup
let providerName, providerConfig;
try {
    const config = ProviderConfig.validate();
    providerName = config.provider;
    providerConfig = config.config;
} catch (error) {
    console.error('Configuration Error:', error.message);
    process.exit(1);
}

const app = express();
const port = process.env.PORT || 3021;

app.use(express.json());
app.use(cors());
app.use(requestLoggerMiddleware);

let llmProvider = null;

// Subagent script paths
const SUBAGENTS = {
    IBM:  path.join(__dirname, 'subagents/ibm-subagent.js'),
    IonQ: path.join(__dirname, 'subagents/ionq-subagent.js'),
};

/**
 * Ask the LLM to classify which provider the question targets.
 * Returns 'IBM', 'IonQ', or 'IBM' as default.
 */
async function classifyProvider(question, logger) {
    const chat = await llmProvider.createChat([]);
    const prompt = `You are a quantum hardware router. Classify which provider this question targets.
Available providers: IBM, IonQ

Question: "${question}"

Reply with ONLY one of these JSON objects — nothing else:
{ "provider": "IBM" }
{ "provider": "IonQ" }

If unsure or the question mentions both, default to IBM.`;

    const sendResult = await llmProvider.sendMessage(chat, prompt);
    const text = await llmProvider.extractTextResponse(sendResult.response || sendResult);

    try {
        const start = text.indexOf('{');
        const end   = text.lastIndexOf('}');
        const parsed = JSON.parse(text.substring(start, end + 1));
        const provider = parsed.provider === 'IonQ' ? 'IonQ' : 'IBM';
        logger.log(`[Dispatcher] Routed to: ${provider}`);
        return provider;
    } catch {
        logger.log('[Dispatcher] Classification failed, defaulting to IBM');
        return 'IBM';
    }
}

/**
 * Spawn a subagent process, send it the question via stdin,
 * and return its answer from stdout.
 *
 * Protocol: subagent writes __RESULT__\n<json> to stdout so we can
 * extract the result even when the LLM response contains curly braces
 * inside code blocks (which broke the old indexOf/lastIndexOf approach).
 */
function callSubagent(provider, question, history, logger, noLocal = false) {
    const timeoutMs = parseInt(process.env.SUBAGENT_TIMEOUT_MS || '120000');

    return new Promise((resolve, reject) => {
        const scriptPath = SUBAGENTS[provider];
        logger.log(`[Dispatcher] Spawning ${provider} subagent: ${scriptPath}${noLocal ? ' (local LLM bypassed)' : ''}`);

        const child = spawn('node', [scriptPath], {
            env: { ...process.env },
            stdio: ['pipe', 'pipe', 'pipe'],
        });

        // Kill the subagent if it doesn't finish within the timeout
        const timer = setTimeout(() => {
            child.kill('SIGTERM');
            reject(new Error(`${provider} subagent timed out after ${timeoutMs}ms`));
        }, timeoutMs);

        // Send question + history (+ noLocal flag) to subagent stdin
        child.stdin.write(JSON.stringify({ question, history: history || [], noLocal }));
        child.stdin.end();

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', chunk => { stdout += chunk; });
        child.stderr.on('data', chunk => { stderr += chunk; });

        child.on('close', code => {
            clearTimeout(timer);
            if (stderr) logger.log(`[${provider} subagent stderr] ${stderr.trim()}`);

            // Primary: look for sentinel line written by subagent
            const sentinelIdx = stdout.indexOf('__RESULT__\n');
            if (sentinelIdx !== -1) {
                try {
                    resolve(JSON.parse(stdout.slice(sentinelIdx + '__RESULT__\n'.length).trim()));
                    return;
                } catch { /* fall through to legacy extraction */ }
            }

            // Fallback: extract first complete JSON object (legacy subagents)
            const start = stdout.indexOf('{');
            const end   = stdout.lastIndexOf('}');
            if (start !== -1 && end > start) {
                try {
                    resolve(JSON.parse(stdout.substring(start, end + 1)));
                    return;
                } catch { /* fall through to error */ }
            }
            reject(new Error(`${provider} subagent returned invalid JSON (exit ${code}): ${stdout.substring(0, 300)}`));
        });

        child.on('error', err => { clearTimeout(timer); reject(new Error(`Failed to spawn ${provider} subagent: ${err.message}`)); });
    });
}

// --- Chat Endpoint ---
app.post('/chat', async (req, res) => {
    // Guard: provider not ready yet (startup race)
    if (!llmProvider) {
        return res.status(503).json({ status: 'error', answer: 'Server is still starting up — try again in a moment.' });
    }

    // Input validation
    const { question, history, noLocal } = req.body;
    if (!question || typeof question !== 'string') {
        return res.status(400).json({ status: 'error', answer: 'No question provided.' });
    }
    if (question.length > 32768) {
        return res.status(400).json({ status: 'error', answer: 'Question too long (max 32 KB).' });
    }
    if (history !== undefined && !Array.isArray(history)) {
        return res.status(400).json({ status: 'error', answer: 'history must be an array.' });
    }

    try {
        req.logger.log(`[Chat] Question received (${question.length} chars)`);

        // 1. Classify which provider to route to
        const llmLimiter = await getLLMLimiter(providerName);
        const provider = await llmLimiter(() => classifyProvider(question, req.logger));

        // 2. Spawn the appropriate subagent and get the answer
        const result = await callSubagent(provider, question, history, req.logger, !!noLocal);

        return res.json({
            status: 'complete',
            answer: result.answer,
            metadata: { provider, ...result.metadata }
        });

    } catch (error) {
        req.logger.error('Error in dispatcher:', error);
        res.status(500).json({
            status: 'error',
            answer: 'Sorry, there was an error processing your request.'
        });
    }
});

// Health check
app.get('/health', (req, res) => res.json({ status: 'ok', subagents: Object.keys(SUBAGENTS) }));

// --- Server Startup ---
app.listen(port, async () => {
    const startupLogger = createLogger('startup');
    startupLogger.log('╔══════════════════════════════════════════════════════════╗');
    startupLogger.log('║   Quantum Hardware MCP Dispatcher Agent                  ║');
    startupLogger.log('╚══════════════════════════════════════════════════════════╝');
    startupLogger.log(`\nServer starting at http://localhost:${port}`);
    startupLogger.log(`LLM Provider: ${providerName}`);
    startupLogger.log(`Subagents: ${Object.keys(SUBAGENTS).join(', ')}`);

    try {
        llmProvider = await ProviderFactory.createProvider(providerName, providerConfig);
        const metadata = llmProvider.getMetadata();
        startupLogger.log(`Model: ${metadata.model || 'unknown'}`);
        startupLogger.log(`\n✓ Dispatcher ready at http://localhost:${port}\n`);
    } catch (error) {
        startupLogger.error('\nFATAL: Failed to start dispatcher:', error.message);
        process.exit(1);
    }
});

process.on('SIGINT',  () => { console.log('\n👋 Dispatcher shutting down.'); process.exit(0); });
process.on('SIGTERM', () => { console.log('\n👋 Dispatcher shutting down.'); process.exit(0); });

// Made with Bob
