#!/usr/bin/env node

const readline = require('readline');
const fs = require('fs').promises;
const path = require('path');
require('dotenv').config();

const AGENT_URL = process.env.QUANTUM_AGENT_URL || 'http://localhost:3021';
const chatHistory = [];
let noLocalMode = false; // toggled by /nolocal command

// Create readline interface for user input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '\n⚛️  Quantum Query> '
});

/**
 * Processes file references in the input text and replaces them with file contents
 * @param {string} input - User input that may contain @/path/to/file references
 * @returns {Promise<string>} Input with file contents inserted
 */
async function processFileReferences(input) {
    // Match @/path/to/file or @./relative/path patterns
    const fileRefPattern = /@(\.?\/[^\s]+)/g;
    const matches = [...input.matchAll(fileRefPattern)];
    
    if (matches.length === 0) {
        return input;
    }
    
    let processedInput = input;
    const fileContents = [];
    
    for (const match of matches) {
        const filePath = match[1];
        const fullMatch = match[0];
        
        try {
            // Resolve relative paths from current working directory
            const resolvedPath = path.resolve(process.cwd(), filePath);
            // Prevent reading files outside the working directory
            if (!resolvedPath.startsWith(process.cwd())) {
                throw new Error('Access denied: path is outside the working directory');
            }
            const content = await fs.readFile(resolvedPath, 'utf-8');
            
            // Format the file content with clear delimiters
            const formattedContent = `\n\n[File: ${filePath}]\n\`\`\`\n${content}\n\`\`\`\n`;
            fileContents.push({ match: fullMatch, content: formattedContent, path: filePath });
            
            console.log(`📎 Attached file: ${filePath} (${content.length} bytes)`);
        } catch (error) {
            const errorMsg = `\n[Error reading file ${filePath}: ${error.message}]\n`;
            fileContents.push({ match: fullMatch, content: errorMsg, path: filePath });
            console.log(`⚠️  Warning: Could not read file ${filePath}: ${error.message}`);
        }
    }
    
    // Replace all file references with their contents
    for (const { match, content } of fileContents) {
        processedInput = processedInput.replace(match, content);
    }
    
    return processedInput;
}

/**
 * Sends a chat message to the Quantum Hardware agent server
 * @param {string} question - The user's question
 * @param {Object} resumeContext - Optional context for resuming from a query
 * @returns {Promise<string>} The agent's answer
 */
async function chat(question, resumeContext = null) {
    try {
        const requestBody = {
            question,
            history: chatHistory,
            noLocal: noLocalMode
        };
        
        // Add resume context if continuing from a query
        if (resumeContext) {
            requestBody.resumeContext = resumeContext;
        }
        
        const response = await fetch(`${AGENT_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();
        
        // Handle query response
        if (data.status === 'query') {
            console.log('\n🤔 Agent Query:', data.query);
            
            // Show suggestions if available
            if (data.suggestions && data.suggestions.length > 0) {
                console.log('\n💡 Suggestions:');
                data.suggestions.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
            }
            
            // Get user's answer using a promise-based approach
            const userAnswer = await new Promise((resolve) => {
                rl.question('\n📝 Your answer> ', (answer) => {
                    resolve(answer.trim());
                });
            });
            
            // Resume with user's answer
            return await chat(userAnswer, {
                sessionId: data.context.sessionId,
                query: data.query
            });
        }
        
        // Handle final answer (both new format and backward compatible)
        const answer = data.answer || (data.status === 'complete' ? data.answer : 'No response');
        
        // Add to chat history in standardized format
        chatHistory.push({
            role: 'user',
            content: question
        });
        chatHistory.push({
            role: 'assistant',
            content: answer
        });

        return answer;
    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            return `Error: Cannot connect to agent server at ${AGENT_URL}. Make sure the agent server is running.`;
        }
        return `Error: ${error.message}`;
    }
}

/**
 * Standardize history entry to common format
 * @param {Object} entry - History entry
 * @returns {Object} Standardized entry
 */
function standardizeHistoryEntry(entry) {
    // Already standardized
    if (entry.role && entry.content && !entry.parts) {
        return entry;
    }
    
    // Gemini format with parts
    if (entry.role && entry.parts) {
        return {
            role: entry.role === 'model' ? 'assistant' : entry.role,
            content: entry.parts[0]?.text || ''
        };
    }
    
    return entry;
}

/**
 * Saves the current chat history to a Markdown file
 * @param {string} filePath - Destination file path (relative or absolute)
 */
async function saveChatToFile(filePath) {
    const resolvedPath = path.resolve(process.cwd(), filePath);
    const lines = chatHistory.map(entry => {
        const heading = entry.role === 'user' ? '## User' : '## Assistant';
        return `${heading}\n\n${entry.content}`;
    });
    const markdown = lines.join('\n\n---\n\n') + '\n';
    await fs.writeFile(resolvedPath, markdown, 'utf-8');
    console.log(`\n✅ Chat saved to: ${resolvedPath}\n`);
}

/**
 * Polls a quantum job at regular intervals until complete or aborted by keypress.
 * Implements Jack's spec: chat tells the agent to check the job; agent uses
 * job_status / ionq_job_status tools and returns results when done.
 *
 * @param {string} provider     - 'IBM' or 'IonQ'
 * @param {string} jobId        - job ID returned by submit_job / ionq_submit_job
 * @param {number} intervalSecs - seconds between each poll (default 10)
 */
async function pollJob(provider, jobId, intervalSecs) {
    let providerName, statusTool, resultTool;
    switch (provider.toUpperCase()) {
        case 'IONQ':
            providerName = 'IonQ';
            statusTool   = 'ionq_job_status';
            resultTool   = 'ionq_job_results';
            break;
        case 'IBM':
        default:
            providerName = 'IBM';
            statusTool   = 'job_status';
            resultTool   = 'job_results';
            break;
    }

    console.log(`\n🔄 Polling ${providerName} job ${jobId} every ${intervalSecs}s`);
    console.log('   Press any key to abort.\n');

    let aborted = false;
    let iteration = 0;

    // Pause readline and enable raw keypress so a single keypress aborts polling
    rl.pause();
    try { process.stdin.setRawMode(true); process.stdin.resume(); } catch (_) {}

    // Prompt sent to the agent on each poll iteration
    const pollPrompt =
        `Use the ${statusTool} tool to check the status of ${providerName} job id ${jobId}. ` +
        `If the status is DONE, also call ${resultTool} and return the COMPLETE raw measurement counts plus your interpretation. ` +
        `If the job is not done yet, your final answer is the current job status only — do not guess results.`;

    while (!aborted) {
        iteration++;
        console.log(`⏱  Poll #${iteration} — checking ${providerName} job ${jobId}...`);

        const answer = await chat(pollPrompt);
        console.log('🤖', answer, '\n');

        // Auto-stop when the agent confirms the job is finished
        const lower = answer.toLowerCase();
        const isDone = (lower.includes('done') || lower.includes('counts') || lower.includes('measurement'))
                    && !lower.includes('queued')
                    && !lower.includes('running');

        if (isDone) {
            console.log('✅ Job complete — polling stopped.\n');
            break;
        }

        // Wait for the interval or abort on any keypress
        const wasAborted = await new Promise(resolve => {
            let settled = false;
            const finish = (val) => { if (!settled) { settled = true; resolve(val); } };

            const timer = setTimeout(() => finish(false), intervalSecs * 1000);
            process.stdin.once('data', () => { clearTimeout(timer); finish(true); });
        });

        if (wasAborted) {
            console.log('\n⛔ Polling aborted.\n');
            aborted = true;
        }
    }

    try { process.stdin.setRawMode(false); } catch (_) {}
    rl.resume();
}

/**
 * Displays welcome message and instructions
 */
function displayWelcome() {
    console.log('\n╔══════════════════════════════════════════════════════════╗');
    console.log('║       Quantum Hardware Agent Chat Interface              ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log('\nConnected to:', AGENT_URL);
    console.log('\nCommands:');
    console.log('  - Type your quantum hardware-related questions');
    console.log('  - Use @/path/to/file to insert file contents inline');
    console.log('  - Type "/exit" or "/quit" to end the session');
    console.log('  - Type "/clear" to clear chat history');
    console.log('  - Type "/save @/path/to/file" to save chat history to a Markdown file');
    console.log('  - Type "/poll <job_id> [interval_secs]" to poll IBM job until done');
    console.log('  - Type "/poll IBM|IonQ <job_id> [interval_secs]" to specify provider');
    console.log('  - Type "/nolocal" to toggle bypassing the local Qiskit code model');
    console.log('  - Type "/help" to see this message again');
    console.log('\nExamples:');
    console.log('  - "List available quantum backends"');
    console.log('  - "Get status of quantum device ibm_brisbane"');
    console.log('  - "Show queue information for backend"');
    console.log('  - "What quantum computers are available?"');
    console.log('  - "Check calibration data for a device"');
    console.log('  - "Get properties of quantum backend"');
}

/**
 * Processes user commands
 * @param {string} input - User input
 * @returns {boolean} True if chat should continue, false to exit
 */
async function processInput(input) {
    const trimmed = input.trim().toLowerCase();

    // /save @/path/to/file  — save chat history to Markdown (Jack's feature, PR #14)
    if (input.trim().startsWith('/save')) {
        if (chatHistory.length === 0) {
            console.log('\n⚠️  Nothing to save — chat history is empty.\n');
            return true;
        }

        const parts = input.trim().split(/\s+/);
        let filePath = parts[1] && parts[1].startsWith('@')
            ? parts[1].slice(1).trim()
            : null;

        if (!filePath) {
            filePath = await new Promise((resolve) => {
                rl.question('\n💾 Save to file> ', (answer) => {
                    resolve(answer.trim());
                });
            });
            if (!filePath) {
                console.log('\nSave cancelled.\n');
                return true;
            }
        }

        try {
            await fs.stat(path.resolve(process.cwd(), filePath));
            const confirm = await new Promise((resolve) => {
                rl.question('\n⚠️  File exists. Overwrite? (y/N)> ', (answer) => {
                    resolve(answer.trim());
                });
            });
            if (confirm !== 'y' && confirm !== 'Y') {
                console.log('\nSave cancelled.\n');
                return true;
            }
        } catch (_) {
            // File does not exist — proceed
        }

        try {
            await saveChatToFile(filePath);
        } catch (error) {
            console.log(`\n❌ Could not save chat: ${error.message}\n`);
        }
        return true;
    }

    // /poll <job_id> [interval]
    // /poll IBM <job_id> [interval]
    // /poll IonQ <job_id> [interval]
    if (input.trim().toLowerCase().startsWith('/poll')) {
        const tokens = input.trim().split(/\s+/).slice(1); // everything after /poll

        // Detect optional provider prefix
        let provider = 'IBM';
        if (tokens.length > 0 && ['ibm', 'ionq'].includes(tokens[0].toLowerCase())) {
            provider = tokens.shift(); // consume provider token
        }

        const jobId = tokens[0];
        const intervalSecs = tokens[1] && !isNaN(parseInt(tokens[1])) ? parseInt(tokens[1]) : 10;

        if (!jobId) {
            console.log('\n⚠️  Usage: /poll <job_id> [interval_secs]');
            console.log('         /poll IBM <job_id> [interval_secs]');
            console.log('         /poll IonQ <job_id> [interval_secs]\n');
            return true;
        }

        await pollJob(provider, jobId, intervalSecs);
        return true;
    }

    // /nolocal — toggle bypass of local Qiskit code model
    if (trimmed === '/nolocal') {
        noLocalMode = !noLocalMode;
        console.log(`\n${noLocalMode ? '⚡ Local Qiskit model BYPASSED' : '🧠 Local Qiskit model ENABLED'} — toggled for this session.\n`);
        return true;
    }

    switch (trimmed) {
        case '/exit':
        case '/quit':
            console.log('\n👋 Goodbye!\n');
            return false;

        case '/clear':
            chatHistory.length = 0;
            console.log('\n✓ Chat history cleared.\n');
            return true;

        case '/help':
            displayWelcome();
            return true;

        case '':
            return true;

        default:
            console.log('\n⏳ Processing...\n');
            // Process file references before sending to chat
            const processedInput = await processFileReferences(input);
            const answer = await chat(processedInput);
            console.log('🤖 Answer:', answer);
            return true;
    }
}

/**
 * Main chat loop
 */
async function main() {
    displayWelcome();
    
    rl.prompt();

    rl.on('line', async (line) => {
        const shouldContinue = await processInput(line);
        
        if (!shouldContinue) {
            rl.close();
            process.exit(0);
        }
        
        rl.prompt();
    });

    rl.on('close', () => {
        console.log('\n👋 Chat session ended.\n');
        process.exit(0);
    });

    // Handle Ctrl+C gracefully
    process.on('SIGINT', () => {
        console.log('\n\n👋 Chat session interrupted.\n');
        process.exit(0);
    });
}

// Start the chat
main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});

// Made with Bob
