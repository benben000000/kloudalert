# AGENTS.md - Repository Rule & System Governance

> [!IMPORTANT]
> **KEY SYSTEM RULE FOR ALL AI MODELS AND SESSIONS**
> This file and `.agent/state.json` govern all automated actions in this repository. Every AI assistant, model (Gemini, Claude, GPT, or others), agent, or session entering this workspace MUST read and strictly obey these rules.

---

## Rule 1: Automatic State Resumption & Self-Execution
1. At the start of **EVERY** new session, conversation, or model switch:
   - Read `AGENTS.md` (this file) and `file:///d:/lnn-mobile-weather/.agent/state.json`.
   - Determine the active task status, last completed step, and next pending action.
   - **Do NOT wait for user prompting** to resume work. Immediately execute the next pending action in `.agent/state.json`.
2. Update `.agent/state.json` after completing or verifying each action to ensure continuous state persistence across sessions.

---

## Rule 2: Zero False Positives & Empirical Verification
1. **NO FAKE PASSES**: Never report or assume a command, build, test, or lint passed without actual execution and output verification.
2. **MANDATORY LOGGING**: Every terminal command, test execution, build script, SAST security scan, or linter invocation MUST route through `.agent/verifier/runner.py` or write raw stdout/stderr/exit codes into `file:///d:/lnn-mobile-weather/.agent/logs/`.
3. **LOG INSPECTION**: Always inspect and verify log output files in `.agent/logs/` before marking a task step as completed in `state.json`.

---

## Rule 3: Repository Integration & Multi-Repo Scope
1. This workspace manages four core repositories located under `repos/`:
   - `repos/improve` (`https://github.com/shadcn/improve.git`)
   - `repos/ponytail` (`https://github.com/DietrichGebert/ponytail.git`)
   - `repos/obsidian-mind` (`https://github.com/breferrari/obsidian-mind.git`)
   - `repos/openscience` (`https://github.com/synthetic-sciences/openscience.git`)
2. All automated orchestrations, code enhancements, security scans, and accessibility checks must operate seamlessly across these integrated components.

---

## Rule 4: Full-Stack, Security, Privacy, & Accessibility Standards
1. **Full-Stack Development**: All code modifications must pass linter checks, syntax verification, and unit test runners (`.agent/skills/fullstack.py`).
2. **Security & Privacy Scanning**: Perform SAST analysis, secret scanning, and dependency vulnerability checks (`.agent/skills/security.py`).
3. **Accessibility & Quality**: Enforce HTML/DOM accessibility rules and quality validation (`.agent/skills/accessibility.py`).

---

## State & Verification Pointers
- Machine State: `file:///d:/lnn-mobile-weather/.agent/state.json`
- Config & Tools: `file:///d:/lnn-mobile-weather/.agent/config.json`
- Verification Engine: `file:///d:/lnn-mobile-weather/.agent/verifier/runner.py`
- Execution Logs: `file:///d:/lnn-mobile-weather/.agent/logs/`
