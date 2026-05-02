# Starter Code App

A template for building AI Agents in Python.

## Structure

```
├── src/
│   ├── agent.py        # Main agent loop
│   ├── tools.py        # Tool definitions
│   └── config.py       # Configuration
├── scripts/
│   ├── setup_hooks.sh  # One-time hook installer
│   ├── log_hook.py     # AI tool hook handler
│   └── submit_log.py   # Submits logs on git push
├── requirements.txt
├── .env.example
├── AGENTS.md           # Rules for using AI coding agents
├── JOURNAL.md          # Weekly journal — product journey & learnings
└── WORKLOG.md          # Technical decisions, task assignments, brainstorming
```

## Getting Started

### 1. Clone and setup

```bash
git clone <repo-url>
cd <repo>

# Install git pre-push hook (required, run once)
bash scripts/setup_hooks.sh
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your `ANTHROPIC_API_KEY`. The `AI_LOG_*` variables are pre-filled.

### 3. Run

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# or: venv\Scripts\activate    # Windows

pip install -r requirements.txt
python -m src.agent
```

## Weekly Journal

Update **[JOURNAL.md](./JOURNAL.md)** at the end of every week to document your product-building journey:

- Features shipped
- AI tools used and how they helped
- Hardest problem of the week and how you solved it
- What you'd do differently
- Plan for next week

> JOURNAL.md **must be updated** before each PR. It is your learning record for the course.

## Worklog

Update **[WORKLOG.md](./WORKLOG.md)** whenever your team makes a technical decision or changes direction:

- **Technical decisions** — why did you choose this approach over alternatives?
- **Task assignments** — who does what, by when
- **Brainstorming** — options considered, pros/cons, conclusion
- **Important bugs** — root cause and fix

See each file for the format and examples.

## AI Logging

Prompts and tool calls are **automatically logged** when you use any supported AI tool (Claude Code, Cursor, Codex, Gemini, Copilot). No manual steps needed after running `setup_hooks.sh`.

See [AGENTS.md](./AGENTS.md) for details.

## Virtual Lab Pipeline

The MVP pipeline is now:

1. Teacher prompt
2. RAG retrieval from `science_db/`
3. `ScriptingAgent` creates the grounded experiment script
4. `SimulatorAgent` converts the script plus the same RAG context into runnable p5.js files

Ingest/update the multimodal RAG index first:

```bash
python -m src.agents.scripting.ingest_cli --all
```

Recommended per-agent model config in `.env`:

```bash
SCRIPTING_PROVIDER_BACKEND=openai
SCRIPTING_LLM_MODEL=gpt-4o
SIMULATOR_PROVIDER_BACKEND=openai
SIMULATOR_LLM_MODEL=gpt-5.5
```

Generate a complete simulation:

```bash
python -m src.pipeline.generate_experiment_cli "So sánh nhiệt độ sôi và trạng thái phân tử của oxygen, ethanol, nước, thủy ngân và sắt" --out-dir generated/phase_change --simulator-model gpt-5.5
```

Run the local Streamlit UI:

```bash
streamlit run demo_virtual_lab.py
```

The UI lets a teacher enter one prompt, then automatically runs RAG, scripting,
simulator generation, writes `index.html` and `sketch.js`, creates a standalone
`experiment_standalone.html`, and previews the experiment in the browser.

Run only the simulator stage from an existing script:

```bash
python -m src.agents.simulator.cli --script script.md --out-dir generated/from_script --model gpt-5.5
```

The simulator output contract requires `index.html` and `sketch.js`; the parser validates that the generated sketch defines `setup()`, `draw()`, creates a canvas, and that HTML loads p5.js plus `sketch.js`.
