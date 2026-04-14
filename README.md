# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it supports Claude and Ollama as LLM backends and provides a Chainlit web GUI as conversational interface.

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; the pipeline builds, submits, and QA-checks the workflow automatically.
- **Image & video generation** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing** — reference-based editing, inpainting, upscaling, and more.
- **Multi-stage pipeline** — Triage routes requests; a lightweight Researcher (Ollama by default) resolves templates, model paths, and sampler settings; the Brain (Claude by default) assembles the workflow, executes it, and runs vision QA.
- **Persistent chat history** — Chainlit SQLAlchemy datalayer stores conversation threads and messages in PostgreSQL; uploaded files are persisted to a local MinIO S3 bucket.
- **FAISS memory** — long-term memory via mem0 + local Ollama embeddings (`nomic-embed-text`).
- **Hugging Face model management** — search, check local availability, and download models on demand.
- **Chainlit web GUI** — interact via a browser-based chat UI; images and videos are delivered inline.
- **Multiple LLM backends** — Claude and Ollama, configurable per pipeline stage.
- **50+ workflow templates** — from Comfy-Org, loaded and patched automatically.
- **Skills system** — drop shell/Python scripts into `skills/` and they become agent-callable tools.
- **ComfyUI extension** — a companion custom node ([agentY-comfyui-extension](https://github.com/szprivate/agentY-comfyui-extension)) lets you send images directly from ComfyUI to agentY and receive responses in real time.

---

## Requirements

- **Python 3.11+**
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- An **Anthropic API key** (for Claude) _and/or_ a local **Ollama** installation
- **PostgreSQL** — required for the Chainlit datalayer (chat history persistence)
- **Docker** — used to run PostgreSQL and MinIO via `docker-compose.yml` (or point to existing instances)
- (Optional) Slack app credentials for Slack integration

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/szprivate/agentY.git
cd agentY
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up PostgreSQL (Chainlit datalayer)

Chainlit uses a SQLAlchemy datalayer to persist conversation threads, messages, and file attachments. A running PostgreSQL instance is required.

**Quick start with Docker:**

```bash
docker run -d \
  --name agenty-db \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  postgres:16
```

Or point `DATABASE_URL` at any existing PostgreSQL instance you already have running.

Chainlit will create its tables automatically on first launch when `DATABASE_URL` is set.

### 4. Set up MinIO (file storage)

Uploaded files (images attached in chat) are persisted to **MinIO**, an S3-compatible object store that runs locally in Docker. The `docker-compose.yml` ships ready to use.

**Start MinIO with Docker Compose:**

```bash
docker compose up -d minio createbuckets
```

This starts two short-lived services:

| Service | Purpose |
|---|---|
| `minio` | S3-compatible API on port `9000`, web console on port `9001` |
| `createbuckets` | One-shot init container that creates the `chainlit` bucket on first run |

**Web console:** open [http://localhost:9001](http://localhost:9001) and log in with `minioadmin` / `minioadmin`.

Default credentials (can be overridden with env vars — see step 5):

| Env var | Default |
|---|---|
| `MINIO_ENDPOINT_URL` | `http://localhost:9000` |
| `MINIO_ACCESS_KEY` | `minioadmin` |
| `MINIO_SECRET_KEY` | `minioadmin` |
| `MINIO_BUCKET` | `chainlit` |

> **`run_agent.ps1` handles this automatically.** When Docker is available the script runs `docker compose up -d minio createbuckets` before launching Chainlit, so you normally don't need to start MinIO manually.

---

### 5. Configure secrets

Copy the example env file and fill in your values:

```bash
cp .env_example .env
```

Edit `.env`:

```dotenv
# Hugging Face token (for gated models)
HF_TOKEN=hf_...

# LLM backends
ANTHROPIC_API_KEY=sk-ant-...

# ComfyUI (leave blank if no API key is set)
COMFYUI_API_KEY=comfyui-...

# Chainlit datalayer — PostgreSQL connection string
DATABASE_URL=postgresql://root:root@localhost:5432/postgres

# Chainlit web UI credentials
CHAINLIT_USERNAME=yourname
CHAINLIT_PASSWORD=yourpassword

# Chainlit auth secret — generate once with: chainlit create-secret
CHAINLIT_AUTH_SECRET="your-generated-secret"

# MinIO file storage (optional — defaults match docker-compose.yml)
# MINIO_ENDPOINT_URL=http://localhost:9000
# MINIO_ACCESS_KEY=minioadmin
# MINIO_SECRET_KEY=minioadmin
# MINIO_BUCKET=chainlit

# Slack integration (optional)
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=
```

Generate the Chainlit auth secret (run once):

```bash
chainlit create-secret
```

Paste the output into `CHAINLIT_AUTH_SECRET` in your `.env`.

### 6. Configure defaults

Edit `config/settings.json` to point to your ComfyUI instance and set default LLMs:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",

  "llm": {
    "pipeline": {
      "researcher":            "ollama,qwen3.5:9b",
      "brain":                 "claude,claude-haiku-4-5",
      "triage":                "ollama,qwen3.5:9b",
      "planner":               "ollama,qwen3.5:9b",
      "learnings":             "ollama,qwen3.5:9b",
      "llm_functions":         "qwen3.5:9b",
      "executor_vision_model": "qwen3.5:9b"
    }
  }
}
```

Each `"pipeline"` value uses the format `"provider,model"` (e.g. `"ollama,qwen3.5:9b"` or `"claude,claude-haiku-4-5"`). `llm_functions` and `executor_vision_model` accept a bare Ollama model name.

---

## Usage

### PowerShell (Windows)

The script creates the virtual environment and installs dependencies automatically on first run.

```powershell
# Default — uses LLMs from settings.json, opens GUI on http://localhost:8000
.\run_agent.ps1

# Custom port
.\run_agent.ps1 -Port 8080

# Auto-reload on source changes (dev mode)
.\run_agent.ps1 -Watch

# Override the Researcher LLM
.\run_agent.ps1 -LlmResearcher "ollama,qwen3-coder:32b"

# Override the Brain LLM
.\run_agent.ps1 -LlmBrain "claude,claude-sonnet-4-5"

# Show help
.\run_agent.ps1 -Help
```

### Python (any OS)

```bash
# Default
python -m src.main

# Override LLMs via CLI flags
python -m src.main --researcher-llm ollama --researcher-ollama-model qwen3-coder:32b
python -m src.main --brain-llm claude --brain-anthropic-model claude-sonnet-4-5

# Skip the Brain stage
python -m src.main --skip-brain
```

Type messages at the `You:` prompt. Type `quit` or `exit` to stop.

---

## Architecture

### Multi-stage pipeline

```
User request
    │
    ▼
┌────────┐   intent   ┌────────────┐   BrainBriefing JSON   ┌───────┐   workflow   ┌─────────┐
│ Triage ├───────────►│ Researcher ├────────────────────────►│ Brain ├─────────────►│ ComfyUI │
└────────┘            └────────────┘                         └───┬───┘              └────┬────┘
 Ollama (default)      Ollama (default)                          │ Claude (default)      │
 • classify intent     • resolve template                        │ • assemble workflow   │
 • route to handler    • resolve model paths                     │ • patch & validate    │
                       • resolve sampler settings                │ • submit & poll       │
                       • produce BrainBriefing                   │ • vision QA           │
                                                                 │ • deliver via Chainlit│
                                                                 ▼                      ▼
                                                              output_images/    Chainlit GUI
                                                              (+ PostgreSQL history)
```

1. **Triage** classifies the incoming message and routes it — to the image generation pipeline, a direct skill, or a general assistant response.
2. **Researcher** receives the user request and produces a validated **BrainBriefing** JSON (template, input images, model paths, prompts, resolution).
3. **Brain** receives the BrainBriefing, loads the selected workflow template, patches node values, submits the prompt to ComfyUI, waits for completion, runs vision QA on the output, and delivers the result.

The `--skip-brain` flag returns the Researcher's BrainBriefing directly, useful for debugging or inspection.

### Memory

Long-term memory is stored in a local FAISS index (`memory/agenty_memory.faiss`) via **mem0** with **nomic-embed-text** embeddings served by Ollama. The agent automatically saves and retrieves relevant context across sessions.

### LLM configuration priority

Each value is resolved in order — first match wins:

1. CLI flag (`-LlmResearcher` / `--researcher-llm`)
2. Environment variable
3. `config/settings.json`
4. Hard-coded default

---

## Chainlit Web GUI

The web GUI is the primary interface. Launch it with:

```powershell
.\run_agent.ps1
```

Then open [http://localhost:8000](http://localhost:8000) in your browser. Log in with the `CHAINLIT_USERNAME` / `CHAINLIT_PASSWORD` values from `.env` (defaults: `yourname` / `yourpassword`).

You can attach images directly in the chat — they are forwarded to ComfyUI as input assets.

### Datalayer (chat persistence)

When `DATABASE_URL` is set, Chainlit persists all conversation threads and messages to PostgreSQL via `SQLAlchemyDataLayer`. Uploaded files (images, etc.) are stored in **MinIO** via an S3-compatible storage client — both services are started automatically by `run_agent.ps1` when Docker is available.

Previous conversations are accessible from the sidebar. Without `DATABASE_URL` the app still works but history is not retained between restarts. Without MinIO, file attachments are not persisted across sessions.

---

## Adding Custom Workflow Templates

Use the helper scripts to register a workflow JSON and make it available to the agent:

```powershell
# Register a new workflow template
.\add_workflow.ps1 path\to\your_workflow_api.json

# Remove a registered template
.\remove_workflow.ps1 your_workflow_api
```

`add_workflow.ps1` parses the workflow, extracts node metadata, and adds an entry to `config/workflow_templates.json`. Custom templates live in `comfyui_workflow_templates_custom/templates/`.

---

## Project Structure

```
agentY/
├── src/
│   ├── main.py                 Entry point and CLI
│   ├── agent.py                Agent factories, LLM config, system prompt loading
│   ├── chainlit_app.py         Chainlit GUI entry point; datalayer init
│   ├── executor.py             Skill executor
│   ├── pipeline.py             Triage → Researcher → Brain pipeline and BrainBriefing schema
│   ├── tools/
│   │   ├── agent_control.py    Restart/stop commands intercepted by the agent loop
│   │   ├── comfyui.py          Workflow template loading/patching, node inspection, prompt submission
│   │   ├── file_tools.py       Plain-text file reader/writer
│   │   ├── huggingface.py      HF Hub: model search, info, local check, download
│   │   ├── image_handling.py   Image upload/download, resolution detection, visual analysis
│   │   ├── iterate.py          Iterative generation helpers
│   │   ├── memory_tools.py     Agent-facing FAISS memory read/write tools
│   │   └── shell.py            Cross-platform shell execution for skill scripts
│   └── utils/
│       ├── agentY_server.py    Lightweight Flask bridge for ComfyUI extension callbacks
│       ├── chat_summary.py     Conversation summarisation utilities
│       ├── comfyui_client.py   Singleton HTTP client for the ComfyUI REST API
│       ├── comfyui_interrupt_hook.py  Halts agent loop after submit_prompt for async polling
│       ├── comfyui_poller.py   Async polling loop for ComfyUI job completion
│       ├── costs.py            Token-cost computation helpers
│       ├── learnings.py        Brain-learnings extraction and storage
│       ├── llm_functions.py    Structured LLM call helpers (non-agent)
│       ├── memory.py           FAISS / mem0 memory initialisation and access
│       ├── models.py           Model registry helpers
│       ├── secrets.py          Reads .env via dotenv_values (never injects into os.environ)
│       ├── triage.py           Intent classification and request routing
│       ├── workflow_parser.py  Workflow JSON analysis and template registration
│       └── workflow_signal.py  Signal/event bus for async workflow events
├── config/
│   ├── settings.json           ComfyUI URL, LLM defaults, polling intervals
│   ├── models.json             Model shortname → path table (injected into system prompts)
│   ├── workflow_templates.json Workflow template metadata
│   ├── brainbrief_example.json Example BrainBriefing for prompt injection
│   └── system_prompts/
│       ├── system_prompt.brain.md
│       ├── system_prompt.researcher.md
│       ├── system_prompt.triage.md
│       ├── system_prompt.planner.md
│       ├── system_prompt.learnings.md
│       ├── system_prompt.qaChecker.md
│       └── system_prompt.info.md
├── comfyui_workflow_templates_official/   Comfy-Org template library (git-ignored)
├── comfyui_workflow_templates_custom/     Your custom workflow templates
├── skills/                     Drop-in skill scripts (shell/Python)
├── memory/                     FAISS index for long-term agent memory
├── output_images/              Generated outputs
├── output_workflows/           Archived workflow JSON files
├── public/                     Static assets served by Chainlit
├── chainlit.md                 Chainlit welcome screen content
├── .env_example                Template for .env secrets
├── requirements.txt
├── run_agent.ps1               Windows launcher (starts Chainlit GUI)
├── add_workflow.ps1            Register a custom workflow template
└── remove_workflow.ps1         Remove a registered workflow template
```

> The ComfyUI custom node lives in its own repo: **[agentY-comfyui-extension](https://github.com/szprivate/agentY-comfyui-extension)**.
> Clone it into `ComfyUI/custom_nodes/agentY_bridge` to get the Send to agentY node.

---

## License

MIT
