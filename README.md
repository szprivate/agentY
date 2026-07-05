# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it supports Claude and Ollama as LLM backends and provides a Chainlit web GUI as conversational interface.

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; the pipeline builds, submits, and QA-checks the workflow automatically.
- **Image & video generation** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing** — reference-based editing, inpainting, upscaling, and more.
- **Persistent chat history** — Chainlit SQLAlchemy datalayer stores conversation threads and messages in PostgreSQL; uploaded files are persisted to a local MinIO S3 bucket.
- **FAISS memory** — long-term memory via mem0 + local Ollama embeddings (`nomic-embed-text`).
- **Hugging Face model management** — search, check local availability, and download models on demand.
- **Chainlit web GUI** — interact via a browser-based chat UI; images and videos are delivered inline.
- **Multiple LLM backends** — Claude and Ollama, configurable per pipeline stage.
- **Skills system** — drop shell/Python scripts into `skills/` and they become agent-callable tools.
- **ComfyUI extension** — a companion custom node ([agentY-comfyui-extension](https://github.com/szprivate/agentY-comfyui-extension)) lets you send images directly from ComfyUI to agentY and receive responses in real time.

---

## Requirements

- **Python 3.11+**
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- **Docker** — used to run PostgreSQL and MinIO via `docker-compose.yml` (or point to existing instances)
- An **Anthropic API key** (for Claude) _and/or_ a local **Ollama** installation

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/szprivate/agentY.git
cd agentY
```

### 2. Run the install script (recommended)

```powershell
.\install_agent.ps1
```

---

### Manual setup

If you prefer to set things up step by step, continue below.

### 3. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install asyncpg boto3
```

### 4. Set up PostgreSQL (Chainlit datalayer)

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

### 5. Set up MinIO (file storage)

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

Default credentials (can be overridden with env vars - see step 6):

| Env var | Default |
|---|---|
| `MINIO_ENDPOINT_URL` | `http://localhost:9000` |
| `MINIO_ACCESS_KEY` | `minioadmin` |
| `MINIO_SECRET_KEY` | `minioadmin` |
| `MINIO_BUCKET` | `chainlit` |

> **`run_agent.ps1` handles this automatically.** When Docker is available the script runs `docker compose up -d minio createbuckets` before launching Chainlit, so you normally don't need to start MinIO manually.

---

### 6. Configure secrets

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
```

Generate the Chainlit auth secret (run once):

```bash
chainlit create-secret
```

Paste the output into `CHAINLIT_AUTH_SECRET` in your `.env`.

> `install_agent.ps1` handles steps 4-6 automatically, including secret generation and `.env` population.

### 7. Configure defaults (optional)

Edit `config/settings.json` to point to your ComfyUI instance and set default LLMs:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",

  "llm": {
    "pipeline": {
      "query_templates":            "ollama,qwen3.5:9b",
      "assemble_workflow":                 "claude,claude-haiku-4-5",
      "detect_user_intent":                "ollama,qwen3.5:9b",
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

# Override the query_templates LLM
.\run_agent.ps1 -Llmquery_templates "ollama,qwen3-coder:32b"

# Override the assemble_workflow LLM
.\run_agent.ps1 -Llmassemble_workflow "claude,claude-sonnet-4-5"

# Show help
.\run_agent.ps1 -Help
```

---

### Memory

Long-term memory is stored in a local FAISS index (`memory/agenty_memory.faiss`) via **mem0** with **nomic-embed-text** embeddings served by Ollama. The agent automatically saves and retrieves relevant context across sessions.

### LLM configuration priority

Each value is resolved in order — first match wins:

1. CLI flag (`-Llmquery_templates` / `--query_templates-llm`)
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

---

## Adding Custom Workflow Templates

Use the helper scripts to register a workflow JSON and make it available to the agent:

```powershell
# Register a new workflow template (also generates a SKILL.md)
.\scripts\add_workflow.ps1 path\to\your_workflow_api.json

# Remove a registered template (also removes its skill directory)
.\scripts\remove_workflow.ps1 your_workflow_api
```

`scripts/add_workflow.ps1` parses the workflow, extracts node metadata, adds an entry to `config/workflow_templates.json`, and calls `scripts/build_skill.py` to generate a `SKILL.md` in the matching `skills/` directory. Custom templates live in `comfyui_workflow_templates_custom/templates/`.

---

## License

MIT
