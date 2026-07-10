# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it supports Claude and Ollama as LLM backends and is driven from a **chat panel that lives inside ComfyUI** — a sidebar tab provided by a small companion custom node.

> **The UI is now native to ComfyUI.** The old Chainlit web GUI (with its Postgres
> thread store and MinIO file storage, all in Docker) has been removed. You chat
> with the agent in ComfyUI's left sidebar; conversations persist to a local
> **SQLite** file; and instead of showing generated media inline, the agent drops
> a **`LoadImage` / video-loader node onto the graph** for every result — ready to
> wire straight into your next workflow.

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; the pipeline builds, submits, and QA-checks the workflow automatically.
- **Image & video generation** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing** — reference-based editing, inpainting, upscaling, and more.
- **Results as graph nodes** — every generated image/video is added to the open ComfyUI graph as a `LoadImage` / video-loader node (staged into ComfyUI's input dir), instead of being shown inline. The chat carries the agent's *text*.
- **Persistent chat history** — threads, messages, and the per-thread image gallery are stored in a self-contained local **SQLite** database (`memory/conversations.sqlite`). No Docker, Postgres, or S3.
- **Slash commands** — `/restart`, `/stop`, `/unload`, `/clear_vram`, `/images`, `/clearhistory`, `/switch_model`, `/add_workflow`, `/remove_workflow`, `/resend` — with an in-panel autocomplete popup.
- **FAISS memory** — long-term memory via mem0 + local Ollama embeddings (`nomic-embed-text`).
- **Hugging Face model management** — search, check local availability, and download models on demand.
- **Multiple LLM backends** — Claude and Ollama, configurable per pipeline stage.
- **Skills system** — drop shell/Python scripts into `skills/` and they become agent-callable tools.

---

## The four repos

agentY ships as a small stack of repositories. The installer below wires them all
up; this is what each one is:

| Repo | Location | Role |
|---|---|---|
| **agentY** (this repo) | your working copy | The Strands chat host / pipeline (`run_agent.ps1`). |
| **[agenty_core](https://github.com/szprivate/agenty_core)** | sibling folder next to `agentY` | Shared ComfyUI/HuggingFace/web/file tool layer + the canonical template/recipe corpus. Installed **editable** (`-e ../agenty_core`); **required**. |
| **[agentY-comfyuiConnect](https://github.com/szprivate/agentY-comfyuiConnect)** | `<ComfyUI>/custom_nodes/` | The chat UI — the **agentY** tab in ComfyUI's left sidebar. |
| **[agentY-mcp](https://github.com/szprivate/agentY-mcp)** | sibling folder next to `agentY` | The alternative **MCP-server / Claude-Desktop** front end (also consumes `agenty_core`). Optional. |

---

## Architecture

```
ComfyUI  (your browser)
  └─ agentY sidebar tab  ── agentY-comfyuiConnect  (in <ComfyUI>/custom_nodes/)
        │  HTTP + SSE (default http://127.0.0.1:5000)
        ▼
  agentY chat host  ── src/agenty_ui_server.py  →  src/utils/agentY_server.py
        │  runs the Strands pipeline; persists to SQLite
        │  tool layer ── ../agenty_core  (editable install)
        ├──HTTP/WS──►  ComfyUI  (submit workflows, stage outputs into /input)
        └──►  memory/conversations.sqlite  (threads, messages, gallery, resume state)
```

The chat host is a normal agentY process (Claude/Ollama, same `config/settings.json`). The ComfyUI custom node is only the **frontend + a graph-load hook** — it talks to the host over HTTP/SSE and injects loader nodes when the agent produces output.

---

## Requirements

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (Python 3.11+ env manager) and **git** on your PATH
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- An **Anthropic API key** (for Claude) _and/or_ a local **Ollama** installation
- A **Hugging Face token** (for gated-model downloads)

No Docker, Postgres, or MinIO.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/szprivate/agentY.git
cd agentY
```

### 2. Run the installer (recommended)

```powershell
.\install_agent.ps1
```

The installer sets up the **whole stack** in one pass:

1. checks for `git` + `uv`;
2. clones the sibling repos it needs — **agenty_core** (required) and **agentY-mcp** (optional) — next to `agentY` if they aren't there already, and fast-forwards them if they are;
3. creates agentY's `.venv` (via `uv`) and installs `requirements.txt` (which pulls in `agenty_core` editable);
4. copies `.env_example` → `.env` and **prompts** you for `HF_TOKEN`, `ANTHROPIC_API_KEY`, and the optional `COMFYUI_API_KEY` / `DASHSCOPE_API_KEY` (Enter keeps an existing value);
5. **finds your ComfyUI** (auto-detects common paths, otherwise asks) and clones **agentY-comfyuiConnect** into its `custom_nodes/`, optionally pointing `settings.json` at your ComfyUI URL;
6. sets up **agentY-mcp**'s own venv + `.env` and reuses the tokens you just entered.

Useful flags:

```powershell
.\install_agent.ps1 -ComfyUIPath "D:\ai\ComfyUI"   # skip ComfyUI auto-detection
.\install_agent.ps1 -SkipMcp                        # don't set up agentY-mcp
.\install_agent.ps1 -SkipComfyNode                  # headless host only, no ComfyUI node
.\install_agent.ps1 -NonInteractive                 # no prompts (CI / re-runs)
.\install_agent.ps1 -Help
```

<details>
<summary><b>Manual setup</b> (instead of the installer)</summary>

```powershell
# agenty_core must sit next to agentY (requirements.txt installs it editable)
git clone https://github.com/szprivate/agenty_core.git ..\agenty_core

# agentY itself
uv venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
uv pip install -r requirements.txt
copy .env_example .env

# the ComfyUI chat UI (restart ComfyUI afterwards)
git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect

# (optional) the MCP / Claude Desktop front end
git clone https://github.com/szprivate/agentY-mcp.git ..\agentY-mcp
```
</details>

### 3. Configure secrets

The installer prompts for these; to edit them later, open `.env`:

```dotenv
HF_TOKEN=hf_...                 # Hugging Face token (for gated model downloads)
ANTHROPIC_API_KEY=sk-ant-...    # for Claude
COMFYUI_API_KEY=comfyui-...     # only if your ComfyUI requires auth / uses API nodes
DASHSCOPE_API_KEY=...           # Alibaba Model Studio (DashScope) — for Qwen models (optional)

# Optional
# AGENTY_UI_HOST=127.0.0.1
# AGENTY_UI_PORT=5000
# AGENTY_CONVERSATION_DB=./memory/conversations.sqlite
```

### 4. The ComfyUI chat UI

The installer clones [`agentY-comfyuiConnect`](https://github.com/szprivate/agentY-comfyuiConnect)
into ComfyUI's `custom_nodes/` for you. If you skipped that step (or ComfyUI
wasn't found), install it by hand and restart ComfyUI once:

```powershell
git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect
```

After the restart, ComfyUI's left sidebar shows an **agentY** tab.

### 5. Configure defaults (optional)

Edit `config/settings.json` to point at your ComfyUI instance and set default LLMs:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",
  "conversation_db": "./memory/conversations.sqlite",
  "llm": {
    "pipeline": {
      "query_templates":   "ollama,qwen3-coder:30b",
      "assemble_workflow": "claude,claude-haiku-4-5",
      "detect_user_intent":"ollama,qwen3.6:27b"
    }
  }
}
```

Each `"pipeline"` value uses the format `"provider,model"` — `"claude,claude-haiku-4-5"`, `"ollama,qwen3.5:9b"`, or `"dashscope,qwen-plus"`. **`dashscope`** routes to **Alibaba Model Studio** (Qwen models over its OpenAI-compatible API) — set `DASHSCOPE_API_KEY` in `.env`; tune the endpoint/default under the `dashscope` block in `settings.json` (defaults to the International endpoint). Aliases `qwen` / `modelstudio` / `alibaba` also work.

---

## Usage

Start the chat host (it builds the pipeline and serves the sidebar backend):

```powershell
# Default — backend on http://127.0.0.1:5000
.\run_agent.ps1

# Custom port / bind address
.\run_agent.ps1 -Port 5001
.\run_agent.ps1 -BindHost 0.0.0.0

# Override a stage's LLM for this run
.\run_agent.ps1 -LlmQueryTemplates "ollama,qwen3-coder:32b"
.\run_agent.ps1 -LlmAssembleWorkflow "claude,claude-sonnet-4-5"

# Help
.\run_agent.ps1 -Help
```

Then open **ComfyUI**, click the **agentY** tab in the left sidebar, and chat:

- *"Generate a cinematic wide shot of Tokyo at night."*
- *"Edit this photo to make it daytime."* (attach an image with 📎)
- *"Make 5 variations of a red sports car, different angles."*
- *"Upscale the last image with UltimateSD."*

Each finished image/video appears as a **loader node on your graph**. Type `/` in the input for the slash-command menu; use the thread dropdown to revisit past conversations.

> If the backend runs on a non-default URL, set it in the browser console:
> `localStorage.agentY_backend = "http://host:port"`.

### LLM configuration priority

Each value resolves in order — first match wins: **CLI flag → environment variable → `config/settings.json` → hard-coded default.**

### Memory

Long-term memory is stored in a local FAISS index (`memory/agenty_memory.faiss`) via **mem0** with **nomic-embed-text** embeddings served by Ollama. Conversation threads are separate — they live in the SQLite store above.

---

## Adding Custom Workflow Templates

```powershell
# Register a new workflow template (also generates a SKILL.md)
.\scripts\add_workflow.ps1 path\to\your_workflow_api.json

# Remove a registered template (also removes its skill directory)
.\scripts\remove_workflow.ps1 your_workflow_api
```

You can also do this from the chat with `/add_workflow <path>` and `/remove_workflow <name>`. Custom templates live in `comfyui_workflow_templates_custom/templates/`.

---

## License

MIT
