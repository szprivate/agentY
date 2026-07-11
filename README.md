# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it runs on Claude, Ollama, or Alibaba/DashScope (Qwen) models and is driven from a **chat panel that lives inside ComfyUI** — a sidebar tab provided by a small companion custom node.

> **The UI is native to ComfyUI.** The old Chainlit web GUI (with its Postgres
> thread store and MinIO file storage, all in Docker) has been removed. You chat
> with the agent in ComfyUI's left sidebar; conversations persist to a local
> **SQLite** file; and instead of showing generated media inline, the agent drops
> a **`LoadImage` / video-loader node onto the graph** for every result — ready to
> wire straight into your next workflow.

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; a free **Orchestrator agent** builds, submits, and QA-checks the workflow automatically.
- **Free-agent orchestration** — one Orchestrator owns each turn with the full toolset. It calls tools directly, **delegates** to specialists (research / assembly / info / story / DOP / planner / web), spawns ad-hoc subagents, and can even **author skills live**. No brittle intent classifier or fixed routing.
- **Image & video generation** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing** — reference-based editing, inpainting, upscaling, and more.
- **Results as graph nodes** — every generated image/video is added to the open ComfyUI graph as a `LoadImage` / video-loader node (staged into ComfyUI's input dir), instead of being shown inline. The chat carries the agent's *text*.
- **Canvas hook nodes** — annotate the graph with **`agentY hook`** nodes ("sweep the seed 6×", "upscale then add film grain") and let the agent run them; chain hooks for multi-step tasks and **bake** a chain into reusable native ComfyUI **subgraphs** (see [Canvas nodes](#canvas-nodes)).
- **Persistent chat history** — threads, messages, and the per-thread image gallery are stored in a self-contained local **SQLite** database (`memory/conversations.sqlite`). No Docker, Postgres, or S3.
- **Slash commands** — `/restart`, `/stop`, `/unload`, `/clear_vram`, `/images`, `/clearhistory`, `/switch_model`, `/add_workflow`, `/remove_workflow`, `/resend` — with an in-panel autocomplete popup.
- **In-panel Settings & token usage** — edit auth keys (`.env`) and `config/settings.json`, and review per-model token cost, from ComfyUI's own Settings panel (no file editing required).
- **FAISS memory** — long-term memory via mem0 + local Ollama embeddings (`nomic-embed-text`).
- **Hugging Face model management** — search, check local availability, and download models on demand.
- **Multiple LLM backends** — Claude, Ollama, and Alibaba/DashScope (Qwen), configurable per pipeline stage.

---

## The four repos

agentY ships as a small stack of repositories. The installer below wires them all
up; this is what each one is:

| Repo | Location | Role |
|---|---|---|
| **agentY** (this repo) | your working copy | The Strands chat host / pipeline (`run_agent.ps1`). |
| **[agenty_core](https://github.com/szprivate/agenty_core)** | sibling folder next to `agentY` | Shared ComfyUI/HuggingFace/web/file tool layer + the canonical template/recipe corpus. Installed **editable** (`-e ../agenty_core`); **required**. |
| **[agentY-comfyuiConnect](https://github.com/szprivate/agentY-comfyuiConnect)** | `<ComfyUI>/custom_nodes/` | The **agentY** sidebar tab **and** the canvas nodes (`agentY hook`, `agentY python`). |
| **[agentY-mcp](https://github.com/szprivate/agentY-mcp)** | sibling folder next to `agentY` | The alternative **MCP-server / Claude-Desktop** front end (also consumes `agenty_core`). Optional. |

---

## Architecture

```
ComfyUI  (your browser)
  ├─ agentY sidebar tab   ┐
  └─ agentY hook / python │── agentY-comfyuiConnect  (in <ComfyUI>/custom_nodes/)
     nodes on the canvas  ┘
        │  HTTP + SSE (default http://127.0.0.1:5000)
        ▼
  agentY chat host  ── src/agenty_ui_server.py  →  src/utils/agentY_server.py
        │  Orchestrator agent (+ specialist delegates, Executor stage)
        │  tool layer ── ../agenty_core  (editable install)
        ├──HTTP/WS──►  ComfyUI  (submit workflows, run Vision-QA, stage outputs into /input)
        └──►  memory/conversations.sqlite  (threads, messages, gallery, resume state)
```

Each user turn is owned by the **Orchestrator** agent (a normal Claude/Ollama/Qwen model, per `config/settings.json`). It has the full toolset and can call the specialist agents as delegates. When it finishes assembling a workflow it hands off to the **Executor** (ComfyUI submission → completion polling → optional Ollama Vision-QA → staging outputs as loader nodes). The ComfyUI custom node is the **frontend + canvas nodes + a graph-load hook** — it talks to the host over HTTP/SSE.

---

## Requirements

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (Python 3.11+ env manager) and **git** on your PATH
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- At least one LLM backend: an **Anthropic API key** (Claude), a local **Ollama** install, and/or a **DashScope / Alibaba Model Studio key** (Qwen)
- A **Hugging Face token** (for gated-model downloads)
- **Ollama** with `nomic-embed-text` pulled if you want long-term FAISS memory

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

# the ComfyUI sidebar + canvas nodes (restart ComfyUI afterwards)
git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect

# (optional) the MCP / Claude Desktop front end
git clone https://github.com/szprivate/agentY-mcp.git ..\agentY-mcp
```
</details>

### 3. Configure secrets

The installer prompts for these; to edit them later, open `.env` **or** use the in-panel Settings (see below):

```dotenv
HF_TOKEN=hf_...                 # Hugging Face token (for gated model downloads)
ANTHROPIC_API_KEY=sk-ant-...    # for Claude
DASHSCOPE_API_KEY=...           # Alibaba Model Studio (DashScope) — for Qwen models
COMFYUI_API_KEY=comfyui-...     # only if your ComfyUI requires auth / uses API nodes

# Optional
# AGENTY_UI_HOST=127.0.0.1
# AGENTY_UI_PORT=5000
# AGENTY_CONVERSATION_DB=./memory/conversations.sqlite
# AGENTY_PYTHON_NODE_DISABLED=1   # make the agentY python node a no-op (see Canvas nodes)
```

### 4. The ComfyUI sidebar + canvas nodes

The installer clones [`agentY-comfyuiConnect`](https://github.com/szprivate/agentY-comfyuiConnect)
into ComfyUI's `custom_nodes/` for you. If you skipped that step (or ComfyUI
wasn't found), install it by hand and restart ComfyUI once:

```powershell
git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect
```

After the restart you get, from the one node pack:
- the **agentY** tab in ComfyUI's left sidebar (the chat panel);
- the **agentY** node category with **`agentY hook`** and **`agentY python`** (see [Canvas nodes](#canvas-nodes));
- an **Open agentY Settings…** entry and a **token-usage** view in ComfyUI's Settings panel.

### 5. Configure defaults (optional)

`config/settings.json` points at your ComfyUI instance and sets the per-stage LLMs. The **Orchestrator** is the model that drives each turn; the other keys set the specialist delegates and the Executor's Vision-QA model. Any value is `"provider,model"`:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",
  "conversation_db": "./memory/conversations.sqlite",
  "llm": {
    "pipeline": {
      "orchestrator":          "dashscope,qwen3.6-flash",  // drives each turn
      "assemble_workflow":     "dashscope,qwen3.6-flash",  // workflow-assembly delegate
      "query_templates":       "dashscope,qwen3.6-flash",  // template/recipe research delegate
      "executor_vision_model": "dashscope,qwen3.6-flash"   // Vision-QA of results
      // …info, story, search_web, dop, planner, learnings, error_checker, llm_functions…
    },
    "dashscope": {
      // Public International endpoint; for mainland China use
      // https://dashscope.aliyuncs.com/compatible-mode/v1
      "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
      "model": "qwen-plus"
    }
  }
}
```

Each `"provider,model"` value can be `"claude,claude-haiku-4-5"`, `"ollama,qwen3-coder:30b"`, or `"dashscope,qwen3.6-flash"`. **`dashscope`** routes to **Alibaba Model Studio** (Qwen over its OpenAI-compatible API) — set `DASHSCOPE_API_KEY` in `.env`; aliases `qwen` / `modelstudio` / `alibaba` also work. You can change any stage live from chat with `/switch_model` (e.g. `/switch_model orchestrator claude,claude-sonnet-4-5`).

---

## Usage

Start the chat host (it builds the agent and serves the sidebar backend):

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

### Canvas nodes

Installing `agentY-comfyuiConnect` adds two nodes under the **agentY** category. They let you drive the agent *from the graph itself*:

- **`agentY hook`** — an instruction attached to the canvas. Wire any node's output into its **auto-growing `anchor` input(s)** and type a directive. Two purposes:
  - *directive* — annotate an existing node ("sweep the seed 6×", "iterate the files in this folder"); the agent expands and runs your on-canvas graph.
  - *workflow-standin* — the agent generates and runs a workflow (or Python script) from the prompt, using the wired input(s) if any.

  Its `passthrough` **outputs also auto-grow**, and all slots are type-agnostic, so one hook can gather several inputs and export several results — image, video, **or scalars (string/int/float)** — to the next hook. Wire hooks output→input to build a **multi-step chain**. A hook is inert on a normal *Queue Prompt* (it's a pure passthrough the agent removes before running), so it never affects a manual run. Toggle `ignore` to disable a hook without deleting it.

- **`agentY python`** — runs an agent-authored Python snippet as a node. It's used by *baking* (below): a value the agent computed at runtime (e.g. a video's length) is placed here so it becomes a genuine, re-runnable output. ⚠️ **It executes arbitrary Python whenever the graph runs** — meant for your own, self-hosted, agent-built workflows; don't run baked workflows from untrusted sources. Set `AGENTY_PYTHON_NODE_DISABLED=1` to make it a no-op.

**Bake a chain into subgraphs.** Turn on `bake_to_canvas` on your standin hooks. When you ask the agent to run the graph, it nests each stage's generated workflow into a ComfyUI **subgraph** (with inputs/outputs matching the hook's slots), **adds** those subgraphs to your canvas *next to the hook nodes* (nothing is removed), and wires them to mirror the chain. The result is a self-contained native workflow you can re-run **without the agent** — the multi-step task, "baked."

### LLM configuration priority

Each value resolves in order — first match wins: **CLI flag → environment variable → `config/settings.json` → hard-coded default.** You can also change any stage live with `/switch_model <stage> <provider,model>`.

### In-panel settings & token usage

Open ComfyUI's **Settings** panel → **agentY**:
- **Open agentY Settings…** edits your auth keys (`.env`) and everything in `config/settings.json` (models per stage, directories, toggles) with a comment-preserving save — no file editing.
- **Open Token Usage…** (also the **📊** button in the chat panel's top bar) breaks down cost per model / per agent role from the persisted token log, with a **🗑 Clear log** button to purge it.

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

You can also do this from the chat with `/add_workflow <path>` (or `/add_workflow canvas <name>` to register the open graph) and `/remove_workflow <name>`. Custom templates live in `comfyui_workflow_templates_custom/templates/`; the shared template/recipe corpus lives in **agenty_core**.

---

## License

MIT
