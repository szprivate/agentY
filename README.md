# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it runs on Claude, Ollama, Alibaba/DashScope (Qwen), OpenAI (GPT), or Google (Gemini) models and is driven from a **chat panel that lives inside ComfyUI** — a sidebar tab provided by a small companion custom node.

> **The UI is native to ComfyUI.** The old Chainlit web GUI (with its Postgres
> thread store and MinIO file storage, all in Docker) has been removed. You chat
> with the agent in ComfyUI's left sidebar; conversations persist to a local
> **SQLite** file; and instead of showing generated media inline, the agent drops
> a **`LoadImage` / video-loader node onto the graph** for every result — ready to
> wire straight into your next workflow.

> 📖 **New to agentY?** The [**Using agentY guide**](docs/using-agentY.md) is a
> screenshot-driven tour of the whole thing — chat, the canvas **hook system**,
> settings, MCP, and token usage.

![agentY sidebar chat next to the ComfyUI graph](docs/images/overview.png)

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; a free **Orchestrator agent** builds and submits the workflow automatically.
- **Output QA against *your* briefing** — write a checklist (and wire in mood/reference images) as a `qa` canvas hook, a reusable file, or `/qa` in chat; a separate QA agent judges every finished image/video criterion by criterion and re-generates what missed. Nothing runs without a briefing. See [Checking outputs](docs/using-agentY.md#checking-outputs-qa).
- **Free-agent orchestration** — one Orchestrator owns each turn with the full toolset. It calls tools directly, **delegates** to specialists (research / assembly / info / story / DOP / planner / web), spawns ad-hoc subagents, and can even **author skills live**. No brittle intent classifier or fixed routing.
- **Custom-node creator** — point the agent at a model's GitHub repo (`create_custom_node`) and it clones the repo, reads its docs + inference code, and writes a self-contained **ComfyUI custom-node pack** (`__init__.py`, `nodes.py`, `requirements.txt`, `README.md`, `pyproject.toml`) into `output/custom_nodes/<name>/` — ready to publish as its own repo.
- **Image & video generation** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing** — reference-based editing, inpainting, upscaling, and more.
- **Results as graph nodes** — every generated image/video is added to the open ComfyUI graph as a `LoadImage` / video-loader node (staged into ComfyUI's input dir), instead of being shown inline. The chat carries the agent's *text*.
- **Canvas hook nodes** — annotate the graph with **`agentY hook`** nodes ("sweep the seed 6×", "upscale then add film grain") and let the agent run them; chain hooks for multi-step tasks, **bake** a chain into reusable native ComfyUI **subgraphs**, or run an interactive **iterative-refine loop** (one gen per turn, feeding each result back in). See [Canvas nodes](#canvas-nodes) and the [hook system guide](docs/using-agentY.md#the-hook-system).
- **MCP support** — call tools from external **MCP servers** (config-driven `config/mcp.json`, `http`/`sse`/`stdio`, with `none` / `header` / **OAuth** auth). Ships with **Magnific** wired via OAuth (one-click *Authorize…* in Settings). See [MCP servers](docs/using-agentY.md#mcp-servers).
- **Persistent chat history** — threads, messages, and the per-thread image gallery are stored in a self-contained local **SQLite** database (`memory/conversations.sqlite`). No Docker, Postgres, or S3.
- **Slash commands** — `/restart`, `/stop`, `/unload`, `/clear_vram`, `/images`, `/clearhistory`, `/switch_model`, `/add_workflow`, `/remove_workflow`, `/resend`, `/qa` — with an in-panel autocomplete popup.
- **In-panel Settings & token usage** — edit auth keys (`.env`) and your settings (saved to `config/settings.local.json`), and review per-model token cost, from ComfyUI's own Settings panel (no file editing required).
- **FAISS memory** — long-term memory via mem0 + local Ollama embeddings (`nomic-embed-text`).
- **Hugging Face model management** — search, check local availability, and download models on demand.
- **Multiple LLM backends** — Claude, Ollama, Alibaba/DashScope (Qwen), OpenAI (GPT), and Google (Gemini), configurable per pipeline stage. Models are picked by **tier** (six of them) with per-role overrides for the exceptions; `/switch_model` and the composer's picker target either, and the model list is **discovered live** from each provider (a vendor appears only when its API key is set), so it never goes stale.

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
        ├──HTTP/WS──►  ComfyUI  (submit workflows, QA the outputs, stage them into /input)
        └──►  memory/conversations.sqlite  (threads, messages, gallery, resume state)
```

Each user turn is owned by the **Orchestrator** agent (a normal Claude / Ollama / Qwen / GPT / Gemini model, per `config/settings.default.toml` + `settings.local.json`). It has the full toolset and can call the specialist agents as delegates. When it finishes assembling a workflow it hands off to the **Executor** (ComfyUI submission → completion polling → [output QA](docs/using-agentY.md#checking-outputs-qa) when you have set a briefing → staging outputs as loader nodes). The ComfyUI custom node is the **frontend + canvas nodes + a graph-load hook** — it talks to the host over HTTP/SSE.

---

## Requirements

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (Python 3.11+ env manager) and **git** on your PATH
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- At least one LLM backend: an **Anthropic API key** (Claude), a local **Ollama** install, a **DashScope / Alibaba Model Studio key** (Qwen), an **OpenAI API key** (GPT), and/or a **Google Gemini API key**
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
5. **finds your ComfyUI** (auto-detects common paths, otherwise asks) and clones **agentY-comfyuiConnect** into its `custom_nodes/`, optionally pointing `settings.local.json` at your ComfyUI URL;
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
OPENAI_API_KEY=sk-...           # for OpenAI (GPT) models
GEMINI_API_KEY=...              # for Google Gemini models (GOOGLE_API_KEY also works)
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

`config/settings.default.toml` holds the committed defaults; put your machine's values (ComfyUI URL/paths, model choices, private endpoints) in `config/settings.local.json` (gitignored, deep-merged over the defaults).

Models are chosen by **tier**, not one dropdown per role: set the six `llm.tiers` values and every role inherits from one of them. `llm.pipeline` underneath is per-role **overrides** — leave a role blank to inherit, fill one in only when that single job wants something different. Resolution for any role is *env var → override → tier → built-in default*. Any model value is `"provider,model"`:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",
  "conversation_db": "./memory/conversations.sqlite",
  "llm": {
    "tiers": {
      "orchestrator":      "dashscope,qwen3.7-max",   // drives every turn
      "research_assembly": "dashscope,qwen3.6-plus",  // templates, graph building, repair
      "fast_utility":      "dashscope,qwen3.6-flash", // info, search, planner, learnings, …
      "vision":            "dashscope,qwen3-vl-flash",// reads the images YOU provide
      "qa_judge":          "dashscope,qwen3-vl-plus", // grades finished outputs
      "coder":             "dashscope,kimi-k2.7-code" // scripts and custom nodes
    },
    "pipeline": {
      // per-role overrides; blank = inherit from the tier. Usually all blank.
      "video_agent": "dashscope,qwen3-vl-plus"
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

Each `"provider,model"` value can be `"claude,claude-opus-4-8"`, `"ollama,qwen3-coder:30b"`, `"dashscope,qwen3.6-flash"`, `"openai,gpt-4o"`, or `"google,gemini-2.5-pro"`. **`dashscope`** routes to **Alibaba Model Studio** (Qwen over its OpenAI-compatible API) — set `DASHSCOPE_API_KEY` in `.env`; aliases `qwen` / `modelstudio` / `alibaba` also work. **`openai`** and **`google`** (alias `gemini`) route to OpenAI and Google Gemini respectively (Gemini via its OpenAI-compatible endpoint) — set `OPENAI_API_KEY` / `GEMINI_API_KEY` in `.env`. You can change any stage live from chat with `/switch_model` (e.g. `/switch_model orchestrator claude,claude-opus-4-8`); its model picker is discovered live from each configured provider, so only vendors whose key is set appear.

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

- **`agentY hook`** — an instruction attached to the canvas. Wire any node's output into its **auto-growing `anchor` input(s)** and type a directive. Purposes:
  - *inline_parameter* — annotate an existing node ("sweep the seed 6×", "iterate the files in this folder"); the agent expands and runs your on-canvas graph.
  - *make_workflow* — the agent generates and runs a workflow (or Python script) from the prompt, using the wired input(s) if any.
  - *text* — the agent writes a string answer and drops a wireable `agentY text` node carrying it.
  - *iterate* — an interactive **refinement loop**: the agent runs the graph one generation per turn, feeds each result back into the wired `LoadImage`, and asks for your next prompt (you can jump back to an earlier generation) until you say stop.
  - *qa* — your **quality briefing** for the graph: the directive is the checklist, the wired anchors are reference/mood images. A separate QA agent judges every produced image/video against it criterion by criterion, and re-generates a failing output against exactly what it missed. See [Checking outputs](docs/using-agentY.md#checking-outputs-qa).

  Its `out` **output** is type-agnostic, so one hook can gather several inputs and produce a result — image, video, **or scalars (string/int/float)** — for the next hook. Wire hooks output→input to build a **multi-step chain**. `freeze` decides whether the agent keeps the hook live (injects the value at run time) or bakes it into a plain workflow. A hook is inert on a normal *Queue Prompt* (it's a pure passthrough the agent removes before running), so it never affects a manual run. **Bypass** (`Ctrl+B`) or mute a hook to disable it without deleting it — the agent skips hooks in those modes.

- **`agentY python`** — runs an agent-authored Python snippet as a node. It's used by *baking* (below): a value the agent computed at runtime (e.g. a video's length) is placed here so it becomes a genuine, re-runnable output. ⚠️ **It executes arbitrary Python whenever the graph runs** — meant for your own, self-hosted, agent-built workflows; don't run baked workflows from untrusted sources. Set `AGENTY_PYTHON_NODE_DISABLED=1` to make it a no-op.

**Bake a chain into subgraphs.** Turn on `bake_to_canvas` on your make_workflow hooks. When you ask the agent to run the graph, it nests each stage's generated workflow into a ComfyUI **subgraph** (with inputs/outputs matching the hook's slots), **adds** those subgraphs to your canvas *next to the hook nodes* (nothing is removed), and wires them to mirror the chain. The result is a self-contained native workflow you can re-run **without the agent** — the multi-step task, "baked."

### LLM configuration priority

Each value resolves in order — first match wins: **CLI flag → environment variable → `config/settings.local.json` → `config/settings.default.toml` → hard-coded default.** Committed defaults live in `settings.default.toml`; per-machine values (paths, model pins, private endpoints) go in the gitignored `settings.local.json`, which is deep-merged over the defaults. You can also change any stage live with `/switch_model <stage> <provider,model>`.

### In-panel settings & token usage

Open ComfyUI's **Settings** panel → **agentY** → **Open agentY Settings…**. That one row is the entire agentY section; everything else is inside the modal:
- Your auth keys (`.env`) and every setting — **model tiers** and per-role overrides, directories, toggles — in **collapsible groups**, with the rarely-touched ones behind **Show advanced settings**. Changed values are saved as overrides in `config/settings.local.json`, leaving the committed defaults untouched — no file editing.
- **MCP servers** (`config/mcp.json`, with per-server status + an **Authorize…** button for OAuth) and model **pricing**.
- **Viewers** — the **message-history log**, the **long-term-memory editor**, and the **token usage** breakdown (cost per model / per agent role, with a **🗑 Clear log** button). Token usage is also the **📊** button in the chat panel's top bar.

The chat panel's top bar also has a **🖼 autograph toggle** — flip whether finished workflows/results are auto-loaded onto the canvas, live (no restart).

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

## Documentation

- [**Using agentY**](docs/using-agentY.md) — the full, screenshot-driven usage
  guide: chat, the canvas **hook system**, settings, MCP, token usage, memory,
  and model configuration.

---

## License

MIT
