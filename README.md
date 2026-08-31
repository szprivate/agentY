# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it runs on Claude, Ollama, Alibaba/DashScope (Qwen), OpenAI (GPT), or Google (Gemini) models and is driven from a **chat panel that lives inside ComfyUI** — a sidebar tab provided by a small companion custom node.

> **The UI is native to ComfyUI.** You chat with the agent in ComfyUI's left
> sidebar; conversations persist to a local **SQLite** file; and every result is
> dropped onto the graph as a **`LoadImage` / video-loader node** — ready to wire
> straight into your next workflow.

> 📖 **New to agentY?** The [**Using agentY guide**](docs/using-agentY.md) is a
> screenshot-driven tour of the whole thing — chat, the canvas **hook system**,
> settings, MCP, and token usage.

![agentY sidebar chat next to the ComfyUI graph](docs/images/overview.png)

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; the agent researches a template, builds the workflow, runs it, and puts the result on your graph.
- **Canvas hook nodes** — annotate the graph with **`agentY hook`** nodes ("sweep the seed 6×", "upscale then add film grain") and let the agent run them. Chain hooks for multi-step tasks, **bake** a chain into native ComfyUI **subgraphs**, stop mid-chain with a `review` hook to pick what continues, or run an interactive **iterative-refine loop**. See [Canvas nodes](#canvas-nodes) and the [hook system guide](docs/using-agentY.md#the-hook-system).
- **Output QA against *your* briefing** — a checklist as a `qa` hook, a reusable file, or `/qa` in chat; a separate QA agent judges every finished image and video criterion by criterion and re-generates what missed. Nothing runs without a briefing. See [Checking outputs](docs/using-agentY.md#checking-outputs-qa).
- **The countable half is measured, not eyeballed** — an **`agentY qa briefing`** node puts ratio, resolution, sharpness, grain, clipping and **likeness** on dropdowns, each settled from the file before the model is asked anything. A vision model is handed a *resized* copy, so "is this 16:9?" is a question it cannot actually answer; likeness is a real score (a face embedding, or a perceptual metric for a place or product) because "match the reference" is what people write most and models answer worst.
- **It fixes the shape rather than re-rolling** — if your briefing says 16:9 and the graph says 1024x1024, agentY walks upstream to the node that decides the size the picture is *made* at (never a downstream resize) and sets it, before the run. On a failure it does the same first, and where nothing in the graph governs what failed it **declines the retry** rather than paying to be told the same thing twice. See [It fixes the shape](docs/using-agentY.md#it-fixes-the-shape-rather-than-re-rolling).
- **A briefing per stage** — an unwired briefing judges everything a run makes; wire its `out` into a hook and it judges only that stage, so the reference frames and the video they feed can have different standards. See [One briefing per stage](docs/using-agentY.md#one-briefing-per-stage).
- **A ranking score that learns your taste** — separate from the pass/fail gates, one 0-1 number orders a run's outputs, shown when a `review` hook stops the chain. Every review you answer is a preference label, and `scripts/fit_fitness_weights.py` **refuses to install weights that don't beat the defaults** on reviews it held back. See [Which of these is best?](docs/using-agentY.md#which-of-these-is-best).
- **Refine loops on your own graph** — *"change the prompt until the woman's position matches the original frame"*. No hook nodes: the agent runs **your** open graph, judges each output against your condition, rewrites one value and goes again, within a budget you set (`[refine] max_runs`, default 4). See [Loops](docs/using-agentY.md#loops-keep-trying-until-its-right).
- **Image & video generation and editing** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo and many others; reference-based editing, inpainting, upscaling.
- **Results as graph nodes** — every generated image or video is added to the open graph as a loader node (staged into ComfyUI's input dir). The chat carries the agent's *text*.
- **Web references, straight onto the canvas** — *"search the web for images of this car, from every angle"*. Downloading **is** showing: the agent searches, picks, downloads into ComfyUI's `input` dir and drops each keeper on your graph, skipping watermarked previews and anything that isn't really an image. See [Finding references](docs/using-agentY.md#finding-reference-images-on-the-web).
- **Screenshots of your canvas** — drawn by the ComfyUI page itself, so it is your graph as **you** have it: your layout, groups, colours and collapsed nodes, with the prompt text painted back in and your view restored before the browser paints. See [Screenshots](docs/using-agentY.md#screenshots-of-your-workflow).
- **MCP support** — call tools from external **MCP servers** (`config/mcp.json`; `http`/`sse`/`stdio`, with `none` / `header` / **OAuth** auth). Ships with **Magnific** wired via OAuth. See [MCP servers](docs/using-agentY.md#mcp-servers).
- **Slack bridge (optional, off by default)** — a **second line** into the agent, never instead of the panel. Every turn is mirrored to your DM as it runs, *including ones you start in the panel*, and a DM back drives the same conversation. Outbound Socket Mode, so nothing has to be reachable from the internet. See [Slack](docs/slack.md).
- **Persistent chat history** — threads, messages and the per-thread gallery in a local **SQLite** file (`memory/conversations.sqlite`).
- **FAISS memory** — long-term memory via mem0 + local Ollama embeddings (`nomic-embed-text`).
- **Slash commands** — `/restart`, `/stop`, `/unload`, `/clear_vram`, `/images`, `/project_memory`, `/clearhistory`, `/switch_model`, `/add_workflow`, `/remove_workflow`, `/resend`, `/qa` — with an in-panel autocomplete popup.
- **In-panel Settings & token usage** — auth keys (`.env`), settings (`config/settings.local.json`) and per-model token cost, all from ComfyUI's own Settings panel.
- **Hugging Face model management** — search, check local availability, download on demand.
- **Multiple LLM backends** — Claude, Ollama, Alibaba/DashScope (Qwen), OpenAI (GPT) and Google (Gemini). Models are picked by **tier** (six of them) with per-role overrides for the exceptions, and the list is **discovered live** from each provider, so a vendor appears only when its key is set.
- **Custom-node creator** (⚠️ *experimental, untested*) — point the agent at a model's GitHub repo and it reads the docs and inference code and writes a self-contained **ComfyUI node pack** into `output/custom_nodes/<name>/`. See [Building a node](docs/using-agentY.md#building-a-node-for-a-new-model).

---

## The four repos

agentY ships as a small stack of repositories. The installer below wires them all
up; this is what each one is:

| Repo | Location | Role |
|---|---|---|
| **agentY** (this repo) | your working copy | The Strands chat host / pipeline (`run_agent.ps1` / `run_agent.sh`). |
| **[agenty_core](https://github.com/szprivate/agenty_core)** | sibling folder next to `agentY` | Shared ComfyUI/HuggingFace/web/file tool layer + the canonical template/recipe corpus. Installed **editable** (`-e ../agenty_core`); **required**. |
| **[agentY-comfyuiConnect](https://github.com/szprivate/agentY-comfyuiConnect)** | `<ComfyUI>/custom_nodes/` | The **agentY** sidebar tab **and** the canvas nodes (`agentY hook`, `agentY qa briefing`, `agentY python`). |
| **[agentY-mcp](https://github.com/szprivate/agentY-mcp)** | sibling folder next to `agentY` | The alternative **MCP-server / Claude-Desktop** front end (also consumes `agenty_core`). Optional. |

---

## Architecture

```
ComfyUI  (your browser)
  ├─ agentY sidebar tab   ┐
  └─ agentY hook / python │── agentY-comfyuiConnect  (in <ComfyUI>/custom_nodes/)
     nodes on the canvas  ┘
        │  HTTP + SSE (default :5000, or :5001 on macOS — see Ports)
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

- **Windows**, or **macOS 14+ on Apple Silicon**. Each has its own installer and
  launcher (`.ps1` / `.sh`); everything above them is the same code. On a Mac run
  `xcode-select --install` first — `insightface` and `sam3` ship as source and are
  compiled during the install.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (Python 3.11+ env manager) and **git** on your PATH
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- At least one LLM backend: an **Anthropic API key** (Claude), a local **Ollama** install, a **DashScope / Alibaba Model Studio key** (Qwen), an **OpenAI API key** (GPT), and/or a **Google Gemini API key**
- A **Hugging Face token** (for gated-model downloads)
- **Ollama** with `nomic-embed-text` pulled if you want long-term FAISS memory

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/szprivate/agentY.git
cd agentY
```

### 2. Run the installer (recommended)

```powershell
.\install_agent.ps1     # Windows
```
```bash
./install_agent.sh      # macOS
```

Same seven stages either way, and a test keeps them in step by comparing the two
scripts' stages, flags and prompts. The installer sets up the **whole stack** in
one pass:

1. checks for `git` + `uv`;
2. clones the sibling repos it needs — **agenty_core** (required) and **agentY-mcp** (optional) — next to `agentY` if they aren't there already, and fast-forwards them if they are;
3. creates agentY's `.venv` (via `uv`), sorts out **torch** — on Windows it offers the CUDA build when it sees an NVIDIA GPU, because the wheel on PyPI is CPU-only and that makes SAM3 grounding take about a minute a call; on a Mac the PyPI wheel already carries Metal, so it just reports whether MPS was found — and installs `requirements.txt` (which pulls in `agenty_core` editable);
4. copies `.env_example` → `.env` and **prompts** you for `HF_TOKEN`, `ANTHROPIC_API_KEY`, and the optional `COMFYUI_API_KEY` / `DASHSCOPE_API_KEY` (Enter keeps an existing value);
5. **finds your ComfyUI** (auto-detects common paths, otherwise asks) and clones **agentY-comfyuiConnect** into its `custom_nodes/`, optionally pointing `settings.local.json` at your ComfyUI URL;
6. sets up **agentY-mcp**'s own venv + `.env` and reuses the tokens you just entered;
7. **checks the result** — every dependency agentY names is import-tested in the venv that will run it, and anything missing is listed with what it costs.

Useful flags:

```powershell
.\install_agent.ps1 -ComfyUIPath "D:\ai\ComfyUI"   # skip ComfyUI auto-detection
.\install_agent.ps1 -SkipMcp                        # don't set up agentY-mcp
.\install_agent.ps1 -SkipComfyNode                  # headless host only, no ComfyUI node
.\install_agent.ps1 -NonInteractive                 # no prompts (CI / re-runs)
.\install_agent.ps1 -SkipTorch                      # don't offer the CUDA torch build
.\install_agent.ps1 -TorchIndexUrl "https://download.pytorch.org/whl/cu126"
.\install_agent.ps1 -Help
```
```bash
./install_agent.sh --comfyui-path ~/ComfyUI
./install_agent.sh --skip-mcp
./install_agent.sh --skip-comfy-node
./install_agent.sh --non-interactive
./install_agent.sh --help
```

There is no `--skip-torch` on macOS, and nothing is missing: the PyPI wheel is
already the right build there, so there is no second one to decline.

That last step is also a standalone command, worth running whenever a feature is
mysteriously doing nothing: most of these packages are somebody else's dependency
too, so a gap only shows up on the machine that resolved differently.

```powershell
.venv\Scripts\python.exe scripts\check_env.py        # full report
.venv\Scripts\python.exe scripts\check_env.py --gpu  # + is torch actually on CUDA?
```
```bash
.venv/bin/python scripts/check_env.py --gpu           # macOS: reports MPS
```

The launcher runs it quietly on every start and speaks up only when something
required is missing.

<details>
<summary><b>Manual setup</b> (instead of the installer)</summary>

```powershell
# agenty_core must sit next to agentY (requirements.txt installs it editable)
git clone https://github.com/szprivate/agenty_core.git ..\agenty_core

# agentY itself. --python names the interpreter on purpose: with a conda env
# active (miniconda auto-activates `base`), uv installs into that one instead.
uv venv .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128     # NVIDIA GPUs; skip on CPU
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py              # confirm it all imports
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
# AGENTY_UI_PORT=5000        # 5001 on macOS; unset = whatever config/ says
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
- the **agentY** node category with **`agentY hook`**, **`agentY qa briefing`** and **`agentY python`** (see [Canvas nodes](#canvas-nodes));
- an **Open agentY Settings…** entry in ComfyUI's Settings panel — the one door to
  everything else (auth keys, model tiers, MCP servers, pricing, and the log /
  memory / token-usage viewers).

Keeping it current is automatic: every launcher start fast-forwards agentY,
`agenty_core` **and** this extension (see [Staying current](#staying-current)).

### Staying current

The launcher checks each remote at startup and fast-forwards the three repos
that make up agentY. It never touches a checkout with uncommitted changes or
unpushed commits, it is `--ff-only` (no merge, rebase or reset), and being offline
is not an error — whatever it declines to do, it says so. `requirements.txt`
changes trigger a reinstall before the app starts.

Opt out with `auto_update = false` in settings, `-NoUpdate` / `--no-update`, or
`AGENTY_NO_UPDATE=1`. Set `comfyui_dir` if your ComfyUI isn't next to agentY.

### Ports

The chat host serves on **5000**, and on **5001 on macOS**. The split is not a
preference: on a stock Mac, ControlCenter's AirPlay Receiver already listens on
`*:5000` (and `*:7000`), and it *answers* — a `403` from `Server: AirTunes/...`
rather than a refused connection — so a sidebar pointed at 5000 there reports the
host as down while the host is running perfectly well beside it.

You do not have to keep the two ends in step. The host registers the port it
actually bound with the ComfyUI extension on every start, and the sidebar asks for
it, so `--port`, `AGENTY_UI_PORT` and `agent_server_url` all reach the panel
without anything being configured twice.

To pin a port yourself, set `agent_server_url` in `config/settings.local.json` (or
in the panel's own settings dialog) — a value you chose wins on every platform.
`--port` on the launcher overrides even that, for one run.

If you would rather have 5000 back on a Mac, turn off **System Settings → General
→ AirDrop & Handoff → AirPlay Receiver** and set the port explicitly.

### OpenMP on macOS

If the agent host dies with `Abort trap: 6` and this above it:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

then two OpenMP runtimes are loaded. torch and faiss-cpu each bundle their own
`libomp.dylib` in their macOS wheels, and the host imports both — torch for SAM3
grounding, faiss for the memory index. The second one to *run OpenMP work* aborts
the process. It depends on which library reaches OpenMP first at runtime rather
than on import order, so it presents as an intermittent crash.

`./install_agent.sh` and `./run_agent.sh` fix this by leaving one runtime:
faiss's copy becomes a symlink to torch's (both are LLVM libomp at the same ABI;
the original is kept as `libomp.dylib.orig`). Reinstalling faiss restores its own
copy, which is why both scripts re-apply it. `scripts/check_env.py` reports the
condition if it ever comes back.

`KMP_DUPLICATE_LIB_OK=TRUE` also silences it and is deliberately **not** used
here — its own error text warns it "may cause crashes or silently produce
incorrect results", and wrong vector-search answers nobody notices are worse than
a crash somebody does.

**Windows is not affected.** There the two ship different implementations —
torch `libiomp5md.dll` (Intel), faiss-cpu `vcomp140.dll` (Microsoft) — and
neither trips the other's duplicate check.

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
# Default — backend on the port config/settings.default.toml names
# (5000, or 5001 on macOS, where AirPlay Receiver holds 5000)
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

The same switches on macOS, spelled the way a shell spells them:

```bash
./run_agent.sh
./run_agent.sh --port 5001
./run_agent.sh --host 0.0.0.0
./run_agent.sh --llm-query-templates "ollama,qwen3-coder:32b"
./run_agent.sh --llm-assemble-workflow "claude,claude-sonnet-4-5"
./run_agent.sh --help
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

`agentY-comfyuiConnect` adds a set of nodes under the **agentY** category, so you can drive the agent *from the graph itself*:

- **`agentY hook`** — an instruction attached to the canvas. Wire any node's output into its **auto-growing `anchor` input(s)** and type a directive. Seven purposes:
  - *inline_parameter* — annotate an existing node ("sweep the seed 6×", "iterate the files in this folder"); the agent expands and runs your on-canvas graph.
  - *make_workflow* — the agent generates and runs a workflow (or Python script) from the prompt, using any wired input.
  - *text* — the agent writes a string answer and drops a wireable `agentY text` node carrying it.
  - *general_request* — free-form: the agent decides the action itself (answer, generate, run, compute).
  - *iterate* — an interactive **refinement loop**, one generation per turn, each result fed back into the wired `LoadImage`.
  - *qa* — your **quality briefing**: the directive is the checklist, the anchors are reference images. See [Checking outputs](docs/using-agentY.md#checking-outputs-qa).
  - *review* — a deliberate **stop** between stages, so you pick what continues. See [Review](docs/using-agentY.md#review-stop-and-pick-what-continues).

  Its `out` **output** is type-agnostic, so one hook can gather several inputs and produce an image, a video **or a scalar** for the next one; wire hooks output→input to build a **multi-step chain**. One `remember` switch decides whether what a hook produced outlives the run — [baked into a subgraph](docs/using-agentY.md#the-keep-switch-should-this-outlive-the-run) for `make_workflow`, memorized for the rest. A hook is inert on a normal *Queue Prompt*, so it never affects a manual run, and **Bypass** (`Ctrl+B`) or mute disables one without deleting it.

- **`agentY qa briefing`** — the checkable half of a briefing as controls rather than prose: aspect ratio, minimum resolution, sharpness, grain, clipping, black and frozen frames, **likeness** against the images wired into `reference`, and a retry budget. Each is settled by measuring the finished file; `notes` carries what needs judgement. Anything left on `any` is not checked, and an empty node enforces nothing. Wire `out` into a hook to judge only that stage. See [the qa briefing node](docs/using-agentY.md#ticking-the-boxes-the-agenty-qa-briefing-node).

- **`agentY python`** — runs an agent-authored Python snippet as a node, so a value computed at runtime becomes a genuine re-runnable output. ⚠️ **It executes arbitrary Python whenever the graph runs** — meant for your own, self-hosted, agent-built workflows; don't run baked workflows from untrusted sources. `AGENTY_PYTHON_NODE_DISABLED=1` makes it a no-op.

Also: **`agentY collector`** (hand the agent a batch of on-disk files), **`agentY add tag`** (name a reference so `#hero_face` resolves), **`agentY load item`** and **`agentY expand image batch`** — see [the guide](docs/using-agentY.md#the-agenty-python-node--collectors).

**Bake a chain into subgraphs.** Turn `remember` on for your make_workflow hooks and each stage's generated workflow is nested into a ComfyUI **subgraph** (inputs/outputs matching the hook's slots), **added** beside the hook nodes — nothing is removed — and wired to mirror the chain. The result is a native workflow you can re-run **without the agent**.

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
- [**Slack**](docs/slack.md) — setting up the optional Slack bridge: creating the
  app, the scopes it needs, the allow-list, and what a turn looks like once it
  reaches a DM.

---

## License

MIT
