![agentY sidebar chat next to the ComfyUI graph](docs/images/agenty_logo_small.png)

agentY ([aˈɡɛnt  ʏpsilɔn]) is an AI agent harness designed to assist with Visual Effects (VFX) work and image generation.

**Interface & Integration:**
- Operates via its own side panel inside ComfyUI or through external applications like Claude Desktop.
- Includes a detachable image, video, and 3D viewer directly within ComfyUI for real-time visual feedback.

**Core Capabilities:**
- Builds and drives ComfyUI workflows and multi-step image generation pipelines.
- Controls 3D and VFX software (Blender, Maya, Houdini, Nuke) via Model Context Protocol (MCP) connections.
- Creates motion graphics using Remotion and edits video content.
- Trains LoRAs and Gaussian Splats through 3rd-party applications.
- Generates entire 3D worlds from single images using WorldLabs Astra.

**Extensibility:**
- Easily install MCPs using preset settings or by instructing the agent directly.

**LLM Compatibility:** 
- Supports a wide range of providers, including Claude, GPT, Qwen, DeepSeek, and local inference options like Ollama and Strata.

> 📖 **New here?** The [**Using agentY guide**](docs/using-agentY.md) is a screenshot-driven tour of everything below, and the [**video tutorials**](docs/tutorials.md) show the main features in about a minute each.


## Installation

### 1. Clone the repository

```bash
git clone -b stable https://github.com/szprivate/agentY.git
cd agentY
```

### 2. Run the installer

```powershell
.\install_agent.ps1     # Windows
```
```bash
./install_agent.sh      # macOS
```


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


## Uninstalling

An install only creates folders (no registry entries, services or shortcuts), so the
uninstaller only removes folders, and only ones it has identified as agentY's (git origin
`github.com/szprivate/<repo>`, or agentY marker files). It shows a plan first, with sizes
and what it keeps, and asks before each group:

```powershell
.\uninstall_agent.ps1 -DryRun        # Windows: only print the plan and the checks
.\uninstall_agent.ps1                # ... then run it for real
```
```bash
./uninstall_agent.sh --dry-run       # macOS
./uninstall_agent.sh
```

- **Removed (default yes):** the running host, agentY + agentY-core + agentY-mcp (code and venvs), the
  agentY panel in ComfyUI's `custom_nodes`.
- **Asked separately (default no):** your data (conversations, memory, settings, `.env` keys, MCP
  tokens, output), with a backup zip next to the folder offered first; the bEpic viewer, which works
  without agentY; `winget uninstall` of ComfyUI Desktop.
- **Never removed:** ComfyUI itself, models, Ollama, Strata, Blender, and anything it cannot verify.
  A folder with an agentY name but a foreign git origin is listed and left alone.
- **Printed as hints only:** the `agentY` entry in Claude Desktop's config, `uv cache clean`, and the
  Deadline plugin (`python scripts/install_deadline_plugin.py --uninstall`, run it **before** agentY
  is deleted).
- `-Yes` / `--yes` skips the questions: it removes the first group, your data after a backup, and the
  panel, and keeps the viewer (`-IncludeViewer`, `-KeepUserData`, `-NoBackup` adjust that).
  If something is running from these folders (ComfyUI, Claude Desktop, the agent window) it says so
  and removes nothing - close it and run it again.

It also works on **older installs** (`agenty_core`, a copied `comfyui-agent-canvas` panel, the
Chainlit data layer) and **without the agentY folder**: pass `-Path` / `--path`, or run it from
inside the install and it finds it. To fetch and run it on its own (dry run first):

```powershell
$u = "https://raw.githubusercontent.com/szprivate/agentY/stable/uninstall_agent.ps1"
& ([scriptblock]::Create((irm $u))) -DryRun -Path D:\AI\agentY
```
```bash
curl -fsSL https://raw.githubusercontent.com/szprivate/agentY/stable/uninstall_agent.sh -o /tmp/uninstall_agent.sh
bash /tmp/uninstall_agent.sh --dry-run --path ~/agentY
```

(The `stable` copy exists from the first release that contains the uninstaller; until then use
`dev` in the URL.)

## Usage

Start the agent:

```powershell
.\run_agent.ps1        # Windows
```
```bash
./run_agent.sh         # macOS
```

Then open **ComfyUI**, click the **agentY** tab in the left sidebar, and chat:

- *"Generate a cinematic wide shot of Tokyo at night."*
- *"Edit this photo to make it daytime."* (attach an image with 📎)
- *"Make 5 variations of a red sports car, different angles."*
- *"Upscale the last image with UltimateSD."*

Type `/` for the slash commands. Settings live in ComfyUI's **Settings → agentY → Open agentY Settings…**. Every start also updates agentY; `-Help` / `--help` lists the launcher's options.


## Features

- **Text → workflow → result** — describe it; the agent picks a template, builds the graph, runs it and puts the output on your canvas.
- **Learns workflows from the web** — for something it doesn't know yet, it searches the web and builds the workflow from what it finds.
- **Works with every LLM** — Claude, GPT, Gemini, Qwen (DashScope) or local Ollama models, mixed per role.
- **Image & video** — Flux, WAN, Qwen, HunyuanVideo and more: generate, edit, inpaint, upscale.
- **Hook nodes** — put instructions on the canvas ("sweep the seed 6×"), chain them into pipelines, bake a chain into plain ComfyUI subgraphs.
- **Inspect-and-improve loops** — the agent judges each output against your condition, changes one thing and runs again, within a budget you set.
- **Output QA** — a briefing per run or per stage; ratio, resolution, sharpness and likeness are measured, the rest is judged, misses are re-generated.
- **Review stops** — pause a chain, pick the outputs that continue; your picks train a ranking score.
- **Parallel agents** — several conversations at once, or one agent per shot of a sequence.
- **Agent spawning** — it starts sub-agents and specialists (vision, video, coder, web search) when a job calls for one.
- **Tagging system** — name a reference once (`#hero_face`) and use it in any prompt or hook.
- **Memory** — long-term memory across sessions, per-project notes, and chat history that survives restarts.
- **Slack support** — follow and drive the same conversation from a DM (optional, off by default).
- **3D apps** — builds models in Blender, Maya and Houdini (photo to model, destruction sims) and checks proportions against the photo.
- **MCP, both ways** — uses tools from external MCP servers (Magnific, Blender, Nuke, After Effects, …; presets in the installer and Settings) and ships its own MCP server for Claude Desktop ([agentY-mcp](https://github.com/szprivate/agentY-mcp)).
- **Sees your canvas** — reads and edits the open graph, takes screenshots of it, fetches web reference images onto it.
- **Model management** — finds, checks and downloads the model files a workflow needs from Hugging Face.
- **Your own templates and skills** — register any workflow as a template; save a procedure as a skill.
- **Settings in ComfyUI** — keys, model choices, MCP servers and token cost per model, all in one panel.


## Requirements

- **Windows**, or **macOS 14+ on Apple Silicon** (on a Mac run `xcode-select --install` first)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** and **git** on your PATH
- A running **ComfyUI** (default: `http://127.0.0.1:8188`)
- At least one LLM: an **Anthropic**, **OpenAI**, **Google Gemini** or **DashScope** API key, or a local **Ollama**
- A **Hugging Face token** (for gated-model downloads)


## Documentation

- [**Using agentY**](docs/using-agentY.md) — the full guide: chat, hooks, QA, settings, MCP, memory, models, troubleshooting.
- [**Video tutorials**](docs/tutorials.md) — seven one-minute videos: install, panel and settings, tags and hooks, Atlas, motion graphics, Gaussian splats, LoRA training.
- [**Execution modes**](docs/execution_modes.md) — local, hybrid or Comfy Cloud: where open-weight models run, what it costs, what the cloud cannot do.
- [**Slack**](docs/setup/slack.md) — setting up the Slack bridge.
- [**Strata**](docs/setup/strata.md) — a local model for the agents, sharing one GPU with ComfyUI.
- [**Reference**](docs/reference.md) — architecture, ports, security, what the agent may run, model tiers, macOS notes.


## License

MIT
