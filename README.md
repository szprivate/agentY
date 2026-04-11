# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it supports Claude and Ollama as LLM backends and integrates with Slack as a conversational interface.

---

## Features

- **Natural language → ComfyUI workflow** — describe what you want; the pipeline builds, submits, and QA-checks the workflow automatically.
- **Image & video generation** — Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing** — reference-based editing, inpainting, upscaling, and more.
- **Two-agent pipeline** — a lightweight Researcher (Ollama by default) resolves templates, model paths, and sampler settings; the Brain (Claude by default) assembles the workflow, executes it, and runs vision QA.
- **Hugging Face model management** — search, check local availability, and download models on demand.
- **Slack integration** — runs as a Slack bot via Socket Mode; send a DM and get images/videos back directly.
- **Multiple LLM backends** — Claude and Ollama, configurable per pipeline stage.
- **50+ workflow templates** — from Comfy-Org, loaded and patched automatically.
- **Skills system** — drop shell/Python scripts into `skills/` and they become agent-callable tools.
- **ComfyUI extension** — a companion custom node ([agentY-comfyui-extension](https://github.com/szprivate/agentY-comfyui-extension)) lets you send images directly from ComfyUI to agentY and receive responses in real time.

---

## Requirements

- **Python 3.11+**
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- An **Anthropic API key** (for Claude) _and/or_ a local **Ollama** installation
- (Optional) Slack app credentials for the bot interface

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

### 3. Configure secrets

Copy the example env file and fill in your values:

```bash
cp .env_example .env
```

Edit `.env`:

```dotenv
HF_TOKEN=hf_...
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=
ANTHROPIC_API_KEY=sk-ant-...
COMFYUI_API_KEY=comfyui-...
```

### 4. Configure defaults

Edit `config/settings.json` to point to your ComfyUI instance and set default LLMs:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",

  "llm": {
    "pipeline": {
      "researcher": "ollama,qwen3.5:9b",
      "brain": "claude,claude-haiku-4-5"
    }
  }
}
```

The `"researcher"` and `"brain"` values use the format `"provider,model"`.

---

## Usage

### PowerShell (Windows)

The script creates the virtual environment and installs dependencies automatically on first run.

```powershell
# Default — uses LLMs from settings.json
.\run_agent.ps1

# Override the Researcher
.\run_agent.ps1 -LlmResearcher "ollama,qwen3-coder:32b"

# Override the Brain
.\run_agent.ps1 -LlmBrain "claude,claude-sonnet-4-5"

# Skip the Brain stage (return Researcher output directly)
.\run_agent.ps1 -SkipBrain

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

### Two-agent pipeline

```
User request
    │
    ▼
┌──────────┐   BrainBriefing JSON   ┌───────┐   workflow   ┌─────────┐
│Researcher├────────────────────────►│ Brain ├─────────────►│ ComfyUI │
└──────────┘                        └───┬───┘              └────┬────┘
  Ollama (default)                      │ Claude (default)      │
  • resolve template                    │ • assemble workflow   │
  • resolve model paths                 │ • patch & validate    │
  • resolve sampler settings            │ • submit & poll       │
  • produce BrainBriefing               │ • vision QA           │
                                        │ • deliver to Slack    │
                                        ▼                      ▼
                                     output/              Slack DM
```

1. **Researcher** receives the user request and produces a validated **BrainBriefing** JSON (template, input images, model paths, prompts, resolution).
2. **Brain** receives the BrainBriefing, loads the selected workflow template, patches node values, submits the prompt to ComfyUI, waits for completion, runs vision QA on the output, and delivers the result.

The `--skip-brain` flag returns the Researcher's BrainBriefing directly, useful for debugging or inspection.

### LLM configuration priority

Each value is resolved in order — first match wins:

1. CLI flag (`-LlmResearcher` / `--researcher-llm`)
2. Environment variable
3. `config/settings.json`
4. Hard-coded default

---

## Slack Integration

agentY uses **Slack Socket Mode** (WebSocket-based, no public HTTP endpoint needed).

### Prerequisites

| Credential | Where to get it |
|---|---|
| `SLACK_BOT_TOKEN` (xoxb-…) | App → OAuth & Permissions → Bot User OAuth Token |
| `SLACK_APP_TOKEN` (xapp-…) | App → Basic Information → App-Level Tokens (scope: `connections:write`) |

### Slack app setup

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps) → **Create New App → From scratch**.
2. **OAuth & Permissions → Bot Token Scopes** — add: `chat:write`, `im:history`, `im:read`, `im:write`, `files:read`, `files:write`.
3. **Basic Information → App-Level Tokens** — generate a token with the `connections:write` scope.
4. **Socket Mode** — enable it.
5. **Event Subscriptions** — enable, subscribe to `message.im` under **bot events**.
6. **Install to Workspace** and copy the Bot token.
7. Add `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` to your `.env`.
8. Start the agent — Socket Mode activates automatically when both tokens are present.

---

## Project Structure

```
agentY/
├── src/
│   ├── main.py                 Entry point and CLI
│   ├── agent.py                Agent factories, LLM config, system prompt loading
│   ├── pipeline.py             Researcher → Brain pipeline and BrainBriefing schema
│   ├── tools/
│   │   ├── comfyui.py          Workflow template loading/patching, node inspection, prompt submission
│   │   ├── file_tools.py       Plain-text file reader
│   │   ├── huggingface.py      HF Hub: model search, info, local check, download
│   │   ├── image_handling.py   Image upload/download, resolution detection, visual analysis
│   │   ├── shell.py            Cross-platform shell execution for skill scripts
│   │   └── slack_tools.py      Slack messaging and file upload
│   └── utils/
│       ├── comfyui_client.py   Singleton HTTP client for the ComfyUI REST API
│       ├── comfyui_interrupt_hook.py  Halts agent loop after submit_prompt for async polling
│       ├── secrets.py          Reads .env via dotenv_values (never injects into os.environ)
│       └── slack_server.py     Slack Socket Mode server
├── config/
│   ├── settings.json           ComfyUI URL, LLM defaults, polling intervals
│   ├── models.json             Model shortname → path table (injected into system prompts)
│   ├── workflow_templates.json Workflow template metadata
│   ├── workflow_types.json     Task type definitions
│   ├── system_prompt.*.md      System prompt templates (filenames configurable via the
│   │                            `system_prompts` mapping in `config/settings.json`)
│   └── brainbrief_example.json Example BrainBriefing for prompt injection
├── comfyui_workflows/          Custom workflow JSON files
├── comfyui_workflow_templates_official/  Comfy-Org template library (git-ignored)
├── skills/                     Drop-in skill scripts (shell/Python)
├── output/                     Generated outputs
├── .env_example                Template for .env secrets
├── requirements.txt
└── run_agent.ps1               Windows launcher script
```

> The ComfyUI custom node lives in its own repo: **[agentY-comfyui-extension](https://github.com/szprivate/agentY-comfyui-extension)**.
> Clone it into `ComfyUI/custom_nodes/agentY_bridge` to get the Send to agentY node.

---

## License

MIT
