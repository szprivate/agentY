# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it supports Claude and Ollama as LLM backends and integrates with Slack as a conversational interface.

---

## Features

- **Natural language → ComfyUI workflow**: Describe what you want; the pipeline builds and queues the workflow automatically.
- **Image & video generation**: Supports Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing**: Reference-based editing, inpainting, upscaling, and more.
- **Hugging Face model management**: Search, check local availability, and download models on demand.
- **Slack integration**: Runs as a Slack bot — send a DM and get images/videos back directly in the channel.
- **Multiple LLM backends**: Claude and Ollama supported; configurable per agent stage.
- **Workflow templates**: 50+ pre-built templates from Comfy-Org as a starting point.
- **Single source of truth for models**: `config/models.json` drives the model table injected into every agent's system prompt automatically.
- **Two-agent pipeline (very experimental)**: A lightweight Researcher (Ollama by default) resolves templates, model paths, and sampler settings; the Brain (Claude by default) does the high-value work — workflow assembly, node wiring, execution, and vision QA.

---

## Requirements

- **Python 3.11+**
- A running **ComfyUI** instance (default: `http://127.0.0.1:8188`)
- An **Anthropic API key** (for the default Claude backend) _or_ a local **Ollama** installation
- (Optional) Slack app credentials for the Slack bot interface
- (Optional) [ngrok](https://ngrok.com/) for exposing the Slack event listener

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/szprivate/agentY.git
cd agentY
```

### 2. Create a virtual environment and install dependencies

**Using `uv` (recommended):**

```bash
uv venv
uv pip install -r requirements.txt
```

**Using standard `pip`:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example below into a `.env` file in the project root and fill in your values:

```dotenv
# -- Anthropic (Brain agent, or single-agent mode) -----------------------
ANTHROPIC_API_KEY=sk-ant-...          # Required when using any Claude backend

# -- Ollama (Researcher agent by default) --------------------------------
OLLAMA_HOST=http://localhost:11434    # Optional – defaults to localhost

# -- Hugging Face --------------------------------------------------------
HF_TOKEN=hf_...                       # Required for gated model downloads

# -- ComfyUI API (Comfy.org cloud, optional) -----------------------------
API_KEY_COMFY_ORG=                    # Optional – leave empty for local access

# -- Slack (optional) ----------------------------------------------------
SLACK_BOT_TOKEN=xoxb-...             # Bot User OAuth Token
SLACK_MEMBER_ID=U0123456789          # Your Slack member ID
SLACK_SIGNING_SECRET=...             # Signing secret for request verification
NGROK_AUTH_TOKEN=...                 # ngrok auth token for the public tunnel
```

All LLM settings (models, mode, history window) can also be set in `config/settings.json` under the `"llm"` key — env vars always take precedence.

### 4. Configure ComfyUI connection

Edit `config/settings.json` to point to your ComfyUI instance if it is not running on the default address:

```json
{
  "comfyui_url": "http://127.0.0.1:8188"
}
```

---

## Usage

### Interactive CLI

```bash
# Default: pipeline mode (Researcher=Ollama, Brain=Claude)
python -m src.main

# Override individual pipeline agents
python -m src.main --researcher-ollama-model qwen3-coder:32b
python -m src.main --brain-anthropic-model claude-sonnet-4-5

# Run both pipeline agents on Ollama (fully local)
python -m src.main --researcher-llm ollama --brain-llm ollama --brain-ollama-model qwen3-vl:30b

# Legacy single-agent mode
python -m src.main --mode single
python -m src.main --mode single --llm claude
python -m src.main --mode single --llm ollama --ollama-model llama3.2
```

Type messages at the `You:` prompt. Type `quit` or `exit` to stop.

### PowerShell helper script (Windows)

The script automatically creates the virtual environment and installs dependencies on first run:

```powershell
# Default pipeline mode
.\run_agent.ps1

# Override Researcher model
.\run_agent.ps1 -ResearcherOllamaModel qwen3-coder:32b

# Override Brain model
.\run_agent.ps1 -BrainAnthropicModel claude-sonnet-4-5

# Legacy single-agent
.\run_agent.ps1 -Mode single -OllamaModel llama3.2
```

### Slack bot

When `SLACK_BOT_TOKEN` and `SLACK_MEMBER_ID` are set, the agent starts a Flask server on port `3000` (configurable via `SLACK_SERVER_PORT`) and exposes it via ngrok. Direct-message the bot in Slack to interact with it — generated images and videos are uploaded back to the conversation automatically.

---

## Slack App Setup

Follow these steps to create and configure the Slack app that agentY uses as its conversational interface.

### 1. Create a Slack app

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps) and click **Create New App → From scratch**.
2. Give it a name (e.g. `agentY`) and choose your workspace.

### 2. Add Bot Token Scopes

Navigate to **OAuth & Permissions → Scopes → Bot Token Scopes** and add:

| Scope | Purpose |
|-------|---------|
| `chat:write` | Post and update messages |
| `im:history` | Read DM message history |
| `im:read` | List DM conversations |
| `im:write` | Open DM channels |
| `files:read` | Download files shared by users |
| `files:write` | Upload generated images / videos |

### 3. Install the app to your workspace

Click **Install to Workspace** (or **Reinstall** if you change scopes later) and approve the permissions. Copy the **Bot User OAuth Token** (`xoxb-...`) shown on the OAuth & Permissions page.

### 4. Get the Signing Secret

Go to **Basic Information → App Credentials** and copy the **Signing Secret**.

### 5. Find your Slack Member ID

In Slack, click your profile picture → **Profile → ⋮ (More) → Copy member ID**. This is your `SLACK_MEMBER_ID` (`U0123456789` format).

### 6. Set up ngrok

agentY uses [ngrok](https://ngrok.com/) to expose the local Flask server so Slack can deliver events.

1. [Download and install ngrok](https://ngrok.com/download).
2. Sign up for a free ngrok account and copy your **Auth Token** from [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).
3. Add it to your `.env` as `NGROK_AUTH_TOKEN`.

> Without `NGROK_AUTH_TOKEN` a random URL is still generated on each run, but it changes every restart. A paid ngrok plan lets you reserve a static domain so the Request URL never changes.

### 7. Add credentials to `.env`

```dotenv
SLACK_BOT_TOKEN=xoxb-...          # Bot User OAuth Token from step 3
SLACK_SIGNING_SECRET=...          # Signing Secret from step 4
SLACK_MEMBER_ID=U0123456789       # Your personal member ID from step 5
NGROK_AUTH_TOKEN=...              # ngrok auth token from step 6
```

### 8. Start the agent

```bash
python -m src.main
```

When `SLACK_BOT_TOKEN` and `SLACK_MEMBER_ID` are present, the agent automatically starts the Flask server on port `3000` and opens an ngrok tunnel. The public Request URL is printed to the console:

```
============================================================
  SLACK EVENT SUBSCRIPTIONS - Request URL
============================================================
  https://xxxx.ngrok-free.app/slack/events
============================================================
```

### 9. Configure Event Subscriptions in Slack

1. In your app's settings go to **Event Subscriptions** and toggle **Enable Events** on.
2. Paste the URL printed in the previous step into the **Request URL** field. Slack will immediately send a verification challenge — the agent responds automatically, and the field turns green.
3. Under **Subscribe to bot events** add `message.im`.
4. Click **Save Changes** and reinstall the app if prompted.

### 10. Send the bot a DM

Find the bot in Slack (search for its name), open a direct message, and send any request — for example:

> _"Generate a photorealistic image of a red panda in a forest"_

Generated images and videos are uploaded back to the conversation automatically.

---

## Project Structure

```
agentY/
├── src/
│   ├── main.py              # Entry point, CLI arg parsing, Slack server bootstrap
│   ├── agent.py             # Agent factories (Researcher, Brain, single), LLM config
│   ├── pipeline.py          # Two-agent Pipeline: Researcher → Brain handoff
│   ├── comfyui_client.py    # HTTP client for the ComfyUI REST API
│   ├── slack_server.py      # Flask + ngrok Slack Events API handler
│   └── tools/               # Agent tool implementations
│       ├── execution.py     # Workflow execution & status polling
│       ├── history.py       # ComfyUI history / results
│       ├── huggingface.py   # HF model search & download
│       ├── models.py        # Local model enumeration
│       ├── prompt.py        # Prompt helpers
│       ├── queue.py         # Queue management
│       ├── slack_tools.py   # Slack messaging & file upload
│       ├── system.py        # System / server info
│       ├── upload.py        # Image upload to ComfyUI
│       ├── view.py          # Image/video download & preview
│       ├── workflow_builder.py  # Dynamic workflow construction
│       └── workflows.py     # Workflow template loading
├── comfyui_workflows/       # Custom workflow JSON files
├── comfyui_workflow_templates_official/  # Comfy-Org template library
├── config/
│   ├── settings.json              # ComfyUI URL, LLM defaults
│   ├── models.json                # Model shortname → path table (injected into all prompts)
│   ├── system_prompt.researcher.md  # Researcher system prompt
│   ├── system_prompt.brain.md       # Brain system prompt
│   ├── system_prompt.claude.md      # Single-agent Claude prompt
│   └── system_prompt.qwencode.md    # Single-agent Ollama prompt
├── output/                  # Downloaded outputs (view_image save_to target)
├── requirements.txt
├── run_agent.ps1            # Windows launcher script
└── .env                     # Local secrets (never commit this)
```

---

## LLM Backends

### Pipeline mode (default)

Two agents run in sequence. Each can use a different LLM backend:

| Agent | Role | Default backend | Default model |
|-------|------|-----------------|---------------|
| Researcher | Template/model/sampler resolution | Ollama | `qwen3-coder:32b` |
| Brain | Workflow assembly, execution, QA | Claude | `claude-sonnet-4-5` |

Override via CLI flags or `config/settings.json`:

```bash
# Use a different Researcher model
python -m src.main --researcher-ollama-model qwen3-coder:7b

# Use a stronger Brain model
python -m src.main --brain-anthropic-model claude-sonnet-4-5

# Run the Brain on Ollama instead of Claude
python -m src.main --brain-llm ollama --brain-ollama-model qwen3-vl:30b
```

Or set defaults in `config/settings.json`:

```json
"llm": {
  "pipeline": {
    "researcher_ollama_model": "qwen3-coder:32b",
    "brain_anthropic_model": "claude-sonnet-4-5"
  }
}
```

### Single-agent mode (legacy)

One model handles everything. Activate with `--mode single`:

```bash
python -m src.main --mode single              # Claude
python -m src.main --mode single --llm ollama # Ollama
```

Or set `AGENT_MODE=single` / `"agent_mode": "single"` in `settings.json`.

---

## Configuration Reference

All LLM settings can be set either as environment variables or in `config/settings.json` under the `"llm"` key. **Env vars always win.**

### Pipeline agents

| Env var | settings.json key | Default | Description |
|---------|-------------------|---------|-------------|
| `AGENT_MODE` | `agent_mode` | `pipeline` | `pipeline` or `single` |
| `RESEARCHER_LLM` | `pipeline.researcher_llm` | `ollama` | Researcher backend |
| `RESEARCHER_OLLAMA_MODEL` | `pipeline.researcher_ollama_model` | `qwen3-coder:32b` | Researcher Ollama model |
| `RESEARCHER_ANTHROPIC_MODEL` | `pipeline.researcher_anthropic_model` | `claude-haiku-4-5` | Researcher Anthropic model |
| `BRAIN_LLM` | `pipeline.brain_llm` | `claude` | Brain backend |
| `BRAIN_ANTHROPIC_MODEL` | `pipeline.brain_anthropic_model` | `claude-sonnet-4-5` | Brain Anthropic model |
| `BRAIN_OLLAMA_MODEL` | `pipeline.brain_ollama_model` | `qwen3-vl:30b` | Brain Ollama model |

### Single-agent mode

| Env var | settings.json key | Default | Description |
|---------|-------------------|---------|-------------|
| `AGENT_LLM` | `single_agent.llm` | `claude` | LLM backend |
| `OLLAMA_MODEL` | `single_agent.ollama_model` | `qwen3-vl:30b` | Ollama model |

### Shared LLM settings

| Env var | settings.json key | Default | Description |
|---------|-------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | — | **Required** for any Claude backend |
| `ANTHROPIC_MODEL` | `anthropic.model` | `claude-haiku-4-5` | Fallback Anthropic model |
| `ANTHROPIC_MAX_TOKENS` | `anthropic.max_tokens` | `4096` | Max tokens per response |
| `OLLAMA_HOST` | `ollama.host` | `http://localhost:11434` | Ollama server URL |
| `AGENT_HISTORY_WINDOW` | `history_window` | `40` | Sliding conversation window |

### Services

| Env var | Default | Description |
|---------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token for gated model downloads |
| `API_KEY_COMFY_ORG` | — | Comfy.org cloud API key |
| `SLACK_BOT_TOKEN` | — | Slack Bot User OAuth Token |
| `SLACK_MEMBER_ID` | — | Your Slack member ID |
| `SLACK_SIGNING_SECRET` | — | Slack request signing secret |
| `NGROK_AUTH_TOKEN` | — | ngrok auth token |
| `SLACK_SERVER_PORT` | `3000` | Port for the Slack event server |

---

## License

MIT
