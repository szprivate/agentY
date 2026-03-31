# agentY

An AI agent that constructs and executes [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows through natural language. Built on the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), it supports Claude and Ollama as LLM backends and integrates with Slack as a conversational interface.

---

## Features

- **Natural language → ComfyUI workflow**: Describe what you want; the agent builds and queues the workflow automatically.
- **Image & video generation**: Supports Flux, WAN2.1/2.2, Qwen, HunyuanVideo, and many other models.
- **Image editing**: Reference-based editing, inpainting, upscaling, and more.
- **Hugging Face model management**: Search, check local availability, and download models on demand.
- **Slack integration**: Runs as a Slack bot — send a DM and get images/videos back directly in the channel.
- **Multiple LLM backends**: Claude (default, with prompt-caching) or a local Ollama model.
- **Workflow templates**: 50+ pre-built templates from Comfy-Org as a starting point.

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
# -- LLM -----------------------------------------------------------------
ANTHROPIC_API_KEY=sk-ant-...          # Required when using the Claude backend
ANTHROPIC_MODEL=claude-haiku-4-5      # Optional – override the default model
ANTHROPIC_MAX_TOKENS=4096             # Optional

# -- Ollama (alternative backend) ----------------------------------------
OLLAMA_HOST=http://localhost:11434    # Optional – defaults to localhost
OLLAMA_MODEL=qwen3-vl:30b            # Optional – defaults to qwen3-vl:30b

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
# Default (Claude backend)
python -m src.main

# Explicitly choose a backend
python -m src.main --llm claude
python -m src.main --llm ollama
```

Type messages at the `You:` prompt. Type `quit` or `exit` to stop.

### PowerShell helper script (Windows)

The script automatically creates the virtual environment and installs dependencies on first run:

```powershell
.\run_agent.ps1
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
│   ├── main.py              # Entry point, CLI loop, Slack server bootstrap
│   ├── agent.py             # Strands Agent setup, system prompt, LLM config
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
│       ├── userdata.py      # bEpic path management
│       ├── users.py         # User data helpers
│       ├── view.py          # Image/video download & preview
│       ├── workflow_builder.py  # Dynamic workflow construction
│       └── workflows.py     # Workflow template loading
├── comfyui_workflows/       # Custom workflow JSON files
├── comfyui_workflow_templates_official/  # Comfy-Org template library
├── config/
│   └── settings.json        # ComfyUI URL, output dir, model paths
├── output/                  # Generated files saved here
├── requirements.txt
├── run_agent.ps1            # Windows launcher script
└── .env                     # Local secrets (never commit this)
```

---

## LLM Backends

| Backend | Env var to set | Default model |
|---------|---------------|---------------|
| `claude` (default) | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` |
| `ollama` | `OLLAMA_HOST`, `OLLAMA_MODEL` | `qwen3-vl:30b` |

Switch backends at runtime with `--llm`:

```bash
python -m src.main --llm ollama
```

Or set the default via the environment:

```dotenv
AGENT_LLM=ollama
```

---

## Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✓ (Claude) | — | Anthropic API key |
| `ANTHROPIC_MODEL` | | `claude-haiku-4-5` | Claude model ID |
| `ANTHROPIC_MAX_TOKENS` | | `4096` | Max tokens per response |
| `AGENT_LLM` | | `claude` | Default LLM backend |
| `AGENT_HISTORY_WINDOW` | | `40` | Sliding conversation window size |
| `HF_TOKEN` | | — | Hugging Face token for gated models |
| `API_KEY_COMFY_ORG` | | — | Comfy.org API key |
| `SLACK_BOT_TOKEN` | | — | Slack bot token |
| `SLACK_MEMBER_ID` | | — | Slack member ID |
| `SLACK_SIGNING_SECRET` | | — | Slack signing secret |
| `NGROK_AUTH_TOKEN` | | — | ngrok auth token |
| `SLACK_SERVER_PORT` | | `3000` | Port for the Slack event server |
| `OLLAMA_HOST` | | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | | `qwen3-vl:30b` | Ollama model ID |

---

## License

MIT
