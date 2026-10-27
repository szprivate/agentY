# agentY reference

How the pieces fit, plus ports, security and configuration files. For day-to-day
use see [Using agentY](using-agentY.md).

## The five repos

| Repo | Location | Role |
|---|---|---|
| **agentY** | your working copy | The agent host and pipeline (`run_agent.ps1` / `run_agent.sh`). |
| **[agentY-core](https://github.com/szprivate/agentY-core)** | next to `agentY` | Shared tool layer (ComfyUI, Hugging Face, web, files) and the workflow template corpus. Required. |
| **[agentY-comfyuiConnect](https://github.com/szprivate/agentY-comfyuiConnect)** | `<ComfyUI>/custom_nodes/` | The sidebar chat tab and the canvas nodes. |
| **[agentY-mcp](https://github.com/szprivate/agentY-mcp)** | next to `agentY` | Optional MCP server for Claude Desktop, built on `agentY-core`. |
| **[ComfyUI-ImageViewer](https://github.com/szprivate/ComfyUI-ImageViewer)** | `<ComfyUI>/custom_nodes/` | The bEpic viewer, agentY's standard picture/video viewer. Installed with agentY; released with it. |

## Architecture

```
ComfyUI  (your browser)
  ├─ agentY sidebar tab   ┐
  └─ agentY canvas nodes  ┘── agentY-comfyuiConnect
        │  HTTP + SSE (port 5000, or 5001 on macOS)
        ▼
  agentY host
        │  Orchestrator agent + specialist agents
        │  tools ── ../agentY-core
        ├──HTTP/WS──►  ComfyUI  (run workflows, check outputs, stage results)
        └──►  memory/conversations.sqlite
```

The **Orchestrator** agent owns each turn. It has the tools and calls specialist
agents (template research, workflow building, vision, QA, coder, web search). A
finished workflow goes to ComfyUI; outputs are checked against your QA briefing
and staged onto the graph as loader nodes.

## Releases

agentY is five repositories that only work as a set. A release pins that set:

- every repository gets the tag `v<version>` and a GitHub release;
- each repository's **`main`** (`master` in agentY-core and ComfyUI-ImageViewer) and **`stable`** branch
  are moved to that commit;
- `release.toml` in agentY records the version and each sibling's commit;
- `requirements.lock` records the Python package versions it was tested with.
  The installers and the launcher install with it as constraints.

**Channels.** `update_channel` (Settings ▸ Updates, or `AGENTY_UPDATE_CHANNEL`):

| channel | follows | for |
|---|---|---|
| `stable` (default) | each repository's `stable` branch | everyone using agentY |
| `dev` | each repository's `dev` branch, every commit | working on agentY itself |

**Branches.** New work goes to `dev` first. `main` and `stable` only move when a
release is made, and a release always covers all five repositories under one
version number. The rules for coding agents are in `AGENTS.md` in each repository.

A stable machine never goes backwards: a checkout that is already past the
release is left where it is, and says so. If the repositories are not the
released set, the launcher prints one `[release]` line naming which.

**Rolling back.** `git checkout v1.0.0` in each repository, and start with
`-NoUpdate` / `--no-update`.

**Making a release** (from a `dev` machine, with every repository on its `dev`
branch, pushed, and the tests run):

```powershell
.venv\Scripts\python.exe scripts\make_release.py 1.1.0 --dry-run   # what it would do
.venv\Scripts\python.exe scripts\make_release.py 1.1.0             # do it
.venv\Scripts\python.exe scripts\make_release.py --show            # the current release
.venv\Scripts\python.exe scripts\make_release.py --check           # is this machine on it?
```

It releases each repository at the commit it has checked out, writes
`release.toml` and `requirements.lock`, tags, moves `main` and `stable`, and
creates the GitHub releases (needs the `gh` CLI).

**The Claude Code plugin.** agentY-mcp is also a Claude Code plugin (tools, skills,
agents, commands; audit in `docs/plugin_audit.md` there). Its `skills/` is a mirror of
agentY's `skills/`, made by `scripts/sync_plugin_skills.py` (host-only skills are listed
with a reason in `config/plugin_skills.json`; a shipped skill that names a tool the MCP
server lacks needs a note there, a test checks). A release does this first, for
`--dry-run` too (reported, nothing written): it syncs the skills, sets the plugin,
marketplace and `.mcpb` versions to the release version, refreshes the `.mcpb` tool list,
runs `claude plugin validate`, and commits and pushes that on agentY-mcp's branch before
tagging, so the tagged commit is the plugin as released. It refuses to start when
agentY-mcp has uncommitted changes in those files, and stops if the manifest is invalid.
Between releases, `scripts/sync_plugin_skills.py --check` says whether the plugin is
stale; agentY-mcp's tests fail on drift too.

Users set the plugin up with the installer's optional step "Set up agentY for Claude
Code" (off by default; it runs `claude plugin marketplace add <agentY-mcp>` and
`claude plugin install agenty@agentY`, or prints those two commands when the `claude`
CLI is not on PATH). The marketplace is the local checkout, read in place, so the
launcher's `git pull` of agentY-mcp updates the plugin.

**The tool layer's name.** The repository and folder are `agentY-core`; the
Python package inside is `agenty_core`. An install from before the rename is
moved over on its next start: the folder is renamed and a link is left under the
old name.

## Ports

The host listens on **5000**, and on **5001 on macOS**, where AirPlay Receiver
already holds 5000. The host tells the ComfyUI extension which port it bound, so
the panel finds it without configuration.

To pin a port, set `agent_server_url` in `config/settings.local.json`, set
`AGENTY_UI_PORT`, or pass `-Port` / `--port` to the launcher for one run.

### ComfyUI alive but port 8188 not listening (Windows)

Symptom: the ComfyUI process is running, but nothing connects and `netstat -ano | findstr :8188`
shows no `LISTENING` line. The ComfyUI log has `Accept failed on a socket` right before,
usually with `OSError: [WinError 64]`. Cause: on Windows asyncio's Proactor loop closes the
listening socket for good when one client resets while being accepted (CPython gh-93821); a
storm of aborted connections from browsers or tests is enough. The agentY ComfyUI extension
patches this at startup (`_proactor_accept.py` in agentY-comfyuiConnect): such accept errors
are logged once as `accept ... failed with a transient error ... still listening` and
serving continues. If you see the symptom with the extension installed, ComfyUI has not yet
been restarted with that version. The agentY host on :5000 is Flask/werkzeug and is not affected.

## Security

The host listens on localhost only. Three protections are on by default, all
under `[security]` in the settings:

- **Only the panel may call the host.** Requests need the `Origin` of the page
  ComfyUI served, and the host only answers to an IP address or `localhost`. Add
  your machine's name to `security.allowed_hosts` if you use one.
- **A session token** in `.agenty_token` next to the checkout. ComfyUI hands it to
  the panel, which sends it with every call. It is kept across restarts; delete
  the file to rotate it.
- **API keys are never sent to the browser.** The settings page gets a mask, not
  the values.

The host also warns at startup when a key in `.env` is older than
`security.api_key_max_age_days` (30; `0` turns it off).

Other programs (Claude Code, MCP clients) can start turns on a running host and read
the result back: see [turn-api.md](integrations/turn-api.md).

To run the host on a server for a team (binding beyond loopback, a service, TLS behind a
reverse proxy, a token per person, headless mode, a visible turn queue), see
[server.md](setup/server.md); for Slack channels with several people, [slack.md](setup/slack.md).

## What the agent may run

`run_script` runs **one program, with no shell**: pipes, redirects and `&&` chains
are refused. The program must be on the allowed list (python, ffmpeg/ffprobe, git,
and the usual listing and search tools; extend with
`security.shell_allowed_commands`), and paths must stay inside the project,
ComfyUI's folders or a temp file (`security.shell_extra_roots`).

Python can still do anything Python can do, so three tools **ask you first** in the
panel, showing the exact command: `run_script`, `iterate` and
`install_custom_node`. Answer *allow once* or *allow for this session*.

With no panel open to ask (a Slack turn, an unattended machine) the request is
declined. `security.unattended_tool_policy = "allow"` changes that;
`security.ask_before_tools` is the list of tools that ask.

## Configuration files

| file | holds |
|---|---|
| `.env` | API keys and tokens |
| `config/settings.default.toml` | the committed defaults — don't edit |
| `config/settings.local.json` | your overrides, merged over the defaults (gitignored) |
| `config/mcp.json` | MCP servers (no secrets) |
| `config/pricing.json` | your model prices |
| `config/qa/` | named QA briefings |

Everything in `settings.local.json` can be set from the Settings dialog. By hand,
model choices look like this:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",
  "llm": {
    "tiers": {
      "orchestrator":      "dashscope,qwen3.7-max",
      "research_assembly": "dashscope,qwen3.6-plus",
      "fast_utility":      "dashscope,qwen3.6-flash",
      "vision":            "dashscope,qwen3-vl-flash",
      "qa_judge":          "dashscope,qwen3-vl-plus",
      "coder":             "dashscope,kimi-k2.7-code"
    },
    "pipeline": {
      "video_agent": "dashscope,qwen3-vl-plus"   // per-role override
    }
  }
}
```

Each value is `"provider,model"`: `claude`, `ollama`, `dashscope` (aliases `qwen`,
`modelstudio`, `alibaba`), `openai`, `google` (alias `gemini`).

#### DashScope: key, plan and endpoint must match

Alibaba Model Studio has three kinds of access, each with its **own key and its own base URL**
(a mismatch answers 401). The installer detects which one a key belongs to (the `sk-sp-` prefix,
then one free `GET /models` per documented endpoint) and writes `DASHSCOPE_BASE_URL` to `.env`;
Settings > Connections has a plan/region picker and a Detect button. Non-interactive installs only
detect, else keep the default (Singapore pay-as-you-go).

| Access | Key | Region | `DASHSCOPE_BASE_URL` |
|---|---|---|---|
| Pay-as-you-go | workspace key | Singapore (default) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| | | Beijing | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| | | US (Virginia) | `https://dashscope-us.aliyuncs.com/compatible-mode/v1` |
| | | Hong Kong | `https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1` |
| Token Plan | `sk-sp-...` | Singapore | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| | | Beijing | `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| Coding Plan | `sk-sp-...` | Singapore | `https://coding-intl.dashscope.aliyuncs.com/v1` |
| | | Beijing | `https://coding.dashscope.aliyuncs.com/v1` |

Workspace-dedicated hosts (`https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1`,
also Tokyo and Frankfurt) are pay-as-you-go too; set `DASHSCOPE_BASE_URL` by hand. The table lives
in `agenty_core/dashscope_endpoints.py`; nothing else hardcodes a host.

**Terms caveat.** Alibaba restricts Token Plan and Coding Plan keys to interactive AI coding tools.
Its docs list "automated scripts, application backends calling the API directly" as unsupported
and call use outside that scope "a violation or abuse" (subscription suspension, key ban). agentY is
a backend, so a plan key here may break Alibaba's terms; the installer says so. Also, the plans
document only the chat APIs: **image/video generation** (`dashscope_generate`, the native
`/api/v1`) and **embeddings** (`text-embedding-v4`, long-term memory) need a pay-as-you-go key.
Set it beside the main one:

```
DASHSCOPE_API_KEY=sk-sp-...                # chat models (the plan)
DASHSCOPE_BASE_URL=https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_PAYG_API_KEY=sk-...              # generation + embeddings
DASHSCOPE_PAYG_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

Generation and the DashScope embedder use the `PAYG` pair when set, else the main pair; a plan key
alone is refused with a message naming `DASHSCOPE_PAYG_API_KEY`.

A value resolves as: CLI flag → environment variable → `settings.local.json` →
`settings.default.toml` → built-in default.

### Adding a custom OpenAI-compatible provider

Any service with an OpenAI-style `/v1` API (Mercury, OpenRouter, Together, vLLM, LM Studio...) can be
used as a provider. In Settings, **Authentication > + Add API key > Other service...**: give the service
name, the key and the server URL, and tick *OpenAI-compatible LLM provider* (and *can read images* if its
models take image input). The key and URL are written to `.env` as `<NAME>_API_KEY` and `<NAME>_BASE_URL`;
the registry entry goes to `llm.custom_providers.<id>` in `config/settings.local.json`:

```json
{"llm": {"custom_providers": {"mercury": {"label": "Mercury", "key_env": "MERCURY_API_KEY",
                                           "url_env": "MERCURY_BASE_URL", "vision": false}}}}
```

The id is the lowercased service name and may not be a built-in provider name (`claude`, `anthropic`,
`dashscope`/`qwen`/`alibaba`/`modelstudio`, `openai`/`gpt`, `google`/`gemini`, `ollama`, `bedrock`). A tier
then reads `"mercury,<model id>"`. The model menus list the models from `GET {url}/models`; if that fails the
provider is still offered and a model id can be typed. Under Connections the provider has a **Test** button
and a **Remove** button (Remove drops the registry entry; the keys stay in `.env` until removed under
Authentication). A missing key or URL is reported by variable name when the agent is built.

## Checking the environment

```powershell
.venv\Scripts\python.exe scripts\check_env.py --gpu     # Windows
```
```bash
.venv/bin/python scripts/check_env.py --gpu              # macOS
```

Lists every dependency agentY needs and whether it imports, and whether torch
found the GPU. The launcher runs a quiet version on every start and reinstalls
`requirements.txt` when something is missing or the file changed.

## OpenMP on macOS

If the host dies with `OMP: Error #15 … libomp.dylib already initialized`, torch
and faiss-cpu each loaded their own OpenMP runtime. `./install_agent.sh` and
`./run_agent.sh` prevent this by making faiss's copy a symlink to torch's (the
original is kept as `libomp.dylib.orig`). Reinstalling faiss undoes it, so run the
launcher again. `scripts/check_env.py` reports the condition. Windows is not
affected.
