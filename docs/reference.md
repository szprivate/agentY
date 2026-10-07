# agentY reference

How the pieces fit, plus ports, security and configuration files. For day-to-day
use see [Using agentY](using-agentY.md).

## The four repos

| Repo | Location | Role |
|---|---|---|
| **agentY** | your working copy | The agent host and pipeline (`run_agent.ps1` / `run_agent.sh`). |
| **[agentY-core](https://github.com/szprivate/agentY-core)** | next to `agentY` | Shared tool layer (ComfyUI, Hugging Face, web, files) and the workflow template corpus. Required. |
| **[agentY-comfyuiConnect](https://github.com/szprivate/agentY-comfyuiConnect)** | `<ComfyUI>/custom_nodes/` | The sidebar chat tab and the canvas nodes. |
| **[agentY-mcp](https://github.com/szprivate/agentY-mcp)** | next to `agentY` | Optional MCP server for Claude Desktop, built on `agentY-core`. |

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

agentY is four repositories that only work as a set. A release pins that set:

- every repository gets the tag `v<version>` and a GitHub release;
- each repository's **`main`** (`master` in agentY-core) and **`stable`** branch
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
release is made, and a release always covers all four repositories under one
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
`modelstudio`, `alibaba`), `openai`, `google` (alias `gemini`). For mainland China
set `llm.dashscope.base_url` to `https://dashscope.aliyuncs.com/compatible-mode/v1`.

A value resolves as: CLI flag → environment variable → `settings.local.json` →
`settings.default.toml` → built-in default.

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
