# Running agentY as a server

agentY is a desktop tool first. This page is for the other way to run it: one host on a
server or a machine in the company network, addressed mainly through
[Slack](slack.md), running ComfyUI workflows without anybody sitting at a ComfyUI tab.
Everything here is **off by default**; an install that sets none of it behaves exactly
as before.

| you want | turn on | section |
|---|---|---|
| listen beyond loopback | `server.bind_host` | [1](#1-where-it-listens) |
| a service that starts with the machine | `deploy/` files | [2](#2-as-a-service) |
| TLS and a public name | a reverse proxy | [3](#3-reverse-proxy-with-tls) |
| a token per person for the HTTP side | `security.user_tokens` | [4](#4-per-user-tokens) |
| ComfyUI and storage somewhere else | `AGENTY_*` variables | [5](#5-where-things-live) |
| no canvas, results to Slack | `server.headless` | [6](#6-headless-mode) |
| a visible line for turns | `server.max_running_turns` | [7](#7-the-turn-queue) |
| health checks | `/agentY/health?deep=1` | [8](#8-health) |

## 1. Where it listens

By default the host binds to `127.0.0.1`. To change that, in order of priority:

- **Launchers** (`run_agent.ps1` / `.sh`): the flag (`-BindHost 0.0.0.0` / `--host 0.0.0.0`),
  then `server.bind_host` in `config/settings.local.json` or `AGENTY_BIND_HOST`, then loopback.
- **A service running `python -m src.agenty_ui_server` directly**: `--host`, then
  `AGENTY_UI_HOST`, then `server.bind_host` / `AGENTY_BIND_HOST`, then the host of
  `agent_server_url` (loopback).

`0.0.0.0` listens on every interface. Do that only behind a reverse proxy or on a
network you trust, with `security.user_tokens` on. A request is checked three ways before
it gets in (see [Security](reference.md#security)): the `Host` header must be an IP, `localhost`
or a name in `security.allowed_hosts`; a browser `Origin` must match the `Host`; and the
request needs a token.

Prefer binding to loopback and letting the proxy (below) face the network.

## 2. As a service

Both services run `python -m src.agenty_ui_server`, the same host the launcher starts.
The launcher's start-up chores (update check, model refresh, dependency sync) are not
repeated on every service restart: run `run_agent.ps1` / `run_agent.sh` by hand once after
an update, then restart the service.

**Linux (systemd).** Copy `deploy/agenty.service` to `/etc/systemd/system/`, edit the user,
checkout path and python path, and put the settings in `/etc/agenty/agenty.env` (start from
`deploy/agenty.env.example`, `chmod 600`):

```
sudo systemctl daemon-reload
sudo systemctl enable --now agenty
journalctl -u agenty -f
```

**Windows (NSSM).** In an elevated PowerShell, with [NSSM](https://nssm.cc) installed:

```
.\deploy\install-service.ps1 -EnvFile C:\agenty\agenty.env
nssm start agentY
```

The script creates the service, logs to `.logs\service.*.log` (rotated), restarts it after
a crash and stops it with Ctrl+C so a running turn winds down. `-Remove` undoes it. Run the
service as a normal account (`nssm edit agentY`, Log on) so it reaches network shares with
that account's rights, not LocalSystem's.

## 3. Reverse proxy with TLS

The host speaks plain HTTP and streams chat as Server-Sent Events. A proxy has to add TLS
and **must not buffer or time out** the stream. Ready-made configs:

- `deploy/Caddyfile.example` - Caddy, certificates automatic
- `deploy/nginx.conf.example` - nginx

In both, the public name must be in `security.allowed_hosts`:

```json
{ "security": { "allowed_hosts": ["agenty.example.com"], "user_tokens": true } }
```

The proxy passes `Authorization` through untouched. Keep the host itself on loopback.

## 4. Per-user tokens

The desktop install has one secret (the session token the panel reads from ComfyUI). A
team needs one per person, so you can tell them apart and revoke one without changing the
rest. With `security.user_tokens` (or `AGENTY_USER_TOKENS=1`) on, a request to the HTTP
side (turn API, viewers, scripts) is let in with **either** the session token or one
person's token, sent as `Authorization: Bearer <token>` (or `X-AgentY-Token`).

```
.venv/Scripts/python.exe scripts/agenty_users.py add anna --slack U01ABCDEF --role approver
.venv/Scripts/python.exe scripts/agenty_users.py list
.venv/Scripts/python.exe scripts/agenty_users.py remove anna
.venv/Scripts/python.exe scripts/agenty_users.py usage --by project --days 30
```

- `add` prints the token **once**. `config/users.json` (gitignored) keeps only a SHA-256
  digest. To replace a lost token, run `add` again for the same name.
- `--slack` links the person to their Slack member id; the role `approver` lets them approve
  other people's requests in Slack.
- A turn started through the turn API with a person's token is theirs: it appears in the
  usage ledger under their name and starts from their notes.
- `/agentY/health` stays open (it carries no data) so a proxy or monitor can reach it.

Tokens protect the HTTP side only. In Slack, the allow-list (`SLACK_ALLOWED_USERS`) is the
gate; the two are separate on purpose.

## 5. Where things live

Per-machine settings can come from the environment, so a service needs no
`settings.local.json`:

| variable | setting |
|---|---|
| `AGENTY_COMFYUI_URL` | `comfyui_url` (also exported as `COMFYUI_URL` for the tool layer) |
| `AGENTY_COMFYUI_MODELS_DIR` | `comfyui_models_dir` |
| `AGENTY_COMFYUI_USER_DIR` | `comfyui_user_dir` |
| `AGENTY_COMFYUI_DIR` | `comfyui_dir` |
| `AGENTY_OUTPUT_DIR` | `output_dir` |
| `AGENTY_OUTPUT_WORKFLOWS_DIR` | `output_workflows_dir` |
| `AGENTY_HEADLESS`, `AGENTY_BIND_HOST` | `server.headless`, `server.bind_host` |

Unset or empty variables change nothing. The agent reads ComfyUI's real folders from
ComfyUI itself (`/system_stats`), so for a remote ComfyUI make sure the output and model
folders it reports are visible to agentY at the same path (a shared mount), or set the
variables above to where agentY sees them.

Conversations, notes and the usage ledger are under `memory/` in the checkout
(`memory/conversations.sqlite`, `memory/team/`); back that folder up.

## 6. Headless mode

`server.headless = true` (or `AGENTY_HEADLESS=1`) says nobody has the ComfyUI tab open:

- finished images and video are **not** dropped on a canvas as loader nodes; they are saved
  in the output folder and, for a Slack request, uploaded into the thread;
- a turn does not wait for a canvas snapshot that cannot arrive (that wait is 8 seconds
  otherwise);
- tools that need the page (screenshot of the canvas, the open tabs, saving the open
  workflow, running hooks on the live graph) answer at once: *"this agentY host runs
  headless ... build and run the workflow directly (execute_workflow); results are
  delivered to Slack"*, so the agent changes approach instead of waiting for a timeout.

Everything that works without a canvas works as normal: building and running workflows,
research, the vision and QA agents, MCP tools, Atlas, file tools.

## 7. The turn queue

Turns asked for from Slack and through the turn API wait in one line:

- `server.max_running_turns` (default 2, never more than `parallel_chats`) turns run at
  once; others wait, in order. `server.max_queued_turns` (20) bounds the line; beyond that
  a request is refused with a clear message.
- The requester sees where they stand: in Slack a message in their thread ("Queued, number
  2 in line") that updates as the line moves and goes away when the turn starts; through
  the API the response carries `"status": "queued", "position": n`, and
  `GET /agentY/turns/<id>` keeps saying so. `GET /agentY/queue` lists who is running and
  who is waiting; the Slack word `queue` does the same.
- A conversation already waiting takes a follow-up message into the same turn instead of a
  second place.

Why a queue and not "everything at once": the host *can* run conversations in parallel
(one pipeline per running conversation, with its own buffers), and the panel does. But
every running turn is LLM traffic, and a channel of people can ask five things in a minute.
The line makes the limit a number you choose and tells people where they are, instead of
letting the pool stall silently. Renders still queue one after another on the GPU in
ComfyUI itself (or in Deadline, see [deadline.md](deadline.md)).

On a desktop host (neither `headless` nor `user_tokens`) API turns start at once, as
before. DM turns keep the old "busy" answer unless `slack.queue_dm_turns` is on.

## 8. Health

`GET /agentY/health` is what the panel polls and is unchanged. For a service manager or
proxy check use `GET /agentY/health?deep=1`, which adds:

```json
{ "server": { "headless": true, "user_tokens": true, "bind": "127.0.0.1",
              "queue": {"limit": 2, "running": 1, "waiting": 0},
              "slack": {"enabled": true, "connected": true, "channels": 1},
              "comfyui": {"url": "http://127.0.0.1:8188", "ok": true, "ms": 12} },
  "status": "ok", ... }
```

`status` is `ok` whenever the host answers; read `server.comfyui.ok` for "can it render".

## Security checklist

- Bind to loopback behind TLS; `allowed_hosts` names only your public name.
- `user_tokens` on; one token per person; revoke by `remove`.
- Keep `security.unattended_tool_policy = "deny"`: tools that need approval (running a
  script, installing a node or an MCP server, switching model) are then asked **in the
  Slack thread** and only the requester or an approver can say yes. With `"allow"` they run
  unreviewed.
- `SLACK_ALLOWED_USERS` and `SLACK_ALLOWED_CHANNELS` are lists, not "everyone".
- The usage ledger and notes are plain files in `memory/team/`.

## What is not here

- No web UI of its own: the panel lives in ComfyUI, and in Slack you do not need it.
- No multi-host clustering: one agentY host, one queue.
- `agentY-mcp` keeps reading the token from the local file; pointing it at a remote host
  with a user token is not wired up.
