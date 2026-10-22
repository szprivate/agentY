# Turn API: talking to a running agentY from another program

A top-level agent (Claude Code, any MCP client, a shell script) can hand a
message to the **running agentY host** and get the finished turn back, instead of
driving the ComfyUI browser panel. The turn is run by agentY itself: same
orchestrator, skills and hooks as a turn typed in the panel.

## From an MCP client (agentY-mcp)

```
agenty_turn(message, conversation=None, wait=True, timeout_s=1800,
            attachments=None, title=None)
agenty_turn_status(turn_id, wait_s=0)
agenty_conversations(limit=30)
agenty_reply(request_id, text)      # answer a question a turn is waiting on
agenty_stop(turn_id="", conversation="")
```

`agenty_turn` returns JSON: `status` (`done` | `error` | `waiting` | `running`),
`reply`, `tool_calls` (name, agent, status, short input/result), `files` it
produced, `canvas` edits it made (op, node ids), `waiting` (`{kind, prompt,
request_id}` when it stopped on a question or an open review), `usage` (tokens,
cost), `turn_id`, `conversation`. With `wait=False` it returns the ids at once.

Conversations are addressable: pass the `conversation` id of an earlier result to
continue it; omit it to start a new one. Different conversations run in parallel;
a second message to a conversation that is mid-turn is refused (HTTP 409).

Shell: `python scripts/agenty_turn.py "message"` in agentY-mcp (`-c <id>`,
`--no-wait`, `--status <turn>`, `--conversations`, `--json`).

## In the panel

The turn is tagged `origin: "api"` on the turn bus. A panel with that conversation
open follows it live (the same adoption that follows Slack and shot turns, via
`/agentY/runs`); otherwise the conversation shows its running dot. A turn that
starts a new conversation appears in the list on the panel's next refresh. It asks
the open panel for the canvas, as a Slack turn does, so canvas tools work.

## HTTP

| route | |
|---|---|
| `POST /agentY/turns` `{message, conversation?, title?, attachments?}` | starts a turn, `202 {turn_id, conversation}`; `400` bad input, `409` conversation busy |
| `GET /agentY/turns/<id>?wait=<s>` | the structured result; `wait` (max 120) holds until the turn ends or stops to ask |
| `GET /agentY/turns` | running and recently finished turns |

Finished results are kept for the last 100 turns, in memory (lost on a host restart;
the conversation's message log stays).

## Auth

Same guard as every other route: the `Host` header must be an IP literal or
`localhost`, a browser `Origin` must be the panel's, and the session token must be
sent in `X-AgentY-Token`. Non-browser clients send no Origin, so the token is what
they need. Nothing about the browser protection is relaxed.

The token is the one the panel gets: the host keeps it in `<agentY>/.agenty_token`
(gitignored, reused across restarts). On a new file the host now also restricts it
to the current Windows user (`icacls`); for an existing file run
`icacls .agenty_token /inheritance:r /grant:r %USERNAME%:F` once. The client
(agentY-mcp `src/utils/agenty_host_client.py`) finds it via, in order:
`AGENTY_SESSION_TOKEN`, `AGENTY_TOKEN_FILE`, `$AGENTY_HOME/.agenty_token`, then
`../agentY/.agenty_token` beside the agentY-mcp checkout. The host URL is
`AGENTY_HOST_URL` (default `http://127.0.0.1:5000`). Anyone who can read the file
can already read `.env`, so the file is the trust boundary.
