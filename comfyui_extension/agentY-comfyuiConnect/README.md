# agentY-comfyuiConnect

The **agentY UI for ComfyUI** — a custom node (frontend hooks + a server route,
**no graph nodes of its own**) that connects ComfyUI to the agentY agent. It has
two parts:

1. **Chat sidebar** (`web/agent_chat.js`) — the "agentY" tab in ComfyUI's left
   sidebar. Chat with the agent, browse past conversations, run slash commands,
   and attach images. This **replaces the old Chainlit web GUI.** The agent's
   *text* streams into the panel; every generated **image/video is dropped onto
   the graph as a `LoadImage` / video-loader node** instead of shown inline.
   It talks to the agentY chat host (`src/utils/agentY_server.py`, default
   `http://127.0.0.1:5000`) over HTTP + SSE. Start that host with `run_agent.ps1`.
   Override the backend URL with `localStorage.agentY_backend` if it runs elsewhere.

2. **Auto-open canvas** (`web/agent_canvas.js`) — opens the workflow the agent
   just ran directly on the canvas.

## Auto-open canvas — how it works

- `__init__.py` registers `POST /agent/load_workflow`. It broadcasts the posted
  graph over the websocket as an `agent.load_workflow` event.
- `web/agent_canvas.js` listens for that event and calls `app.loadGraphData(graph)`.
- `agenty_core.tools.comfyui.open_workflow_in_canvas` converts the executed
  API-format workflow to graph format, saves it to `workflows/agent/` (sidebar
  fallback), **and** POSTs it to `/agent/load_workflow` (this route).
- The executor (`src/executor.py::_submit_workflow`) calls that tool on every run
  (gated by `AGENTY_CANVAS_AUTOLOAD`, default on), so every workflow the agent
  runs opens on the canvas automatically.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and **restart ComfyUI** once:

```
cp -r comfyui_extension/agentY-comfyuiConnect  <ComfyUI>/custom_nodes/
```

After the restart the console prints `[agentY-comfyuiConnect] ready …`. Until then, workflows
still land in the Workflows sidebar (one click to open); auto-open just isn't live.

Loading replaces the current canvas graph. To disable the automatic behavior set
`AGENTY_CANVAS_AUTOLOAD=0` for the agentY process (the `open_workflow_in_canvas`
tool remains available for on-demand "show me the workflow" calls).
