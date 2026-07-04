# AgentCanvas — auto-open agent workflows on the ComfyUI canvas

A tiny ComfyUI custom node (server route + frontend hook, **no nodes**) that lets
the agentY pipeline open a workflow directly on the open ComfyUI canvas, so you
see exactly what the agent just ran without clicking through the Workflows sidebar.

## How it works

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
cp -r comfyui_extension/comfyui-agent-canvas  <ComfyUI>/custom_nodes/
```

After the restart the console prints `[AgentCanvas] ready …`. Until then, workflows
still land in the Workflows sidebar (one click to open); auto-open just isn't live.

Loading replaces the current canvas graph. To disable the automatic behavior set
`AGENTY_CANVAS_AUTOLOAD=0` for the agentY process (the `open_workflow_in_canvas`
tool remains available for on-demand "show me the workflow" calls).
