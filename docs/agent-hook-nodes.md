# Agent Hook Nodes — plan & implementation

## Idea

Add an **`AgentYHook`** ComfyUI node that the user drops onto the canvas and wires
from any node's output, then types a natural-language directive into — e.g.
*"create variations for the prompt in the previous node"*, *"iterate through all
the input files in this folder"*, *"sweep the seed of the previous node"*.

- On a **normal Queue Prompt** the hook is inert (it feeds no output node, so
  ComfyUI's validation/execution skips it).
- When the user asks the **agentY agent** to run the workflow, the agent runs the
  **on-canvas graph** and applies each hook's directive to its anchored node,
  expanding the run into a mutated batch.

Confirmed decisions: (1) run the **canvas graph** (not a template), (2) anchor
**by wire**, (3) multiple hooks **multiply** into a Cartesian product with a
safety cap. The node is a **passthrough** (identity), recommended usage = wire the
input only, leave the output dangling.

## Why it fits cleanly

- **Inertness is free.** ComfyUI only executes nodes on the path to an output
  node; an unreferenced hook is skipped by `validate_prompt`/execution.
- **`app.graphToPrompt()`** already yields the exact API-format prompt ComfyUI
  would run — no Python graph→API converter needed.
- The three example directives are all the same shape (*base graph → N mutated
  copies → submit each*), which is exactly what the existing batch executor +
  `_expand_variations` already do. `_submit_workflow` already submits API-format
  prompt JSON.

## Data flow

```
canvas (AgentYHook nodes)
  └─ agent_chat.js.send()
       ├─ _collectCanvasHooks()   → [{directive, mode, anchor_node_id, anchor_type, anchor_widgets}]
       └─ _captureCanvasGraph()   → app.graphToPrompt().output   (API-format prompt)
     POST /agentY/chat { canvas_prompt, canvas_hooks, … }
  └─ agentY_server: _run_pipeline_stream → pipeline.stream_async(canvas_prompt=, canvas_hooks=)
  └─ pipeline._astream_orchestrator:
       ├─ splice_hook_nodes(canvas_prompt) → self._canvas_base_prompt   (clean base)
       └─ _build_orchestrator_input prepends the [CANVAS HOOKS] block
  └─ orchestrator reads the block → apply_canvas_hooks(resolutions=[…])
       └─ build_batch(base, resolutions, cap) → write N prompt files → append_workflow_path(each)
  └─ pipeline drains the workflow-signal mailbox → _execute_workflows_batch → outputs staged as nodes
```

## Files

**New**
- `comfyui_extension/agentY-comfyuiConnect/web/agent_hook.js` — node coloring.
- `src/utils/canvas_hooks.py` — `splice_hook_nodes`, `build_batch`,
  `enumerate_folder`, `describe_hooks`.

**Modified**
- `comfyui_extension/agentY-comfyuiConnect/__init__.py` — the `AgentYHook` node
  (wildcard `anchor` input, `directive`/`mode` widgets, identity passthrough) +
  `NODE_CLASS_MAPPINGS`.
- `comfyui_extension/agentY-comfyuiConnect/web/agent_chat.js` — hook collection +
  graph capture; `send()` ships `canvas_hooks` + `canvas_prompt`.
- `src/pipeline.py` — `apply_canvas_hooks` tool (in `_build_delegation_tools`),
  `canvas_prompt`/`canvas_hooks` kwargs threaded through
  `stream_async` → `_astream_orchestrator`, splice + `[CANVAS HOOKS]` block.
- `src/utils/agentY_server.py` — `/agentY/chat` reads `canvas_prompt`/`canvas_hooks`
  and passes them to `_run_pipeline_stream` → `stream_async`.
- `config/system_prompts/system_prompt.orchestrator.md` — the canvas-hook contract.

## `apply_canvas_hooks` resolution schema

```jsonc
// one entry per hook directive; the batch is their Cartesian product (capped by
// AGENTY_MAX_CANVAS_BATCH, default 25)
{ "target_node_id": "12", "param": "seed", "mode": "sweep_seed", "count": 6, "start": 0? }
{ "target_node_id": "4",  "param": "text", "mode": "value_list", "values": ["…","…"] }
{ "target_node_id": "9",  "param": "image","mode": "folder", "folder": "C:/in", "extensions": ["png","jpg"], "use_full_path": false }
```

## Notes / limits

- **Inline hooks** are supported: `splice_hook_nodes` rewires the downstream input
  back to the hook's `anchor` source before removing the hook, so the graph stays
  connected. Dangling hooks are simply dropped.
- **Folder iterate** sets the loader's filename input to each file (basename by
  default); the files must be reachable by that loader (normally ComfyUI's `input`
  dir). Use `use_full_path` if the node accepts absolute paths.
- **No Vision-QA** in canvas mode (there's no brainbriefing describing intent) —
  variants run and are staged as loader nodes on the graph.
- Legacy template-assembly path is unchanged; canvas mode only activates when the
  graph contains `AgentYHook` nodes with directives.

## Verification (offline)

- `py_compile` of all touched Python files — pass.
- `splice_hook_nodes` (dangling + inline rewire), `build_batch` (product, cap,
  deep-copy isolation), `describe_hooks` — unit tests pass.
- Tool glue: variant files written, queued on the workflow-signal mailbox, drained
  as executor-submittable API prompts — pass.
- `stream_async`/`_astream_orchestrator` accept the new kwargs; `apply_canvas_hooks`
  registered in the delegation toolset — pass.
- `node --check` on the JS (script + ESM) — pass.

Live smoke (needs a running ComfyUI + the SSE host): add an `agentY hook` node,
wire it from a KSampler, type "sweep the seed 4×", ask the agent to run the
workflow → 4 variants execute and stage onto the graph; confirm a normal Queue
Prompt with the hook present still runs unchanged.
