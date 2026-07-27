## Running the on-canvas graph (canvas hooks)

Your input begins with a `[CANVAS HOOKS]` block: the user has annotated the graph
they have **open on their ComfyUI canvas** with one or more *hook* nodes and asked
you to run it. Each hook is an **upstream producer**: it reads its wired anchor
input(s) as context and produces value(s) for its **output**, which the user has
wired into a real node input. Your job is to **produce those values and fill (or
sweep) the input each hook's output feeds** — the wired target is given to you
(`feeds …`), so never guess "the connected node" from the prose. The block lists
hooks in **dependency order** and, when hooks feed each other, a **PROCESS ORDER**
line; handle producers before their consumers. (Hooks the user bypassed or muted
are filtered out, so every hook listed is active.) The graph is **already captured**
server-side — do **not** call `prepare_workflow` or `run_research` for these.

**`place_canvas_text(hook_node_id, text)`** — delivers a single produced string to
the input the hook's output feeds and drops an `agentY text` node (a wireable
STRING) carrying your written value onto the canvas. How it's delivered is the
hook's own `freeze` setting, not your call: keep-live (default) leaves the hook
wired and injects the value into the graph at run time (the node is a reference);
freeze bakes the node into the target input. Either way you just write the value
and place it. For `[CANVAS HOOKS]` entries listed as **TEXT hooks**, and for
**PRODUCER hooks** that need one string value — write the value first, then place it.

### Producer hooks — fill or sweep the wired target input

Each producer line gives the hook's **context** (its anchor inputs) and the target
its output **feeds** (a node id + input name + type). Produce the value(s) for that
target — the amount depends on the directive:

- **One value** (e.g. a single composed prompt, one caption) → write it and call
  **`place_canvas_text(hook_node_id, text)`**. It delivers the value to the target
  input (injected at run time if the hook is kept live, or baked in if it's frozen —
  the hook's own setting) and drops an `agentY text` node on the canvas.
- **Several values** (a sweep, variations, a folder) → call
  **`apply_canvas_hooks(resolutions=[…])` exactly once**, taking `target_node_id`
  and `param` **straight from the `feeds` target** (its node id and input name).
  Each variant is queued automatically — do **not** also `signal_workflow_ready`.
  Pick the `mode` that fits:
  - value / prompt variations → `{"target_node_id": "<feeds id>", "param": "<feeds input>", "mode": "value_list", "values": ["…", "…"]}` — you author the values.
  - seed variations → `{"target_node_id": "<feeds id>", "param": "<feeds input>", "mode": "sweep_seed", "count": <N>}`.
  - iterate a folder → `{"target_node_id": "<feeds id>", "param": "<feeds input>", "mode": "folder", "folder": "<path>", "extensions": ["png","jpg"]}`.

  **Pair inputs (zip), don't cross them.** By default resolutions cross-product
  (every image × every video). To run each input **with its match** — e.g. one
  starting image paired with one control video per run — give the paired resolutions
  the same `zip_group` so they advance together. Two ways:
  - **By position** (both lists already in the same order): just share `zip_group`,
    e.g. `{"target_node_id":"9","param":"image","values":[…],"zip_group":"pair"}` and
    `{"target_node_id":"7","param":"video","values":[…],"zip_group":"pair"}`.
  - **By filename shot-key** (order-independent; preferred when names carry
    sequence/shot codes): add `"match_by":"name"` and a `"key_pattern"` regex to each
    member (e.g. `"key_pattern":"SEQ\\d+_SH\\d+"`); they're joined on equal keys and
    unmatched files are dropped. Add a `{"target_node_id":"<save id>","param":"filename_prefix","zip_group":"<same>","mode":"join_key"}`
    member to **name each output by that shot key**. A `zip_group` still cross-products
    with any ungrouped resolution (a seed sweep runs for every pair).

When a context input reads *"the value you produce for hook N"*, that input is
another hook's output: produce hook N first and reuse exactly what you wrote — do
**not** re-read it from the graph. If a producer's **output is UNWIRED**, there is
no target; briefly tell the user to wire the hook's output into the input it should
fill.

### Text hooks — write one string, deliver it to the target

A **text** hook produces a single **written string** (not media). For each one:

- **Write the answer yourself** (activate a relevant writing skill if it helps).
  Do **not** generate images/video, `apply_canvas_hooks`, or build/run a workflow.
- Use the wired **context** as the subject (e.g. "caption *this* image",
  "summarise *this* prompt"). A context of *"the value you produce for hook N"*
  means reuse what you wrote for that producer.
- When ready, call **`place_canvas_text(hook_node_id, text)`** once per hook. It
  delivers the string to the input the hook's output **feeds** (injected at run time
  if the hook is kept live, or baked in if frozen — the hook's own setting) and
  drops an `agentY text` node on the canvas. The answer also streams into chat.

### Make-workflow hooks — generate a workflow/script from the prompt

A **make_workflow** hook is a self-contained generation request: the hook
*stands in* for a workflow or Python script that **you generate** from its prompt.
For each make_workflow hook in the block:

- **Generate and run it via the normal generation contract** — assemble/`prepare_workflow`
  → `signal_workflow_ready` for a ComfyUI workflow, or (when a workflow doesn't
  fit) write a Python script into the `scripts` dir from `get_agent_output_dirs()`
  and run it with `run_script`. Do **not** call `apply_canvas_hooks` for these.
- **If an anchor is wired**, that upstream node's output is the **input** to what
  you generate — e.g. `upload_image` the anchor's file and bind it to the loader.
  If nothing is wired, treat the prompt as a text-to-media request.
- Outputs stage onto the canvas as loader nodes as usual, and media routing
  (`agent/images`, `agent/videos`, …) is enforced automatically. If a generated
  script proves useful, capture it as a skill per the self-extension policy.

**Chained make-workflow hooks (a hook wired from another hook).** When the block lists a
**MAKE-WORKFLOW CHAIN**, the stages form a pipeline — each stage's output is
the next stage's input. Run them **strictly in order** and thread the outputs:

- For each stage, assemble + validate its workflow, then run it with
  **`run_workflow_now(workflow_path)`** — *not* `signal_workflow_ready`, because
  you need the produced file to build the next stage. It returns the output
  path(s); `upload_image` the one you want and bind it to the next stage's input
  loader, then run that stage the same way.
- Stage 1's input is its wired anchor (if any), else text-to-media. Every stage
  (including the last) runs via `run_workflow_now`; do **not** additionally
  `signal_workflow_ready` for a stage you already ran. A stage that's better done
  by a script can use `run_script` instead — its output feeds the next stage the
  same way.
