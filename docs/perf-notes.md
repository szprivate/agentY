# Where a turn's time goes (measured 2026-10-22) and what to do about it

Method: `.logs/tokens_usage.log` (one line per tool call; rows with `+0in` are the
parallel siblings of the previous LLM call), grouped into turns by gaps over 10
minutes, joined with `.logs/turn_trace.log` (`BEGIN/END req=`). See the "profiling a
run" note for the four logs. Numbers are today's five orchestrator turns
(agent = deepseek-v4.1-flash, 250 log rows).

## Breakdown

| turn | tool rows | LLM calls | wall | quick steps (gap <= 10 s) | slow steps (> 10 s) | ctx first -> last |
|---|---|---|---|---|---|---|
| 00:00 atlas / canvas | 59 | 50 | 11 min | 40 x 6.0 s = 241 s | 9 = 404 s | 205k -> 66k |
| 00:21 camera move | 66 | 66 | 11 min | 48 x 4.6 s = 219 s | 17 = 437 s | 67k -> 108k |
| 13:07 variants + review | 84 | 62 | 14 min | 37 x 5.6 s = 209 s | 24 = 644 s | 166k -> 349k* |
| 13:44 tweak | 13 | 6 | 3 min | 2 | 3 = 181 s | 309k* -> 110k |
| 17:27 retouch | 28 | 15 | 5 min | 7 x 6.1 s = 43 s | 7 = 244 s | 221k* -> 122k |

(* a few single-call spikes to 220-500k are the log flushing a turn-boundary
total on a `skills` / `place_canvas_text` line, not a real 500k prompt: the next
call is back at 57-110k. Do not chase them.)

What the numbers say:

* **A quick step is ~5 s and is pure LLM round trip** (the tool itself answers in
  well under a second: `get_node_schema`, `write_text_file`, `run_script`,
  `set_canvas_node_params`). 30-50 % of every turn is these steps; each replays a
  60-120k context (97 % cached, so the cost is latency, not money).
* **The other half is long tools**: `run_workflow_now` / `run_canvas` renders
  (38-46 s each, four in a row in the 13:07 turn), `render_camera_move`
  (25-57 s), `analyze_image` (12-21 s each, 4 in parallel is fine, 14 of them is
  not), `atlas_create_world` (145 s). Renders are GPU time and not addressable here.
* **Context per call**: a floor of ~60k (system prompt + tool schemas, first call
  of a fresh conversation), growing ~0.6k per step. History compaction already
  holds the carried-over history near its 20k budget, so the per-call size is
  mostly the fixed 60k. Cutting that floor (tool schemas) is the biggest
  per-call lever left, and also the riskiest.
* **Repeats seen** (turn 13:07): 4 x `duplicate_workflow`, then 4 x `update_workflow`,
  4 x `insert_workflow_into_canvas`, 4 x `run_workflow_now`, each issued one per
  step although the per-variant calls are independent (16 steps; ~190 s of the
  turn). 9 x `get_node_schema` across orchestrator + `info`, 6 x `get_canvas_node`
  polls. 00:21 turn: 9 `write_text_file` immediately followed by `run_script`
  (two steps for one throw-away script) and 18 `run_script` in all.
* Everything in `get_node_schema` / `get_workflow_node_info` / `search_nodes` that
  takes a list already batches (`agenty_core/tools/_batch.py`); the repeats are
  the model not using it, not a missing feature.

## Built (on `dev`)

1. `duplicate_workflow(source_path, count=N)` (agenty_core): N fresh-seeded copies
   in one call, `new_paths` returned; `count=1` is byte-identical to before.
   `skills/batch-handoff` now tells the agent to use it. Saves N-1 round trips
   (about 5 s each) per batch.
2. Run ledger override (agentY): a user message that plainly asks to do it again
   ("rerun", "re-render", "run it again", "from scratch", "once more", "start
   over", "new seed") adds a `[THE USER ASKED FOR A RE-RUN ...]` line to the
   `[RUN LEDGER]` block for that turn, so "APPROVED = settled, do not redo" no
   longer answers "nothing left to run". Only the user's own words count (not a
   branch/harness message); the ledger entries stay as the previous round. No gate
   removed: review stops and the plan gate are untouched.

## Proposals not built

* **Fast path for solved jobs.** After a turn runs a workflow successfully, store
  the executed API graph plus a short request signature as a *local* template
  (the deferred "local templates" plan: a gitignored `local/` folder with its own
  `index.json`; discovery is already a recursive `index.json` walk). A later
  request that matches (same task/model, new inputs) would go `get_workflow_template`
  -> patch inputs -> `signal_workflow_ready` and skip research/assemble/validate.
  Not built: needs the write path that skips `workflow_templates`/recipe regen,
  a matching rule that does not misfire, and a decision on which inputs are
  parameters; wrong matches would be worse than the saved time.
* **`run_python_code(code, args)`** in agenty_core: one step instead of
  `write_text_file` + `run_script` (about 9 steps / 45 s in the 00:21 turn).
  Prototyped and dropped: `run_script` is wired into `tool_permissions` (ask),
  `code_sink` and `media_sink` by name, so a new tool needs each of those
  extended or it silently bypasses the approval and the file-publishing hooks.
* **Parallel-variant nudge.** The prompt already says independent calls go out
  together; the 13:07 turn did not. A host-side hint when the last step issued a
  single `duplicate_workflow`/`update_workflow`/`run_workflow_now` and the
  conversation names N variants would fix it deterministically; or one
  `run_variants(path, patches=[...])` tool that duplicates, patches, inserts and
  runs in one call.
* **Cut the 60k floor.** Defer rarely used tool groups behind `load_tools` (it
  exists) so the orchestrator starts with a smaller schema list; measure first
  with a per-tool schema token count.
* **`analyze_image` fan-out cap.** Eleven analyses in the 17:27 turn at 12-21 s
  each; a contact-sheet (several images in one vision call) would be 1 call.
