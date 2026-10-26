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

2b. **The speed-up set (2026-10-22, on `dev`)** - what the proposals of the first
   version of this note listed, built (next section). Numbers are measured here, tests are
   in each repo; none of it is released.

## The speed-up set

### 1. `partner_run` - a partner node built directly from its schema

`partner_run(node_type, inputs, settings, save, name, run=True)` (agenty_core
`tools/partner_run.py`; the orchestrator's copy is the pipeline closure of the same name,
agentY-mcp's is in `src/tools/execution.py`). One call reads the node's `/object_info`
entry (dynamic combos and nested autogrow slots through the helpers in
`assembly_deterministic`: `expand_input_specs`, `selected_option_key`,
`autogrow_member_spec`, `describe_input_names`), loads each file into the slot named
(`LoadImage` / `LoadVideo` / `LoadAudio`, staged into ComfyUI's input folder), sets
`model.*` and plain widgets (an option the node does not offer is an error listing the
options; numbers are clamped and reported), adds the save node (`SaveImage` / `SaveVideo` /
`SaveAudio`; an output node such as Meshy gets none - the host's configured saver is swapped
in by the executor as for any workflow), fills defaults through `harden_node_inputs`,
validates locally and writes the file. `run=True` then goes through the host's own path
(agentY: `run_workflow_now`, so the API key, output tags/sidecars, viewer push, canvas drop
and the plan/review gates all apply; agentY-mcp: `execute_workflow`; headless core: submit
and wait). Prompt: `system_prompt.orchestrator.md` ("Delegates") and the `partner-run` skill
(agentY orchestrator allowlist, agentY-mcp skills). Fixtures from the live ComfyUI:
`agenty_core/tools/tests/fixtures/partner_nodes_object_info.json` (GPT Image 2 / 2.5,
Seedance 2.5 first-last-frame and text-to-video, Kling start-end, Nano Banana, Meshy, the
loaders and savers). Smoke-tested against all 307 partner nodes of the live ComfyUI's
`/object_info` (read-only GET): none crashed the builder; of the 87 that need no input file,
83 build into a graph that passes local validation, and the other 4 (Flux3 image-to-video,
Meshy multi-image, Reve remix, Wan reference) are refused with the autogrow slot they need
named. Nodes whose output is only text / SVG / a voice id have no saver `partner_run` knows
and say so.

### 2. Solved jobs become machine-local templates

`agenty_core/tools/local_templates.py`. Store: `<custom templates>/local/` with its own
`index.json`, git-ignored (`.gitignore` of agentY-core), skipped by the recipe generator
(`Corpus._parse_folders` prunes `local/`), never written into the shared catalog. The read
side needed one line: the catalog (`load_descriptions`) and the loader
(`_find_template_file`) already walked the custom folder recursively; `_load_index` got the
`local/index.json` path. Write paths: `register_workflow_template(file, description=..., local=True)`
(the user said "keep this"), and `src/utils/local_templates_hook.py` after a clean run
(`Pipeline` records the workflows a turn ran once its batch finished with outputs, no failed
member and no QA failure; `partner_run` records its own) - setting `local_templates_auto`
(default on). Each entry carries what a later request needs: a description with the new
inputs, `parameters` (loaders, the prompt, key settings), `output_kind`, the nodes, the words
it was solved for, `uses` / `last_used`. The same structure again is ONE template (use count
up); 200 at most (least recently used goes). `unregister_workflow_template(name)` forgets one.

Retrieval: `Pipeline._format_template_catalog` puts a "YOUR SAVED JOBS" block first for
matching jobs (`match_local`) and the researcher picks it by name; assembly then loads it like
any template. Matching is conservative on purpose: the output kind must be the one asked for,
a model named in the request must be one the job runs, `template_fit` must not reject it, at
least 2 shared words and at least half of the request's words covered (70 % and 4+ words for
an automatically recorded job, so a short vague request never matches one). A wrong match
costs more than the research it saves, so when in doubt it offers nothing.

### 3. Template matching checks kind, not just name

`agenty_core/utils/template_fit.py`: `template_facts` reads a template's graph (UI or API
format, subgraphs included) - output kinds from its save nodes (the catalog files video
templates under `mediaType: image`), a storyboard step and its panel count, model versions,
resolutions, audio (where the graph names it), first-frame vs reference vs text-only;
`request_wants` reads the same from the request; `check` returns `reject` (structural: wrong
output kind, a storyboard the request did not ask for, another model version, text-only for a
frames job, reference mode for a first-frame job), `demote` (patchable: resolution, audio -
with the instruction) or `fit`. Used in three places: the catalog the researcher reads is
marked per request (`[NOT FOR THIS REQUEST: ...]`, `[adjust: ...]`), a pick that is rejected
is handed back to the researcher with the reason (bounded retries, then `build_new`; never
when the user named the template themselves), and `get_workflow_template(name, request=...)`
adds `fit` for hosts without a researcher. A `ready` result carries `template_adjust`.
Regression (`test_template_fit.py`, `test_template_pick.py`): "animate each starting frame
with Seedance 2.5 first-frame, 480p, no audio, cut into one film" rejects
`template_seedance2_storyboard_to_video` (8-panel board, Seedance 2.0, 720p) with those
reasons, and keeps `api_seedance2_5_flf2v` (adjust: 720p -> 480p).

### 4. `run_variants` - parallel variants

`run_variants(workflow_path, variants=[{"label", "set": {"<node>.<input>": v}}, ...], run=True)`
(agenty_core `tools/variants.py`): `duplicate_workflow(count=N)` + one `update_workflow` per
copy (server validation skipped - the run is the check) + all copies queued together; agentY
runs them concurrently through the same executor `run_workflow_now` uses (outputs per
variant, staged on the canvas), agentY-mcp through `execute_workflows_batch`'s executor.
16 steps (4 x duplicate, update, insert, run) become 1. `batch-handoff` and the orchestrator
prompt say so. `update_workflow` now honours `local_only()` like `validate_workflow`.

### 5. Smaller tool set per turn

Measured as `json.dumps(tool_spec)` chars / 4 (there is no tokenizer on this machine, so the
absolute numbers are estimates; before/after is like for like), orchestrator tool list =
`ORCHESTRATOR_TOOLS` + the pipeline's delegation tools, 142 -> 143 tools:

| | tools always in the list | schema tokens |
|---|---|---|
| before (dev at the start) | 63 | 17,761 |
| + the two new tools (`partner_run` 629, `run_variants` 421) | 65 | 19,124 |
| after (more packs) | 47 | 13,483 |

Net -4,278 tokens (-24 %) of tool definitions per call with both new tools included; the pack
catalogue in the prompt grew by ~310 tokens, so about -3,970 per call. Moved behind
`load_tools` (calls in 22,230 orchestrator tool calls in brackets): `dashscope_generate` /
`dashscope_fetch` (0), `show_in_viewer` / `show_in_panel` (13 / 65), `web_search_images` /
`run_web_search` (67 / 16), `file_read` / `inspect_workflow_file` / `find_workflows`
(80 / 184 / 77), `list_mcp_servers` / `add_mcp_server` (35 / 6), `get_comfyui_dirs` /
`get_agent_output_dirs` (166 / 122), `analyze_video` (156, into `video`), `revise_prompt`
(33, into `loops`), `get_history` / `get_prompt_status_by_id` / `submit_prompt` (109 / 30 /
13, into `admin`). A direct call to a held tool still loads its pack and runs (existing
`ToolPacksHook`). Left alone on purpose: what the existing tests pin as "what every turn
needs" (`start_shot`, `download_hf_model`, `name_outputs`, `work_in_open_graph`,
`read_web_page`, `queue`) and `change_graph` (the splice-first hook). Still the biggest always
in: `partner_run` 629, `change_graph` 562, `run_script` 497, `analyze_image` 493. The ~60k
floor is mostly the system prompt (33 KB of markdown) and the skills listing: the next lever
is there, not in the tools.

### 6. Script in one step

`run_script(code="...", args="...")` (agenty_core `tools/shell.py`): the code goes to a temp
file and runs with agentY's interpreter; `command` and `code` are exclusive; the same sandbox
vets the interpreter + file + args (path arguments stay inside the allowed folders). It IS
`run_script`, so nothing is bypassed: the `tool_permissions` ask (the prompt shows the code,
not an empty command), `code_sink` (the Agent code panel gets the snippet) and `media_sink`
(files it writes are published) are all keyed on that name already - tests extend `describe`
and `snippets_for_tool` for the `code` argument. The 9 `write_text_file` + `run_script` pairs
of the camera-move turn become 9 calls.

### 7. Contact-sheet vision

`analyze_image(file_paths=[...])` (agentY and agentY-mcp; the sheet is
`agenty_core/utils/contact_sheet.py`): up to 16 images tiled into ONE numbered JPEG and one
vision call (agentY: through the vision agent; mcp: the sheet comes back as the image); the
answer is per image number and the result carries the number -> file legend. Eleven analyses
at 12-21 s each become one call. The orchestrator prompt says "several images, one question is
one call".

### Needs a host restart / reload

agentY: restart `:5000` (new pipeline tools, tool packs, prompt, skill); nothing under
`comfyui/custom_nodes` changed. agentY-mcp: restart the MCP server (`partner_run`,
`run_variants`, `analyze_image(file_paths)`, `run_script(code)`).

## Proposals not built

* **Cut the 60k floor further.** The tool definitions are now ~13.5k of it; the rest is the
  system prompt and the skills listing (`orchestrator-prompt-split` already moved the
  conditional sections out). A pass on the 33 KB base prompt is the next lever.
* **Host-side nudge** when the last step issued a single `duplicate_workflow` /
  `update_workflow` / `run_workflow_now` and the conversation names N variants: the prompt
  and `run_variants` are the fix for now; a hook would add a guarantee.
* **`insert_workflow_into_canvas` per variant** is not part of `run_variants`: outputs are
  staged on the canvas as loaders already, and inserting the graph changes which file
  `run_workflow_now` runs (the canvas copy). Left to the existing tool.
* **Local-template matching** is by words; a learned ranking (as the review picks train the QA
  score) could replace the cover threshold once there are enough saved jobs.
