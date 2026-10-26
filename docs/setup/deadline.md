# Rendering on a Deadline farm

agentY can send a ComfyUI workflow to a Thinkbox Deadline 10 render farm instead of the local ComfyUI, and follow
the job. Deadline Monitor is then the queue: priorities, pools, retries, logs, several machines at once.

How it works: agentY submits one Deadline job of the plugin `ComfyUI` (API-format workflow as the job's auxiliary
file, one task per run). On the worker the plugin starts a headless ComfyUI, posts the workflow, waits, copies the
results to a network folder, writes a small manifest next to them, and stops ComfyUI again. agentY tools:

| tool | what |
|---|---|
| `deadline_farm_info` | pools, groups, worker counts, and the dedicated ComfyUI pool with its workers and their state |
| `submit_to_deadline` | submit a workflow (`batch=N` for N variations, each with its own seeds) |
| `deadline_job_status` | state, progress, error text, output files (`wait_s` waits for the end) |

## 1. One pool for the machines that have ComfyUI

ComfyUI is installed on only some farm machines, so jobs always go to a dedicated pool, **`comfyui`** by default
(`DEADLINE_COMFYUI_POOL`). `submit_to_deadline` refuses to submit when the pool does not exist or has no workers,
and never falls back to `none` (all workers). Create the pool and assign the machines once:

- In Deadline Monitor: **Tools > Manage Pools** > New > `comfyui`. Then enter **Super User** mode, right-click a
  worker > **Modify Worker Properties > Pools**, add `comfyui` (remove `none` from it if the machine should do
  nothing else).
- Or with the command line (run these yourself, on any machine with the Deadline client):

  ```
  deadlinecommand -AddPool comfyui
  deadlinecommand -AddPoolToSlave RND03,RND04 comfyui          # keep the worker's other pools
  deadlinecommand -SetPoolsForSlave RND03,RND04 comfyui        # or: ONLY comfyui
  deadlinecommand -GetSlaveNamesInPool comfyui                 # check
  ```

A **group** works the same way as an alternative (`-AddGroup`, `-AddGroupToSlave` / `-SetGroupsForSlave`);
agentY sends `DEADLINE_GROUP` with the job as an extra filter, but the pool is the guard.

## 2. Install the plugin into the repository

The plugin is in this repo: `deadline/plugins/ComfyUI/` (`ComfyUI.py`, `.param`, `.options`). **You** install it
(agentY never writes into the repository) with the script, from the agentY checkout:

```
.venv/Scripts/python.exe scripts/install_deadline_plugin.py --dry-run     # shows what would change, writes nothing
.venv/Scripts/python.exe scripts/install_deadline_plugin.py                # installs / updates
.venv/Scripts/python.exe scripts/install_deadline_plugin.py --uninstall    # moves it out again
```

It finds the repository from `--repository`, then `DEADLINE_REPOSITORY`, then `deadlinecommand -GetRepositoryRoot`,
copies the three files to `<repository>\custom\plugins\ComfyUI` (for this studio
`\\bepic-storage01\farm\deadline\custom\plugins\ComfyUI`) and checks the copy byte for byte. An older copy that differs
is moved to `<repository>\custom\_backup\ComfyUI-<time>` first (not left in `plugins\`, so Deadline does not list it).
Run it again after every agentY update that touches `deadline/plugins/ComfyUI/`. It creates no pool and assigns no
worker; the commands for that are printed at the end and are in section 1.

Workers pick the plugin up on their next job (they sync it from the repository). No icon is needed. By hand,
copying the folder to the same place does the same.

Then **Monitor > Tools > Configure Plugins > ComfyUI** (Super User) and set:

| setting | meaning |
|---|---|
| ComfyUI Folder (Windows / Linux / Mac) | the folder with `main.py` (or a portable folder holding `ComfyUI\main.py`) on the workers. Local to each machine, or a shared install. |
| Python Executable | empty = look for `python_embeded`, `.venv`, `venv` next to / inside that folder, then `python` on PATH |
| Extra ComfyUI Arguments | e.g. `--extra-model-paths-config //server/models/extra_model_paths.yaml` to share models |
| Startup Timeout | seconds to wait for ComfyUI to come up (default 300) |
| Keep ComfyUI warm | leave ComfyUI running after the job; the next job on that worker reuses it |
| API Key Environment Variable / API Key File | where partner (API) nodes get the Comfy.org key on the worker |

If the folder does not exist on a worker, the task fails at once with `ComfyUI not installed on <worker>`, so
Deadline's bad-worker detection stops sending the job there. The fix is to take that worker out of the pool.

Workers need ComfyUI with the **same custom nodes and models** the workflow uses (a shared models folder via the
extra-model-paths argument is the usual way). Output and input folders are network paths; on Linux or Mac workers
set up Deadline path mapping for them.

**Partner / API nodes:** set the Comfy.org key in the worker's own environment (machine-wide variable named in
the plugin setting, `COMFY_ORG_API_KEY` by default, then restart the Worker) or in a key file readable only by the
render account. The plugin sends it to ComfyUI as `extra_data.api_key_comfy_org`. It is never written to a job file.

## 3. Configure agentY

All optional; set them in `.env` (the settings dialog's `.env` section lists them):

| key | meaning |
|---|---|
| `DEADLINE_PATH` | the Deadline `bin` folder holding `deadlinecommand` (default `C:\Program Files\Thinkbox\Deadline10\bin` is tried) |
| `DEADLINE_WEBSERVICE_URL` | e.g. `http://deadline-host:8082`. Used when set and reachable, else `deadlinecommand` |
| `DEADLINE_REPOSITORY` | the repository path (informational) |
| `DEADLINE_COMFYUI_POOL` | the ComfyUI pool (default `comfyui`) |
| `DEADLINE_GROUP` | default group |
| `DEADLINE_COMFYUI_PATH` | ComfyUI folder on the workers, sent with each job (a plugin setting for the OS wins) |
| `DEADLINE_OUTPUT_DIR` | **required**: a folder the workers can write (network path); each job gets a subfolder, input files go to `_inputs/` there |
| `DEADLINE_PRIORITY` | default priority (50) |
| `DEADLINE_ROUTE_RUNS` | `1`/`0`: send the agent's normal runs to the farm (section 5). Overrides the setting `deadline.route_runs` |
| `DEADLINE_FETCH_DIR` | with routed runs: copy the results here (next to your other outputs) instead of using them in place on the share |
| `DEADLINE_MAX_WAIT_MINUTES` | with routed runs: stop waiting (and suspend the job) after this long; default 240, 0 = never |

With neither a Web Service nor `deadlinecommand`, the tools answer "Deadline not configured".

## 4. First test

1. `deadline_farm_info` (or ask agentY "is the farm ready?"): `ready: true`, the workers of the pool listed.
2. Submit a tiny workflow (a 512px, 4-step text-to-image) with `submit_to_deadline`; watch it in Monitor and with
   `deadline_job_status(job_id, wait_s=60)`.
3. The result lands in `DEADLINE_OUTPUT_DIR\<name>_<time>\` as `<file>_t0000.png` plus `.agenty_farm_<jobid>_0.json`.

If it fails: the task's log in Monitor shows ComfyUI's own log lines and, at the end, ComfyUI's error text
(`execution_error` with node and exception). A startup failure includes the tail of ComfyUI's log.

## 5. Sending the agent's normal runs to the farm

The three tools above are explicit: you ask for "render this on the farm". Routing makes it the default. Turn on
`deadline.route_runs` in `config/settings.local.json` (or `DEADLINE_ROUTE_RUNS=1` in `.env`; the variable wins, so
`0` switches it off again without touching settings). **It is off by default.**

```json
{ "deadline": { "route_runs": true } }
```

With it on, `execute_workflow`, `execute_workflows_batch` and the chained stages of a hook run (`run_workflow_now`)
queue each workflow as a Deadline job instead of the local ComfyUI. Everything else about a run is unchanged:

| step | how it works on the farm |
|---|---|
| preparing the graph | the same as locally (autogrow wires, chosen save nodes, output naming, viewer tab) |
| cost gate for partner nodes | the same, before anything is queued |
| submission | one job per workflow on the dedicated `comfyui` pool; refused (with the setup hint) if the pool is missing or empty |
| progress | "queued, waiting for a free worker" / "rendering 1/1 - 40%" lines, in the panel and in Slack, plus a reminder every minute while it waits |
| results | the files the plugin wrote to `DEADLINE_OUTPUT_DIR\<name>_<time>\` are handed to the same collection step as a local run's: QA against your briefing, the viewer, the canvas, the Slack upload, the output tags. With `DEADLINE_FETCH_DIR` set they are copied there first and *those* paths are used |
| a failure | the plugin's ComfyUI error text is parsed back into node id / node type / exception, so the repair loop works as it does locally |
| Stop | suspends the job (it stays in Monitor; nothing is deleted) and the plugin interrupts ComfyUI on the worker |
| a batch | every member is its own job, so they render on different machines at the same time; a failed member is healed and resubmitted like a local one |

What it needs, besides sections 1-3: **the output folder must be visible to the agentY host under the same path the
workers wrote** (`DEADLINE_OUTPUT_DIR` as the host sees it: a UNC path on Windows, a mount on Linux with Deadline path
mapping for the workers). If the host cannot open a result path, the run says "Could not resolve ..." for it.

What it does not do: it does not turn on for runs a person queues in ComfyUI's own Queue button (those are not agentY's);
and partner-node credit settlement is not measured for farm jobs (the worker's own key pays).

Use it with a server host ([server.md](server.md)): a Slack request, queued in the turn queue, renders on the farm, and the
files come back into the thread.

### What is untested

Everything here is tested against a stand-in for Deadline's status calls and for the plugin's output manifest, not
against your farm: the pool `comfyui` does not exist yet and the plugin is not installed. Before relying on it, do
section 4 with a tiny workflow, then turn routing on for one run and watch it in Monitor.
