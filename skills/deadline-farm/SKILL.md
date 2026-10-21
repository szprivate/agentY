---
name: deadline-farm
description: Render a ComfyUI workflow on the studio's Deadline render farm instead of the local ComfyUI. Use for "render this on the farm", "send it to Deadline", "run 20 variations on the farm", "is the farm free", "what happened to my farm job", "get the farm results".
allowed-tools: submit_to_deadline deadline_job_status deadline_farm_info
---

# Rendering on the Deadline farm

The farm runs the workflow on other machines with ComfyUI installed, queued by Deadline (priorities, retries,
several GPUs at once). Use it only when the user asks for the farm (or for many runs they say should not tie up
this machine). Otherwise queue locally as usual. Needs `DEADLINE_*` settings; if a tool answers "Deadline not
configured", say what is missing and stop.

1. **Check the farm first when unsure:** `deadline_farm_info()`. It reports the dedicated ComfyUI pool (default
   `comfyui`): whether it exists, which workers are in it and if they are online. Read `issues`. If the pool is
   missing or empty, tell the user (the message says how to create it); do not submit to another pool.
2. **Get the workflow in API format** (the prompt ComfyUI's `/prompt` takes), the same one you would queue
   locally, with models and nodes that exist on the farm machines. Then
   `submit_to_deadline(workflow=<JSON or .json path>, name="...")`.
   - `batch=N` runs it N times as variations (each with its own seeds); `frames="0-9"` is the same with a list.
   - Leave `pool` empty: it always means the ComfyUI pool. `output_dir` only if the user names a folder.
   - Input images are copied to the farm share for you. Say what you submitted (job id, pool, runs) in one line.
3. **Follow it:** `deadline_job_status(job_id, wait_s=60)` - state Queued / Rendering / Completed / Failed,
   progress, error text. Do not poll in a tight loop: wait with `wait_s`, or tell the user the job id and check
   when asked. A job stuck in Queued with workers online usually waits for a free machine.
4. **When Completed,** the answer lists `outputs` (files on the share) and `output_dir`. Show or open them
   like any other result (`show_in_viewer`). On Failed, report the `errors` text; a "ComfyUI not installed on
   <machine>" error means that machine should leave the pool.

Partner/API nodes work on the farm only when the workers have the Comfy.org key set up; if their tasks fail
for a missing key, say so rather than retrying.
