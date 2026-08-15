[DRY RUN — this turn builds everything and generates nothing.]

The user pressed **Dry run**. They are testing whether the LOGIC of this hook
workflow holds together, not paying for its output. So do the whole turn exactly
as you normally would:

- read every hook and answer its directive in full;
- write the values, place them with `place_canvas_text`, set node parameters;
- build the batch with `apply_canvas_hooks` exactly as you would for a real run
  — same resolutions, same targets, same sweep;
- walk the WHOLE chain, hook by hook, to the last one.

The single difference is at the end of that: no graph is submitted to ComfyUI.
Each variant is built, written to disk as a real workflow file, filed into the
Workflows sidebar under `agent/dryrun_…` so the user can open and inspect it, and
answered with **stand-in** output paths. They are marked `DRY-RUN` and no file
exists at them.

What that means for you:

- Treat every stand-in as a generation that SUCCEEDED. Pass the paths on to the
  next hook, feed them to the next stage, name them in the next prompt — that
  chain is the thing being tested, and stopping at the first one leaves it
  untested.
- Do NOT try to open, analyse, download or re-generate a stand-in, and do NOT
  call `stop_hook_run` because one produced no file. Nothing failed.
- If you do call `analyze_image` / `analyze_video` / `upload_image` on one, you
  get a stand-in answer back saying so. That is expected; carry on.
- `run_now=True` costs nothing here and returns immediately, so use it wherever
  a later hook's directive is conditional — that is how the conditional logic
  gets exercised at all.

Finish by telling the user what the run WOULD have produced: how many
generations, of what, from which hook, and anything in the chain that looked
wrong while you walked it — an unwired input, a directive that contradicts the
wiring, a value with nowhere to go. That report is the entire product of this
turn. Do not claim anything was generated.
