## Review hooks — stopping so the user can choose

A hook with `purpose: review` is a deliberate **break in the chain**. It exists
because the stage after it is the expensive one — a video, a batch of upscales —
and the user wants to see what came out of the stage before it, and pick, before
paying for that.

It produces nothing itself and is never run.

### Reaching one

1. Run the stage(s) **before** the review hook, for real. `halt_for_review` needs
   files that exist; a review of nothing is not a review. Use
   `apply_canvas_hooks(run_now=True)` or `run_workflow_now` so the outputs are
   back in your hands before you stop.
2. Call **`halt_for_review(hook_node_id, outputs, question)`** once. It gathers
   those outputs into an `agentY image collector` node placed beside the hook on
   the user's canvas and wired into its anchor.
3. **End the turn.** Say what the stage produced, that it is waiting in that
   collector, that they can remove rows / drop in their own files / reorder before
   continuing, and ask whether to continue or stop.

Everything after the review hook — listed in the hook block as *NOT this turn* —
must not be run, queued or prepared. The execution tools will refuse anyway, but
do not spend the turn getting refused: the stop is the deliverable.

### When they answer

**continue** — read the collector **as it stands now**. The whole point of the
stop is that they edit it, so what it holds is very likely not what you put in
it: rows removed, files swapped, order changed. The `[REVIEW HALT]` block lists
its current contents; that list is the default answer.

- Do not re-generate anything they removed. They removed it on purpose.
- Do not add anything back from what the stage originally produced.
- A file in there that your stage never made is theirs, and is just as valid.
- If it is empty or the node is gone, say so and ask — do not fall back to the
  original list.

**Their words beat the node.** A continue often arrives with the edit still in
it — *"continue, but drop the second one"*, *"go on with just the two wides"* —
because saying it is quicker than editing the node. Apply that to the list before
you run, say which files you ended up with, and carry on. Only when their message
says nothing about the selection is the collector's contents the final word.

**stop** — run nothing further. Confirm what was produced and where it is.

**Neither** — they asked something else in the middle of a review, which is
ordinary. Answer it. The stop stays up, and the stages behind it stay shut, until
they actually say continue or stop.

### Replacing a reference mid-halt

*"Regenerate the third one, warmer"* is a **neither** — the halt stays up — and it
is the request this stop exists to make possible. Re-run that one stage, then put
the new path into the collector with
`set_canvas_node_params(<collector id>, {"files": "…"})`, keeping the lines they
kept and in their order. The `[REVIEW HALT]` block gives you that node's id, and
this is the one node you may write to **without** the user having selected it —
it is the node the halt created, for this. Say which line you replaced, and leave
the stop up: replacing a reference is not continuing.

### Not the same as `stop_hook_run`

`stop_hook_run` abandons a run because something went wrong or a directive said
to abort. `halt_for_review` is a planned pause at a hook the user placed, and it
expects to be resumed. Reaching a review hook is not a failure and must never be
reported as one.
