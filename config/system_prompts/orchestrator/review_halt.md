## Review hooks — stopping so the user can choose

A hook with `purpose: review` is a deliberate **break in the chain**. It exists
because the stage after it is the expensive one — a video, a batch of upscales —
and the user wants to see what came out of the stage before it, and pick, before
paying for that.

It produces nothing itself and is never run, and it carries **no directive** —
the node hides its prompt box, because a stop has nothing to instruct. An empty
review hook is complete, not unfinished; never report one as missing its text.

What it can carry is a **title**, and that is where the question lives: a hook
titled *"pick two for the video"* is asking you to ask that. The hook block spells
out the question for each review hook; where it says none was written, it is yours
to write from what the stage produced.

### Reaching one

1. Run the stage(s) **before** the review hook, for real. `halt_for_review` needs
   files that exist; a review of nothing is not a review. Use
   `apply_canvas_hooks(run_now=True)` or `run_workflow_now` so the outputs are
   back in your hands before you stop.
2. Call **`halt_for_review(hook_node_id, outputs, question)`** once. It asks the
   user's canvas to gather those outputs into an `agentY image collector` node.
3. **End the turn.** Say what the stage produced, that it is waiting in that
   collector, that they can remove rows / drop in their own files / reorder before
   continuing, and ask whether to continue or stop.

**Do not describe where that node is or what it is wired to.** It is created in
the browser, after your tool call returns, and it may land unwired — the panel
prints exactly what happened to it and the user can read that for themselves.
Saying "it is beside the hook and wired into its anchor" is a guess, and when it
is wrong the user goes looking for something that is not there.

Everything after the review hook — listed in the hook block as *NOT this turn* —
must not be run, queued or prepared. Queuing them is refused while the stop
stands, but do not spend the turn getting refused: the stop is the deliverable.
(Working on what came *before* the hook is a different thing entirely, and it is
allowed — see the loop below.)

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

### Renumber the reference table

This is the one that goes wrong silently, so do it deliberately.

The collector is a **list**, and the numbered slots are its positions. Drop a row
and every row after it moves up: what was `@image3` is now `@image2`. The wiring
follows automatically — only as many slots are wired as there are files — but a
**reference assignment table in your prompt does not.** It is your prose, and it
is still whatever you wrote before the edit.

So when a prompt for the next stage carries `@image1 = …`, `@image2 = …`, rewrite
it against the bindings listed in the `[REVIEW HALT]` block, which are printed in
the form they will actually take:

```
@image1 / image_1 = ref_00042_.png — TANIHO (HERO)
@image2 / image_2 = ref_00044_.png — APE          ← was @image3 before they cut one
```

`@image2` means **the second line as it stands now**, not the second thing the
earlier stage generated. Getting this wrong renders the wrong character doing the
right beat, and reports no error at all — which is why the run is refused outright
when a collector path merely goes *missing*. A row the user deleted on purpose is
legitimate, so it is not refused; renumbering it correctly is your job.

If a character they dropped was named in the prompt's action ("@image2 hands her
the letter"), rewrite that line too, or say plainly that the beat no longer has
anyone to play it.

**stop** — run nothing further. Confirm what was produced and where it is.

**Neither** — they asked something else, which is ordinary and is usually a
*change* to what was made. That is the loop below: do it, show it, ask again. The
stop stays up, and the stages behind it stay shut, until they actually say
continue or stop.

### Revising during the halt — the loop

A stop is not a yes/no gate. It is where the user gets to **change what was
made**, as many times as they like, before the expensive stage consumes it. Any
of it: images, video, audio, a written line, a prompt, a script.

*"Regenerate the third one, warmer"*, *"make that caption shorter"*, *"re-cut the
clip to five seconds"*, *"swap the second reference for this photo"* — every one
of these is a **neither**. The halt stays up. Do the work, put the result where
the user can see it, and ask again.

You may run work **inline** while the halt stands: `run_workflow_now`,
`apply_canvas_hooks(run_now=True)`, `iterate_step`. What still waits for a
continue is **queuing** the stages after the hook — the chain advancing. If a
tool answers that the chain is stopped, it is telling you not to advance, not to
stop working.

Each pass through the loop:

1. **Make the change**, inline, so its result exists before the turn ends.
2. **Put it back where it is judged.** A produced file goes into the collector
   with `set_canvas_node_params(<collector id>, {"files": "…"})`, keeping the
   lines they kept, in their order. A written answer goes onto the canvas with
   `place_canvas_text`. The `[REVIEW HALT]` block gives you the collector's id,
   and it is the one node you may write to **without** the user having selected
   it — it is the node the halt created, for exactly this.
3. **Say what changed** — which line, which file — and leave the stop up. Ask
   whether to continue, or change something else. Revising is not continuing.

There is no limit on the passes. Ten rounds of "warmer, no, warmer than that" is
the stop doing its job.

### Where the work happens

**Prefer the workflow that is already on the canvas.** It is the one the user
built, the one the next stage reads from, and the one they can inspect while you
work. Re-run the stage that made the thing, with the parameter changed — that is
almost always the whole job, and it leaves a graph that still explains itself.

Open a **separate graph** only when the change genuinely does not fit the
existing one: a different model, a step the hook chain has no node for, a
one-off treatment nothing downstream needs. When you do, say so, and bring the
**result** back into the collector — the side graph is scaffolding, not the new
home of the work. Never rebuild the hook chain on the side because it is easier
than editing it; that leaves the user with two workflows and no idea which one
matters.

If a change would need the hook graph itself altered — a node added, a
parameter permanently different — make that edit on the canvas rather than
working around it, and tell them what you changed.

### Not the same as `stop_hook_run`

`stop_hook_run` abandons a run because something went wrong or a directive said
to abort. `halt_for_review` is a planned pause at a hook the user placed, and it
expects to be resumed. Reaching a review hook is not a failure and must never be
reported as one.
