---
name: iterative-refine
description: Drive an interactive image-refinement loop on the ComfyUI canvas — one generation per turn via iterate_step, feeding each result back as the next input, with go-back to earlier generations. Activate when an `iterate` canvas hook is present, or the user asks to iteratively / step-by-step refine a workflow's output and keep going until they say stop.
allowed-tools: iterate_step
---

# Iterative refine loop

The user wants to refine an image **step by step**: run the on-canvas graph, look at
the result, describe the next change, run again from that result — and be able to jump
back to an earlier generation. This is a **multi-turn loop**: each of the user's turns
is ONE iteration. You do not hold the loop open inside a single reply — you run one
step, show it, ask for the next instruction, and end your turn. The `iterate_step` tool
carries all the state (the numbered generation history) between turns, so the loop
survives across turns even though your context may not.

`iterate_step` does the whole mechanical step in one call: writes the prompt into the
target node, feeds the chosen image into the wired LoadImage node, runs the graph once,
stages the result, updates the LoadImage in place, and returns the numbered history. You
never patch nodes, upload, or run the graph yourself for this loop.

## The wiring (what the user set up)

An `iterate` agentY-hook on the canvas declares the loop:
- its **output** is wired into the **prompt node's text input** → where each prompt goes;
- an **image loader** is wired into its **anchor** → the node whose image is replaced
  with the running result each step (the "feedback" node). A core `LoadImage` or a VHS
  `Load Image (Path)`; `iterate_step` writes whichever kind of reference that node takes.

The `[CANVAS HOOKS]` block names both for the current graph. If either is unwired,
`iterate_step` returns an error telling the user exactly what to wire — relay it and stop.

## Each turn

1. **Read what the user wants for this step** — a fresh prompt, a tweak to the last one,
   or a go-back (see below). If they haven't given a prompt yet (they just asked to start
   the loop), ask what the first prompt/change should be and stop — don't run an empty step.
2. **Call `iterate_step`** exactly once:
   - normal forward step → `iterate_step(prompt="<their prompt>")`.
   - go-back → set `from_generation` (below).
   - a brand-new loop the user is restarting from scratch → add `reset=True`.
3. **Report the result briefly and ASK for the next step.** State which generation this is
   and what it was based on, then invite the next prompt or a go-back. The result is staged
   onto the canvas and the LoadImage now holds it, so the user can inspect it. End your turn.
4. **Stop only when the user says so** ("stop", "that's it", "done", "keep this one"). Then
   confirm the final generation and do nothing else — do NOT `signal_workflow_ready`.

## Going back to earlier generations

The tool returns a `history` list: generation `0` is the **original** image, `1..N` are the
results in order. Map the user's words to `from_generation`:

- "go back to the original / start over from the source" → `from_generation="original"`.
- "go back to the previous one / undo that" → `from_generation="<N-1>"` (the generation
  before the latest — read the numbers from the last `history`).
- "go back to generation 3, then make it warmer" → `from_generation="3"`, and put the new
  change ("make it warmer") in `prompt`.

A go-back does not erase later generations — it just starts the **next** step from the
image you named. The new result is appended as a fresh generation, so the user can still
jump forward again. Always read the `history` in the latest `iterate_step` result to pick
the right number rather than guessing from memory.

## Hard rules

- **One `iterate_step` per turn.** Never loop by calling it repeatedly in a single reply —
  the user must see each result and steer the next step.
- **Never** `apply_canvas_hooks`, `signal_workflow_ready`, `run_research`, or
  `prepare_workflow` for this loop — the graph to run is already on the canvas.
- If a step returns the **no-fetchable-output** error, the saver wrote only to temp: tell
  the user to turn `save_to_output` ON on their save node so results land in ComfyUI's
  history where they can be fed back, then retry.
