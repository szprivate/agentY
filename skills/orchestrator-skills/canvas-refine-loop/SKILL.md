---
name: canvas-refine-loop
description: Run the workflow the user already has open in the ComfyUI canvas over and over, judging each output against a condition they stated and changing one value until it is met — the closed "keep trying until it looks right" loop. Activate when the user asks for a loop / to iterate / to keep trying until an output condition holds, on a graph that is already working. Not for step-by-step refinement they steer turn by turn (that is iterative-refine).
allowed-tools: refine_canvas_until, get_canvas_node, set_canvas_node_params
---

# Refine loop on the open canvas

The user has a workflow that works. They want you to run it, look at what came
out, decide whether it meets a condition they stated, and try again if it does
not — without them pressing Queue between attempts.

> *"Ok let's try a loop — you change the prompt until the woman's position in the
> output matches her position in the original frame."*

`refine_canvas_until` is the whole loop in one call. It queues **their** graph
(patched, never duplicated), judges each output, rewrites the one value it is
varying, and runs again until the condition is met or the budget is spent.

## Which loop is this?

| The user wants | Tool |
| --- | --- |
| "keep trying until *X*" — you judge, they wait | `refine_canvas_until` |
| "now make it warmer" … "now go back to gen 3" — they judge, one step per turn | `iterate_step` (skill: `iterative-refine`) |
| one run of a graph you assembled | `run_workflow_now` |
| change a value and stop | `set_canvas_node_params` |

The tell is a **stopping condition**. "Until it matches", "until there's no text
in it", "until she's on the left" — that is this loop. "Make it warmer" is not.

## Before you call it

**Say the plan in one line and then call.** Three facts, no more: what you will
vary, what you will judge against, and how many runs it may cost. *"I'll vary the
prompt on Positive (#6), judge each result against her position in the loaded
frame, up to 4 runs."* Each run is a real generation, so the cost is the user's to
see coming — but do not stop and ask for permission unless they asked you to.

**Write the condition as something visible in the picture.** It becomes the
criteria every output is graded on, so it has to be checkable by looking:
*"the woman stands in the same place in the frame as in the reference"* is
judgeable; *"looks better"* is not, and the loop will churn.

**Let it find the reference.** With no `references`, it compares against the
images the graph's own loaders hold — which is exactly what "the original frame"
means when the graph is editing that frame. Pass `references` only for an image
that is *not* in the graph.

**Let it find the value.** With no `node_id`, it picks the prompt when there is
exactly one; otherwise it comes back with the candidates. Answer that by calling
again with a `node_id` — do not go hunting through `[CANVAS GRAPH]` first. It
never picks a negative prompt on its own; if the user wants the negative varied,
name it.

**`vary_seed=True`** when the goal depends on the roll — pose, composition,
placement, how many people are in shot. Leave it off for wording, style, content:
you want a changed result to be evidence the *value* did it.

## Reading the result

`status` is the whole answer:

- **`matched`** — a run met the condition. The canvas holds the value that did
  it. Show that output, say what changed about the value, and stop.
- **`missed`** — every run was judged, none passed. Show the closest one, say
  what the judge kept objecting to (it is in `history`), and ask whether to keep
  going or change tack. Calling again continues from where it stopped.
- **`unjudged`** — the judge could not be read. This is **not** a near-miss and
  **not** a pass: the output was never graded. Say exactly that.
- **`stalled`** — no new value was left to try. Usually the condition is not
  reachable by rewording this value; suggest varying something else, or a sharper
  condition.
- **`interrupted`** — the user typed something. Report where it got to, answer
  them, and do not restart it unasked.

`history` carries every run: the value tried, the output, and why it was judged
the way it was. That is what you tell the story from — "run 2 moved her too far,
run 3 landed it" — rather than describing only the last image.

## Hard rules

- **Never `signal_workflow_ready` after this.** Everything already ran. Signalling
  would queue the graph a second time at the end of the turn.
- **Never assemble or research a workflow for this.** The graph to run is the one
  they have open. `prepare_workflow` and `run_research` have no part in it.
- **One call per request.** The tool owns the loop; do not call it repeatedly to
  build your own. Call it a second time only when the user asks you to continue.
- **The user's original value is in `original_value`.** If they want it back,
  `set_canvas_node_params` puts it back — offer that when the loop ended `missed`.
- **The budget is a ceiling, not a target.** If the tool says it capped you, pass
  that on with where the setting lives; do not loop around it by calling again
  unprompted.
