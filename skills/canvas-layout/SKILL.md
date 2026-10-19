---
name: canvas-layout
description: Make the user's open canvas readable - arrange nodes so none overlap, split the graph into chunks, draw and name groups, colour them consistently, and save the workflow. Activate when the user says the graph is messy, cluttered or overlapping, asks to "tidy", "arrange", "organise", "group", "name the groups" or "save" the workflow, or after you have added many nodes to their graph and it no longer reads.
allowed-tools: arrange_canvas, group_canvas_nodes, auto_group_canvas, save_canvas_workflow, load_tools
---

# Making a canvas readable

The tools are in the `layout` pack: `load_tools("layout")` first if they are not
in your list. Everything here is cosmetic - groups and positions change nothing
about what runs - and one Ctrl+Z undoes it.

## The order to work in

1. **Look before you move.** `auto_group_canvas()` (default `apply=false`) changes
   nothing and returns the chunks the structure suggests. Read them.
2. **Fix the names, then draw.** Rename the chunks the way the rules below say and
   send them to `group_canvas_nodes`. If the proposal's titles are already right,
   `auto_group_canvas(apply=true)` draws them as they are.
3. **Arrange.** `group_canvas_nodes` and `auto_group_canvas(apply=true)` arrange by
   default; call `arrange_canvas()` on its own for a graph that needs only moving.
4. **Save** only when asked, or when the work is worth keeping and the workflow is
   still "Unsaved": `save_canvas_workflow(name=...)`.

It moves the user's nodes. On a layout they arranged by hand, do it when they ask
(or tell them in one line what you are about to do); on a graph you just built or
grew, tidy it without asking.

## What a chunk is

A chunk is **one step of the work that a person can name and skip over**. Cut by
function and stage, in the order the data flows:

| chunk | what is in it | colour |
|---|---|---|
| Load models | checkpoint / unet / clip / vae / lora / controlnet loaders | `models` (blue) |
| Input media | LoadImage, LoadVideo, LoadAudio, collectors | `inputs` (slate) |
| Prompt | text, text encoders, conditioning (positive and negative together) | `prompt` (yellow) |
| Sampling | sampler, scheduler, guider, noise, empty latent | `sampling` (purple) |
| Decode & save | VAE decode, post-processing, save / preview | `finish` (green) |
| Hooks | agentY hook / review / loop nodes of one stage | `hooks` (orange) |
| anything else | helpers that belong to none of the above | `other` |

Rules:

* **A graph with two generating stages (image, then video; base, then upscale) gets
  its chunks per stage**, each titled with the stage first: "Flux image · Sampling",
  "WAN video · Sampling". A stage ends where an IMAGE / VIDEO / AUDIO wire leaves it.
* **No chunk of one node unless that node is a whole step** (a loader on its own is
  fine; a lone reroute is not - it belongs to what it feeds).
* **Nothing is in two groups**, and a group never contains nodes of two stages.
  Groups are rectangles: if two chunks interleave, the chunking is wrong.
* **A small graph is not chunked.** Under about four nodes, or one kind of thing,
  a box says nothing - `auto_group_canvas` returns an empty proposal; leave it.
* **Never touch a group the user drew** beyond what they asked: do not rename,
  recolour or delete it. Group around their nodes only when they asked for grouping.

## What a good name is

Name a group **by what it produces or does for the next step**, in the user's
words, two to four words, title case, no node class names, no ids.

* Good: "Load models", "Portrait prompt", "Base render", "4x upscale", "Hero
  image - save", "Mask from reference".
* Weak: "Group 1", "KSampler stuff", "Nodes 4-9", "CLIPTextEncode", "Misc".
* Prefer the result over the mechanism: "Upscaled 4K" beats "UltimateSDUpscale".
* When the proposal's title is generic and you know what the chunk is FOR in this
  graph ("Prompt" -> "Raven portrait prompt"), say that. Keep the generic one when
  you do not know - a plain true name beats a guess.
* Keep names stable: renaming a group later is `group_canvas_nodes` with
  `rename_from`, not a second group.

## Colours

Use the role colours (`"color": "models"` etc.) so the same kind of chunk is the
same colour on every graph. Do not invent a colour per group; a rainbow says
nothing. One extra colour at most, for the chunk the user is asking about.

## Layout

* **Data flows left to right.** Loaders at the left, the save at the right;
  `arrange_canvas` does this by distance from the inputs.
* **Groups sit in flow order** and never overlap each other or loose nodes.
* **Keep hooks beside their stage.** A hook's group sits next to the stage it
  belongs to (same row band, directly after it in the flow), never at the far end
  of the canvas. Name it for the stage: "Hooks - Base render".
* **Tidy after you add.** Growing a graph with `edit_canvas_graph` drops nodes
  beside their neighbour; finish with `arrange_canvas` (or `group_canvas_nodes`
  with `arrange=true`) when the result no longer reads.

## Saving

* An unsaved workflow needs a **short file name for what it does**
  ("flux_portrait_upscale"), not "workflow" or the date.
* The open workflow is saved over its own file only with `overwrite=true`. Without
  it you get the file name back: tell the user which file would be replaced and let
  them decide. A name that is already another workflow is refused, never replaced.
* Say what you saved and where, in one line.
