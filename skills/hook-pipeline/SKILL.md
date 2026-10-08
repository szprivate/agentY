---
name: hook-pipeline
description: Build a hook pipeline on the user's canvas yourself — stages, reviews, loops, parallel branches — out of agentY's own nodes, wired with the execution wire. Activate when the user asks you to set up, design, build or restructure an automation pipeline or a hook chain ("build me a pipeline that…", "set this up as hooks", "turn this into a workflow I can re-run"), or when a request changes what the stages of an existing chain are.
allowed-tools: edit_canvas_graph, set_canvas_node_params, delete_canvas_nodes, get_canvas_node
---

# Building a hook pipeline

A hook pipeline is the user's plan, drawn: stages of work, the order they run in,
where a person looks, what repeats. You can draw it for them. You build it; they
read it and press **agentY hooks** to run it.

**Build, show, stop.** Do not run a pipeline in the turn you build it. It is only
read from the canvas when the user sends a message, and a plan they have not seen
is not theirs yet. End the turn with the pipeline on the canvas and one short
list of its stages.

## 1. Plan the stages first, in the chat

Before any node, write the pipeline as a numbered list and get it right there:

- **One stage = one kind of work with one result.** "Write the screenplay",
  "one reference image per character", "animate the chosen references". If a
  stage needs an *and* to describe it, it is two stages.
- **What does each stage read?** The value of an earlier stage, a file, a node
  already on the canvas.
- **Where does a person need to look before money is spent?** That is a review.
  Put one before every expensive stage whose input was just generated.
- **What repeats until it is right?** That is a loop, and it needs someone to
  judge it: a review inside it.
- **What does not depend on each other?** Those are branches; they run at the
  same time, each in its own conversation.

If the user asked for the pipeline in one line, show this list and build it. If
they are restructuring one they built, show what changes and wait for their go.

## 2. The nodes

| node (`class_type`) | what it is | settings (`params`) |
|---|---|---|
| `AgentYHook` | a stage of work | `directive`, `purpose`, `remember` |
| `AgentYReview` | who looks at the stage before it | `reviewer`: `human` or `agent`; `notes` |
| `AgentYLoopStart` | where a loop begins | none |
| `AgentYLoopBreak` | where it ends | `condition`, `max_rounds`, `forward` |
| `AgentYJoin` | where branches meet again | none |

**The three purposes of a hook** — write them exactly like this, with the spaces:

- `set / sweep parameter` — produces the VALUE of the input its `out` is wired
  to. One value sets it; several run the graph once each. Use it when a node that
  does the generating is already on the canvas and the stage decides what goes
  into it: *"three prompts per character, one run each"*.
- `make workflow` — you build and run a whole workflow from the directive. Use it
  when nothing on the canvas does this stage's work yet: *"upscale 2× and add
  film grain"*, *"animate the chosen references, 5 s each"*.
- `text only` — a written answer and nothing else: *"extract the characters and
  describe each"*.

**A directive is a brief, not a label.** It is all the stage will be told. Write
what to make, from what, how many, and what must be true of the result.

**A review:**

- `reviewer: "human"` — the run STOPS there and waits for the user. `notes` is the
  question they are asked. Use it wherever taste decides: story, casting, look.
- `reviewer: "agent"` — the QA agent judges and the run carries on. `notes` says
  what needs judgement; the measured checks (`aspect_ratio`, `resolution`,
  `sharpness`, `grain`, `likeness`, …) are set as params too. Use it where the
  standard can be written down.

## 3. The three wires

They mean three different things. Getting them mixed up is the usual mistake.

- **`exec` → `exec`: WHEN.** The execution wire. Every node above has an `exec`
  output and an `exec` input (a join has `execs.exec0`, `execs.exec1`, …).
  Stages run in the order this wire is drawn, and nothing else decides order.
  Connect it through EVERY node of the pipeline, reviews and loop nodes included.
- **`out` → `anchors.anchorN`: WHAT IT READS.** A stage that uses an earlier
  stage's result needs that stage's `out` in one of its anchors. Being after it
  on the exec wire does not give it the value.
- **`out` → a real node's input: WHAT IT PRODUCES.** For `set / sweep parameter`,
  wire `out` into the input it fills (a prompt, a seed).

The exec wire only connects to exec. A stage's `out` never goes into an `exec`.

**Forks and joins.** Wire one `exec` output into two stages and you have two
branches. To bring them together, wire each branch's last `exec` into an
`AgentYJoin` (`execs.exec0`, `execs.exec1`) and the join's `exec` into the stage
that needs both. Without a join, branches simply end.

**A loop** is everything on the exec wire between a `AgentYLoopStart` and a
`AgentYLoopBreak`. Put the review that judges it last in the body, just before
the break. With a human review the break's `condition` can stay empty; with an
agent review or none, write the condition.

## 4. Building it

One `edit_canvas_graph` call, all or nothing. Add the nodes in run order, each
`near` the one before it so the chain reads left to right, then the wires.

```json
[
 {"op": "add", "class_type": "AgentYHook", "ref": "story",
  "params": {"purpose": "text only",
             "directive": "Write a 6-shot screenplay about …, no dialogue, 2 s per shot."}},
 {"op": "add", "class_type": "AgentYReview", "ref": "story_ok", "near": "story",
  "params": {"reviewer": "human", "notes": "Is the story right, or what should change?"}},
 {"op": "add", "class_type": "AgentYHook", "ref": "cast", "near": "story_ok",
  "params": {"purpose": "text only",
             "directive": "Extract the characters; age, look and wardrobe for each."}},
 {"op": "add", "class_type": "AgentYHook", "ref": "places", "near": "story_ok",
  "params": {"purpose": "text only", "directive": "Extract the locations."}},
 {"op": "add", "class_type": "AgentYJoin", "ref": "both", "near": "cast"},
 {"op": "add", "class_type": "AgentYHook", "ref": "frames", "near": "both",
  "params": {"purpose": "make workflow",
             "directive": "One start frame per shot, characters and locations as described."}},

 {"op": "connect", "from": "story", "output": "exec", "to": "story_ok", "input": "exec"},
 {"op": "connect", "from": "story_ok", "output": "exec", "to": "cast", "input": "exec"},
 {"op": "connect", "from": "story_ok", "output": "exec", "to": "places", "input": "exec"},
 {"op": "connect", "from": "cast", "output": "exec", "to": "both", "input": "execs.exec0"},
 {"op": "connect", "from": "places", "output": "exec", "to": "both", "input": "execs.exec1"},
 {"op": "connect", "from": "both", "output": "exec", "to": "frames", "input": "exec"},

 {"op": "connect", "from": "story", "output": "out", "to": "cast", "input": "anchors.anchor0"},
 {"op": "connect", "from": "story", "output": "out", "to": "places", "input": "anchors.anchor0"},
 {"op": "connect", "from": "story", "output": "out", "to": "frames", "input": "anchors.anchor0"},
 {"op": "connect", "from": "cast", "output": "out", "to": "frames", "input": "anchors.anchor1"},
 {"op": "connect", "from": "places", "output": "out", "to": "frames", "input": "anchors.anchor2"}
]
```

If the result lists errors, nothing was changed: fix them and send the whole list
again. If it lists an input still unwired, finish the wiring.

**On a canvas that already has nodes**, build around them: a generator the user
already placed is the target of a `set / sweep parameter` stage's `out`, and its
save node is what a review's `anchors.anchor0` reads. Do not add a `make
workflow` stage for work a node on the canvas already does.

**Restructuring an existing chain**: change a stage's settings with
`set_canvas_node_params`, add what is new with `edit_canvas_graph`, remove what
is dropped with `delete_canvas_nodes` — and re-wire the `exec` across the gap,
or the chain ends where the node was.

## 5. Check it before you hand it over

Read the canvas graph back and check, stage by stage:

1. Every node of the pipeline is on the exec wire: one unbroken line from the
   first stage, splitting only where branches are meant.
2. Every stage that uses an earlier result has that stage's `out` in an anchor.
3. Every `set / sweep parameter` stage has its `out` in a real input.
4. Every loop has a start and a break, and something that judges it.
5. Every expensive stage whose input was generated has a review in front of it.

## 6. Hand it over

End the turn. Say, in a short numbered list, what the pipeline does stage by
stage and where it stops for them — and that **agentY hooks** runs it. Offer one
change if you see an obvious one. Do not run it, and do not describe node
positions; they can see the canvas.
