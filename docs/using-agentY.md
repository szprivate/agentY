# Using agentY

A practical guide to driving **agentY** — the AI agent that builds and runs
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows from natural
language, right inside ComfyUI's sidebar. For install/setup see the
[README](../README.md); this guide is about *using* it once it's running.

![agentY sidebar chat next to the ComfyUI graph](images/overview.png)

*The agentY tab lives in ComfyUI's left sidebar. You chat on the left; every
result the agent produces is dropped onto the graph on the right as a loader
node, ready to wire into your next step.*

---

## Contents

- [The big picture](#the-big-picture)
- [Starting a session](#starting-a-session)
- [Staying up to date](#staying-up-to-date)
- [The chat panel](#the-chat-panel)
  - [Talking to a turn that is already running](#talking-to-a-turn-that-is-already-running)
- [Generating & editing](#generating--editing)
  - [Finding reference images on the web](#finding-reference-images-on-the-web)
  - [Marking up an image](#marking-up-an-image)
  - [Asking about a video](#asking-about-a-video)
  - [Seeing the plan first](#seeing-the-plan-first)
  - [Long jobs: batches and background work](#long-jobs-batches-and-background-work)
- [Slash commands](#slash-commands)
- [**The hook system**](#the-hook-system)  ← the most powerful part
  - [Tagging a reference](#tagging-a-reference-the-agenty-add-tag-node)
  - [Asking the agent to change a setting](#asking-the-agent-to-change-a-setting)
  - [Letting the agent edit the whole graph](#letting-the-agent-read-and-edit-the-whole-graph)
  - [Filling a slot nothing is wired to](#filling-a-slot-nothing-is-wired-to)
  - [Loops: keep trying until it's right](#loops-keep-trying-until-its-right)
  - [Screenshots of your workflow](#screenshots-of-your-workflow)
  - [Dry run: check the logic first](#dry-run-check-the-logic-before-you-pay-for-it)
  - [Review: stop and pick what continues](#review-stop-and-pick-what-continues)
  - [The keep switch](#the-keep-switch-should-this-outlive-the-run)
  - [Memorize: produce once](#memorize-produce-once-reuse-until-something-changes)
  - [Naming what a hook produces](#naming-what-a-hook-produces)
- [Slack: a second way in](#slack-a-second-way-in)
- [Checking outputs (QA)](#checking-outputs-qa)
  - [Ticking the boxes instead of writing them](#ticking-the-boxes-the-agenty-qa-briefing-node)
  - [Does it match the reference?](#does-it-match-the-reference)
  - [Which of these is best?](#which-of-these-is-best)
- [The agentY python node & collectors](#the-agenty-python-node--collectors)
- [Settings & secrets](#settings--secrets)
- [MCP servers](#mcp-servers)
- [Token usage & cost](#token-usage--cost)
- [Memory](#memory)
- [Choosing models](#choosing-models)
- [Custom workflow templates](#custom-workflow-templates)
- [Building a node for a new model](#building-a-node-for-a-new-model)
- [Troubleshooting](#troubleshooting)

---

## The big picture

agentY is a **free-agent orchestrator**: one agent owns each turn with the full
toolset. You describe what you want; it researches a template, assembles a
ComfyUI workflow, submits it, waits, optionally QA-checks the result with a
vision model, and stages the output back onto your graph. It can also delegate
to specialists (research / assembly / info / web), spawn subagents, and author
skills live.

Two ways to drive it:

1. **Chat** — type in the sidebar (*"generate a cinematic wide shot of Tokyo at
   night"*).
2. **Canvas hooks** — annotate the graph with **`agentY hook`** nodes and ask
   the agent to run it. This is where agentY becomes a graph-native automation
   tool rather than just a chat box — see [The hook system](#the-hook-system).

Everything is local: chat history lives in a SQLite file, results are staged
into ComfyUI's `input` dir and dropped as nodes, and no Docker/Postgres/S3 is
involved.

---

## Starting a session

1. **Start the agent host** (the SSE backend the sidebar talks to, default
   `http://127.0.0.1:5000`):

   ```powershell
   .\run_agent.ps1
   ```

   Restarting `run_agent.ps1` is how you pick up agent-side code/config changes.
   If the host isn't running, the chat panel shows a **▶ Start server** button.

   On start it also **checks for updates** and fast-forwards agentY, `agenty_core`
   and the ComfyUI sidebar extension — see [Staying up to date](#staying-up-to-date).

2. **Open ComfyUI** in your browser (default `http://127.0.0.1:8188`) and click
   the **agentY** tab in the left sidebar.

3. Start typing. Type `/` for the slash-command menu; use the thread dropdown to
   revisit past conversations.

> If the backend runs on a non-default URL, set it once in the browser console:
> `localStorage.agentY_backend = "http://host:port"`.

### Staying up to date

Each start checks the remotes and fast-forwards the three checkouts that make up
agentY: this repo, `agenty_core`, and the sidebar extension (both the clone
ComfyUI loads and a dev clone beside agentY, if you have one).

It is deliberately timid, because it runs unattended over your working copy:

- a repo with **uncommitted changes** is left completely alone — never stashed,
  never discarded;
- a repo with **unpushed commits** is left alone — never rebased, never reset;
- it is **`--ff-only`**, so a diverged branch reports and stops rather than merging;
- being **offline** is a shrug, not a failed start.

Anything it declines to do it says out loud, so a stale checkout is never silent:

```
[update] agentY is 3 commit(s) behind origin/main - fast-forwarding...
[update] agentY updated -> 1895057
[update] agenty_core has uncommitted changes - skipping (nothing was touched).
[update] agentY-comfyuiConnect is up to date.
```

If the pull touched `requirements.txt`, dependencies are reinstalled before the
app starts. If it touched the extension, you're told to restart ComfyUI (or just
reload the browser, for JS-only changes).

Turn it off with `auto_update = false` (Settings ▸ advanced ▸ Behaviour),
`.\run_agent.ps1 -NoUpdate`, or `AGENTY_NO_UPDATE=1`. If your ComfyUI isn't in an
obvious spot next to agentY, point `comfyui_dir` at it so the extension is found.

> `run_agent.ps1` updates *itself* too, but the copy already running is the old
> one — changes to the launcher apply from the next start.

---

## The chat panel

![The agentY chat panel](images/chat-panel.png)

**Top bar (left → right):**

| Control | What it does |
|---|---|
| **Thread dropdown** | Switch between saved conversations. |
| **➕ New chat** | Start a fresh thread. |
| **🗑 Delete** | Delete the current conversation. |
| **📊 Token usage** | Open the [cost breakdown](#token-usage--cost). |
| **🖼 Auto-graph** | Toggle **autograph** on/off — whether finished workflows/results are loaded onto the canvas automatically. Highlighted when on. Takes effect immediately (no restart). |

**Composer:**

- **📎 Attach** — add an image to your message (for edits, references, etc.).
- **Message box** — your prompt. `/` opens the command menu.
- **➤ Send**.
- **Model** — a live **Switch model…** picker for the orchestrator (only vendors
  whose API key is set appear), and an **agent scope** selector (*All agents* or
  a specific role) so a model switch can target one stage. This is the in-panel
  equivalent of [`/switch_model`](#slash-commands).

### Talking to a turn that is already running

Anything you type while the agent is working becomes a **⏳ chip** above the
composer and is sent when the turn ends. You don't have to wait for it, though:

| On the chip | What happens |
|---|---|
| **↳** | Hands the message to the **running** turn. The agent reads it at its next step and carries on from where it is — it doesn't start over. |
| **Shift + ↳** | Same, but **cancels the step it was about to take** so it reads you first. Nothing already produced is undone. |
| **✕** | Drop it. |

Two things worth knowing. The agent picks the message up **between tool calls**,
so if it is in the middle of a long one — a batch of eleven generations, a
specialist doing research — your words land when that finishes. To stop something
already running, use **⏹ Stop**, which also interrupts ComfyUI. And if the turn
happens to end before the message reaches the agent, nothing is lost: it goes
back on the queue and is sent with the next turn, and the panel says so.

Messages with an attached image stay queue-only — a mid-run message reaches the
agent as text.

---

## Generating & editing

Just describe the outcome. Some examples:

- *"Generate a cinematic wide shot of Tokyo at night."*
- *"Edit this photo to make it daytime."* (attach an image with 📎)
- *"Make 5 variations of a red sports car, different angles."*
- *"Upscale the last image with UltimateSD."*

Each finished image/video appears as a **`LoadImage` / video-loader node on your
graph** (staged into ComfyUI's `input` dir), so the result is immediately
wireable into your next workflow. The chat carries the agent's *text*; the media
lands on the canvas. Toggle **🖼 autograph** off if you'd rather the agent not
place nodes automatically.

### Finding reference images on the web

Ask for references and they arrive on the canvas, the same way generated images
do:

> *"Search the web for images of this car — from every angle, and the interior."*

The agent searches, picks, downloads into ComfyUI's `input` directory, and every
picture it keeps is dropped onto your graph as a loader node. **Downloading is
showing**: there is no extra step, and nothing to ask for. If you name a number
("five options") it stages that many; otherwise it takes the best one or two,
because a reference that is going to feed a generation wants the *right* picture
rather than a pile.

It skips watermarked stock previews where it can — image search is full of them,
and a logo across the middle ruins the picture both to look at and to generate
from.

A file that turns out not to be an image (a hotlink block, a login page served at
`…/photo.jpg`) is not placed: a loader node pointing at an HTML page shows nothing
and fails when the graph runs.

### Marking up an image

> *"Circle the bolts."*  *"Put a red box around the logo."*  *"Show me where the
> damage is."*

The marks are drawn **on top of** your picture — it is your photograph with ink
on it, not a re-generated lookalike, so nothing else in the frame moves or
changes. You choose the shape (circle, box, arrow), the colour, and whether each
mark is numbered or labelled. The result lands on the canvas like any other
output.

Locating the thing you named is the only part that needs a model; everything
after that — de-duplicating overlapping hits, scaling to the image, drawing —
is fixed, so the same request twice gives the same marks.

### Asking about a video

Attach a clip and ask about it. Frames are sampled across its length and read by
a video-understanding agent, so *"what happens in this shot"*, *"when does the
camera start moving"* and *"is the logo visible at the end"* are answerable
without you scrubbing through it. It is the video counterpart of the vision
agent that reads your images, and it uses the same **Vision** tier.

For cutting rather than reading, ask for the shots: agentY can find the cuts in
a clip and write one file per shot, which is what the
[collectors](#the-agenty-python-node--collectors) then hand back to a workflow.

### Seeing the plan first

Anything that takes more than one step — a graph with several hooks, a chain of
stages, a multi-part request — is announced before it happens: the agent writes
the plan into the chat as a short numbered list and then gets on with it. You are
not asked to confirm, because you can read it while it works and interrupt with
**↳** (see [talking to a running turn](#talking-to-a-turn-that-is-already-running))
if it got you wrong.

If you'd rather it **wait**, say so — anywhere the agent reads:

- in your message: *"Show me the plan and wait for my go before you run anything"*
- in a hook directive, where it becomes a standing rule for that graph:
  *"Ask me first before you start generating"*
- in the project's memory, where it applies to every thread on that project

Then nothing runs until you answer: the tools that would generate, queue or
execute refuse for that turn, and the panel says **✋ holding**. Your next message
releases it — a *yes* runs the plan as stated, a change is applied first. One
round trip, not one per step; the next new request asks again. To override a
standing rule for a single turn, just say *"go ahead"* or *"just do it"*.

### Long jobs: batches and background work

*"Run every image in this folder through that workflow."* A run over many inputs
does not block the conversation: it is scheduled as a **batch job** and a
detached worker drives ComfyUI on its own, so you can keep talking while it
works. Ask how it is going and the agent reports progress; ask it to stop and it
stops.

Stages chain. With two workflows, each input goes through the first, its output
feeds the second, and you get one final file per input — which is the usual shape
for *"upscale everything, then add grain"*.

Some things finish long after the turn that started them — an async provider
render, for instance. Those arrive on their own: the file is downloaded, dropped
onto the canvas as a loader node, and a notification tells you it landed, whether
or not you were looking at the panel.

---

## Slash commands

Type `/` in the composer for an autocomplete menu.

| Command | Action |
|---|---|
| `/restart` | Restart the agent pipeline |
| `/stop` | Stop and shut down the agent |
| `/unload` | Unload Ollama models from VRAM |
| `/clear_vram` | Clear ComfyUI GPU VRAM |
| `/images` | List images generated in this thread |
| `/project_memory` | Inspect and forget what is remembered for **this project** (characters, style, named references) |
| `/clearhistory` | Delete all conversation history |
| `/switch_model <target> <provider,model>` | Set a model **tier**, a single role, or `all` (e.g. `/switch_model fast_utility dashscope,qwen3.6-flash`). Saved to `settings.local.json` |
| `/add_workflow <path>` \| `/add_workflow canvas <name>` | Register a workflow template (a JSON file, or the open graph) |
| `/remove_workflow <name>` | Remove a registered template |
| `/resend` | Resend the first user message |

---

## The hook system

Hooks are agentY's headline feature: **instructions you attach to the graph
itself**. Drop an **`agentY hook`** node (category **agentY**), wire it, type a
directive, and ask the agent to run the graph. On a normal **Queue Prompt** a
hook is *inert* — it's an identity passthrough nothing downstream needs, so
ComfyUI never executes it. It only means something when the **agentY agent**
runs the graph.

### Anatomy of a hook

![Two make_workflow hooks wired into a pipeline](images/hook-chain.png)

*Two `agentY hook` nodes wired output → anchor form a two-stage pipeline:
"generate a scene" feeds "animate it".*

- **`anchor` input (auto-grows).** Wire any node's output in as **context /
  input**. Each time you wire one, a fresh empty anchor slot appears, so a single
  hook can gather several inputs. Type-agnostic (image, video, or a scalar).
- **`out` output.** The value(s) the hook produces, which you wire into a real
  node's input (or into the next hook). Also type-agnostic.
- **`directive`** — the natural-language instruction / prompt.
- **`purpose`** — what the hook *is* (below).
- **`remember`** — should what this hook produced outlive the run? (see
  [the keep switch](#the-keep-switch-should-this-outlive-the-run)). It reads
  *bake into subgraph* on `make_workflow` and *memorize result* everywhere else,
  because that is what keeping each of them means.

To **disable a hook without deleting it**, bypass it (`Ctrl+B`) or mute it
(`Ctrl+M`) like any other node — the agent skips hooks in those modes. There's no
separate toggle to remember, and a disabled hook is obvious on the canvas.

> **Mental model:** a hook is an **upstream producer**. It reads its wired anchor
> inputs as context and *produces* the value(s) for its output, which you wire
> into the input it should fill. The agent fills the input the hook's output is
> wired to — it doesn't guess "the connected node" from prose. Wire the output
> where the produced value belongs.

### What the agent can *see* on an anchor

Wire a **Load Image / Load Video** node (or an [agentY
collector](#the-agenty-python-node--collectors)) into an anchor and the agent
sees the picture straight away — the node names a file, so there is something to
look at before anything runs.

Wire **anything else** — a `VAEDecode`, an upscaler, an `ImageBlend`, a mask op —
and there is no file anywhere: that wire carries a **tensor that only exists
during a run**. agentY handles this by *tapping* the wire before the turn starts:
it trims your graph down to just that node's ancestors, renders it, and hands the
agent the resulting file. You'll see a `🔎 Rendering hook input(s)…` line while it
happens.

- Only the **upstream** part runs. Your savers and any unrelated branch are not
  in the tap graph, so nothing lands in your output folder and your pipeline is
  not kicked off. (A tapped `VIDEO` wire is the one exception — ComfyUI has no
  preview-video node, so those go to `output/agentY_tap/`.)
- If the graph has **already run**, ComfyUI serves it from cache and the tap is
  near-instant. On a cold graph it really does render that branch first.
- `IMAGE`, `MASK`, `LATENT` (decoded with the graph's own VAE) and `VIDEO` wires
  are supported. A batch contributes its first few frames.
- Turn it off with **`hook_tap_tensors`** in Settings → Behaviour (or
  `AGENTY_HOOK_TAP=0`). Tuning: `AGENTY_MAX_HOOK_TAPS` (4 wires per turn),
  `AGENTY_HOOK_TAP_FRAMES` (4), `AGENTY_HOOK_TAP_TIMEOUT` (300s).

### Tagging a reference (the `agentY add tag` node)

Wire five images into one hook and every one of them is "an image". The **`agentY
add tag`** node fixes that. It sits *on a wire* — `Load Image → add tag →
anywhere` — and carries two fields:

- **`tag name`** — a short handle for this reference: `hero_face`, `alley_light`.
- **the prompt box** — what the agent should *take* from it: *"the face only —
  not the hair, not the wardrobe"*, *"the light, not the architecture"*.

Both are optional and they do different jobs. The prompt narrows what a reference
is *for*, and it always has: the agent describes that image with your question
instead of describing it whole, and carries the restriction into the prompt it
writes. That is how a reference for the *lighting* stops dictating the
architecture.

The tag **names** it. Once one tag exists anywhere on the canvas, typing `#` in
any hook's prompt box opens a small menu of every tag in the scene — keep typing
to filter, `↑`/`↓` to move, `Enter` or `Tab` to insert, `Esc` to dismiss. So a
directive can say:

```
Put #hero_face in the alley, lit like #alley_light. Wide shot.
```

and each `#name` points at exactly one node. The agent is handed the mapping with
the graph (`#hero_face → node 43 (LoadImage)`), so it resolves the name instead of
guessing which of the five wired inputs you meant. A `#name` that no tag node
carries is flagged in the [dry run](#dry-run-check-the-logic-before-you-pay-for-it)
rather than quietly matched to the nearest input.

**A named reference is an input — you can skip the anchor wire.** Naming a tag in
a hook's prompt hands that hook the reference, the same as wiring it in. The hook
block reports it under `NAMED IN THE DIRECTIVE`, the run keeps that node (and
whatever it takes to produce it) in scope instead of trimming it away, and a
`make_workflow` hook that names one builds an image-to-image job rather than
treating the prompt as text-to-image. So five references and three hooks do not
mean fifteen wires — tag each image once, name the ones each hook needs.

**Making a tag outlive the graph.** Turn on **`remember for the project`** on the
tag node and the reference is written into [project memory](#memory) as a named
entry — the file's path and what you said it is for. From then on `#hero_face`
resolves in a *new* graph too, and a Claude Desktop session on the same ComfyUI
can read it. What it resolves to there is a **file**, not a node: the agent
uploads and wires it rather than anchoring it.

Turning the switch back off stops refreshing that entry but does **not** delete
it — a graph that happens not to contain the tag must never silently forget it.
Forgetting is deliberate and yours: `/project_memory` (or **agentY settings ▸
Viewers ▸ 📌 Project memory**) opens an editor that lists everything remembered
for this project and lets you delete what should no longer be true. It doesn't
let you write — entries are established by the agent or by this switch, so there
is only ever one source for each file.

Two things still want the wire:

- **A reference that has to reach a node in your own graph.** A name cannot make
  ComfyUI carry a value from one node to the next — only a wire does that. So the
  image a sampler branch actually consumes, or the `LoadImage` an `iterate` hook
  swaps each turn, stays wired.
- **A mid-graph tensor** (a `VAEDecode`, an upscaler, a mask op). Those carry no
  file of their own, and only a *wired* anchor gets rendered to disk for the agent
  to look at. Tag a saved image, or wire the tensor into the hook.

Because the node lives *on the wire*, there is no node id to keep in sync:
whatever is plugged into it is what it is about. Rewire it and the tag follows.
Anchoring a hook on the tag node itself is fine too — the agent reports the
`LoadImage` behind it, not the annotation.

> Spaces and a leading `#` are forgiven (`#hero face` and `hero_face` are the same
> tag), so the name you see in the menu is always the one that resolves.

> This node was called **`agentY ref note`** before it grew the tag field. Saved
> graphs keep working and their prompt text is preserved — only the name and the
> extra field changed.

### Asking the agent to change a setting

You can just say it: *"turn QA off"*, *"stop putting workflows on my canvas"*,
*"don't retry failed QA outputs"*, *"let yourself see the whole graph"*. The agent
changes it, tells you what it was before, and says when it takes effect (usually
your next message; `auto_update` at the next server start).

It can only touch a **short, fixed list** — behavioural switches and a few small
numbers:

`canvas_full_graph` · `autoload_workflows_into_canvas` · `hook_scoped_graph` ·
`hook_tap_tensors` · `comfyui_console_lines` · `auto_update` · `memory.enabled` ·
`qa.enabled` · `qa.max_retries` · `qa.max_outputs` · `qa.max_references` ·
`qa.video_frames` · `llm.history_window`

Everything else — model choices, folders, server URLs, API key variables — stays
yours, in the Settings dialog. Not because the agent couldn't write them, but
because a misread sentence that flips `qa.enabled` costs you a QA pass, while one
that rewrites `output_dir` or `comfyui_url` costs you your work or points the app
at the wrong machine. Ask for one of those and it will tell you which setting you
want rather than finding a way to do it.

Changes go to `config/settings.local.json`, never to the committed defaults — so
they survive updates, and undoing one by hand means deleting one line.

### Letting the agent read and edit the whole graph

By default the agent sees only the nodes you have **selected**, and can only
change those. Selecting is how you say *"this one"* — but it also means every
edit starts with "go and click the node first".

Turn on **`canvas_full_graph`** (Settings ▸ Behaviour, or
`AGENTY_CANVAS_FULL_GRAPH=1`) and it sees the whole workflow you have open — every
node, with its id, type, your title and its values — and can change any of them
without you selecting anything. *"Set the sampler to 30 steps"*, *"what does this
graph actually do?"*, *"find the node that's writing to the wrong folder"* all
work directly. A selection still narrows it to what you mean.

It is **off by default because it costs tokens on every canvas turn**, whether or
not the turn was about the graph — roughly 250 for a 20-node workflow, ~1.5k for
200 nodes, capped past that. Worth turning on if you edit graphs by chatting;
leave it off if you mostly generate.

Values in the listing are shortened to fit one line per node; the agent is told to
re-read a truncated value in full before rewriting it, so a long prompt does not
get half-rewritten. Editing never queues the graph — you run it yourself. The one
exception is a loop you asked for, below.

### Screenshots of your workflow

Ask for a picture of the graph and you get one:

> *"send me a screenshot of my workflow on Slack"*

![A workflow photographed by `screenshot_canvas`](images/canvas-screenshot.png)

It is your canvas as **you** have it — your node positions, your groups and
colours, whatever you have collapsed — not a re-drawing of the same workflow from
its JSON. It is cropped to the graph, so there is no empty background around it,
and ComfyUI's own render-stats overlay is left out. The agent takes it, then hands it to Slack if that is where you wanted
it. Your view does not move: the zoom is put back before the browser paints a
frame, so you will not see anything happen.

Two things worth knowing.

**Big graphs come back as an overview.** ComfyUI stops drawing node text below a
certain zoom, so a workflow too large to fit on one page at readable size arrives
showing its shape and wiring with no labels on the nodes. The agent is told to say
so rather than pretend otherwise. If you want to *read* something, select the part
you mean and ask again — a handful of nodes is photographed at full size.

**Prompts are in it, but they are drawn in.** ComfyUI keeps multiline text in HTML
boxes floating *above* the canvas, where a canvas drawing cannot see them, so
agentY paints each one back in at the position and size the real box has. It
matches, and the wrapping is the same — but if you ever see a prompt clipped
where the real one scrolls, that is why.

**It needs the browser open.** The picture is drawn by the ComfyUI page, so a
closed tab means no picture. The agent will tell you that rather than sit waiting.

If several workflows are open in ComfyUI's tabs, the agent is told which ones and
which is active. It only ever reads, edits, runs or photographs the **active** tab
— the others exist as saved state, not as live graphs — so if you mean a different
one, click it first and ask again. It will ask rather than switch your tab for you.

### Filling a slot nothing is wired to

An empty input is invisible in the graph — an unwired slot simply is not there
until something is plugged into it. So a model node with ten free reference slots
and nothing in them looks, to anything reading the workflow, exactly like a node
with no reference slots at all.

That mattered, because it is the shape of a very ordinary request:

> *"Run those two again, but with the photo I just gave you as a reference."*

The agent can now fill a slot **nothing is wired to**. It reads what the node
really has from ComfyUI's own schema, adds a loader for your file, and connects
it — for that run only. Your canvas is untouched: nothing to undo afterwards, and
nothing left behind.

Two things follow from that:

- **Only for some of the runs, if you like.** An empty value leaves the slot
  unwired for that one, so "use the reference on three of the four" is a thing you
  can simply say.
- **It never fakes it.** Naming a file inside the prompt text is not wiring a
  reference — the model never receives the picture, and the run reports success
  anyway. If a reference cannot be wired, you are told why rather than handed a
  render that quietly ignored it.

### Loops: keep trying until it's right

You have a workflow that works and you want the agent to *keep going* until the
output meets a condition. No hook nodes, no template — your own graph, as it is
on the canvas:

> *"Ok let's try a loop — you change the prompt until the woman's position in the
> output matches her position in the original frame."*

The agent runs your graph, looks at what came out, judges it against the condition
you wrote, rewrites the prompt, and runs it again. You watch the prompt change in
your own node between runs. It stops the moment the condition is met.

What makes it work:

- **A condition you can see in the picture.** It becomes the criteria each output
  is graded on — by the same judge as [QA](#checking-outputs-qa), so the same rules
  apply. *"She stands where she does in the reference"* is judgeable; *"make it
  better"* will just churn.
- **The reference is usually already there.** "The original frame" is the image
  your graph loads, and that is what it compares against unless you name another.
- **One value changes.** By default the prompt — it will not pick a *negative*
  prompt on its own, and it never touches your checkpoint, sampler or seed. Say
  which node to vary if your graph has more than one prompt.
- **A budget you set.** `Settings ▸ refine ▸ max_runs` (default **4**) caps how
  many generations one loop may spend. The agent can ask for fewer, never more.
- **You can stop it.** Type anything while it's running and it stops at the end of
  the current generation.

It reports what happened per run, not just the last picture — which value was
tried, what the judge objected to, which run landed it. If none did, your original
prompt is in the report and the agent can put it back.

Your graph needs a saver that writes to ComfyUI's **output** folder, so each result
can be fetched and judged (for the bEpic viewer node that means `save_to_output`
**ON**). Temp-mode previews cannot be read back, and the loop will say so.

This is the *closed* loop — you state the goal once and wait. For the *open* one,
where you look at each result and say what to change next, see
[Iterative refinement](#iterative-refinement-the-iterate-purpose).

### The seven purposes

**1. `inline_parameter`** (default) — annotate an existing node and let the agent
expand + run your on-canvas graph. Great for sweeps and batches:

- *"sweep the seed 6×"*
- *"create 4 prompt variations"*
- *"iterate every file in this folder"*

The agent produces the value(s) for the wired target input. One value → it writes
it; several values (a sweep/variations/folder) → it runs the expanded batch
automatically (capped by `AGENTY_MAX_CANVAS_BATCH`, default 25).

**2. `make_workflow`** — the hook stands in for a **whole workflow (or Python
script)** the agent *generates* from the directive. It builds it, runs it (using
any wired anchor as the input — e.g. an image to edit — else text-to-media), and
stages the result as loader nodes. Use it for self-contained generation steps:

- *"generate a neon cyberpunk city street at night, cinematic wide shot"*
- *"upscale 2× and add film grain"*

**3. `text`** — ask for a **written answer** (no media, no workflow). The agent
writes the string and drops an **`agentY text`** node carrying it, wired where
this hook's output went — so downstream nodes consume it on a normal run. Any
wired anchor is the *subject* of the answer:

- *"write a caption for this image"*
- *"summarise the wired prompt into 8 words"*

**4. `general_request`** — a **free-form** instruction for when the task doesn't
fit the purposes above. The agent treats the directive as an ordinary request —
with any wired anchor as the provided input/context and your graph already
captured — and decides the action itself (answer, generate/edit media, run a
workflow, compute a value). Media results stage onto the canvas, a single produced
value goes to the wired target, a plain question is answered in chat:

- *"what would improve this workflow?"*
- *"take the wired image and give me three different style directions as renders"*

**5. `iterate`** — turn the graph into an **interactive refinement loop**. See
[Iterative refinement](#iterative-refinement-the-iterate-purpose) below.

**6. `qa`** — not a job, a **standard**. The directive is your checklist and the
wired anchors are reference images; every output the graph produces is judged
against them. See [Checking outputs](#checking-outputs-qa).

**7. `review`** — a deliberate **stop**, so you can choose what goes on to the
next stage. See [Review](#review-stop-and-pick-what-continues) below.

### Chaining hooks into pipelines

Wire one hook's **`out`** into another hook's **`anchor`** and you've built a
**pipeline**: each stage's output becomes the next stage's input. The agent runs
the stages strictly in order, feeding real outputs forward (stages after the
first are always image-to-media / edit steps, never fresh text-to-image). The
screenshot above is exactly this: *generate a scene* → *animate it*.

A single hook can also fan several inputs in (multiple anchors) or a stage can
produce several outputs — the agent forwards them all.

### Dry run: check the logic before you pay for it

A chain of hooks is two things at once: a piece of reasoning, and a pile of paid
API calls. The reasoning is what usually goes wrong; the API calls are what
costs. **Dry run** separates them.

Next to ComfyUI's Run button, the **agentY hooks** button has an arrow on its
right. Open it and pick **Dry run** (it is also `Dry run agentY hooks` in the
command palette and under the **Workflow** menu).

The turn then runs *completely normally* — every hook is read and answered, every
value is written and placed on the canvas, every workflow variant is built and
saved to disk — with one thing removed: **no graph is submitted to ComfyUI**.
Where a generation would have happened, the agent is handed a **stand-in**: a
file path marked `DRY-RUN`, with no file behind it.

That last part is what makes it useful on a chain. A second hook whose directive
is *"take the reference frames you just made and queue one video per shot"* still
receives something where the references were, so it runs too — and you find out
whether the second half of your pipeline holds together, which is usually the
half you cannot check any other way. Tools that would open a stand-in
(`analyze_image`, `analyze_video`, `upload_image`) recognise it and answer in
kind rather than failing.

The graphs it builds are **filed where you can look at them**: each one lands in
the Workflows sidebar under `agent/dryrun_…` (one per build — an 18-way sweep is
the same graph eighteen times, a four-stage chain is four different graphs).
Open one and you see the wiring and the exact values the agent wrote into it.
They are *not* swapped onto your open canvas unless you have auto-graphing turned
on — during a dry run, the graph you have open is the thing being tested.

At the end you get an account of what *would* have run: how many generations, of
what, the path of every graph that was built, and which ones were filed. Nothing
is staged onto the canvas, nothing is added to the gallery, and nothing is
written to the hook memory — not even the journal (a result derived from a
stand-in must never be served to a real run later).

Two things a dry run deliberately does not do: it skips the `iterate` purpose
(that loop exists to be looked at, and its result is written back into your own
`LoadImage` node), and it does not run QA — there are no pixels to judge.

The pre-flight check still runs, so a dry run is also where the "this cannot
work" findings show up: an input nothing feeds, a directive naming an anchor slot
that has no wire on it, a hook feeding one image slot while its directive talks
about all of them.

A dry run walks straight **past** a [review hook](#review-stop-and-pick-what-continues)
rather than stopping at one: its outputs are stand-ins, and asking you to choose
between files that don't exist is no kind of review.

### Review: stop and pick what continues

A chain that makes reference frames and then feeds them into a video runs the
whole way through, every time. The video is the expensive half — and by the time
you have seen the references, you have already paid for it.

A hook with **`purpose: review`** is a deliberate break. Put it between the stage
that produces candidates and the stage that consumes them:

```
make_workflow  →  review  →  make_workflow
 "one reference    "which     "animate the
  per character"    two?"      chosen refs"
```

The stage before it runs. What it produced is gathered into an **`agentY image
collector`** node placed beside the hook and wired into its anchor — and the run
**stops there** and asks you.

**That collector is the ballot.** Whatever is in it when you continue is what the
next stage gets:

- **delete the rows** you don't want;
- **add your own files** — a frame you retouched in Photoshop is just as valid as
  one the agent made;
- **reorder them** — the order is the order the next stage receives them in.

Then say **`continue`** in the panel, or press the action-bar button, which turns
amber and reads **Continue with these** while a run is halted (its menu carries
*Continue* and *Stop*). Saying **`stop`** ends the run instead.

Nothing is deleted either way — the files the stage produced stay on disk, and the
collector stays on the canvas as the record of what that stage ran with.

**You can just say it.** Editing the node is the precise way, but *"continue, but
drop the second one"* works too — your words win over the node's contents, and the
agent tells you which files it ended up with.

**Changing things, not just choosing between them.** A stop isn't a yes/no gate.
While it's up you can ask for anything to be *different* — *"regenerate the third
one, warmer"*, *"make that caption shorter"*, *"re-cut the clip to five seconds"*,
*"swap the second reference for this photo"* — images, video, audio, written text
alike. It's neither continue nor stop, so the halt stays up: the agent makes the
change, puts the new result into the collector, tells you what changed, and asks
again.

**As many rounds as you want.** Ten passes of *"warmer — no, warmer than that"* is
the stop doing its job. Nothing advances until you say `continue`, and everything
the agent does meanwhile happens now, in front of you, rather than being queued.

You don't need to select the collector first; the one a halt is waiting on is the
single node the agent may edit unasked, because it's the one it created for this.

**It works on your workflow.** The agent re-runs the stage that made the thing,
in the graph you built — so what you get back came from the same pipeline the next
stage will read from. It'll only open a separate graph when a change genuinely
doesn't fit yours (a different model, a step your chain has no node for), and then
it brings the *result* back into the collector and says that's what it did.

**It follows the wire.** Unwire the collector and wire a different one into the
review hook's anchor and *that* becomes the ballot — handy if you'd rather build
the selection in a collector you already had.

**Reference tags renumber when you delete a row.** The collector is a list and the
numbered slots are its positions, so removing the second image moves everything
after it up: what fed `image_3` now feeds `image_2`. The wiring follows by itself
(only as many slots are wired as there are files), and on resume the agent is
handed the bindings in the form they'll actually take —

```
@image1 / image_1 = ref_00042_.png — TANIHO (HERO)
@image2 / image_2 = ref_00044_.png — APE          ← was @image3 before the cut
```

— with instructions to rewrite any `@imageN` table in the next stage's prompt to
match. Worth a glance at what it says it's running with, though: this is the one
mistake that renders the wrong character doing the right beat and reports no error
at all.

**Anything else keeps the stop up.** Ask a question, change a prompt, go make
coffee: the halt survives until you actually say continue or stop, and the stages
behind it stay shut. There is no timeout — the canvas is the record, and it will
still be waiting next week.

Two review hooks in one chain means two stops. A review hook on one chain doesn't
stop an unrelated chain on the same canvas.

> **`review` vs `qa`.** Same shape, same place in a chain, opposite judge: `qa`
> asks a model against your written criteria and carries on by itself; `review`
> stops and asks *you*. Use `qa` for standards you can write down, `review` for
> the ones you can only recognise on sight.

### Baking a chain into subgraphs

Turn on the keep switch (**bake into subgraph**) on your `make_workflow` hooks.
When you ask the agent to run the graph, it doesn't just execute each stage — it **nests each
generated workflow into a native ComfyUI subgraph** (inputs/outputs matching the
hook's slots), **adds** those subgraphs to your canvas next to the hooks
(nothing is removed), and wires them to mirror the chain. The result is a
self-contained native workflow you can **re-run without the agent** — the
multi-step task, "baked." A value the agent computed at runtime (e.g. a video's
length) is baked in via an [`agentY python`](#the-agenty-python-node--collectors)
node so it reproduces on re-run too.

### The keep switch: should this outlive the run?

One switch, one question: *should what this hook produced outlive the run?*

- **OFF** (default): the agent works it out again next time.
- **ON**: it is kept.

*What* keeping it means follows the `purpose`, because the purposes produce
different things and there is only one sensible way to keep each. That is why the
switch is **labelled** differently — it is not a second decision you make:

| purpose | the switch reads | ON keeps… |
|---|---|---|
| `make_workflow` | **bake into subgraph** | the generated workflow, nested into a ComfyUI **subgraph** placed beside the hook and wired to mirror the chain (see [Baking](#baking-a-chain-into-subgraphs)) — plus the files that run produced, so re-opening the graph re-uses them instead of re-rendering |
| `text`, `inline_parameter`, `general_request` | **memorize result** | everything the hook produced: written values and prompts, scripts, images and videos (by path), written to `agent/memory/` beside the outputs |

**The hook is never rewired.** Whichever way the switch is set, it stays wired
exactly as you drew it and the `agentY text` node is dropped *unconnected* as a
human-readable reference. The hook chain is your graph's readable statement of
what happens, and a switch about keeping a *result* has no business rewriting
it.

> **This was three switches** — `bake_to_canvas`, `freeze` and `memorize` — then
> two. They were always one question asked several ways. Your saved graphs migrate
> when you open them: a hook comes back with the switch set from whichever of the
> old ones its `purpose` actually read.

**When you can't see it.** It is hidden on `qa` and `iterate`, which produce
nothing to keep. Hiding is presentation only — the value is still saved, so
flipping `purpose` back brings it back untouched.

### Memorize: produce once, reuse until something changes

A hook that reads an image and writes a description costs a vision call and a
turn of the agent's attention. Wire it into a graph you iterate on for an
afternoon and you pay for that same description twenty times, for a picture that
never moved. A hook that *generates* — a reference frame, a video — costs far
more than attention.

Turn the switch on and the result is kept. On later runs the written value goes
straight back into the graph and the produced files are re-delivered as that
turn's outputs, and the agent is told the hook is **already done** — no call, no
re-reading the anchors, no re-rendering. The panel says `♻️ reused …`.

**You can decide in hindsight.** You rarely know a result was worth keeping until
you have looked at it, so what a hook produced is written down either way. Turn
the switch on *after* a run you liked and it keeps that run's result — the value
is already there, under the key that run wrote. Turning it off is still the forget
gesture: off, send anything, on again.

It is released the moment the question changes:

| What you change | Released? |
|---|---|
| A different image, or any edit upstream of the hook (however many nodes back) | ✅ |
| Rewiring an anchor, or where the hook's output goes | ✅ |
| The hook's own prompt or `purpose` | ✅ |
| Switching the keep switch **off** — this is how you force a fresh result | ✅ |
| Replacing a file with a different one *of the same name* | ✅ (size + timestamp are part of it) |
| Deleting a remembered image or video from disk | ✅ (the whole entry — four of five frames replayed is a worse answer than doing it again) |
| Anything **downstream** — a save prefix, a sampler after the hook | ❌ (it didn't change what the result is) |

It's stored in **`agent/memory/`** under ComfyUI's own output directory, next to
the `agent/images` and `agent/videos` folders it points at — so a remembered path
and the file it names travel together, and the whole lot switches with the
project. Paths are recorded relative to that output directory, so moving the
folder doesn't strand the entries in it. It's a cache, not a note: it never
appears in [memory](#memory).

### Naming what a hook produces

Say what the outputs *are* in the hook's own prompt and the name travels with
them:

```
Generate one start frame per shot.
role: shot start frame
```

`role: …`, `[role: …]`, or *"tag the outputs as 'hero sheet'"* all work. Then:

- each dropped node is **titled with the role** instead of the filename;
- an **`agentY add tag` node is attached** to it carrying your words, so whatever
  you wire it into next is told what to take from it;
- a small `.agenty.json` file is written **beside the image or video**, so months
  later — in another thread, or via an [agentY
  collector](#the-agenty-python-node--collectors) pointed at the folder — the
  agent still knows what it is instead of looking again.

Without a stated role, the first two still happen using the directive itself
(minus the tag node — agentY won't add nodes to your canvas uninvited). Its `tag
name` field is left empty either way: a name is what you type from the `#` menu,
and one invented out of a sentence is a name nobody chose.

**In a batch, each variant is named separately.** Sweep three character prompts
through one hook and the three frames come back as *"character reference: Anna,
red coat, 30s"*, *"…: Ben, grey suit"*, *"…: Cleo, shaved head"* — named after the
value that produced each one, before it runs. The agent also gets the pairing back
as data (`variants[].made_from` / `variants[].outputs`), so it never has to assume
the files came back in the order they went out. They usually do; they don't when a
generation fails and is repaired, which re-queues it behind the others.

**Feeding those frames to a video model.** The second hook needs somewhere to put
them: a `reference_images`-style input is one wire, so N images have to arrive
through an `ImageBatch` / `BatchImagesNode` or an [agentY image
collector](#the-agenty-python-node--collectors) — **you wire that part**, the
agent fills it. Order then matters, because that's how the prompt addresses them:
for Kling, `@image1`, `@image2`, … refer to the 1st, 2nd, … image on that input.

```
@image1 walks past @image2 in the alley and hands her the letter
```

The agent is told to name them that way rather than describing the characters in
prose and hoping the model matches them up — so which frame is which stays true
from the hook that made it to the shot that uses it.

### Iterative refinement (the `iterate` purpose)

The `iterate` purpose runs an **interactive, multi-turn refine loop** — one
generation per turn, each result fed back in as the next input, so you sculpt an
image step by step in chat.

**Wiring:**

- Wire this hook's **`out` → the prompt node's text input** (each prompt you type
  in chat is written there).
- Wire the **`LoadImage` node's image output → an `anchor`** (this is the loader
  the agent updates in place with each run's result).
- You need a **save node that writes to ComfyUI's history** (a `SaveImage`, or a
  viewer node with its "save to output" toggle on) so the agent can fetch each
  result and feed it forward.

**Using it:** ask the agent to start the loop. Each turn you give the next
prompt/change; the agent writes it in, runs the graph once, replaces the
`LoadImage` path with the new result, shows it, and asks for the next step. You
can **jump back**:

- *"go back to the original image, then make it warmer"*
- *"back to generation 3, then add rain"*

Keep going until you say **stop**. (Driven by the `iterate_step` tool and the
`iterative-refine` skill.)

This is the loop **you** steer, one step per turn. If instead you want to state a
goal once and have the agent keep going on its own until the output meets it —
with no hook node at all — see [Loops: keep trying until it's
right](#loops-keep-trying-until-its-right).

---

## Slack: a second way in

Off by default. Turned on, Slack becomes a **second line** into the same agent —
never a replacement for the panel, and never a separate conversation.

- **Every turn is mirrored to your Slack DM as it runs** — including the ones you
  start in the panel. Queue a render at your desk, walk away, watch it finish on
  your phone.
- **A DM back drives the same conversation.** Reply and the agent answers there
  and in the panel; the thread you are in is the thread it is in.
- **Send it images and video** and it takes them as inputs, exactly as the 📎
  button does.
- **The agent can send you files** — *"send me a screenshot of my workflow on
  Slack"*, one frame out of sixty, a JSON it just wrote. Generated media is
  already mirrored, so this is for the things nothing else would send.

One conversation is one Slack thread. Reply **in the thread** to continue it;
message the bot **at top level** to start a new one.

While a turn is running, what a DM means depends on where you put it. A reply
**in that turn's own thread** is handed to it mid-flight, exactly like
[talking to a running turn](#talking-to-a-turn-that-is-already-running) in the
panel. Anything else — a top-level message, or a reply in a different thread —
is answered **busy** and not queued: there is one agent, and a message written
for another conversation should not be dropped into this one.

The connection is **outbound only** (Socket Mode), so nothing on your machine has
to be reachable from the internet.

It needs a Slack app of your own — bot token, an app-level token, and your member
id — then `slack.enabled` under **Settings ▸ Slack**. It takes effect at the next
agent start.

> **`SLACK_ALLOWED_USERS` is not optional.** Empty means every message is refused,
> deliberately: anyone who could DM the bot would otherwise be able to run
> generations and tools on your machine.

**[Full setup walkthrough → docs/slack.md](slack.md)**

---

## Checking outputs (QA)

agentY can generate a thing. It can also tell you whether the thing is any good —
but only by *your* standard, never an invented one. **With no briefing, no QA
runs at all.**

A **briefing** is two things, because "is this right?" usually is:

- **criteria** — prose or bullets, one checkable statement per line;
- **references** — mood images the output should sit beside without looking out
  of place. A grade or a character look is not something words are good at.

A separate **QA agent** (Settings ▸ llm ▸ pipeline ▸ `qa_checker`) reads every
image and video a run produced and reports **per criterion**: pass, fail, or
`n/a` — with a sentence of evidence for each. This is the one role where a
stronger model pays for itself: it runs once per finished output, and a weak
judge either waves defects through or fails clean work and triggers a pointless
re-render.

### It measures rather than eyeballs

Anything countable is **computed from the file** and handed to the agent as fact,
never left to its eyes: dimensions, aspect ratio (with the nearest standard ratio
named), duration, frame count, fps, format and file size.

This matters more than it sounds. Vision models are famously poor at judging
proportion, and the picture they're shown has been *resized* on the way in — so
"is this 16:9?" asked of the image is a question they cannot actually answer, and
a 9:16 render sails through. Now the criterion is compared against a measured
`aspect ratio: 0.565 — 9:16 (portrait)` and fails with the number quoted. Same for
*"at least 10 seconds"* against a measured 3.36 s clip.

So criteria can be exact, and it's worth making them exact:

```markdown
- 16:9 landscape.
- At least 10 seconds long.
- No visible text anywhere in frame.
```

The same trade applies to how the picture *looks*, not just how big it is.
**Softness, grain and blown highlights** are the complaints people actually make,
and a vision model estimates all three badly from a resized copy — so they are
measured too, and the numbers go to the judge as fact:

- **sharpness**, with the sharpest *region* reported separately. A portrait with
  a soft background reads soft overall, and without that second number the check
  would reject exactly the picture you asked for;
- **grain**, measured on a copy that has had the fine detail smoothed out first,
  so texture is not mistaken for noise;
- **exposure** — the mean, the contrast, and how much of the frame is pinned at
  pure white or pure black. Detail that clipped is gone and cannot be graded back.

### Ticking the boxes: the `agentY qa briefing` node

Everything above is exact, which means none of it needs to be *written*. Drop an
**`agentY qa briefing`** node on the canvas and the technical half is dropdowns
and switches:

| control | what it does |
|---|---|
| `aspect_ratio` | compared against the file's real dimensions, within a rounding tolerance — 1312x736 counts as 16:9 |
| `resolution` | minimum **short** side, which is how "1080p" is usually meant |
| `sharpness` | fails a soft render; a shallow depth of field still passes |
| `grain` | fails visible grain. Leave it on `any` when grain is the look |
| `no_clipping` | fails more than 2% of pixels pinned at pure white or black |
| `no_black_frames` / `no_stalled_motion` | video only: a black sampled frame, or a clip that freezes |
| `likeness` | see [below](#does-it-match-the-reference) |
| `retries` | this briefing's own retry budget |

`notes` carries everything a measurement cannot settle — mood, framing, whether it
looks right — and is read exactly like a `qa` hook's directive. Wire reference
images into `reference`.

Anything left on `any` (or off) is **not checked**, and a node with nothing set
enforces nothing. What is set is decided *before* the model is asked anything, and
the model is then shown the answers and told not to re-judge them — so a
measurement cannot be argued with, and it costs no round trip.

### Does it match the reference?

*"The character must match the reference"* is the criterion people write most
often and the one a vision model answers worst: it will call any two dark-haired
men the same person. Set **`likeness`** on the briefing node and it becomes a
number instead:

- **`must match the reference face`** — a face embedding (ArcFace), compared by
  cosine against every image wired into `reference`. On this machine's own renders
  the same character scores 0.95-0.98 and different characters 0.09-0.54,
  stylised faces included.
- **`must match the reference subject`** — for everything a face cannot answer: a
  location, a product, a grade. This one (DreamSim) was trained on human
  judgements of *diffusion-generated* images, so its notion of "alike" is fitted
  to the pictures agentY actually makes.

A video is compared frame by frame and the best frame counts — a character out of
shot for part of a take still matches. If the comparison cannot be made at all (no
face in the output, no reference with a face in it), it yields **no verdict** and
the written criterion goes to the model instead; doubt never condemns your work.

Both scorers are optional and load only when the control is set — the first run
downloads their weights (~3.6 GB, into `models/` beside the checkout, not your
home folder) and they run on the CPU, because the GPU belongs to ComfyUI.

### Which of these is best?

Everything above is a **gate**: pass or fail, and nothing compensates for
anything. That is right for *"must be 16:9"* and useless for the other question a
run of eight variants raises — *which one?*

So there is a second, separate number: a **technical quality score**, 0 to 1,
from the same measurements. Sharpness, focus, cleanliness, headroom, and likeness
when it was measured, each normalised and weighted. You will see it in two
places: beside each output when a [`review` hook](#review-stop-and-pick-what-continues)
stops the chain to let you choose, and in the facts the QA judge is given.

**It never decides anything.** It is not a gate and it cannot fail an output — a
weighted sum lets a strong feature pay for a weak one, which is exactly what you
want for ordering and exactly what you do not want for a requirement. Your taste
outranks it, always. What it is good for is the thing thumbnails are bad at:
*"4 and 7 are the softest of these"*.

**And it learns what you actually like.** Every time you answer a review hook —
delete the rows you do not want, say continue — agentY writes down which outputs
you kept and which you dropped, with each one's measured features. That is a
preference label, collected from a decision you were making anyway. Once you have
a dozen or so:

```
.venv/Scripts/python.exe scripts/fit_fitness_weights.py            # what would change
.venv/Scripts/python.exe scripts/fit_fitness_weights.py --write    # install, if better
```

It fits the weights to your choices, holds a slice of them back, and **refuses to
install anything that does not beat the defaults on reviews it has not seen**.
Two weights start at zero for exactly this reason — contrast and brightness are
style, not quality, so a hand-set score must not touch them, but if you
consistently keep the darker take, that is learnable and it will be learned.

The labels live in `output/agent/preferences.jsonl` (gitignored — they are your
judgements, not the repository's), and they store the *numbers*, not just the
paths, so they stay usable long after the pictures are deleted. Delete
`config/fitness_weights.json` at any time to go back to the hand-set weights.

### Writing a briefing — three ways

(Four, counting the briefing node above — which *is* a `qa` hook as far as
everything below is concerned, and combines with the rest the same way.)

**1. A `qa` hook on the canvas** (the main one). Drop an `agentY hook`, set
`purpose: qa`, type the checklist in `directive`, and wire your reference images
into its **anchors**.

Wiring is the point. A turn can carry inputs, outputs and references at the same
time, and no amount of careful phrasing reliably keeps them apart — but an image
wired into a QA hook's anchor is unambiguously a *reference*, never another input
to the workflow. Anything that resolves to a file works as an anchor: a
`LoadImage`, an [agentY collector](#the-agenty-python-node--collectors) holding a
whole folder, even a mid-graph node (it gets rendered to a file first, exactly as
in [What the agent can see](#what-the-agent-can-see-on-an-anchor)).

Several QA hooks on one graph **combine** rather than compete — two notes pinned
to one canvas both apply. And the briefing is saved with the workflow, so it is
still there when you reopen it next month.

**2. A named file** — `config/qa/<name>.md`, with mood images in an optional
sibling `<name>.refs/` folder. Reusable across graphs and threads, and it lives
in version control. See `config/qa/README.md`.

**3. `/qa` in the chat panel** — for turns with no canvas graph:

```
/qa                                      show what's active
/qa house-style                          use a named briefing
/qa no text anywhere, warm skin tones    use this as the criteria
/qa off                                  clear it
```

They compose, and they have a precedence. A `qa` hook **wins** over the thread's
`/qa` briefing — it's the more specific, more visible statement. Either can cite
a named file with `@name` and add to it:

> `@house-style plus the logo must stay legible`

### What happens on a failure

Failing outputs are **never withheld** — a verdict is an opinion about your
criteria, not a reason to hide your file. What happens next is
Settings ▸ qa ▸ `max_retries`:

- **`1` (default)** — the failing output is re-generated once, against *exactly*
  the criteria it missed. The seed is rerolled (without that a re-run reproduces
  the same image and the retry is pure waste) and the positive prompt is rewritten
  to address the named defects, keeping subject and intent intact. The rejected
  workflow is left untouched beside it for comparison.
- **`0`** — report the verdict and stop. Use this if you'd rather approve each fix.

Only the picture is adjusted, never the graph: a workflow that ran cleanly isn't
broken, and rebuilding it would invalidate the very verdict that asked for the
retry. Failing a *run* is a different thing, handled by self-healing.

**Say what a failure should cause, in the briefing itself:**

| you write | what happens |
|---|---|
| `retry: 3` | that briefing's own budget, overriding `max_retries` for this run |
| `retry: hook 5` | the fix lives a stage earlier. The agent is handed the outputs that missed, what they missed, and an instruction to produce fresh values for **hook 5** for those variants only — and to leave the ones that passed alone |
| `re-run hook 5 x2` | both |

The runtime can re-roll a generation by itself; it cannot re-write the prompt that
a *previous hook* produced, because that stage is an agent doing creative work.
So `retry: hook N` hands it back with everything needed to act — which is also
why a verdict now travels to the agent at all: before, a spent verdict went to the
log and the one thing that could have fixed the shot never heard about it.

**Two kinds of check, because "does this meet the brief?" is two questions.**
Each output is judged **on its own** against the criteria and the references —
and then, when a run made several, the whole set is judged **together** for the
criteria only a set can answer: one grade across all of them, consistent
character identity, no accidental near-duplicates. The per-file judge is told to
mark set-criteria `n/a` rather than fail an image for the absence of images it
was never shown — a failure no re-generation could fix, since the missing images
were never that image's job.

### The rest of Settings ▸ qa

| setting | what it does |
|---|---|
| `enabled` | master switch (env `AGENTY_QA=0`) |
| `max_outputs` | how many of a run's outputs get checked — a 25-variant sweep would otherwise be 25 strong-model calls. The rest are still delivered, just unchecked |
| `max_references` | reference images sent with each check |
| `video_frames` | frames sampled from a video, sent as **one** labelled sequence so the clip is judged for continuity and drift rather than as unrelated stills |
| `briefing_dir` | where named briefings live |

---

## The agentY python node & collectors

- **`agentY python`** — runs an agent-authored Python snippet as a node. It's the
  companion to [baking](#baking-a-chain-into-subgraphs): a value the agent
  computed at runtime is placed here so it becomes a genuine, re-runnable output.
  ⚠️ **It executes arbitrary Python whenever the graph runs** — meant for your
  own, self-hosted, agent-built workflows; don't run baked workflows from
  untrusted sources. Set `AGENTY_PYTHON_NODE_DISABLED=1` to make it a no-op.
  Like **`agentY text`**, it is placed *by the agent* and is therefore hidden
  from the add-node menu and the double-click search — both stay fully
  functional wherever they already sit on a graph. Turn on ComfyUI's
  **Settings ▸ Enable dev mode options** if you want them offered in search.
- **`agentY collector`** — hand the agent a batch of on-disk files. Their paths
  live in the node (no pre-run needed), so an anchored collector is rendered to
  the agent as its explicit file list — it can bind every path directly. Use it as
  an anchor input to a hook to run one directive across many files. Type `#` in
  the list to pick from the canvas's tags and the project's remembered references;
  what lands in the box is the file's **path**, exactly as the picker would have
  added it, so the list stays literal.

  It takes **images and video in one node**, with an output for each (`images` is
  a stacked IMAGE batch, `videos` a list of VIDEO objects, `paths` the whole list
  as text) — wire whichever you need. It has three outputs rather than one
  because the types are genuinely different and ComfyUI fixes them at
  registration, so no single output could be all three.
- **`agentY load item`** — loads one entry from [project memory](#memory): a
  remembered reference image or clip, or a written fact. Pick it from a dropdown
  of what is actually stored; the `item` output takes the **type of whatever the
  entry is** (IMAGE, VIDEO, or the text), so the same node feeds a sampler, a
  video node or a prompt box. An image or video entry is **previewed on the node**
  without running anything — the file is already on disk. `text` and `path` come
  out alongside, so a graph that wants both doesn't need two nodes.
- **`agentY expand image batch`** — splits an image batch into one image per
  output (`image_1` … `image_8`, plus a `count`).

  You need it whenever a collector feeds a **model node that takes references in
  numbered single-image slots** (`image_1`, `image_2`, …). The collector emits its
  files as one IMAGE *batch*; wire that straight into `image_1` and the node takes
  the **first image and silently ignores the rest** — you hand it five references
  and the render is built from one, with no error anywhere. So:

  ```
  agentY collector ──▶ agentY expand image batch ──┬─▶ image_1
                                                         ├─▶ image_2
                                                         └─▶ image_3
  ```

  A slot past the end of the batch emits nothing rather than repeating the last
  image — repeating would be the same quiet failure in a new costume (the same
  character twice on a reference sheet). `count` tells you how many actually
  arrived, so you can see whether the slots you wired are real. Pre-flight also
  catches the un-expanded case before a run and names this node.

  A plural `images` input is fine as it is — that one takes a batch on purpose.

---

## Settings & secrets

Open ComfyUI's **Settings** (gear, bottom-left) → **agentY** → **Open agentY
Settings…**. That single row is the whole agentY section — everything else lives
inside the modal, which is grouped and collapsed by default so it stays scannable.

Close it by clicking outside the card or pressing **Escape**; **Save** is the only
button.

![agentY application settings](images/settings.png)

- **Viewers** — the message-history log, the long-term-memory editor, and the
  [token usage](#token-usage--cost) breakdown. This is where they open from.
- **Authentication (.env)** — your API keys and host settings, stored in `.env`
  on the agent host. Secrets are masked; tick **Show secret values** to reveal.
  **+ Add auth key** appends a new `.env` variable (e.g. a secret an MCP server
  references) and applies it to the live process.
- **Application settings** — six sections, in the order you are likely to want
  them. **Models** is first and already open, because it is what most people came
  to change:

  | section | what is in it |
  |---|---|
  | **Models** | the six tiers, plus per-role overrides folded underneath |
  | **Connections** | ComfyUI, the agentY host, your Ollama server |
  | **Canvas** | what the agent may see and do on your open graph |
  | **Output checks** | [QA](#checking-outputs-qa) and [refine loops](#loops-keep-trying-until-its-right) |
  | **Slack** | the [second line](slack.md) into the agent |
  | **Updates** | whether agentY updates itself |

  **Show advanced settings** adds five more — Memory, Providers (per-vendor
  tuning), Files & logs, Prompts, Annotation — none of which you need to touch to
  use agentY. Only changed values are written to the gitignored
  `config/settings.local.json`; committed defaults are left untouched.

  `ollama_server_url` in Connections is the single address for *everything* that
  talks to Ollama — agents on a local model, the memory embedder, and the small
  `llm_functions` helper. `OLLAMA_HOST` overrides it.
- **Model pricing (config/pricing.json)** — per-model USD prices per million
  tokens, so the [token-usage](#token-usage--cost) cost column matches your
  endpoint (handy for private/MaaS deployments and models the built-in tables
  don't ship).

### Choosing models: six tiers, not sixteen dropdowns

**Models** is the first section and opens with the panel. You set six **tiers**,
and every role inherits from one of them:

| tier | who uses it |
|---|---|
| **Orchestrator** | drives every turn — routing, tool calls, talking to you |
| **Research & assembly** | template research, workflow assembly, repair, building a graph from scratch |
| **Fast utility** | info lookups, web search, planner, learnings, small structured-output helpers |
| **Vision** | reads the images and video *you* provide |
| **QA judge** | grades finished outputs against your [QA briefing](#checking-outputs-qa) |
| **Coder** | Python scripts and ComfyUI custom nodes |

**Per-role overrides** sits underneath, one row per role, blank by default —
*"— inherit from tier —"*. Fill one in only when a single job wants something
different from the rest of its tier. The group header tells you whether any are
set, so an override can't quietly beat a tier without you knowing.

Why tiers: the roles only really differ along two axes — how much reasoning they
need and whether they must see images. Sixteen dropdowns made you answer the same
question sixteen times.

Two of the groupings are deliberate rather than obvious. **QA judge** is separate
from **Vision** because it runs once per finished output and a weak judge either
waves defects through or fails clean work and triggers a pointless re-render — it
is worth more than the model that merely reads your inputs. **Coder** is its own
tier because it usually wants a code-specialist model.

Resolution for any role: *environment variable → per-role override → tier →
built-in default*. Changes apply on the **next agent start**.

> **Upgrading?** If your `settings.local.json` predates tiers it will have a pin
> per role, and those keep winning (so nothing changes). To lift them into tiers:
> `python scripts/migrate_model_tiers.py --dry-run`, then run it without the flag.
> It writes a timestamped `.bak` and leaves every role resolving to exactly the
> same model.

---

## MCP servers

agentY can call tools from external **MCP** (Model Context Protocol) servers.
They're defined in `config/mcp.json` (tracked; holds no secrets) and edited from
the same settings modal.

![MCP servers section](images/mcp-settings.png)

Each server has:

- a **transport** — `http`, `sse`, or `stdio` (with `command`/`args`);
- a **url** (for http/sse);
- an **auth** mode:
  - **`none`** — no auth;
  - **`header`** — reference `${ENV_VAR}` in the server's `headers`, and store
    the secret in `.env` via **+ Add auth key** above;
  - **`oauth`** — browser sign-in; click **Authorize…** on the server's status
    row.

Saved changes load into the orchestrator on the **next agent start**.

### The Magnific example (OAuth)

`config/mcp.json` ships with **Magnific** wired as an OAuth server:

```json
{ "servers": { "magnific": {
  "enabled": true, "transport": "http",
  "url": "https://mcp.magnific.com", "auth": "oauth"
} } }
```

Its status row shows `http/oauth — needs_auth` until you authorize it:

1. Click **Authorize…** → your browser opens the Magnific sign-in.
2. Approve; the token is stored in the gitignored `config/.mcp_tokens/`.
3. **Restart the agent** so the orchestrator rebuild loads the server's tools.

Startup is always **silent** — if a server has no token yet, it's skipped (no
browser, no hang); authorizing is an explicit one-time action. The OAuth
callback lands on `http://localhost:8199/callback` (tweak the port in
`src/tools/mcp_tools.py` if a provider rejects it).

---

## Token usage & cost

The **📊** button in the chat top bar — or **Viewers ▸ Token usage…** inside
the agentY Settings modal — breaks down token spend and cost from the persisted
token log.

![Token usage overview](images/token-usage.png)

- Filter by **time range** and **model**.
- Stat tiles: input / output / total tokens, **cache read (hits)** and **cache
  hit rate**, **estimated cost**, and total calls.
- A per-model table with input/output, cache read/write, hit rate, and cost —
  and a **🗑 Clear log** to purge it.

Costs use the built-in price tables, overridden by
[`config/pricing.json`](#settings--secrets) where you set your own.

---

## Memory

Long-term memory is a local **FAISS** index (`memory/agenty_memory.faiss`) via
**mem0**. Conversation threads are separate — they live in the SQLite store
(`memory/conversations.sqlite`). Browse/edit both from **Settings → agentY →
Viewers**.

Two models are involved, and they are **not interchangeable** — this trips people
up, because the startup line names both:

```
[memory] FAISS memory layer initialised (embed=ollama:nomic-embed-text, llm=dashscope:qwen3.6-flash)
```

| | what it does |
|---|---|
| **embedder** | turns text into **vectors** so memories can be found by meaning. An embedding model does *only* this — `nomic-embed-text` cannot write a sentence, and a chat model cannot produce embeddings. That's why it is a different model from everything else in agentY, and why it stays on Ollama by default (it's small, local and free). |
| **llm** | rewrites memories in its own words ("fact extraction"), and **only** for `infer` writes — the normal write path never calls it. |

Leave `memory.llm.model` **blank** and it follows the **Fast utility** tier,
bringing that provider's endpoint and API key with it. Fill it in only to run
fact extraction somewhere else. Both live under Settings ▸ *Show advanced
settings* ▸ Memory.

> Changing the embedder model or `embedding_dims` invalidates the index on disk.

---

## Choosing models

Any model value is `"provider,model"`. Providers: `claude`, `ollama`,
`dashscope` (Alibaba Model Studio / Qwen; aliases `qwen` / `modelstudio` /
`alibaba`), `openai`, `google` (alias `gemini`).

Which model runs which job is set by **tier**, with per-role overrides for the
exceptions — see [Choosing models: six tiers](#choosing-models-six-tiers-not-sixteen-dropdowns)
in Settings.

Resolution order for a role (first match wins): **CLI flag → environment variable
→ per-role override (`llm.pipeline`) → tier (`llm.tiers`) → built-in default**,
with `config/settings.local.json` layered over `config/settings.default.toml` at
each step.

Change a role live from chat — this writes a per-role **override**, so it wins
over that role's tier until you clear it:

```
/switch_model orchestrator claude,claude-opus-4-8
```

The scope dropdown next to it offers **All tiers**, each of the six **tiers**,
and each individual **role** (labelled with the tier it belongs to). It is built
from the agent's own tier map at startup, so it always matches Settings ▸ Models.

Switching a **tier** is the normal move. Switching a single **role** writes a
per-role override, which then beats that role's tier until you clear it in
Settings — the reply says so when it happens.

The model list itself is discovered live from each configured provider, so only
vendors whose key is set appear; it never goes stale. Agents the pipeline holds
live (orchestrator, query_templates, info, planner) switch immediately — the rest
apply at the next agent start, and the reply tells you which is which.

---

## Custom workflow templates

Register your own workflows so the agent can retrieve them:

```powershell
.\scripts\add_workflow.ps1 path\to\your_workflow_api.json   # also generates a SKILL.md
.\scripts\remove_workflow.ps1 your_workflow_api             # also removes its skill dir
```

Or from chat: `/add_workflow <path>` (or `/add_workflow canvas <name>` to
register the open graph) and `/remove_workflow <name>`. Custom templates live in
`comfyui_workflow_templates_custom/templates/`; the shared template/recipe
corpus lives in **agenty_core**.

---

## Building a node for a new model

Point the agent at a model's GitHub repo and it writes a ComfyUI node pack for it:

> *"Build me a ComfyUI node for github.com/…"*

The repo is shallow-cloned (weights skipped), the **coder** agent reads its
README, docs and inference code, and writes a complete importable pack —
`__init__.py`, `nodes.py`, `requirements.txt`, `README.md`, `pyproject.toml` —
into `output/custom_nodes/<name>/`, ready to publish as its own repo.

It is for a model that has **no** ComfyUI node yet. If one already exists,
installing it is the better answer, and the agent will say so.

---

## Troubleshooting

- **Panel shows "▶ Start server" / can't connect** — the agent host isn't
  running. Run `.\run_agent.ps1` (or click the button), and check the backend URL
  (`localStorage.agentY_backend`).
- **"The model configured for vision / qa_judge is not multimodal"** — that
  tier is pointed at a text-only model, so it cannot be handed a picture at
  all. Point it at a vision model in Settings ▸ Models (for DashScope that
  means a `-vl-` one). Worth knowing which way each fails: **vision** fails
  loudly, because nothing can be described; **qa_judge** fails *quietly* — a
  judge that cannot see will pass every output rather than condemn work it was
  unable to read, so QA looks like it is running and is not. agentY now says
  so instead of leaving you to notice.
- **Autograph toggle or MCP section does nothing / 404** — those routes live in a
  newer host build; **restart `run_agent.ps1`** so the `:5000` host serves them.
- **Canvas nodes/UI look stale after an update** — the ComfyUI copy of
  `agentY-comfyuiConnect` is separate from your dev clone; `git pull` it in
  `<ComfyUI>/custom_nodes/` and reload ComfyUI.
- **"🚫 … refused this generation on content grounds"** — the provider's own
  filter rejected it (copyright, likeness, safety), not a workflow problem. Every
  API model has one and they all word it differently; agentY recognises them and
  **re-runs with a fresh seed** rather than sending the repair agent after a graph
  that was never broken — twice for a rejected *result*, once for a rejected
  *prompt*, since a prompt the provider read and refused rarely reads differently
  the second time. Set `AGENTY_POLICY_RETRIES` to change that budget (`0` disables
  the retries). When they're spent you get the provider's own words and what to
  change; the seed is only part of it, since several of these APIs ignore the seed
  and simply aren't deterministic.
- **"No output files found in ComfyUI history"** (esp. with a custom save node) —
  the agent harvests results from ComfyUI's history, which requires the save node
  to write there. For the `iterate` loop and result staging, make sure your save
  node actually saves to the output dir (e.g. a viewer's "save to output" toggle
  on).
- **A hook did nothing on Queue Prompt** — that's by design: hooks are inert on a
  normal run. Ask the **agentY agent** to run the graph.
- **Model switch had no effect** — model-per-stage changes apply on the next
  agent start; use `/switch_model` for a live change.

---

*See also: [README](../README.md).*
