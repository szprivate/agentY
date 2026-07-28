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
- [The chat panel](#the-chat-panel)
- [Generating & editing](#generating--editing)
- [Slash commands](#slash-commands)
- [**The hook system**](#the-hook-system)  ← the most powerful part
- [Checking outputs (QA)](#checking-outputs-qa)
- [The agentY python node & collectors](#the-agenty-python-node--collectors)
- [Settings & secrets](#settings--secrets)
- [MCP servers](#mcp-servers)
- [Token usage & cost](#token-usage--cost)
- [Memory](#memory)
- [Choosing models](#choosing-models)
- [Custom workflow templates](#custom-workflow-templates)
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

2. **Open ComfyUI** in your browser (default `http://127.0.0.1:8188`) and click
   the **agentY** tab in the left sidebar.

3. Start typing. Type `/` for the slash-command menu; use the thread dropdown to
   revisit past conversations.

> If the backend runs on a non-default URL, set it once in the browser console:
> `localStorage.agentY_backend = "http://host:port"`.

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
- **`bake_to_canvas`** — *make_workflow only*; bake the result into a reusable
  subgraph (see [Baking](#baking-a-chain-into-subgraphs)).
- **`freeze`** — *inline_parameter / text only*; keep the hook live vs. bake the
  value in (see [Freeze](#freeze-keep-live-vs-bake-in)).

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

### The six purposes

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

### Chaining hooks into pipelines

Wire one hook's **`out`** into another hook's **`anchor`** and you've built a
**pipeline**: each stage's output becomes the next stage's input. The agent runs
the stages strictly in order, feeding real outputs forward (stages after the
first are always image-to-media / edit steps, never fresh text-to-image). The
screenshot above is exactly this: *generate a scene* → *animate it*.

A single hook can also fan several inputs in (multiple anchors) or a stage can
produce several outputs — the agent forwards them all.

### Baking a chain into subgraphs

Turn on **`bake_to_canvas`** on your `make_workflow` hooks. When you ask the
agent to run the graph, it doesn't just execute each stage — it **nests each
generated workflow into a native ComfyUI subgraph** (inputs/outputs matching the
hook's slots), **adds** those subgraphs to your canvas next to the hooks
(nothing is removed), and wires them to mirror the chain. The result is a
self-contained native workflow you can **re-run without the agent** — the
multi-step task, "baked." A value the agent computed at runtime (e.g. a video's
length) is baked in via an [`agentY python`](#the-agenty-python-node--collectors)
node so it reproduces on re-run too.

### Freeze: keep live vs. bake in

For `inline_parameter` / `text` hooks, **`freeze`** controls what happens to the
value the agent produces:

- **OFF — keep hook live** (default): the hook stays wired as you drew it; the
  agent **injects the produced value at run time** and drops the `agentY text`
  node *unconnected*, as a human-readable reference.
- **ON — freeze into graph**: the agent **bakes** the `agentY text` node into the
  wired target input and takes over the hook's downstream link — yielding a
  plain, self-contained workflow you can re-run yourself (at the cost of
  bypassing the hook).

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

### Writing a briefing — three ways

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
- **Collector nodes** (`agentY image collector` / `agentY video collector`) —
  hand the agent a batch of on-disk files. Their paths live in the node (no
  pre-run needed), so an anchored collector is rendered to the agent as its
  explicit file list — it can bind every path directly. Use them as anchor inputs
  to a hook to run one directive across many files.

---

## Settings & secrets

Open ComfyUI's **Settings** (gear, bottom-left) → **agentY** → **Open agentY
Settings…**. That single row is the whole agentY section — everything else lives
inside the modal, which is grouped and collapsed by default so it stays scannable.

Close it by clicking outside the card or pressing **Escape**; **Save** is the only
button.

![agentY application settings](images/settings.png)

- **Viewers** — the message-history log, the long-term-memory editor, and the
  [token usage](#token-usage--cost) breakdown. This is the one place they open
  from (they used to be duplicated as separate rows in ComfyUI's Settings).
- **Authentication (.env)** — your API keys and host settings, stored in `.env`
  on the agent host. Secrets are masked; tick **Show secret values** to reveal.
  **+ Add auth key** appends a new `.env` variable (e.g. a secret an MCP server
  references) and applies it to the live process.
- **Application settings** — only two groups are shown by default: **Connections**
  (ComfyUI, the agentY host, and your Ollama server) and **Models & providers**.
  Everything else — ComfyUI paths, output & logs, behaviour toggles, memory, output
  QA, system prompts, per-provider tuning — is behind **Show advanced settings**.
  Only changed values are written to the gitignored `config/settings.local.json`;
  committed defaults are left untouched.

  `ollama_server_url` in Connections is the single address for *everything* that
  talks to Ollama — agents on a local model, the memory embedder, and the small
  `llm_functions` helper. (The older `llm.ollama.host` still works when it's blank;
  `OLLAMA_HOST` overrides both.)
- **Model pricing (config/pricing.json)** — per-model USD prices per million
  tokens, so the [token-usage](#token-usage--cost) cost column matches your
  endpoint (handy for private/MaaS deployments and models the built-in tables
  don't ship).

### Choosing models: six tiers, not sixteen dropdowns

Under **Models & providers** you set six **tiers**, and every role inherits from
one of them:

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
from the agent's own tier map at startup, so it always matches Settings ▸ Models
& providers.

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

## Troubleshooting

- **Panel shows "▶ Start server" / can't connect** — the agent host isn't
  running. Run `.\run_agent.ps1` (or click the button), and check the backend URL
  (`localStorage.agentY_backend`).
- **Autograph toggle or MCP section does nothing / 404** — those routes live in a
  newer host build; **restart `run_agent.ps1`** so the `:5000` host serves them.
- **Canvas nodes/UI look stale after an update** — the ComfyUI copy of
  `agentY-comfyuiConnect` is separate from your dev clone; `git pull` it in
  `<ComfyUI>/custom_nodes/` and reload ComfyUI.
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
