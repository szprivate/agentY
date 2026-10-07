# agentY reference

The notes that used to sit in the README: how the pieces fit, and the details of ports, security and configuration. For day-to-day use see [Using agentY](using-agentY.md).

## The four repos

agentY ships as a small stack of repositories. The installer below wires them all
up; this is what each one is:

| Repo | Location | Role |
|---|---|---|
| **agentY** (this repo) | your working copy | The Strands chat host / pipeline (`run_agent.ps1` / `run_agent.sh`). |
| **[agenty_core](https://github.com/szprivate/agenty_core)** | sibling folder next to `agentY` | Shared ComfyUI/HuggingFace/web/file tool layer + the canonical template/recipe corpus. Installed **editable** (`-e ../agenty_core`); **required**. |
| **[agentY-comfyuiConnect](https://github.com/szprivate/agentY-comfyuiConnect)** | `<ComfyUI>/custom_nodes/` | The **agentY** sidebar tab **and** the canvas nodes (`agentY hook`, `agentY qa`, `agentY python`). |
| **[agentY-mcp](https://github.com/szprivate/agentY-mcp)** | sibling folder next to `agentY` | The alternative **MCP-server / Claude-Desktop** front end (also consumes `agenty_core`). Optional. |


## Architecture

```
ComfyUI  (your browser)
  ├─ agentY sidebar tab   ┐
  └─ agentY hook / python │── agentY-comfyuiConnect  (in <ComfyUI>/custom_nodes/)
     nodes on the canvas  ┘
        │  HTTP + SSE (default :5000, or :5001 on macOS — see Ports)
        ▼
  agentY chat host  ── src/agenty_ui_server.py  →  src/utils/agentY_server.py
        │  Orchestrator agent (+ specialist delegates, Executor stage)
        │  tool layer ── ../agenty_core  (editable install)
        ├──HTTP/WS──►  ComfyUI  (submit workflows, QA the outputs, stage them into /input)
        └──►  memory/conversations.sqlite  (threads, messages, gallery, resume state)
```

Each user turn is owned by the **Orchestrator** agent (a normal Claude / Ollama / Qwen / GPT / Gemini model, per `config/settings.default.toml` + `settings.local.json`). It has the full toolset and can call the specialist agents as delegates. When it finishes assembling a workflow it hands off to the **Executor** (ComfyUI submission → completion polling → [output QA](using-agentY.md#checking-outputs-qa) when you have set a briefing → staging outputs as loader nodes). The ComfyUI custom node is the **frontend + canvas nodes + a graph-load hook** — it talks to the host over HTTP/SSE.

### Staying current

The launcher checks each remote at startup and fast-forwards the three repos
that make up agentY. It never touches a checkout with uncommitted changes or
unpushed commits, it is `--ff-only` (no merge, rebase or reset), and being offline
is not an error — whatever it declines to do, it says so. `requirements.txt`
changes trigger a reinstall before the app starts.

Opt out with `auto_update = false` in settings, `-NoUpdate` / `--no-update`, or
`AGENTY_NO_UPDATE=1`. Set `comfyui_dir` if your ComfyUI isn't next to agentY.

### Ports

The chat host serves on **5000**, and on **5001 on macOS**. The split is not a
preference: on a stock Mac, ControlCenter's AirPlay Receiver already listens on
`*:5000` (and `*:7000`), and it *answers* — a `403` from `Server: AirTunes/...`
rather than a refused connection — so a sidebar pointed at 5000 there reports the
host as down while the host is running perfectly well beside it.

You do not have to keep the two ends in step. The host registers the port it
actually bound with the ComfyUI extension on every start, and the sidebar asks for
it, so `--port`, `AGENTY_UI_PORT` and `agent_server_url` all reach the panel
without anything being configured twice.

To pin a port yourself, set `agent_server_url` in `config/settings.local.json` (or
in the panel's own settings dialog) — a value you chose wins on every platform.
`--port` on the launcher overrides even that, for one run.

If you would rather have 5000 back on a Mac, turn off **System Settings → General
→ AirDrop & Handoff → AirPlay Receiver** and set the port explicitly.

### Security

The chat host listens on localhost. That is not the same as being private: any
page in any browser tab can call a local server, and this one holds your `.env`
and an agent that can run commands. Three things stand in the way, all on by
default and all configurable under `[security]` in
`config/settings.default.toml` (or the panel's Settings ▸ Security).

**Only the panel may call the host.** Requests are refused unless their `Origin`
is the page ComfyUI served, and the host will not answer to a name that is not an
IP address or `localhost` (that check is what stops DNS rebinding — add your
machine's real name to `security.allowed_hosts` if you use one). Nothing needs
configuring for a LAN install: the panel builds the backend address from its own
hostname, so both ends agree by construction.

**A session token.** It lives in `.agenty_token` beside the checkout; ComfyUI
reads it from there and hands it to the panel, which sends it on every call. This
is what stops a script, or another machine, which has no browser to be honest
about where it came from.

The token is **kept across restarts**, on purpose. A ComfyUI tab reads it once, at
page load, and the host gets restarted far more often than the tab does — so a
token that changed every start would silently break every open panel and offer no
cure but a reload nobody knew to do. It buys nothing either: the file is
owner-only and sits beside `.env`, so anyone who can read one can read every key
you own. Delete `.agenty_token` to rotate it deliberately. A panel that somehow
has the wrong one asks ComfyUI again and retries by itself, so opening the tab
before the host exists still works.

**Your API keys are never sent to the browser.** `GET /agentY/settings` returns a
mask, not the values. Type over a field to replace a key; leave it alone and it
stays as it is.

Alongside those, the host warns at startup when a key has been in `.env` for more
than `security.api_key_max_age_days` (30 by default, `0` to switch it off). A key
does not expire on its own, so rotating on a clock is the only protection that
does not depend on noticing a leak. Pasting a new value restarts the clock by
itself — there is nothing to acknowledge, and the ledger in `config/key_ages.json`
stores fingerprints, never the keys.

### What the agent may run

`run_script` executes **one program, with no shell**. Pipes, redirects, `&&`
chains and `$(…)` are refused rather than silently mangled, so the whole class of
quoting and chaining attacks is gone — there is no shell left to attack. The
program must be one of a named set (python, ffmpeg/ffprobe, git, and the usual
listing and searching tools; add more with `security.shell_allowed_commands`),
and path arguments must stay inside the project, ComfyUI's directories or a temp
file (`security.shell_extra_roots` adds more).

That is a real limit and it is not containment, which is worth being plain about:
`python` is on the list because skills need it, and a Python process can do
anything Python can do. So the tools whose effects leave this process —
`run_script`, `iterate`, and `install_custom_node`, which clones and installs code
from the internet — **stop and ask you first**, in the panel, showing the exact
command. Answer *allow once* or *allow for this session*; anything else, including
closing the tab, declines.

If no panel is open to ask — a turn driven from Slack, or a machine left alone —
the request is declined and the agent is told to ask you. Set
`security.unattended_tool_policy = "allow"` if you deliberately want an agent
running with nobody watching. The list itself is `security.ask_before_tools`;
empty it to never be asked.

### OpenMP on macOS

If the agent host dies with `Abort trap: 6` and this above it:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

then two OpenMP runtimes are loaded. torch and faiss-cpu each bundle their own
`libomp.dylib` in their macOS wheels, and the host imports both — torch for SAM3
grounding, faiss for the memory index. The second one to *run OpenMP work* aborts
the process. It depends on which library reaches OpenMP first at runtime rather
than on import order, so it presents as an intermittent crash.

`./install_agent.sh` and `./run_agent.sh` fix this by leaving one runtime:
faiss's copy becomes a symlink to torch's (both are LLVM libomp at the same ABI;
the original is kept as `libomp.dylib.orig`). Reinstalling faiss restores its own
copy, which is why both scripts re-apply it. `scripts/check_env.py` reports the
condition if it ever comes back.

`KMP_DUPLICATE_LIB_OK=TRUE` also silences it and is deliberately **not** used
here — its own error text warns it "may cause crashes or silently produce
incorrect results", and wrong vector-search answers nobody notices are worse than
a crash somebody does.

**Windows is not affected.** There the two ship different implementations —
torch `libiomp5md.dll` (Intel), faiss-cpu `vcomp140.dll` (Microsoft) — and
neither trips the other's duplicate check.

### Configuring defaults

`config/settings.default.toml` holds the committed defaults; put your machine's values (ComfyUI URL/paths, model choices, private endpoints) in `config/settings.local.json` (gitignored, deep-merged over the defaults).

Models are chosen by **tier**, not one dropdown per role: set the `llm.tiers` values and every role inherits from one of them. `llm.pipeline` underneath is per-role **overrides** — leave a role blank to inherit, fill one in only when that single job wants something different. Resolution for any role is *env var → override → tier → built-in default*. Any model value is `"provider,model"`:

```jsonc
{
  "comfyui_url": "http://127.0.0.1:8188",
  "conversation_db": "./memory/conversations.sqlite",
  "llm": {
    "tiers": {
      "orchestrator":      "dashscope,qwen3.7-max",   // drives every turn
      "research_assembly": "dashscope,qwen3.6-plus",  // templates, graph building, repair
      "fast_utility":      "dashscope,qwen3.6-flash", // info, search, planner, learnings, …
      "vision":            "dashscope,qwen3-vl-flash",// reads the images YOU provide
      "qa_judge":          "dashscope,qwen3-vl-plus", // grades finished outputs
      "coder":             "dashscope,kimi-k2.7-code" // scripts and custom nodes
    },
    "pipeline": {
      // per-role overrides; blank = inherit from the tier. Usually all blank.
      "video_agent": "dashscope,qwen3-vl-plus"
    },
    "dashscope": {
      // Public International endpoint; for mainland China use
      // https://dashscope.aliyuncs.com/compatible-mode/v1
      "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
      "model": "qwen-plus"
    }
  }
}
```

Each `"provider,model"` value can be `"claude,claude-opus-4-8"`, `"ollama,qwen3-coder:30b"`, `"dashscope,qwen3.6-flash"`, `"openai,gpt-4o"`, or `"google,gemini-2.5-pro"`. **`dashscope`** routes to **Alibaba Model Studio** (Qwen over its OpenAI-compatible API) — set `DASHSCOPE_API_KEY` in `.env`; aliases `qwen` / `modelstudio` / `alibaba` also work. **`openai`** and **`google`** (alias `gemini`) route to OpenAI and Google Gemini respectively (Gemini via its OpenAI-compatible endpoint) — set `OPENAI_API_KEY` / `GEMINI_API_KEY` in `.env`. You can change any stage live from chat with `/switch_model` (e.g. `/switch_model orchestrator claude,claude-opus-4-8`); its model picker is discovered live from each configured provider, so only vendors whose key is set appear.

### Canvas nodes

`agentY-comfyuiConnect` adds a set of nodes under the **agentY** category, so you can drive the agent *from the graph itself*:

- **`agentY hook`** — an instruction attached to the canvas. Wire any node's output into its **auto-growing `anchor` input(s)** and type a directive. Seven purposes:
  - *inline_parameter* — annotate an existing node ("sweep the seed 6×", "iterate the files in this folder"); the agent expands and runs your on-canvas graph.
  - *make_workflow* — the agent generates and runs a workflow (or Python script) from the prompt, using any wired input.
  - *text* — the agent writes a string answer and drops a wireable `agentY text` node carrying it.
  - *general_request* — free-form: the agent decides the action itself (answer, generate, run, compute).
  - *iterate* — an interactive **refinement loop**, one generation per turn, each result fed back into the wired `LoadImage`.
  - *qa* — your **quality briefing**: the directive is the checklist, the anchors are reference images. See [Checking outputs](using-agentY.md#checking-outputs-qa).
  - *review* — a deliberate **stop** between stages, so you pick what continues. See [Review](using-agentY.md#review-stop-and-pick-what-continues).

  Its `out` **output** is type-agnostic, so one hook can gather several inputs and produce an image, a video **or a scalar** for the next one; wire hooks output→input to build a **multi-step chain**. One `remember` switch decides whether what a hook produced outlives the run — [baked into a subgraph](using-agentY.md#the-keep-switch-should-this-outlive-the-run) for `make_workflow`, memorized for the rest. A hook is inert on a normal *Queue Prompt*, so it never affects a manual run, and **Bypass** (`Ctrl+B`) or mute disables one without deleting it.

- **`agentY qa`** — everything QA on one node. The checkable half as controls rather than prose: aspect ratio, minimum resolution, sharpness, grain, clipping, black and frozen frames, **likeness** against the images wired into `reference`, and a retry budget. Each is settled by measuring the finished file; `notes` carries what needs judgement. Anything left on `any` is not checked, and an empty node enforces nothing. Wire what you want checked into `judge` (a hook's `out`, an IMAGE, a collector, a path) and what to compare it against into `reference`; unwired, it judges everything the run produces. See [the qa node](using-agentY.md#the-agenty-qa-node).

- **`agentY python`** — runs a Python snippet as a node, written by you or by the agent (`run_python_node`); its outputs carry the values, and files it saves become the run's outputs. Baking uses it so a value computed at runtime becomes a genuine re-runnable output. ⚠️ **It executes arbitrary Python whenever the graph runs** — meant for your own, self-hosted, agent-built workflows; don't run baked workflows from untrusted sources. `AGENTY_PYTHON_NODE_DISABLED=1` makes it a no-op.

Also: **`agentY collector`** (hand the agent a batch of on-disk files), **`agentY add tag`** (name a reference so `#hero_face` resolves), **`agentY load item`** and **`agentY expand image batch`** — see [the guide](using-agentY.md#the-agenty-python-node--collectors).

**Bake a chain into subgraphs.** Turn `remember` on for your make_workflow hooks and each stage's generated workflow is nested into a ComfyUI **subgraph** (inputs/outputs matching the hook's slots), **added** beside the hook nodes — nothing is removed — and wired to mirror the chain. The result is a native workflow you can re-run **without the agent**.

### LLM configuration priority

Each value resolves in order — first match wins: **CLI flag → environment variable → `config/settings.local.json` → `config/settings.default.toml` → hard-coded default.** Committed defaults live in `settings.default.toml`; per-machine values (paths, model pins, private endpoints) go in the gitignored `settings.local.json`, which is deep-merged over the defaults. You can also change any stage live with `/switch_model <stage> <provider,model>`.

### In-panel settings & token usage

Open ComfyUI's **Settings** panel → **agentY** → **Open agentY Settings…**. That one row is the entire agentY section; everything else is inside the modal:
- Your auth keys (`.env`) and every setting — **model tiers** and per-role overrides, directories, toggles — in **collapsible groups**, with the rarely-touched ones behind **Show advanced settings**. Changed values are saved as overrides in `config/settings.local.json`, leaving the committed defaults untouched — no file editing.
- **MCP servers** (`config/mcp.json`, with per-server status + an **Authorize…** button for OAuth) and model **pricing**.
- **Viewers** — the **message-history log**, the **long-term-memory editor**, and the **token usage** breakdown (cost per model / per agent role, with a **🗑 Clear log** button). Token usage is also the **📊** button in the chat panel's top bar.

The chat panel's top bar also has a **🖼 autograph toggle** — flip whether finished workflows/results are auto-loaded onto the canvas, live (no restart).

### Memory

Long-term memory is stored in a local FAISS index (`memory/agenty_memory.faiss`) via **mem0**, with the embedder you picked at install (built-in local, Ollama, or a provider — see [Choosing the embedder](using-agentY.md#choosing-the-embedder)). Conversation threads are separate — they live in the SQLite store above.

## Adding Custom Workflow Templates

```powershell
# Register a new workflow template (also generates a SKILL.md)
.\scripts\add_workflow.ps1 path\to\your_workflow_api.json

# Remove a registered template (also removes its skill directory)
.\scripts\remove_workflow.ps1 your_workflow_api
```

You can also do this from the chat with `/add_workflow <path>` (or `/add_workflow canvas <name>` to register the open graph) and `/remove_workflow <name>`. Custom templates live in `comfyui_workflow_templates_custom/templates/`; the shared template/recipe corpus lives in **agenty_core**.
