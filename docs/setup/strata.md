# Strata: a local model for the agents

[Strata](https://github.com/Niko1221/Strata) runs a GGUF model on this machine behind an
OpenAI-compatible API. agentY knows it as the provider `strata`: any tier or role can run on it,
it costs nothing per token, and agentY stops and starts it around ComfyUI renders so the two can
share one graphics card.

## Install

1. Download Strata from its [releases page](https://github.com/Niko1221/Strata/releases) and install
   it (tested: Strata 0.2.7, engine 0.1.52 cuda13).
2. Strata keeps its models and components under `%LOCALAPPDATA%` by default. To put them on another
   drive (models are 20+ GB; nothing large belongs on C:), start it through a small wrapper that sets
   its folders, e.g. `D:\AI\tools\strata\strata-d.cmd`:

   ```bat
   @echo off
   set STRATA_DATA_DIR=D:\AI\tools\strata
   set STRATA_COMPONENTS_DIR=D:\AI\tools\strata\components
   set TEMP=D:\AI\tools\strata\_tmp
   set TMP=D:\AI\tools\strata\_tmp
   "D:\AI\tools\strata\terminal\current\bin\strata.cmd" %*
   ```

3. Download a model once and start it by hand to see that it runs:

   ```
   strata-d.cmd launch --family qwen36 --model UD-Q4_K_XL --context 128k --parallel 1 --no-tui --yes
   strata-d.cmd launch --stop
   ```

## Settings

**Settings > Connections > Strata (local model server)**:

| setting | default | |
|---|---|---|
| Server URL (`base_url`) | `http://127.0.0.1:8080/v1` | `STRATA_BASE_URL` in `.env` overrides it |
| `model` | `qwen3.6-35b-a3b-ud-q4_k_xl` | offered in the model menus while Strata is not running |
| `launcher` | `D:\AI\tools\strata\strata-d.cmd` | the `strata` command or your wrapper; blank = `strata` on PATH |
| `launch_args` | `launch --family qwen36 --model UD-Q4_K_XL --context 128k --parallel 1 --no-tui --yes` | what agentY starts it with |
| `think` | off | Qwen's reasoning, for every Strata agent (see below) |
| `vision` | on | the model reads images |
| `autostart` | on | start Strata when an agent needs it and it is not running |
| Share the GPU with ComfyUI (`gpu_sharing`) | on | see below |
| `start_timeout` / `request_timeout` | 600 s / 1800 s | |

**Test** lists what Strata serves (`GET /v1/models`); **Start** and **Stop** do what they say (a
start waits for ComfyUI's queue to be empty first). No API key is needed on this machine; put
`STRATA_API_KEY` in `.env` if your server wants one.

Then pick the model for a tier under **Models**, group *Strata (local)* - or by hand,
`"orchestrator": "strata,qwen3.6-35b-a3b-ud-q4_k_xl"` in `llm.tiers`.

**Thinking stays off.** agentY sends `chat_template_kwargs: {enable_thinking: false}` with every
call. With thinking on, Qwen 3.6 invented a tool call in a test, took twice as long and lost the
prompt cache from one call to the next. A role set to reason under *Models* (💭) still reasons.
Other request fields can be added as `llm.strata.extra_body` (an object) in `settings.local.json`.

## Model and window

`qwen36 / UD-Q4_K_XL` (Qwen 3.6 35B-A3B, 4-bit) on a 32 GB card, `--parallel 1` (two slots do not
fit). Measured on an RTX 5090:

| window | prompt read (cold) | writing | a 50k-token prompt |
|---|---|---|---|
| `--context 64k` | 6,000-10,000 tok/s | 100-200 tok/s | 8 s to the first token |
| `--context 128k` | 1,700-2,800 tok/s | 30-210 tok/s (slower as the prompt grows) | 23 s |
| `--context 256k` | 650-1,000 tok/s | 13-50 tok/s | 51 s |

64k is fast and holds an ordinary turn; 128k holds long turns (an orchestrator call in an open graph
with project memory is ~47k tokens) at a third of the speed. Strata reuses the prompt between calls:
a follow-up call reads only what is new (cached tokens are counted as cache hits in the token log).
With thinking off, three orchestrator test tasks took 16, 34 and 72 s (2-6 tool calls each, no tool
errors).

## One GPU for Strata and ComfyUI

Strata holds the whole card while it runs (~31 GB), and so does a render: both at once starve each
other. With `gpu_sharing` on (the default), and only when a tier or role runs on Strata:

- **Before a render** that needs the card - anything agentY queues (run, batches, the canvas
  probe, Atlas, the python node), and the Queue button / prompt loop on the page - agentY waits for
  the Strata calls in flight to finish, runs `strata launch --stop` and waits until the card is
  free. Graphs of API nodes only leave Strata running.
- **At the next agent call** after that, agentY waits until ComfyUI's queue is empty, asks ComfyUI
  to unload its models, starts Strata with `launch_args`, and waits until it answers. The panel
  shows *Starting local model… (Strata)* meanwhile.

This works when the orchestrator itself runs on Strata: the stop happens inside the tool call that
queues the render, the start before the following call. Agents that run side by side (parallel
workflow preparation) all wait for the one start and then queue on Strata's single slot.

Turn `gpu_sharing` off when Strata has a card of its own.

## Limits

- **A start costs time.** 30 s with a warm disk cache, up to 6 min on a cold one - after every
  render that stopped it. The first call after a start also re-reads its whole prompt (no cache
  yet). A conversation that renders often is faster on a hosted model.
- **One call at a time** (`--parallel 1`). Parallel agents wait their turn; their calls have a long
  timeout (`request_timeout`) for that reason.
- LoRA and splat training use the card outside ComfyUI's queue and do not stop Strata.
- Strata must be on this machine for the start/stop; a Strata on another machine works as a
  provider with `gpu_sharing` off.
