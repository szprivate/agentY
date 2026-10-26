# Execution modes: local, hybrid, cloud

agentY can run the open-weight models of a workflow on this machine's GPU, on **Comfy Cloud**, or pick between the two.
The mode is one setting, `execution.mode` (Settings ▸ Execution); the installer chooses a starting value from your
hardware.

| Mode | Open-weight models (SDXL, Flux, Wan, LTX, LoRAs ...) | Needs |
|---|---|---|
| `local` (default) | on this machine's ComfyUI, exactly as before. Comfy Cloud is never used. | a GPU |
| `hybrid` | locally when every model is on disk **and** fits the GPU's free VRAM; otherwise on Comfy Cloud if it has every model and node; otherwise the run is refused with what is missing | `COMFYUI_API_KEY` |
| `cloud` | on Comfy Cloud | `COMFYUI_API_KEY` and a Comfy Cloud subscription |

**Partner (API) nodes are not affected by the mode.** Seedance, GPT Image, Kling, Nano Banana and the rest need no
GPU on your side: their models run on the providers' servers. In *every* mode, `local` included, the request is sent from your
local ComfyUI under the usual partner-credit checks. A
workflow that mixes partner nodes with open-weight models is routed by its open-weight part alone. (If no local
ComfyUI is reachable at all, a partner-only workflow in `hybrid` / `cloud` runs on Comfy Cloud instead; the partner
nodes work there too.)

## Choosing the mode

Strongest first:

1. **For one turn**, in words: "run this in the cloud", "use Comfy Cloud" → cloud; "keep it local", "run it on my GPU"
   → local. The agent sets this with the tool `set_execution_mode_for_turn` (it lasts that turn, 30 minutes at most, and
   does not touch the setting). `get_execution_mode` shows what is in force, whether a key is set and the balance.
2. The environment variable `AGENTY_EXECUTION_MODE` (a deployment pins it).
3. The setting `execution.mode` in `config/settings.local.json` (Settings ▸ Execution).
4. `local`.

## What the installer does

`install_agent.ps1` / `install_agent.sh` (same stages, same flag `-ExecutionMode` / `--execution-mode`) detect the
hardware with `scripts/hardware_profile.py` and ask **one** question with the recommendation pre-selected:

| Detected | Recommended |
|---|---|
| NVIDIA, 12 GB or more | local |
| NVIDIA, 6 to 12 GB | hybrid |
| NVIDIA under 6 GB; no GPU; AMD / Intel GPU (Windows, Linux) | cloud |
| Apple Silicon (unified memory) | hybrid |
| Intel Mac | cloud |

The answer is saved to `config/settings.local.json`. For hybrid and cloud the installer then asks for the Comfy API key
(`COMFYUI_API_KEY`, from platform.comfy.org). With `-NonInteractive` nothing is asked: the mode stays `local` and no
ComfyUI is installed. A re-run keeps an earlier answer on Enter.

**No ComfyUI found?** The installer offers **ComfyUI Desktop**, Comfy's official one-click app
(https://www.comfy.org/download): on Windows `winget install --id Comfy.ComfyUI-Desktop --exact --silent`, on an Apple
Silicon Mac `brew install --cask comfy`, otherwise the download page opens. It asks first (default yes for hybrid / cloud,
no for local). Desktop creates its ComfyUI only when started once, so start it, finish its first-run setup, and run the
installer again to add the sidebar node.

**Several ComfyUI installs?** The installer lists every one it can find (the usual folders, Desktop's `ComfyUI-Installs`,
the data folder named in Desktop's `config.json` (`basePath`), and ComfyUI processes that are running), each with its
version or git commit, and you pick one by number, type another path, or press Enter to skip. It installs the sidebar node
only into the one you chose; if you decline them all, nothing is installed and the summary says so. With `-NonInteractive`
it uses `-ComfyUIPath`, or the only install found, and otherwise installs nothing.
In `cloud` mode a local ComfyUI is still useful because partner API nodes run in it without a GPU. A ComfyUI with no
usable GPU has to be started with `--cpu` (the flag exists in ComfyUI's `main.py`; CPU is for everything, slow, which
does not matter for partner nodes).

## How a run is routed

Both executors (agentY and agentY-mcp) share one seam with the Deadline farm: `execute_workflow`, the batch tools and
the hook stages all go through `_submit_workflow`. After the usual preparation and the partner-credit gate, it asks
`agenty_core.utils.exec_route.decide(workflow)`; no LLM is involved:

1. `local`: nothing is analysed; today's path (or the Deadline farm, if switched on).
2. The workflow's nodes are classified from `/object_info` (partner / core / custom) and its model files are
   found (inputs ending in `.safetensors .ckpt .pt .gguf ...`).
3. **No model files and only partner + core nodes:** the local ComfyUI.
4. **hybrid**: all models on disk (`config/models.json` inventory) and the estimate fits the free VRAM
   (`/system_stats`; estimate = largest model file + a quarter of the others + `vram_margin_gb`, default 2 GB, because
   ComfyUI loads stage by stage) → local.
5. Else Comfy Cloud, **if** it has every model file (`/api/experiment/models`) and every node class
   (`/api/object_info`); both lists are cached for 6 hours (memory and `.cache/comfy_cloud.json`).
6. Else a refusal, `error_kind: cannot_run_here`, listing `missing_models` (not on this machine),
   `missing_on_cloud_models`, `missing_on_cloud_nodes` and a hint: download the models (`find_hf_file` +
   `download_hf_model`, which check the drive and the size), use a partner model (`partner_run`), or swap models.

A third backend, `comfy_cloud`, sits next to local and `deadline`. Its references look like `cloud:<job id>` where the
executors expect a `prompt_id`; a cloud failure carries the failing node and exception, so the heal loop treats it like a
local one, and outputs are downloaded into `<output_dir>/comfy_cloud/` and then follow the usual path (QA, canvas loader
nodes, viewer, Slack).

**Every run says where it ran:** a line `📍 ran_on: local | comfy_cloud | deadline`, and for cloud runs the cost
(`cost $0.12`), measured from the Comfy credit balance before and after. (Plain `local` mode stays as quiet as before.)

## Cost

Before every cloud run (`agenty_core.utils.cloud_run.gate`):

- the balance is read; below `execution.min_balance_usd` (default $0.50) the run is refused (`insufficient_credits`);
- the estimate is the partner nodes' price badges plus GPU time: Comfy Cloud bills it per second (standard
  workflows on an RTX 6000 Pro: about 0.27 credits/s, $0.077/min, $4.62/h), so the estimate is
  `execution.cloud_gpu_usd_per_second` times an expected runtime of `cloud_gpu_min_seconds` (20) to
  `cloud_gpu_max_seconds_image` (120) or `cloud_gpu_max_seconds_video` (600) - about $0.03-0.15 for an image
  and up to $0.77 for a video run;
- an estimate at or above the partner confirmation level (`confirm_above_usd`, default $1) is refused with
  `confirm_cost` until the user says yes; then the agent repeats it with `confirm_cost=True`;
- afterwards the real spend is measured and added to the turn's usage like partner spend.

## What Comfy Cloud cannot do

One list, `agenty_core.utils.execution_mode.CLOUD_LIMITS`, which the `execution-modes` skill repeats. In `cloud` mode
these answer `error_kind: local_only` with the reason instead of failing silently:

- SAM3 masks / finding things in pictures (this GPU)
- Gaussian splat training with Brush (`train_splat`)
- local LoRA training (the hosted `fal` trainer works in any mode)
- custom nodes Comfy Cloud does not have (the run is refused naming them)
- canvas-only features (hooks, canvas edits, screenshots) happen on this machine's ComfyUI in every mode; only the model
  run moves

## The Comfy Cloud API used

Plain REST at `https://cloud.comfy.org`, key in the `X-API-Key` header (no MCP involved); client in
`agenty_core/utils/comfy_cloud.py`. Sources: docs.comfy.org/development/cloud/overview, .../api-reference and
https://docs.comfy.org/openapi-cloud.yaml. The v1 Cloud API is marked experimental / superseded by Comfy API v2 for
workflow submission, but v1 is the one with models, nodes and queue management; the client isolates it in one file.

| Need | Call |
|---|---|
| submit | `POST /api/prompt` `{"prompt": ..., "extra_data": {"api_key_comfy_org": key}}` → `prompt_id` |
| poll | `GET /api/job/{id}/status` (`waiting_to_dispatch` … `completed` / `error` / `cancelled`) |
| outputs, errors | `GET /api/jobs/{id}` (`outputs`, `execution_error`) |
| download | `GET /api/view` → 302 to a signed URL, fetched without the key |
| upload inputs | `POST /api/upload/image` (multipart) |
| has the models / nodes | `GET /api/experiment/models[/folder]`, `GET /api/object_info` |
| cancel | `POST /api/queue` `{"delete": [id]}` |

The key is never printed or put in an error message. Settings: see `[execution]` in `config/settings.default.toml`.

## Tests

`agentY-core`: `python -m unittest agenty_core.tools.tests.test_execution_modes` (router, precedence, cost gate, client
against stubbed responses). `agentY`: `tests/test_execution_modes.py` (executor seam, partner nodes in every mode),
`tests/test_hardware_profile.py` (recommendation rules, installer parity). No test spends credits.
