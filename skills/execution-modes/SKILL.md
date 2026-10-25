---
name: execution-modes
description: Where the open-weight models of a run execute - this machine's GPU or Comfy Cloud. Use for "run this in the cloud", "use Comfy Cloud", "keep it local", "run it on my GPU", "why did that run in the cloud", "what did that cost", "it says a model is missing / does not fit", or any run that came back with ran_on, cannot_run_here or local_only.
allowed-tools: set_execution_mode_for_turn get_execution_mode
---

# Execution modes: local, hybrid, cloud

The mode (Settings > Execution, written by the installer) decides where the **open-weight models**
(checkpoints, UNets, LoRAs, text encoders ...) of a run execute:

- **local** (default): on this machine's ComfyUI, as always. Comfy Cloud is never used.
- **hybrid**: locally when every model is on disk and fits the GPU's free VRAM; otherwise on Comfy Cloud
  if it has every model and node; otherwise the run is refused with what is missing.
- **cloud**: open-weight work goes to Comfy Cloud (this machine needs no GPU).

**Partner (API) nodes - Seedance, GPT Image, Kling, Nano Banana, Flux Pro ... - are not affected by the
mode.** They need no GPU, run through the local ComfyUI in every mode (local included), under the same
credit checks as always. A workflow that mixes partner nodes and open-weight models is routed by its
open-weight part alone. Never tell the user a mode "turns partner models off".

You do not choose the backend; `execute_workflow` and the batch tools route by the rules above. You only:

1. **Override the mode for the turn** when the user says where a run should happen. "run this in the cloud",
   "use Comfy Cloud" -> `set_execution_mode_for_turn("cloud")`; "keep it local", "on my GPU" ->
   `set_execution_mode_for_turn("local")`; "let it pick" -> `"hybrid"`; `"default"` removes the override.
   Call it before the run, once. It lasts this turn and does not change the setting.
2. **Look** with `get_execution_mode()` when asked what the mode is, whether a key is set, or the balance.
3. **Tell the user where it ran and what it cost.** Run results carry a line `ran_on: local | comfy_cloud |
   deadline` and, for cloud runs, `cost $x.xx` (measured from the Comfy credit balance). Say it in one
   short sentence with the result. A refusal `confirm_cost` or `insufficient_credits` works exactly as for
   partner nodes: give the estimate and ask; only after a yes, run again with `confirm_cost=True`.

## When a run is refused (`error_kind: cannot_run_here`)

The answer lists `missing_models` (not on this machine), `missing_on_cloud_models`, `missing_on_cloud_nodes`
and a `hint`. Offer, in this order: download the missing models (`find_hf_file` then `download_hf_model`;
they check the drive and free space first), or a partner-model alternative (`partner_run`), or swap to
models the cloud has. Do not retry the same graph. No Comfy API key (`COMFYUI_API_KEY`) is the other
common cause: tell the user to add it under Settings > Authentication.

## What Comfy Cloud cannot do (`error_kind: local_only`)

- SAM3 masks / finding things in pictures (runs on this GPU)
- Brush splat training (`train_splat`)
- local LoRA training (the hosted `fal` trainer works in any mode)
- custom nodes Comfy Cloud does not have
- canvas-only features (hooks, canvas edits, screenshots) happen on this machine's ComfyUI in every mode;
  only the model run moves

In cloud mode these answer `local_only` with the reason. Say so plainly; if the user wants it, switch the
turn to local (`set_execution_mode_for_turn("local")`) and retry. Never fail silently or substitute.

Outputs of a cloud run land exactly like local ones (canvas loader nodes, the viewer, Slack), and a cloud
failure goes through the same repair loop as a local one.
