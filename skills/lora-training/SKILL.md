---
name: lora-training
description: Train a LoRA for an image or video model - a person, pet, product, style or motion - with the agent doing every preparation step and the USER approving the dataset before anything trains. Use for "train a LoRA of my dog", "make a style LoRA from these pictures", "teach Wan this motion", "fine-tune Z-Image / FLUX.2 / Qwen-Image / LTX on my photos", "make a character LoRA", "is my LoRA overfitting", "stop the training", "how is the training going".
allowed-tools: lora_dataset_prepare lora_dataset_set_captions lora_dataset_review train_lora lora_training_status lora_training_stop show_in_viewer analyze_image view_image download_image
---

# LoRA training: dataset -> the user's approval -> training -> test

Training is the last step of a long preparation. The tools do the preparation; **the user's approval of the
dataset is a hard gate** (`train_lora` refuses an unapproved dataset, and one edited after approval). Never
approve on the user's behalf, never start training "to save time".

## 0. Should it be a LoRA at all?
A LoRA is for an identity, style, product or motion that must come back *reliably across many prompts*. For
one picture's look, or a one-off, reference images / an edit model / IP-Adapter is cheaper and faster. Say so
if the request is really that, and let the user choose.

## 1. Choose the model and the backend
`lora_training_status()` (no job) lists the presets (steps, rank, VRAM need, caption style), the installed
backends and the GPU. Pick by what the user will render with:

| model | backend | notes |
|---|---|---|
| `z_image`, `flux2_klein_4b/9b`, `qwen_image`, `flux1_dev`, `sdxl`, `sd15` | `ai-toolkit` (default) or `musubi` | image; ai-toolkit downloads the base model from Hugging Face on first use (say the size, ask) |
| `wan22_t2v_14b`, `wan22_5b`, `ltx2` | `ai-toolkit` (musubi for Wan) | video: needs CLIPS, not pictures; LTX needs ~32 GB INT8 (uncertain: the preset says so) |
| `flux2_dev`, `wan22_image` | `fal` (hosted, **costs money**) or a big card | `flux2_dev` locally wants 48 GB |
LTX's official trainer (`Lightricks/LTX-2` `ltx-trainer`) is not driven from here: Linux / WSL2 / farm only.
The dataset folder this tool writes (`trainer/`: files + .txt captions) is what it reads.

## 2. Build the dataset (`lora_dataset_prepare`)
Sources: the user's files / folder, canvas outputs (their file paths), web pictures only with the user's consent
(`download_image` first). The tool drops duplicates, blurry and tiny pictures and watermarked ones, crops to
the model's aspect buckets, cuts long clips, captions, and writes the manifest.
Size and quality per kind (fewer *good* beats more mediocre):
- **character / pet**: 15-40 pictures, varied angles, light, clothes, backgrounds; the face sharp in most.
  One outfit or one background in every picture bakes in. Mixed subjects (two people) ruin it: remove.
- **style**: 25-60 pictures of the style on DIFFERENT subjects. Same subject everywhere teaches the subject.
- **product**: 15-30, clean cuts from every angle, consistent branding visible, varied backgrounds.
- **motion** (video): 10-30 clips of the SAME motion, 2-5 s, the preset's frame count and fps, steady framing.
Read the `warnings` in the result and tell the user the real ones (too few, too similar, mixed subjects).

**Captions** (`needs_captions` -> look at the pictures, write them with `lora_dataset_set_captions`, ids to text):
describe what is visible and should stay *changeable* (setting, pose, clothes, light, camera, motion); do NOT
describe what the LoRA is meant to learn (the face/body of a character, the style's look, the product's shape).
Do not write the trigger word: the trainer prepends it. Style per model is in the preset (`caption_style`):
Z-Image / FLUX.2 / Qwen: natural sentences; SDXL / SD1.5: short phrases; video: subject, action and camera
move in one paragraph (and the sound for LTX).

## 3. The review gate (`lora_dataset_review`)
`lora_dataset_review(dataset)` opens the viewer's **Dataset panel** (grid with captions, edit, include or drop
items, crop, bulk replace, *Approve & train*, *Ask the agent*). It returns `waiting_for_user`: **end your turn**
with one short message (what is in the set, the trigger word, the warnings). The user's approval or their note
arrives as the next message. A note ("remove the blurry ones", "rename the trigger") -> make the change
(`lora_dataset_set_captions`, or tell them to do it in the panel), then review again.
No viewer (Slack, Claude Desktop, MCP): the result says so; send the contact sheet (`send_to_slack` / the
chat) with the captions and trigger word and wait for "approve". Only then
`lora_dataset_review(action="approve", user_message=<their words>)`.

## 4. Train (`train_lora`)
`train_lora(dataset, model=..., backend=...)`; use `dry_run=True` first when anything is uncertain: it shows the
pre-flight (free GPU memory, disk - never C:, base model present, backend installed) without starting.
- Local backends are installed into their own venv on the job's first stage (several GB, takes a while).
- A missing base model: tell the user the size, and only with a yes pass `allow_download=True`.
- `train_lora` returns at once with a job id and refuses a repeat: `already_running` (a run on this dataset is
  still going: follow it with `lora_training_status`, do not start another) and `already_trained` (a finished run,
  e.g. v3, has the same dataset, model and steps: tell the user and ask). Only on the user's explicit wish pass
  `new_run=True`. Never start a run twice because the panel looked unchanged: the Training tab switches to a new run
  by itself and names it (v1...vN).
- **Hosted (`fal`) costs money.** The result `confirm_cost` carries the estimate (an estimate; fal.ai/pricing is
  the truth). Tell the user the price, wait for a yes, call again with `confirm_cost=True`.
- The job runs detached and survives a restart. Tell the user it started and that the Training tab of the
  Dataset panel shows step, ETA, loss and the sample strip. You are notified at each sample and at the end;
  do not poll in a loop.
`steps`, `rank`, `learning_rate`, `options={...}` override the preset; change them only for a reason you can say.

## 5. Reading the signals (`lora_training_status`, the samples)
- **Loss** is noisy and flattens early; a flat loss alone is not "done". Judge by the **samples**: they render
  fixed prompts with the same seed, so the LoRA "arrives" step by step.
- **Under-trained**: samples still look like the base model, the trigger word does nothing -> more steps / higher rate.
- **Overfit**: the prompts without the trigger word also show the subject; backgrounds and poses of the training
  set repeat; faces/products turn plastic or artefacted -> take an earlier checkpoint (`lora_training_stop`
  with `keep_last_checkpoint=True` keeps the newest saved one) or train again with fewer steps / lower rank.
- A video LoRA is checked on motion, not on a still frame.

## 6. After the run
The finished LoRA is copied into ComfyUI's `loras/agentY/` and registered (`agent/lora_registry.json`: name,
trigger word, base model, recommended strength 1.0). `status.result.test_plan` lists prompts, a seed and
strengths: render the same prompts and seed with strength 0 (without) and 0.8 / 1.0 with a LoraLoader on the
base model's usual workflow, look at the pictures yourself (`analyze_image` with both), and give the user a
verdict in two sentences. `result.comparison_sheet` shows the first / middle / last checkpoint samples.
The user chooses the checkpoint to keep; the others stay in the run folder. (musubi-tuner LoRAs for some models
need its `convert_lora` script for ComfyUI - if a LoRA has no effect, say so rather than "it failed".)

## Costs and limits
Local training is free of API cost but occupies the GPU for minutes to hours: it frees ComfyUI's VRAM first, so
tell the user ComfyUI will not render meanwhile. Stop with `lora_training_stop`. Never kill other processes.
