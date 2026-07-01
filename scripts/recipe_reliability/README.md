# recipe_reliability

A harness that drives the local workflow-recipe intents through the real
researcher -> brain -> execute pipeline (headless, no Chainlit), captures the
ComfyUI execution outcome, and classifies each recipe so we can tell an
*agent-build* bug from an *environment* problem.

## Usage

```bash
# One recipe / a task family / everything local:
python -m scripts.recipe_reliability.run --only image_edit__qwen_image
python -m scripts.recipe_reliability.run --task "Image Edit"
python -m scripts.recipe_reliability.run            # all execution==local recipes

# Per-recipe timeout (seconds) and a quick subset:
python -m scripts.recipe_reliability.run --task "Image Edit" --limit 1 --timeout 840
```

Results stream into `report.json` after every recipe (so a hang never loses
progress). Each entry has the intent, duration, the interactive events seen, the
ComfyUI statuses/errors, and the classified `outcome`:

- `pass` - ComfyUI executed the workflow with no error.
- `agent_build_fail` - the Brain could not assemble/validate a workflow.
- `comfyui_exec_error` - the Brain built + queued it, but ComfyUI failed at run.
- `resource_oom` - execution reached the sampler but ran out of GPU VRAM
  (environment/resource, not an agent bug - the graph was structurally valid).
- `missing_model` - a required model file is not installed (environment).
- `timeout` / `no_execution` - inspect the log.

Only `agent_build_fail` (and some `comfyui_exec_error`) are agent-fixable; the
rest are environment issues.

## Input images

Image-input recipes need real inputs. The harness synthesizes a pool of distinct
test images (PIL) in ComfyUI's input dir and feeds each recipe **as many inputs
as its recipe type exposes** (`boundary_ports` IMAGE count) - e.g. Qwen-Image
and FireRed edits get 3 reference images, others get 1. This exercises the
multi-input wiring, not just the single-image path.

## Fixes this loop produced (agenty_core)

Running the sweep on ComfyUI v0.27 surfaced (and drove fixes for) three
agent-level reliability bugs, all in `agenty_core.tools.comfyui`:

1. **Subgraph flattening** (`_flatten_subgraphs`). 93/95 official templates wrap
   their whole graph in a ComfyUI *subgraph* node (11 nest them) whose
   `class_type` is a UUID. ComfyUI's /prompt API rejects those ("Node not
   found"). The converter now inlines every subgraph recursively into concrete
   nodes with rewired links before API conversion.
2. **Boundary I/O synthesis.** Subgraph templates expose bare IMAGE boundary
   *ports* instead of `LoadImage` / `SaveImage` nodes; the flattener synthesizes
   a `LoadImage` for each unconnected IMAGE input and a `SaveImage` for each
   terminal IMAGE output, restoring the classic-template invariant the Brain and
   Researcher expect.
3. **Model-name resolution** (`_resolve_model_names`). Template widget filenames
   drift from the live server's (`ae.safetensors` / `FLUX1/ae.safetensors` vs
   object_info's `FLUX1\ae.safetensors`); model inputs are now snapped to the
   exact object_info string so separator/folder drift is not mistaken for a
   missing model.

The pipeline also now registers image paths embedded in a plain-text (headless)
message into `last_user_input_images` (`Pipeline._scan_media_paths`), so the
Brain wires the real input file when driven without Chainlit.

The remaining common blocker is `resource_oom`: the large edit models
(Flux 2 dev ~32B, Qwen-Edit ~20B, Flux 2 Klein 9B) can exceed 34 GB VRAM at
1 MP. That is a hardware limit, not an agent bug.

## vision_qa_test.py - vision-QA RL-style loop

Tests the vision-QA agent (`src.executor._vision_qa`) in isolation - no ComfyUI,
no diffusion. It feeds the agent controlled `(intent, image[, input image])`
cases with a KNOWN ground-truth verdict (PIL-generated: red square, blue circle,
recolor edits, text), parses the PASS/FAIL it returns, and scores verdict
accuracy + false-pass / false-fail. Reward = correct verdicts; tune
`config/system_prompts/system_prompt.qaChecker.md` (or `_vision_qa`) and re-run.

```bash
python -m scripts.recipe_reliability.vision_qa_test           # all cases
python -m scripts.recipe_reliability.vision_qa_test --limit 1 # validate setup
```

### Second known blocker: Ollama vision

Vision QA (and memory) run on Ollama. Currently Ollama's model runner is
crashing (`0xc0000005` access violation): **every** `/api/chat` call returns 500
- text and image alike - so the QA loop cannot run. Restart Ollama (and verify a
multimodal model such as `qwen3-vl:*` responds to an image) before running this.
The configured `executor_vision_model` is `gemma4:12b`; if it turns out not to be
multimodal once Ollama is healthy, switch it to a `qwen3-vl` model.
