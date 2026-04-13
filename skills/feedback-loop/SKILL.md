---
name: feedback-loop
description: Handle follow-up requests when triage routes directly to Brain (no Researcher pass). Activated for param_tweak, chain, and correction intents.
allowed-tools: 
---

# Feedback Loop — Follow-up Request Handler

Activate this skill whenever the incoming prompt starts with:
> **Follow-up request (intent: `param_tweak` | `chain` | `feedback`)**

The conversation summary block at the top of your context provides:
- `TASK` — what was done in the prior round
- `TEMPLATE` — the workflow template used
- `WORKFLOW_FILE` — the archived final workflow JSON (full path on disk)
- `INPUT_PATHS` — original input images/videos
- `OUTPUT_PATHS` — generated outputs from the prior round
- `STATUS` / `ERRORS` — outcome of the prior round

---

## Intent: `param_tweak`

The user wants to adjust one or more parameters of the last run (style, resolution, seed, strength, steps, cfg, prompt, LoRA, etc.) without changing the overall task.

**Steps:**

1. Read `WORKFLOW_FILE` from the summary — this is the patched workflow on disk, ready to re-use.
2. Identify exactly which parameter(s) the user wants to change from their message.
3. Call `patch_workflow(WORKFLOW_FILE, patches)` with only the targeted changes.
   - Do NOT reload the template from scratch — patch the existing archived workflow.
4. Call `validate_workflow(WORKFLOW_FILE)` and fix any errors.
5. Call `submit_prompt(WORKFLOW_FILE)` → `get_prompt_status_by_id(prompt_id)` once.
6. Vision QA runs automatically via the pipeline executor.

**Example tweaks:**
- "make it more saturated" → adjust cfg or prompt
- "use a different seed" → patch the seed node
- "higher resolution" → patch width/height nodes
- "add a LoRA" → patch or add LoRA loader node

---

## Intent: `chain`

The user wants to pipe the last output into a new workflow (e.g., "now upscale it", "turn it into a video", "make a 3D model from it").

**Steps:**

1. Read `OUTPUT_PATHS` from the summary — these are the files to use as input.
2. Identify the new task type from the user's message.
3. Activate the **workflow-templates** skill to select the appropriate new template.
4. Call `get_workflow_template(template_name)` to get the new workflow file path.
5. Upload input files and assemble the new workflow as normal (follow the Brain's main steps 3–8).
   - Set INPUT_PATHS from `OUTPUT_PATHS` of the prior round.
6. Vision QA runs automatically via the pipeline executor.

**No Researcher pass is needed** — the task is unambiguous from context.

---

## Intent: `correction`

The user is correcting a mistake the agent made (wrong template, wrong model, bad output, failed tool call, etc.).

**Steps:**

1. Read `ERRORS` and `STATUS` from the summary, plus the user's correction message, to identify the root cause.
2. Determine the minimum fix:
   - **Wrong template** → re-select with **workflow-templates** skill and restart from step 2 of the Brain's main flow.
   - **Patch/validation error** → re-patch `WORKFLOW_FILE` with corrected parameters → re-validate → re-submit.
   - **Quality failure** → re-run with different seed or adjusted parameters.
   - **QA failure** → retry from step 7 (Vision QA) using `OUTPUT_PATHS`.
3. Apply the minimum fix — do not redo steps that succeeded.
4. Vision QA runs automatically via the pipeline executor.

---

## General rules

- Always acknowledge the intent and the prior context briefly (one line) before acting.
- Never ask "should I proceed?" — act immediately.
- Keep status messages concise and include the intent type, e.g. `[param_tweak] patching seed…`.
- If the summary is missing or `STATUS: error` for unrecoverable reasons, ask the user one clarifying question then proceed.
