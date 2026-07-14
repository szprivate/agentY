---
name: spawn-subagent
description: When and how to delegate a job to a fresh, specialized subagent — plan first, then spawn an executor with a MINIMAL toolset and a single skill. Use for complex/multi-step/batch jobs, or anything that keeps looping in the main context.
allowed-tools: spawn_subagent, run_planner
---

# Spawning a specialized subagent

Spawn a subagent to run a self-contained job in a **fresh, lean context** instead
of doing it inline. A subagent with only the handful of tools its task needs
carries far fewer tool definitions per call — less context, faster, and small
models pick the right tool far more reliably from ~6 options than from ~60. Its
context is also clean and stable (cache-friendly), and a spawned run is bounded,
so it can't spiral your main conversation.

## When to spawn (and when not to)

**Spawn** when the task is:
- **multi-step / batch** — e.g. "apply this edit to N images", "generate a set,
  upscale each, then compile", a storyboard, a from-scratch node build;
- **self-contained** — you can hand it one complete instruction and get back one
  result;
- **prone to looping** — a job that has been making many tool calls or running
  away in the main context.

**Do NOT spawn** for a simple one-shot generation or a quick question — handle
those inline. Spawning has overhead; use it when the isolation pays for itself.

## The procedure — plan first, then spawn

1. **Draft a short plan** (2–6 concrete steps). For a genuinely multi-stage job
   you may call `run_planner` to decompose it. Keep it tight.
2. **Confirm the plan with the user** when the job is large, expensive, or
   irreversible (many generations, downloads, long video). For routine jobs skip
   the gate and proceed. Presenting the plan also lets the user correct the route
   before any compute is spent.
3. **Pick the MINIMAL toolset** — the exact tool names the plan needs, nothing
   else. Typical bundles:
   - same-op over N images → `upload_image_multiple` (stage them all in one call),
     `get_workflow_template`, `apply_brainbriefing`, `validate_workflow`,
     `duplicate_workflow`, `signal_workflow_ready`
   - from-scratch build → `get_workflow_recipe`, `get_node_schema`,
     `add_workflow_node`, `update_workflow`, `validate_workflow`,
     `signal_workflow_ready`
4. **Pick a skill** that encodes the procedure (e.g. `batch-handoff` for a
   same-op-over-N-images job) so the subagent follows fixed steps instead of
   improvising.
5. **Spawn it** with an explicit, self-contained task plus the minimal tools and
   the skill:
   ```
   spawn_subagent(
     task="<the full plan as ONE instruction, with the concrete inputs/paths>",
     tools=[<minimal tool names>],
     skill="<skill name>",
     model="<optional stronger/caching model for a hard reasoning job>",
   )
   ```
6. **Fold the result back** — relay what the subagent produced. If it reports a
   blocker, decide the next step; don't silently re-do its work inline.

## Rules

- Prefer an explicit `tools` list over a preset `toolset` — the whole point is a
  lean context.
- Give the subagent a **complete** task: it cannot ask you questions, and it
  cannot spawn further subagents (depth-1).
- One subagent per self-contained job; don't fan out unless the jobs are truly
  independent.
- The subagent's terminal step for a generation is `signal_workflow_ready` — make
  sure that tool is in its `tools` list.
