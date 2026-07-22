---
name: self-extension
description: How to extend yourself — capture a reusable procedure as a skill (create_skill), spawn a subagent for a heavy self-contained job (spawn_subagent), or author a ComfyUI node pack from a model's GitHub repo (create_custom_node) — plus the hard safety policy (never edit your own live code; write a proposal instead). Activate when the user asks you to build/save a skill, spawn a subagent, or turn a GitHub model into a node.
allowed-tools: create_skill, list_skills, remove_skill, spawn_subagent, create_custom_node, list_generated_nodes, run_script, write_text_file, run_planner
---

# Self-extension

- `create_skill(name, description, instructions, allowed_tools?)` — when you work
  out a **reusable multi-step procedure**, save it as a skill so you (and future
  turns) can reload it via the `skills` tool instead of re-deriving it. Your
  authored skills appear in `<available_skills>` from the next turn.
- `list_skills()` / `remove_skill(name)` — manage what you've authored.
- `spawn_subagent(task, toolset?, model?, tools?, skill?)` — isolate a heavy,
  multi-step, or self-contained sub-task in a fresh, lean context; it runs to
  completion and returns its text (subagents cannot spawn further subagents).
  **ONLY call this when the user's current message explicitly asks you to use or
  spawn a subagent.** For all normal work — staging inputs, building/duplicating
  workflows, batch handoff, research — use your own tools directly; never
  delegate routine steps to a subagent. If you call it without an explicit user
  request it will refuse (it is disarmed for that turn).
  When the user *has* asked: **prefer a MINIMAL explicit `tools` list + a `skill`**
  over a preset `toolset` — a subagent with only the ~6 tools its job needs
  carries far less context and picks tools far more reliably than the full set.
  Activate the **`spawn-subagent` skill** for when-and-how-to-spawn rules (plan
  first, scope the toolset, optional user approval for big jobs).
- `create_custom_node(github_url, node_name?, notes?)` — when the user points you
  at a **model's GitHub repo that has no existing ComfyUI node**, run the coder
  agent (custom-node-from-github skill): it clones the repo, reads the docs +
  inference code, and writes a self-contained node pack into
  `output/custom_nodes/<name>/` (the
  user can then publish it as its own repo). Relay the returned `agent_summary`,
  especially any "Unresolved / TODO" items it flagged. Use `list_generated_nodes()`
  to see packs already created. This authors code for the user to review/publish —
  it does not install the node into the live ComfyUI.

**How far self-extension may go — the safety policy:**

- **Capturing capability as a skill is always allowed.** When a script or
  procedure works, turn it into a skill with `create_skill` (it lands under
  `skills/_scratch/`, is reversible via `remove_skill`, and is data — not live
  code). This is the default way to "add a script to your toolset". Keep the
  script itself in `output/scripts` and have the skill invoke it via `run_script`.
- **You may NOT edit your own code (`src/`, `agenty_core/`) live.** Those are
  imported by the running server (and by another app), so a live edit can break
  everything with no review. If you believe a change to the agent's own code is
  warranted, do **not** write into `src/` or `agenty_core/`. Instead write a
  **proposal**: save the intended change (a diff or a full replacement file plus a
  short rationale) under `output/proposals/`, and tell the user it's ready for
  review. A human applies, tests, and restarts. Promoting a `_scratch` skill into
  the committed `skills/` set is likewise a human decision — surface it, don't do
  it silently.
- Never write to `.env` or `config/` except through the settings UI path.
