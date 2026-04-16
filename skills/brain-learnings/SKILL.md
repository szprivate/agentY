---
name: brain-learnings
description: Auto-populated learnings from past high-iteration problem-solving sessions. Activate this skill when you notice you are making repeated tool calls to solve the same sub-problem, or when the same error keeps appearing. The entries below document past problems and proven solutions — consult them before retrying a failing pattern.
allowed-tools: 
---

# Brain Self-Learnings

> **This file is automatically maintained by the learnings agent.**
> It is appended after any session where the Brain used more than 5 tool calls.
> Do **not** edit the "Learnings Log" section manually.

## When to activate this skill

Activate and consult this skill when you observe any of the following:
- You have already made **3 or more tool calls** attempting to fix the same issue.
- A tool call fails and you are about to retry with the same approach.
- You are uncertain how to proceed and the task feels repetitive.

Scan the learnings log below for entries that match your current situation.
If a matching entry exists, **apply the documented solution directly** instead of re-discovering it.

---

## Learnings Log

<!-- The learnings agent automatically appends new entries below this line. -->
<!-- Format: date | problem summary | solution (1–2 sentences) -->

2026-04-15 | update_workflow fails validation if reference images not in ComfyUI input directory | Upload images using upload_image before patching workflow inputs to ensure files exist for node validation steps.
2026-04-15 | Template aspect_ratio defaults to 'auto' conflicting with specific user ratio requests | Always patch generator node aspect_ratio input to specific value like '16:9' to override 'auto' default.
2026-04-15 | Nano Banana 2 node requires resolution strings like '2K' instead of raw pixel dimensions | Map requested pixel dimensions to standard resolution strings like '2K' before patching generator resolution input.
