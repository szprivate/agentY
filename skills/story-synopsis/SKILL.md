---
name: story-synopsis
description: Write a very short synopsis (a logline of a few words up to ~2 short sentences) capturing a storyline's premise. Activate whenever the user wants a new story idea, logline, premise, or storyline seed.
allowed-tools:
---

# Story Synopsis

Produce a **very short** synopsis: a logline. A few words up to two short sentences — never a full story.

This is a pure text-writing task: load this skill and write the synopsis yourself
(you the orchestrator, or a spawned writer subagent). No tools, no generation.

## What a synopsis captures
- The premise in its most compressed form: **who** the protagonist is, **what** they want or face, and the **central tension or hook**.
- Genre/tone flavour, conveyed through word choice rather than stated labels.
- No scene-by-scene detail, no resolution spelled out — just the seed of a story.

## How to write it
- Lead with the synopsis itself — no preamble like "Here's a story idea".
- Honour any user constraints: genre, tone, setting, characters, era, length.
- When the user gives little direction, make confident, specific creative choices (a concrete protagonist and situation beat a vague one).
- Favour concrete nouns and strong verbs over abstraction.
- Keep it self-contained and evocative.

## Length
- Default: a single logline sentence.
- Hard ceiling: ~2 short sentences. If you're writing more, you're in scene-description territory (the `story-scene` skill), not synopsis.

## After the synopsis
Optionally add one short line offering the next step, e.g. "Want me to expand this into scene descriptions?" — this points toward the `story-scene` skill.

## Examples (shape, not content to copy)
- "A retired lighthouse keeper discovers the light has been guiding something out of the sea — and it has finally arrived."
- "Two rival street magicians in 1920s Cairo must team up when their tricks start coming true."

## Scope
Stay textual — this skill only writes the premise. If the user then wants images or
video, that's a separate step: generate it via the normal generation contract
(`prepare_workflow` → `signal_workflow_ready`) or a spawned generation subagent, not here.
