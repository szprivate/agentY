---
name: assemble-workflow-learnings
description: Consult long-term memory before assembling or patching a ComfyUI workflow, and especially before retrying a failing pattern (repeated tool calls to fix the same assembly sub-problem, or the same validation error recurring). Every prior session's problem→solution lessons live in long-term memory (local FAISS). Call the memory_read tool with a query naming the problem — e.g. "CLIPLoader path validation", "apply_brainbriefing null prompt node", "Kling multi-shot duration limit", "UNETLoader Qwen weight_dtype" — to retrieve proven fixes. After resolving a new recurring failure, persist a one-sentence lesson with memory_write.
allowed-tools: 
---

Workflow-assembly learnings are stored in **long-term memory** (the local FAISS
store) — the single source of truth — not in this file. Keeping every lesson in
one searchable, inspectable, editable place stops the file and the store from
drifting apart (they used to, badly).

- **Before retrying a failing assembly step**, call `memory_read` with a short
  query describing the problem to pull proven solutions from past sessions
  (e.g. a recurring validation error, a loader that rejects a model path, a
  template whose prompt node is null).
- **After you resolve a recurring failure**, call `memory_write` with one concise
  sentence capturing the problem and its fix, so it sticks across future sessions.

A learnings pass also runs automatically after substantial assembly sessions and
adds new lessons to the same store, so what you save here compounds over time.
