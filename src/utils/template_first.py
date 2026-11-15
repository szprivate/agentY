"""Templates come before models: a turn's first lookup waits for ``prepare_workflow``.

Seen on Sebastian's Mac, 2026-11-15, asked twice for "an LTX 2.5 SDR-to-HDR
workflow": both turns began with ``find_local_models("ltx-2.5")``, then
``load_tools("models")`` and two or three ``hf_search`` calls, then
``find_workflows`` / ``inspect_workflow_file`` on the templates it found by hand,
then a ``web_search`` - eight to ten calls before (or instead of) the one call
that does the job. The library had two templates for exactly that request
(``ltx2_5_vfx_sdr_to_hdr`` and ``api_ltx_video_sdr2hdr``), and ``prepare_workflow``
picks one, puts the user's files into its loaders, sets its save nodes, and says
which model files are missing. The prompt says "your FIRST step is
prepare_workflow"; a model that wants to know whether the model is there asks
that first anyway.

So the redirect sits where that decision arrives. Until ``prepare_workflow`` has
been called in a turn, a model / template / web lookup is handed back - not run -
with the call to make instead. A first version let the same call through when it
was sent again, and the model on the Mac simply sent it again (2026-11-15), so a
lookup now waits for ``prepare_workflow`` every time, up to :data:`MAX_HOLDS`
per turn (then they run, so a turn can never be locked out). A question that
really is about models ("which LTX files do I have?") is told apart by the user's
own words and never held.

Not armed at all when the turn plainly asks about models or downloads, when it is
a hook-pipeline or open-graph turn (the graph is there; nothing is built), or when
the agent has no ``prepare_workflow``.
"""

from __future__ import annotations

import re

# The calls that look for a model, a template or a recipe by hand.
LOOKUP_TOOLS = frozenset({
    "find_local_models", "check_model", "download_hf_model",
    "hf_search", "hf_repo", "hf_file", "find_hf_file",
    "find_workflows", "inspect_workflow_file",
    "get_workflow_catalog", "get_workflow_template", "list_workflow_recipes", "get_workflow_recipe",
})
# Tool packs whose only job is such a lookup.
LOOKUP_PACKS = frozenset({"models", "inspect"})
# A web search counts when it is looking for a workflow.
_WEB_TOOLS = frozenset({"web_search"})
_WORKFLOW_WORDS = re.compile(r"workflow|comfy ?ui|template|\bnodes?\b|custom node", re.I)

# The turn asks about models / files themselves: lookups are the job.
_MODEL_QUESTION = re.compile(
    r"\b(?:which|what)\s+(?:\w+\s+){0,2}(?:models?|checkpoints?|loras?|files?|weights)\b"
    r"|\binstalled\b|\bdownload|\bhugging\s?face\b|\bhf\b|\bdo i have\b|\bis there an?\b.*\bmodel\b"
    r"|\bmodel (?:file|folder)s?\b|\bfind (?:the |a )?model\b",
    re.I)
# Turns that work on a graph that is already there.
_GRAPH_TURN = re.compile(r"\[CANVAS HOOKS|\[OPEN GRAPH MODE\]")

GATE_TOOL = "prepare_workflow"
MAX_HOLDS = 4


def _user_text(messages) -> str:
    """The text of the last user message (the turn's request and its pinned blocks)."""
    for msg in reversed(list(messages or [])):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        parts = [b.get("text") for b in msg.get("content") or []
                 if isinstance(b, dict) and isinstance(b.get("text"), str)]
        if parts:
            return "\n".join(parts)
        # a message of tool results only is not the request; keep looking
    return ""


def armed_for(user_text: str) -> bool:
    """Should this turn's first lookup wait for ``prepare_workflow``?"""
    text = str(user_text or "")
    if not text.strip():
        return False
    if _GRAPH_TURN.search(text):
        return False
    # Only the user's own words decide "this is a question about models": the
    # pinned blocks above them are the runtime's.
    own = text.rsplit("\n\n", 1)[-1] if "\n\n" in text else text
    return not _MODEL_QUESTION.search(own)


def is_lookup(tool_name: str, tool_input) -> bool:
    """Does this call look for a model, template or workflow by hand?"""
    name = str(tool_name or "")
    inp = tool_input if isinstance(tool_input, dict) else {}
    if name in LOOKUP_TOOLS:
        return True
    if name == "load_tools":
        return str(inp.get("pack") or "").strip().lower() in LOOKUP_PACKS
    if name in _WEB_TOOLS:
        return bool(_WORKFLOW_WORDS.search(str(inp.get("query") or "")))
    return False


def message(tool_name: str) -> str:
    return (
        f"NOT RUN - nothing was looked up. Templates come before models: call "
        f"prepare_workflow(request, staged_inputs) first. It searches every workflow template "
        f"(the custom ones, the official ComfyUI ones and the jobs already solved on this machine), "
        f"picks the one that fits, puts the user's files into its loader nodes and its outputs into "
        f"the save nodes, and reports any model file that is missing - so you look models up only "
        f"for what it names, and build from scratch only when it returns build_new or the user asked "
        f"for a workflow built from scratch. Do NOT send {tool_name} again first: every model, "
        f"template or workflow lookup waits until prepare_workflow has run in this turn. "
        f"Your next call is prepare_workflow.")


class TemplateFirstHook:
    """Hand back the model/template lookups made before ``prepare_workflow`` in a turn."""

    def __init__(self) -> None:
        self._state: dict = {}

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent
        registry.add_callback(BeforeInvocationEvent, self._on_turn)
        registry.add_callback(BeforeToolCallEvent, self._on_tool)

    def _on_turn(self, event, **kwargs) -> None:  # noqa: ARG002
        agent = getattr(event, "agent", None)
        try:
            text = _user_text(getattr(event, "messages", None) or getattr(agent, "messages", None))
        except Exception:  # noqa: BLE001
            text = ""
        self._state[id(agent)] = {"armed": armed_for(text), "holds": 0}

    def _on_tool(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            if getattr(event, "cancel_tool", False):
                return
            st = self._state.get(id(event.agent))
            if not st or not st.get("armed"):
                return
            use = event.tool_use or {}
            name = str(use.get("name") or "")
            if name == GATE_TOOL:
                st["armed"] = False        # the templates have been asked: lookups are follow-up work
                return
            if GATE_TOOL not in getattr(event.agent.tool_registry, "registry", {}):
                st["armed"] = False        # nothing to send it to
                return
            if not is_lookup(name, use.get("input")):
                return
            st["holds"] = st.get("holds", 0) + 1
            if st["holds"] >= MAX_HOLDS:
                st["armed"] = False        # never lock a turn out
            event.cancel_tool = message(name)
        except Exception:  # noqa: BLE001 - never cost a call
            pass
