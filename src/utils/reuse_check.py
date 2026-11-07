"""Before a run: say so when the pass it is about to make already exists.

A depth pass, a mask or a pose pass of a 97-frame plate costs minutes of GPU,
and the same plate is often worked on again: "new background" today, "new
character, same shot" tomorrow, in another conversation. Every output now
records what it was made from (``agenty_core.utils.dependencies``), so before a
graph runs this checks each of its outputs: is a pass of the same kind, from the
same file, with the same settings (frames, model, mask subject) already on disk?

If so, the run is held ONCE, with a message naming the file that exists. The
agent then either wires that file in instead, or calls the same tool again
unchanged, and that second call runs: the user may want a fresh pass, and this
never decides that for them. Renders are never held - a new seed is a new picture.

Deterministic, no model call: a dictionary lookup in sidecars already on disk.
``[dependencies] reuse_check = false`` (or ``AGENTY_REUSE_CHECK=0``) switches it off.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("agentY.reuse_check")

#: Tools that run a workflow file, and the input that names it.
RUN_TOOLS = {"signal_workflow_ready": "workflow_path", "run_workflow_now": "workflow_path",
             "run_variants": "workflow_path", "submit_prompt": "workflow_path"}


def enabled() -> bool:
    raw = os.environ.get("AGENTY_REUSE_CHECK", "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes", "on")
    try:
        from src.utils.settings import load_settings
        return bool((load_settings().get("dependencies") or {}).get("reuse_check", True))
    except Exception:  # noqa: BLE001
        return True


def _api_graph(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    if isinstance(doc, dict) and isinstance(doc.get("prompt"), dict):
        doc = doc["prompt"]
    if not isinstance(doc, dict) or isinstance(doc.get("nodes"), list):
        return {}          # a UI-format file: nothing to compare without converting it
    return doc


def _age(created: str) -> str:
    import time
    try:
        t = time.mktime(time.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return ""
    s = max(0, time.time() - t)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} days ago"


def _frames(params: dict) -> str:
    cap = params.get("frame_load_cap")
    skip = params.get("skip_first_frames") or 0
    if isinstance(cap, int) and cap > 0:
        start = int(skip) if isinstance(skip, int) else 0
        return f" frames {start}-{start + cap - 1}"
    return ""


def message(hits: list[dict]) -> str:
    """The note the agent reads when a run is held."""
    lines = []
    for h in hits:
        src = os.path.basename(h["source"])
        when = _age(h.get("created", ""))
        lines.append(f"- {h['kind']} of {src}{_frames(h.get('params') or {})} already exists: "
                     f"{h['existing'][0]}" + (f" ({when})" if when else ""))
    return ("Not run yet - this graph makes passes that are already on disk, from the same file with "
            "the same settings:\n" + "\n".join(lines) +
            "\nUse the existing file instead (load it where the pass would go, or skip that branch), "
            "and tell the user you reused it. If a fresh pass is wanted (the user asked for one, or the "
            "settings differ in a way that matters), call the same tool again unchanged and it runs.")


def check(tool_name: str, tool_input: dict) -> list[dict]:
    """The existing passes a run of *tool_name* would make again ([] = nothing to say)."""
    key = RUN_TOOLS.get(str(tool_name or ""))
    if not key:
        return []
    path = str((tool_input or {}).get(key) or "")
    if not path or not os.path.isfile(path):
        return []
    graph = _api_graph(path)
    if not graph:
        return []
    from agenty_core.utils import dependencies
    return dependencies.existing_for_graph(graph)


try:
    from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry

    class ReuseCheckHook(HookProvider):
        """Hold a run once when the pass it makes already exists (see the module docstring)."""

        def __init__(self) -> None:
            self._held: set = set()

        def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
            registry.add_callback(BeforeInvocationEvent, self._on_start)
            registry.add_callback(BeforeToolCallEvent, self._on_before)

        def _on_start(self, event=None, **kwargs) -> None:  # noqa: ANN001, ARG002
            self._held = set()

        def _on_before(self, event: BeforeToolCallEvent, **kwargs) -> None:  # noqa: ARG002
            try:
                use = getattr(event, "tool_use", None) or {}
                name = str(use.get("name") or "")
                if name not in RUN_TOOLS or not enabled():
                    return
                inp = use.get("input") or {}
                key = (name, json.dumps(inp, sort_keys=True, default=str))
                if key in self._held:
                    return                        # asked again unchanged: the user's call, run it
                hits = check(name, inp)
                if not hits:
                    return
                self._held.add(key)
                event.cancel_tool = message(hits)
            except Exception as exc:  # noqa: BLE001 - never cost a run
                logger.debug("reuse check skipped: %s", exc)
except Exception:  # noqa: BLE001
    ReuseCheckHook = None  # type: ignore[assignment]
