"""The common changes go through ``change_graph`` first.

``change_graph`` adds a ControlNet, a hires pass, a FreeU with every wire set -
and was called in six of some thirty attempts that needed it (benchmark of
2026-10-13), 41 calls going to the hand-wiring tool instead, where a small model
then loses a wire. Naming the tool in the prompt, and then in the builder's
result, changed nothing: a model that has decided to add a node reaches for the
tool that adds nodes.

So the redirect sits where that decision arrives. When a hand-wiring call
(``edit_canvas_graph``, ``update_workflow``) ADDS a node that belongs to one of
the common changes, it is handed back once - not run - with the one call that
does the whole change. Once per operation and per invocation: the same call sent
again runs as it is, because a model that wants a different wiring than the
standard one must be able to have it, and ``change_graph`` answering "cannot"
has to leave a way on. A turn that has already called ``change_graph`` for the
operation is not redirected at all: what follows it is finishing work.

Pure where it can be: :func:`operation_for` reads a tool call and says which
operation it amounts to.
"""

from __future__ import annotations

import json

# Node class -> the change_graph operation that adds it, wired.
_CLASS_TO_OPERATION = {
    "FreeU": "patch_model", "FreeU_V2": "patch_model", "LoraLoaderModelOnly": "patch_model",
    "LoraLoader": "lora",
    "ControlNetLoader": "controlnet", "ControlNetApply": "controlnet", "ControlNetApplyAdvanced": "controlnet",
    "VAEEncode": "image_to_image",
    "LatentUpscale": "hires_fix", "LatentUpscaleBy": "hires_fix",
    "UpscaleModelLoader": "model_upscale", "ImageUpscaleWithModel": "model_upscale",
    "PreviewImage": "preview_output",
}
_HAND_TOOLS = ("edit_canvas_graph", "update_workflow")


def _as_list(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except ValueError:
            return []
    if isinstance(value, dict):
        value = [value]
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def added_classes(tool_name: str, tool_input) -> list:
    """The node classes a hand-wiring call would ADD."""
    inp = tool_input if isinstance(tool_input, dict) else {}
    if tool_name == "edit_canvas_graph":
        return [str(o.get("class_type") or "") for o in _as_list(inp.get("ops"))
                if str(o.get("op") or "").lower() == "add" and o.get("class_type")]
    if tool_name == "update_workflow":
        return [str(n.get("class_type") or "") for n in _as_list(inp.get("add_nodes")) if n.get("class_type")]
    return []


def operation_for(tool_name: str, tool_input) -> tuple:
    """``(operation, [classes])`` when the call adds nodes of ONE common change,
    else ``("", [])``. Two different changes in one call are left alone: that is
    somebody building something of their own."""
    if tool_name not in _HAND_TOOLS:
        return "", []
    classes = added_classes(tool_name, tool_input)
    found = {_CLASS_TO_OPERATION[c] for c in classes if c in _CLASS_TO_OPERATION}
    if len(found) != 1:
        return "", []
    return next(iter(found)), [c for c in classes if c in _CLASS_TO_OPERATION]


def message(operation: str, classes: list, tool_name: str, tool_input) -> str:
    path = str((tool_input or {}).get("workflow_path") or "").strip() if tool_name == "update_workflow" else ""
    extra = ', params={"class_type": "%s"}' % classes[0] if operation == "patch_model" else ", params={...}"
    where = f', workflow_path="{path}"' if path else ""
    return (f"NOT RUN - nothing was changed. Adding {', '.join(classes)} is the common change "
            f"\"{operation}\", and one call makes all of it, with every wire set:\n"
            f"  change_graph(operation=\"{operation}\"{extra}{where})\n"
            "Call that now. If it answers \"cannot\", or you need a wiring different from the standard "
            "one on purpose, send this same call again - it will run as it is.")


class SpliceFirstHook:
    """Hand back, once, a hand-wiring call that a single ``change_graph`` call covers."""

    def __init__(self) -> None:
        self._seen: dict = {}

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent
        registry.add_callback(BeforeInvocationEvent, self._on_turn)
        registry.add_callback(BeforeToolCallEvent, self._on_tool)

    def _state(self, agent) -> set:
        return self._seen.setdefault(id(agent), set())

    def _on_turn(self, event, **kwargs) -> None:  # noqa: ARG002
        self._seen.pop(id(getattr(event, "agent", None)), None)

    def _on_tool(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            if getattr(event, "cancel_tool", False):
                return
            use = event.tool_use or {}
            name = str(use.get("name") or "")
            done = self._state(event.agent)
            if name == "change_graph":
                done.add(str((use.get("input") or {}).get("operation") or "").strip().lower())
                return
            if "change_graph" not in getattr(event.agent.tool_registry, "registry", {}):
                return                      # nothing to send it to
            operation, classes = operation_for(name, use.get("input"))
            if not operation or operation in done:
                return
            done.add(operation)
            event.cancel_tool = message(operation, classes, name, use.get("input"))
        except Exception:  # noqa: BLE001 - never cost a call
            pass
