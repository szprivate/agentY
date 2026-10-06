"""Rarely used tools join a conversation when it needs them.

Every tool definition goes out with every model call of the orchestrator. Some
are called almost never — in three weeks and 4,900 orchestrator tool calls:
annotate_image 0, bake_hooks_to_canvas 0, create_custom_node 0, start_batch_job
0, upload_file_to_url 0, create_skill 0, spawn_subagent 1, calculator 11 — and
together they are about a fifth of the tool definitions. With the setting
``tool_packs_on_demand`` (the default) they are grouped into packs by purpose:
the prompt lists each pack with its tools, and ``load_tools(pack)`` puts one in
the tool list.

The same rules as MCP servers (src/tools/mcp_on_demand.py), for the same
reasons: which packs a conversation has is read off its own history at the start
of every turn (pooled agents, restarts), and a direct call to a tool whose pack
is not loaded loads the pack and runs — the names are in the prompt, so a model
may well call one straight away.
"""
from __future__ import annotations

import json
import logging

from strands import tool
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.types.tools import ToolContext

logger = logging.getLogger("agentY.tools")

TOOL_NAME = "load_tools"

# pack -> (what it is for, its tools). Only tools the orchestrator rarely calls
# and that nothing in an ordinary turn depends on.
PACKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "batch": ("run one workflow over many inputs unattended (a folder of images, a shot list)",
              ("start_batch_job", "get_batch_status", "stop_batch_job", "list_batch_jobs")),
    "extend": ("save a working procedure as a skill, write a custom ComfyUI node, or start a "
               "sub-agent (only when the user asks for one)",
               ("create_skill", "list_skills", "remove_skill", "create_custom_node",
                "list_generated_nodes", "spawn_subagent")),
    "annotate": ("circle, box or arrow things in an image (an overlay, not a re-render)",
                 ("annotate_image",)),
    "bake": ("bake a hook chain's workflows into subgraphs on the canvas (a hook's `bake` switch)",
             ("bake_hooks_to_canvas",)),
    "upload": ("PUT a local file to a presigned URL — the upload step of an MCP flow (Magnific)",
               ("upload_file_to_url",)),
    "calc": ("exact arithmetic and symbolic maths", ("calculator",)),
}
_PACK_OF = {name: pack for pack, (_why, names) in PACKS.items() for name in names}


def enabled() -> bool:
    """Setting ``tool_packs_on_demand`` (default on)."""
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("tool_packs_on_demand", True))
    except Exception:  # noqa: BLE001
        return True


def name_of(tool_obj) -> str:
    """A tool's name, for a decorated function and for a module-style tool."""
    name = getattr(tool_obj, "tool_name", None)
    if name:
        return str(name)
    spec = getattr(tool_obj, "TOOL_SPEC", None)
    if isinstance(spec, dict) and spec.get("name"):
        return str(spec["name"])
    return str(getattr(tool_obj, "__name__", "")).rsplit(".", 1)[-1]


def split(tools: list) -> tuple[list, dict]:
    """``(always, held)``: the tools every conversation has, and
    ``{pack: [tool, …]}`` for the ones that wait to be loaded."""
    always, held = [], {}
    for t in tools:
        pack = _PACK_OF.get(name_of(t))
        if pack:
            held.setdefault(pack, []).append(t)
        else:
            always.append(t)
    return always, held


def catalogue(held: dict) -> str:
    """The prompt section naming each pack and its tools."""
    lines = [f"- **{pack}** — {PACKS[pack][0]}: " + ", ".join(f"`{name_of(t)}`" for t in tools)
             for pack, tools in held.items()]
    if not lines:
        return ""
    return ("\n\n## More tools — load on demand\n"
            f"These tools are yours but NOT in your list until you call `{TOOL_NAME}(pack)` — "
            "once per conversation; they then stay for the rest of it. Load a pack when a "
            "request needs it (and only then); its tools then appear with their full "
            "parameters.\n" + "\n".join(lines) + "\n")


def packs_in(messages) -> set:
    """The packs a conversation has loaded or called a tool of."""
    found = set()
    for msg in messages or ():
        content = msg.get("content") if isinstance(msg, dict) else None
        for block in content if isinstance(content, list) else ():
            use = block.get("toolUse") if isinstance(block, dict) else None
            if not isinstance(use, dict):
                continue
            name = str(use.get("name") or "")
            if name == TOOL_NAME:
                pack = str((use.get("input") or {}).get("pack") or "").strip().lower()
                if pack in PACKS:
                    found.add(pack)
            elif name in _PACK_OF:
                found.add(_PACK_OF[name])
    return found


def _held(agent) -> dict:
    return getattr(agent, "_tool_packs", None) or {}


def loaded(agent) -> set:
    got = getattr(agent, "_tool_packs_loaded", None)
    if got is None:
        got = set()
        try:
            agent._tool_packs_loaded = got
        except Exception:  # noqa: BLE001
            pass
    return got


def load(agent, pack: str) -> list:
    """Put *pack*'s tools in *agent*'s list. Returns their names."""
    tools = _held(agent).get(pack)
    if not tools:
        raise LookupError(f"no tool pack named {pack!r} (there are: {', '.join(_held(agent)) or 'none'})")
    reg = agent.tool_registry
    missing = [t for t in tools if name_of(t) not in reg.registry]
    if missing:
        reg.process_tools(missing)
    loaded(agent).add(pack)
    return [name_of(t) for t in tools]


def unload(agent, pack: str) -> None:
    reg = agent.tool_registry
    for t in _held(agent).get(pack) or ():
        reg.registry.pop(name_of(t), None)
        reg.dynamic_tools.pop(name_of(t), None)
    loaded(agent).discard(pack)


def sync(agent) -> set:
    """Give *agent* the packs its current conversation uses, and only those."""
    wanted = packs_in(getattr(agent, "messages", None)) & set(_held(agent))
    for pack in loaded(agent) - wanted:
        unload(agent, pack)
    for pack in wanted - loaded(agent):
        try:
            load(agent, pack)
        except Exception as exc:  # noqa: BLE001
            logger.info("tool pack %s not loaded: %s", pack, exc)
    return loaded(agent)


@tool(context=True)
def load_tools(pack: str, tool_context: ToolContext) -> str:
    """Load one of your extra tool packs for this conversation.

    The packs and the tools in each are listed in your prompt under "More tools —
    load on demand". Call this once before the first use of a pack's tools; they
    are then in your list for the rest of the conversation.

    Args:
        pack: The pack's name, e.g. "batch".
    """
    pack = str(pack or "").strip().lower()
    try:
        names = load(tool_context.agent, pack)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})
    return json.dumps({"ok": True, "pack": pack, "tools": names,
                       "message": "In your list now: " + ", ".join(names) + ". Call them directly."})


class ToolPacksHook(HookProvider):
    """Each turn: the conversation's packs. Each call to a held tool that isn't
    loaded: load its pack and run the call."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        registry.add_callback(BeforeInvocationEvent, self._on_turn)
        registry.add_callback(BeforeToolCallEvent, self._on_tool)

    def _on_turn(self, event: BeforeInvocationEvent, **kwargs) -> None:  # noqa: ARG002
        try:
            sync(event.agent)
        except Exception as exc:  # noqa: BLE001 — never break a turn over this
            logger.warning("tool packs: sync failed: %s", exc)

    def _on_tool(self, event: BeforeToolCallEvent, **kwargs) -> None:  # noqa: ARG002
        if event.selected_tool is not None:
            return
        try:
            name = str((event.tool_use or {}).get("name") or "")
            pack = _PACK_OF.get(name)
            if not pack or pack not in _held(event.agent):
                return
            load(event.agent, pack)
            found = event.agent.tool_registry.registry.get(name)
            if found is not None:
                event.selected_tool = found
        except Exception as exc:  # noqa: BLE001 — the unknown-tool message follows
            logger.info("tool packs: could not load for %s: %s", event.tool_use, exc)
