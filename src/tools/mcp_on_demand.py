"""MCP tools on demand: a server's tools join a conversation when it needs them.

Every MCP tool definition goes out with every model call of the orchestrator.
Magnific and Blender alone are 224 tools — about 86k tokens, two thirds of what
each step of a conversation sends — and they are called in about one step in a
hundred. So with the setting ``mcp_tools_on_demand`` (the default) the
orchestrator starts with none of them: its prompt lists each connected server and
its tool names, and ``use_mcp_server(name)`` puts that server's tools in its list.

Which servers a conversation has is read off the conversation itself, at the start
of every turn (:func:`sync`): a server it loaded or called a tool of is loaded, any
other is dropped. Pipelines are pooled — the agent that answers this conversation
answered another one a minute ago — and a conversation outlives a restart, so
nothing is remembered anywhere but in its history.

A model that calls ``<server>__<tool>`` without loading the server first (it saw
the name in the list) gets the server loaded on the spot instead of an unknown
tool: :class:`MCPOnDemandHook` hands the call its tool.
"""
from __future__ import annotations

import logging

from strands import tool
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.types.tools import ToolContext

from src.tools import mcp_tools as _mcp

logger = logging.getLogger("agentY.mcp")

SEP = "__"
TOOL_NAME = "use_mcp_server"
# Tool lists by server, for the client they came from (filled at startup by
# load_mcp_tools): the tool objects are shared by every pooled agent.
_LISTS = _mcp._LISTS


def enabled() -> bool:
    """Setting ``mcp_tools_on_demand`` (default on)."""
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("mcp_tools_on_demand", True))
    except Exception:  # noqa: BLE001
        return True


def _configured() -> set:
    servers = _mcp.load_mcp_config().get("servers") or {}
    return {n for n, sc in servers.items() if isinstance(sc, dict) and sc.get("enabled")}


def server_tools(server: str) -> list:
    """The live tool objects of *server*, connecting it first if it is configured
    but not connected (signed in since the start, say). Raises with the reason."""
    client = _mcp._CLIENTS.get(server)
    if client is None:
        sc = (_mcp.load_mcp_config().get("servers") or {}).get(server)
        if not isinstance(sc, dict):
            raise LookupError(f"no MCP server named {server!r}")
        if not sc.get("enabled"):
            raise LookupError(f"MCP server {server!r} is switched off in agentY Settings ▸ MCP servers")
        try:
            client, tools = _mcp._connect(server, sc, interactive=False)
        except _mcp._AuthRequired:
            _mcp._STATUS[server] = "needs_auth"
            raise LookupError(f"MCP server {server!r} needs a browser sign-in — "
                              "agentY Settings ▸ MCP servers ▸ Authorize…") from None
        _mcp._CLIENTS[server] = client
        _mcp._STATUS[server] = f"connected ({len(tools)})"
        _LISTS[server] = (id(client), tools)
        return tools
    cached = _LISTS.get(server)
    if cached and cached[0] == id(client):
        return cached[1]
    tools = client.list_tools_sync()
    _LISTS[server] = (id(client), tools)
    return tools


def _server_of(name: str, servers: set) -> str | None:
    head, sep, _ = str(name or "").partition(SEP)
    return head if sep and head in servers else None


def servers_in(messages, servers: set | None = None) -> set:
    """The servers a conversation has loaded or called a tool of."""
    servers = _configured() if servers is None else servers
    found = set()
    for msg in messages or ():
        content = msg.get("content") if isinstance(msg, dict) else None
        for block in content if isinstance(content, list) else ():
            use = block.get("toolUse") if isinstance(block, dict) else None
            if not isinstance(use, dict):
                continue
            name = use.get("name")
            if name == TOOL_NAME:
                wanted = str((use.get("input") or {}).get("server") or "").strip()
                if wanted in servers:
                    found.add(wanted)
            else:
                s = _server_of(name, servers)
                if s:
                    found.add(s)
    return found


def loaded(agent) -> set:
    got = getattr(agent, "_mcp_loaded", None)
    if got is None:
        got = set()
        try:
            agent._mcp_loaded = got
        except Exception:  # noqa: BLE001
            pass
    return got


def load(agent, server: str) -> list:
    """Put *server*'s tools in *agent*'s list. Returns their names."""
    tools = server_tools(server)
    reg = agent.tool_registry
    names = []
    for t in tools:
        name = t.tool_name
        if name not in reg.registry:
            reg.register_tool(t)
        names.append(name)
    loaded(agent).add(server)
    return names


def unload(agent, server: str) -> None:
    reg = agent.tool_registry
    prefix = server + SEP
    for table in (reg.registry, reg.dynamic_tools):
        for name in [n for n in table if n.startswith(prefix)]:
            table.pop(name, None)
    loaded(agent).discard(server)


def sync(agent) -> set:
    """Give *agent* the servers its current conversation uses, and only those."""
    servers = _configured()
    wanted = servers_in(getattr(agent, "messages", None), servers)
    for s in loaded(agent) - wanted:
        unload(agent, s)
    for s in wanted - loaded(agent):
        try:
            load(agent, s)
        except Exception as exc:  # noqa: BLE001 — it says why when the agent asks for it
            logger.info("mcp[%s]: not loaded for this conversation: %s", s, exc)
    return loaded(agent)


def catalogue() -> str:
    """The prompt section listing what each connected server offers."""
    lines = []
    for server in sorted(_configured()):
        if server not in _mcp._CLIENTS:
            continue
        try:
            names = _mcp._tool_names(server, server_tools(server))
        except Exception:  # noqa: BLE001
            continue
        lines.append(f"- **{server}** ({len(names)} tools): " + ", ".join(names))
    if not lines:
        return ""
    return (
        "\n\n## MCP servers — tools load on demand\n"
        f"These servers are connected, but their tools are NOT in your list until you "
        f"call `{TOOL_NAME}(server)` — once per conversation; they then stay for the "
        "rest of it. Load a server as soon as a request needs it (and only then); "
        f"after loading, its tools are named `<server>{SEP}<tool>` with their full "
        "parameters. What each one offers:\n" + "\n".join(lines) + "\n"
        "A server missing here may still be configured but not connected: "
        "`list_mcp_servers` says why.\n")


@tool(context=True)
def use_mcp_server(server: str, tool_context: ToolContext) -> str:
    """Load an external MCP server's tools for this conversation.

    The servers and the tools each offers are listed in your prompt under "MCP
    servers — tools load on demand". Call this once before the first use of a
    server's tools; they are then in your list, as ``<server>__<tool>``, for the
    rest of the conversation. ``list_mcp_servers`` tells you about servers that
    are configured but not connected.

    Args:
        server: The server's name, e.g. "magnific".
    """
    import json
    server = str(server or "").strip()
    agent = tool_context.agent
    try:
        names = load(agent, server)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc),
                           "what_to_do": "Call list_mcp_servers to see what is set up."})
    return json.dumps({"ok": True, "server": server, "tools": len(names),
                       "message": f"{len(names)} tool(s) of {server} are in your list now, "
                                  f"named {server}{SEP}<tool>. Call them directly."})


class MCPOnDemandHook(HookProvider):
    """Each turn: the conversation's servers. Each call to a server's tool that
    isn't loaded: load it and run the call."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        registry.add_callback(BeforeInvocationEvent, self._on_turn)
        registry.add_callback(BeforeToolCallEvent, self._on_tool)

    def _on_turn(self, event: BeforeInvocationEvent, **kwargs) -> None:  # noqa: ARG002
        try:
            sync(event.agent)
        except Exception as exc:  # noqa: BLE001 — never break a turn over this
            logger.warning("mcp on demand: sync failed: %s", exc)

    def _on_tool(self, event: BeforeToolCallEvent, **kwargs) -> None:  # noqa: ARG002
        if event.selected_tool is not None:
            return
        try:
            name = str((event.tool_use or {}).get("name") or "")
            server = _server_of(name, _configured())
            if not server:
                return
            load(event.agent, server)
            found = event.agent.tool_registry.registry.get(name)
            if found is not None:
                event.selected_tool = found
        except Exception as exc:  # noqa: BLE001 — the unknown-tool message follows
            logger.info("mcp on demand: could not load for %s: %s", event.tool_use, exc)
