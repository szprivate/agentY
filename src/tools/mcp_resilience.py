"""Keep MCP tool calls from wedging the agent: timeout, reconnect, guards.

``harden(name, client, ...)`` wraps one strands ``MCPClient`` *instance* so every
tool call made through it (the orchestrator's ``MCPAgentTool``s and
``call_mcp_tool`` alike) gets:

* **a timeout** — ``timeout_s`` per server in ``config/mcp.json`` (default 600),
  ``tool_timeouts`` maps a tool-name glob to its own limit (render tools run
  longer). A timed-out call returns a clear error and the session is restarted,
  because the server is still busy with the abandoned request.
* **a reconnect** — a server process that died (killed, crashed) leaves a session
  whose every call fails with an exception that has no text. That is detected,
  the stdio process is restarted once and the call retried. If the restart
  fails the error says so and the server's status stops saying "connected".
* **readable errors** — the exception type is named when ``str()`` is empty.
* **guards** — small data rules (``guards`` in the server's config, plus
  ``_BUILTIN_GUARDS``) that refuse a call before it reaches the server, e.g.
  ``init_remotion_project`` on a folder that already holds a project.
"""
from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import logging
import os
import threading
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger("agentY.mcp")

DEFAULT_TIMEOUT_S = 600
# Render tools run far longer than a normal call; a server can override.
_DEFAULT_TOOL_TIMEOUTS = {"render_*": 1800}
# Slack on top of the protocol-level read timeout before the call is abandoned.
_GRACE_S = 30

# Rules every server gets (a server's own ``guards`` are added to these). A rule
# refuses a call of ``tool`` (name glob, the server's own name for it) when a path
# argument (first of ``path_args`` that is given) is a folder holding any of the
# ``refuse_if_exists`` files. A relative path is taken from the server's ``cwd``.
_BUILTIN_GUARDS: list[dict] = [
    {
        "tool": "init_remotion_project",
        "path_args": ["directory", "dir", "path", "projectPath", "project_path",
                      "outputDir", "output_dir", "targetDir", "target_dir"],
        "refuse_if_exists": ["package.json", "src/Root.tsx"],
        "message": ("Refused: {path} already holds a project ({found}). "
                    "init_remotion_project would run npm install and overwrite it. "
                    "Use that project's own local CLI instead, from inside the folder: "
                    "`npx remotion render` / `npx remotion still` (and `npx remotion studio`)."),
    },
]

_CLOSED_NAMES = {"ClosedResourceError", "BrokenResourceError", "EndOfStream",
                 "BrokenPipeError", "ConnectionResetError", "ConnectionAbortedError",
                 "MCPClientInitializationError", "CancelledError"}

# the exception of the current tool call, set by the patched error handler
_last_exc: contextvars.ContextVar = contextvars.ContextVar("mcp_last_exc", default=None)


def describe(exc: BaseException) -> str:
    """``str(exc)``, or the exception's type when that is empty."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    code = getattr(getattr(exc, "error", None), "code", None)
    return code == 408 or "timed out" in str(exc).lower()


def _is_closed(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, EOFError)):
        return True
    if type(exc).__name__ in _CLOSED_NAMES:
        return True
    low = str(exc).lower()
    return "connection to the mcp server was closed" in low or "session is not running" in low \
        or "not initialized" in low or "connection closed" in low


def _error(tool_use_id: str, text: str) -> dict:
    return {"status": "error", "toolUseId": tool_use_id, "content": [{"text": text}]}


def _timeout_for(sc: dict, tool: str) -> float:
    for pat, secs in {**_DEFAULT_TOOL_TIMEOUTS, **(sc.get("tool_timeouts") or {})}.items():
        if fnmatch.fnmatch(tool, pat):
            return float(secs)
    return float(sc.get("timeout_s") or DEFAULT_TIMEOUT_S)


def check_guards(sc: dict, tool: str, arguments: dict | None) -> str | None:
    """The refusal text when a guard rule blocks this call, else ``None``."""
    args = arguments if isinstance(arguments, dict) else {}
    for rule in _BUILTIN_GUARDS + list(sc.get("guards") or []):
        if not isinstance(rule, dict) or not fnmatch.fnmatch(tool, str(rule.get("tool") or "")):
            continue
        raw = next((args[k] for k in rule.get("path_args") or [] if isinstance(args.get(k), str)
                    and args[k].strip()), None)
        if raw is None:
            continue
        base = Path(os.path.expandvars(str(sc.get("cwd") or "."))).expanduser()
        folder = Path(os.path.expandvars(raw)).expanduser()
        if not folder.is_absolute():
            folder = base / folder
        found = [f for f in rule.get("refuse_if_exists") or [] if (folder / f).exists()]
        if found:
            return str(rule.get("message") or "Refused: {path} already holds {found}.").format(
                path=str(folder), found=", ".join(found), tool=tool)
    return None


def harden(name: str, client, get_config, set_status=None) -> None:
    """Wrap *client* (a strands ``MCPClient``) in place. ``get_config()`` returns the
    server's current config dict; ``set_status(text)`` records connected/error."""
    lock = threading.Lock()
    orig_async = client.call_tool_async
    orig_sync = client.call_tool_sync
    orig_handle = client._handle_tool_execution_error

    def status(text: str) -> None:
        if set_status:
            try:
                set_status(text)
            except Exception:  # noqa: BLE001
                pass

    def restart() -> str | None:
        """Stop and start the session again. ``None`` on success, else the reason."""
        with lock:
            try:
                try:
                    client.stop(None, None, None)
                except Exception as exc:  # noqa: BLE001  stop re-raises the close error
                    logger.debug("mcp[%s]: stop while restarting: %s", name, describe(exc))
                client.start()
                n = len(client.list_tools_sync())
            except Exception as exc:  # noqa: BLE001
                reason = describe(exc)
                status(f"error: session closed, restart failed: {reason}")
                logger.warning("mcp[%s]: restart failed: %s", name, reason)
                return reason
            status(f"connected ({n})")
            logger.info("mcp[%s]: session restarted", name)
            return None

    def handle(tool_use_id, exception):
        holder = _last_exc.get()
        if holder is not None:
            holder.append(exception)
        result = orig_handle(tool_use_id, exception)
        if not str(exception).strip():
            result = _error(tool_use_id, f"Tool execution failed: {describe(exception)}")
        elif _is_timeout(exception):
            result = _error(tool_use_id, f"Tool execution failed: {describe(exception)}")
        return result

    client._handle_tool_execution_error = handle

    def dead_text(reason: str) -> str:
        return (f"MCP server {name!r} is not running and could not be restarted ({reason}). "
                "Its tools are unavailable until it can start (check its command in "
                "Settings > MCP servers).")

    def timeout_text(tool: str, secs: float) -> str:
        return (f"MCP tool {name}/{tool} did not finish within {int(secs)} s and was abandoned; "
                f"the server session was restarted. Retry with a smaller job, or raise "
                f"`timeout_s` / `tool_timeouts` for {name!r} in config/mcp.json.")

    async def call_async(tool_use_id, name_, arguments=None, read_timeout_seconds=None, meta=None):
        sc = get_config() or {}
        refusal = check_guards(sc, name_, arguments)
        if refusal:
            return _error(tool_use_id, refusal)
        secs = read_timeout_seconds.total_seconds() if read_timeout_seconds else _timeout_for(sc, name_)
        for attempt in (0, 1):
            if not client._is_session_active():
                reason = await asyncio.to_thread(restart)
                if reason:
                    return _error(tool_use_id, dead_text(reason))
            holder: list = []
            _last_exc.set(holder)
            try:
                res = await asyncio.wait_for(
                    orig_async(tool_use_id=tool_use_id, name=name_, arguments=arguments,
                               read_timeout_seconds=timedelta(seconds=secs), meta=meta),
                    secs + _GRACE_S)
            except asyncio.TimeoutError:
                await asyncio.to_thread(restart)
                return _error(tool_use_id, timeout_text(name_, secs))
            except Exception as exc:  # noqa: BLE001  e.g. session not running
                holder.append(exc)
                res = _error(tool_use_id, f"Tool execution failed: {describe(exc)}")
            exc = holder[0] if holder else None
            if exc is None:
                return res
            if _is_timeout(exc):
                await asyncio.to_thread(restart)
                return _error(tool_use_id, timeout_text(name_, secs))
            if _is_closed(exc) and attempt == 0:
                logger.warning("mcp[%s]: session dead (%s) — restarting and retrying %s",
                               name, describe(exc), name_)
                reason = await asyncio.to_thread(restart)
                if reason:
                    return _error(tool_use_id, dead_text(reason))
                continue
            return res
        return res  # pragma: no cover

    def call_sync(tool_use_id, name_, arguments=None, read_timeout_seconds=None, meta=None):
        sc = get_config() or {}
        refusal = check_guards(sc, name_, arguments)
        if refusal:
            return _error(tool_use_id, refusal)
        secs = read_timeout_seconds.total_seconds() if read_timeout_seconds else _timeout_for(sc, name_)
        for attempt in (0, 1):
            if not client._is_session_active():
                reason = restart()
                if reason:
                    return _error(tool_use_id, dead_text(reason))
            holder: list = []
            _last_exc.set(holder)
            try:
                res = orig_sync(tool_use_id=tool_use_id, name=name_, arguments=arguments,
                                read_timeout_seconds=timedelta(seconds=secs), meta=meta)
            except Exception as exc:  # noqa: BLE001
                holder.append(exc)
                res = _error(tool_use_id, f"Tool execution failed: {describe(exc)}")
            exc = holder[0] if holder else None
            if exc is None:
                return res
            if _is_timeout(exc):
                restart()
                return _error(tool_use_id, timeout_text(name_, secs))
            if _is_closed(exc) and attempt == 0:
                reason = restart()
                if reason:
                    return _error(tool_use_id, dead_text(reason))
                continue
            return res
        return res  # pragma: no cover

    client.call_tool_async = call_async
    client.call_tool_sync = call_sync
    client._agenty_hardened = True
