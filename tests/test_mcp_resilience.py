"""MCP call robustness: dead-session reconnect, timeout, readable errors, guards."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from src.tools import mcp_resilience as r


class ClosedResourceError(Exception):
    """Empty-text exception, like anyio's."""


class FakeClient:
    """Mimics the bits of strands' MCPClient that harden() touches."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)   # per attempt: dead | timeout | hang | ok
        self.active = True
        self.starts = 0
        self.stops = 0
        self.calls = 0
        self.fail_start = False

    def _is_session_active(self):
        return self.active

    def stop(self, *a):
        self.stops += 1
        self.active = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("spawn failed")
        self.starts += 1
        self.active = True

    def list_tools_sync(self):
        if not self.active:
            raise RuntimeError("not running")
        return [object(), object()]

    def _handle_tool_execution_error(self, tool_use_id, exception):
        return {"status": "error", "toolUseId": tool_use_id,
                "content": [{"text": f"Tool execution failed: {exception}"}]}

    def _next(self):
        self.calls += 1
        b = self.behaviours.pop(0)
        if b == "dead":
            return b, ClosedResourceError()
        if b == "timeout":
            return b, TimeoutError("Timed out while waiting for response")
        return b, None

    async def call_tool_async(self, tool_use_id, name, arguments=None,
                              read_timeout_seconds=None, meta=None):
        b, exc = self._next()
        if b == "hang":
            await asyncio.sleep(60)
        if exc:
            return self._handle_tool_execution_error(tool_use_id, exc)
        return {"status": "success", "toolUseId": tool_use_id, "content": [{"text": "done"}]}

    def call_tool_sync(self, tool_use_id, name, arguments=None,
                       read_timeout_seconds=None, meta=None):
        b, exc = self._next()
        if exc:
            return self._handle_tool_execution_error(tool_use_id, exc)
        return {"status": "success", "toolUseId": tool_use_id, "content": [{"text": "done"}]}


def make(behaviours, sc=None):
    client = FakeClient(behaviours)
    states = []
    r.harden("srv", client, lambda: sc or {}, states.append)
    return client, states


def run(coro):
    return asyncio.run(coro)


class CalledLikeStrands(unittest.TestCase):
    """strands' MCPAgentTool calls with keywords (tool_use_id=, name=, arguments=, read_timeout_seconds=)."""

    def test_keyword_call_async(self):
        c, _ = make(["ok"])
        res = run(c.call_tool_async(tool_use_id="t1", name="x", arguments={}, read_timeout_seconds=None))
        self.assertEqual(res["status"], "success")

    def test_keyword_call_sync(self):
        c, _ = make(["ok"])
        self.assertEqual(c.call_tool_sync(tool_use_id="t1", name="x", arguments={})["status"], "success")


class DeadSession(unittest.TestCase):
    def test_restart_and_retry_async(self):
        c, states = make(["dead", "ok"])
        res = run(c.call_tool_async("t1", "render_still", {}))
        self.assertEqual(res["status"], "success")
        self.assertEqual((c.stops, c.starts, c.calls), (1, 1, 2))
        self.assertEqual(states[-1], "connected (2)")

    def test_restart_and_retry_sync(self):
        c, _ = make(["dead", "ok"])
        self.assertEqual(c.call_tool_sync("t1", "x", {})["status"], "success")
        self.assertEqual(c.starts, 1)

    def test_retries_only_once(self):
        c, _ = make(["dead", "dead"])
        res = run(c.call_tool_async("t1", "x", {}))
        self.assertEqual(res["status"], "error")
        self.assertIn("ClosedResourceError", res["content"][0]["text"])
        self.assertEqual(c.calls, 2)

    def test_inactive_session_restarts_before_call(self):
        c, _ = make(["ok"])
        c.active = False
        self.assertEqual(run(c.call_tool_async("t", "x", {}))["status"], "success")
        self.assertEqual(c.starts, 1)

    def test_restart_failure_reports_and_flags_status(self):
        c, states = make(["dead"])
        c.fail_start = True
        res = run(c.call_tool_async("t", "x", {}))
        text = res["content"][0]["text"]
        self.assertIn("could not be restarted", text)
        self.assertIn("spawn failed", text)
        self.assertTrue(states[-1].startswith("error:"))

    def test_empty_exception_text_is_readable(self):
        c, _ = make([])
        res = c._handle_tool_execution_error("t", ClosedResourceError())
        self.assertEqual(res["content"][0]["text"], "Tool execution failed: ClosedResourceError")


class Timeouts(unittest.TestCase):
    def test_hang_is_abandoned_and_session_restarted(self):
        c, _ = make(["hang"], {"timeout_s": 0.05})
        orig = r._GRACE_S
        r._GRACE_S = 0.05
        try:
            res = run(c.call_tool_async("t", "x", {}))
        finally:
            r._GRACE_S = orig
        self.assertEqual(res["status"], "error")
        self.assertIn("did not finish", res["content"][0]["text"])
        self.assertEqual(c.starts, 1)

    def test_protocol_timeout_restarts_without_retry(self):
        c, _ = make(["timeout", "ok"])
        res = run(c.call_tool_async("t", "x", {}))
        self.assertIn("did not finish", res["content"][0]["text"])
        self.assertEqual((c.calls, c.starts), (1, 1))

    def test_timeout_lookup(self):
        sc = {"timeout_s": 100, "tool_timeouts": {"slow_*": 900}}
        self.assertEqual(r._timeout_for(sc, "x"), 100)
        self.assertEqual(r._timeout_for(sc, "slow_a"), 900)
        self.assertEqual(r._timeout_for({}, "x"), r.DEFAULT_TIMEOUT_S)
        self.assertEqual(r._timeout_for({}, "render_still"), 1800)


class Guards(unittest.TestCase):
    def test_init_refused_on_existing_project(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "package.json").write_text("{}")
            c, _ = make(["ok"])
            res = run(c.call_tool_async("t", "init_remotion_project", {"directory": d}))
            self.assertEqual(res["status"], "error")
            self.assertIn("npx remotion render", res["content"][0]["text"])
            self.assertEqual(c.calls, 0)

    def test_init_allowed_on_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            c, _ = make(["ok"])
            res = run(c.call_tool_async("t", "init_remotion_project", {"directory": d}))
            self.assertEqual(res["status"], "success")

    def test_root_tsx_counts_and_relative_uses_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "proj" / "src").mkdir(parents=True)
            (Path(d) / "proj" / "src" / "Root.tsx").write_text("")
            msg = r.check_guards({"cwd": d}, "init_remotion_project", {"path": "proj"})
            self.assertIn("Root.tsx", msg)

    def test_custom_guard_from_config(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "keep.txt").write_text("")
            sc = {"guards": [{"tool": "wipe_*", "path_args": ["dir"],
                              "refuse_if_exists": ["keep.txt"], "message": "no {path}"}]}
            self.assertTrue(r.check_guards(sc, "wipe_all", {"dir": d}).startswith("no "))
            self.assertIsNone(r.check_guards(sc, "other", {"dir": d}))


if __name__ == "__main__":
    unittest.main()
