"""A missing or broken MCP server is one friendly line plus a log file, never a traceback."""
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from src.tools import mcp_tools as m


def _cfg(**servers):
    return {"servers": {n: {"enabled": True, "auth": "none", **sc} for n, sc in servers.items()}}


class StartupErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for patcher in (mock.patch.object(m, "_ROOT", Path(self.tmp.name)),
                        mock.patch.dict(m._STATUS, clear=True),
                        mock.patch.dict(m._CLIENTS, clear=True),
                        mock.patch.dict(m._REASONS, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _load(self, cfg):
        with mock.patch.object(m, "load_mcp_config", return_value=cfg), \
                self.assertLogs("agentY.mcp", level="WARNING") as cap:
            tools = m.load_mcp_tools()
        return tools, [r.getMessage() for r in cap.records]

    def test_missing_key_is_skipped_without_starting(self):
        cfg = _cfg(elevenlabs={"transport": "stdio", "command": sys.executable,
                               "env": {"ELEVENLABS_API_KEY": "${ELEVENLABS_API_KEY}"}})
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("ELEVENLABS_API_KEY", None)
            with mock.patch("strands.tools.mcp.MCPClient") as client:
                _, lines = self._load(cfg)
        client.assert_not_called()
        self.assertEqual(lines, ["MCP elevenlabs: not connected — needs ELEVENLABS_API_KEY in .env "
                                 "(details: .logs/mcp_elevenlabs.log)"])
        self.assertIn("ELEVENLABS_API_KEY", m._STATUS["elevenlabs"])
        self.assertTrue((Path(self.tmp.name) / ".logs" / "mcp_elevenlabs.log").is_file())

    def test_preset_required_key_without_placeholder(self):
        cfg = _cfg(elevenlabs={"transport": "stdio", "command": sys.executable})
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("ELEVENLABS_API_KEY", None)
            _, lines = self._load(cfg)
        self.assertIn("needs ELEVENLABS_API_KEY in .env", lines[0])

    def test_command_not_found(self):
        cfg = _cfg(npxy={"transport": "stdio", "command": "npx-does-not-exist-xyz"})
        _, lines = self._load(cfg)
        self.assertIn("command 'npx-does-not-exist-xyz' not found", lines[0])
        cfg = _cfg(x={"transport": "stdio", "command": "uvx"})
        with mock.patch.object(m.shutil, "which", return_value=None):
            _, lines = self._load(cfg)
        self.assertIn("install uv", lines[0])

    def test_missing_folder_env(self):
        cfg = _cfg(bepic_worlds={"transport": "stdio", "command": sys.executable,
                                 "cwd": "${BEPIC_WORLDS_DIR}"})
        with mock.patch.dict("os.environ", {"BEPIC_WORLDS_DIR": ""}), \
                mock.patch.dict(m._BUILTIN_CACHE, {"BEPIC_WORLDS_DIR": ""}), \
                mock.patch.dict(m._BUILTINS, {"BEPIC_WORLDS_DIR": lambda: ""}):
            _, lines = self._load(cfg)
        self.assertEqual(len(lines), 1)
        self.assertIn("BEPIC_WORLDS_DIR", lines[0])
        self.assertNotIn("Traceback", lines[0])

    def test_timeout_reason(self):
        from strands.types.exceptions import MCPClientInitializationError
        cfg = _cfg(slow={"transport": "http", "url": "http://localhost:9/mcp"})
        err = MCPClientInitializationError("background thread did not start in 30 seconds")
        with mock.patch.object(m, "_connect", side_effect=err):
            _, lines = self._load(cfg)
        self.assertIn("no answer in time", lines[0])

    def test_group_wrapped_server_exit_names_the_key(self):
        """The real server's own stderr lands in the log and the reason comes from it."""
        script = Path(self.tmp.name) / "fake_server.py"
        script.write_text(textwrap.dedent("""
            import sys
            print("Traceback (most recent call last):", file=sys.stderr)
            print("ValueError: FAKE_SERVICE_API_KEY environment variable is required", file=sys.stderr)
            sys.exit(1)
        """), encoding="utf-8")
        cfg = _cfg(fake={"transport": "stdio", "command": sys.executable, "args": [str(script)]})
        _, lines = self._load(cfg)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("needs FAKE_SERVICE_API_KEY in .env", lines[0])
        log = (Path(self.tmp.name) / ".logs" / "mcp_fake.log").read_text(encoding="utf-8")
        self.assertIn("FAKE_SERVICE_API_KEY environment variable is required", log)  # server stderr
        self.assertIn("client failed to initialize", log)                            # strands traceback
        self.assertIn("McpError", log)

    def test_stdio_stderr_goes_to_the_errlog(self):
        sc = {"transport": "stdio", "command": sys.executable}
        sink = object()
        with mock.patch("mcp.stdio_client") as stdio:
            m._transport_callable(sc, None, sink)()
        self.assertIs(stdio.call_args.kwargs["errlog"], sink)

    def test_package_missing_reason(self):
        reason = m._friendly_reason("p", {"transport": "stdio", "command": "x"},
                                    RuntimeError("ModuleNotFoundError: No module named 'foo'"))
        self.assertIn("package", reason)

    def test_status_and_test_server_show_the_reason(self):
        cfg = _cfg(elevenlabs={"transport": "stdio", "command": sys.executable,
                               "env": {"ELEVENLABS_API_KEY": "${ELEVENLABS_API_KEY}"}})
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("ELEVENLABS_API_KEY", None)
            self._load(cfg)
            with mock.patch.object(m, "load_mcp_config", return_value=cfg):
                st = m.mcp_status()["elevenlabs"]
            res = m.test_server("elevenlabs", cfg["servers"]["elevenlabs"])
        self.assertEqual(st["reason"], "needs ELEVENLABS_API_KEY in .env")
        self.assertEqual(st["log"], ".logs/mcp_elevenlabs.log")
        self.assertTrue(st["state"].startswith("error:"))
        self.assertEqual(res["error"], "needs ELEVENLABS_API_KEY in .env")
        # with the key typed into the form it is no longer refused up front
        with mock.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "k"}):
            m._check_startable("elevenlabs", cfg["servers"]["elevenlabs"])


if __name__ == "__main__":
    unittest.main()
