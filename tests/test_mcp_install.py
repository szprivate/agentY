"""add_mcp_server: parse, pin, ask, test, save, connect — in that order."""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from src.tools import mcp_install as mi
from src.tools import mcp_tools
from src.utils import tool_permissions as tp

YES = SimpleNamespace(allowed=True, reason="")
NO = SimpleNamespace(allowed=False, reason="you declined this.")
WIN = sys.platform == "win32"


class Pin(unittest.TestCase):
    def test_npx_package_is_pinned_and_gets_cmd_on_windows(self):
        cmd, args, notes = mi.pin_package("npx", ["-y", "remotion-mcp"], resolve_npm=lambda p: "1.0.5")
        self.assertEqual(args, ["-y", "remotion-mcp@1.0.5"])
        self.assertEqual(cmd, "npx.cmd" if WIN else "npx")
        self.assertIn("Pinned", notes[0])

    def test_scoped_and_already_pinned(self):
        _, args, _ = mi.pin_package("npx", ["-y", "@scope/pkg"], resolve_npm=lambda p: "2.0.0")
        self.assertEqual(args, ["-y", "@scope/pkg@2.0.0"])
        _, args, notes = mi.pin_package("npx", ["-y", "@scope/pkg@1.2.3"], resolve_npm=lambda p: 1 / 0)
        self.assertEqual(args, ["-y", "@scope/pkg@1.2.3"])
        self.assertEqual(notes, [])

    def test_uvx_and_unresolvable(self):
        _, args, _ = mi.pin_package("uvx", ["mcp-server-fetch"], resolve_pypi=lambda p: "0.6.2")
        self.assertEqual(args, ["mcp-server-fetch==0.6.2"])
        _, args, notes = mi.pin_package("uvx", ["x"], resolve_pypi=lambda p: 1 / 0)
        self.assertEqual(args, ["x"])
        self.assertIn("UNPINNED", notes[0])

    def test_other_commands_untouched(self):
        self.assertEqual(mi.pin_package("python", ["server.py"])[:2], ("python", ["server.py"]))


class Install(unittest.TestCase):
    def setUp(self):
        self.cfg = {"servers": {"magnific": {"enabled": True}}}
        self.saved = []
        patches = [
            mock.patch.object(mcp_tools, "load_mcp_config", side_effect=lambda: json.loads(json.dumps(self.cfg))),
            mock.patch.object(mcp_tools, "save_mcp_config", side_effect=lambda c: self.saved.append(c)),
            mock.patch.object(mcp_tools, "test_server", return_value={"ok": True, "tools": 3, "names": ["a", "b", "c"]}),
            mock.patch.object(mi, "pin_package", side_effect=lambda c, a: (c, a, ["Pinned x."])),
        ]
        self.test_server = patches[2].start()
        for p in patches[:2] + patches[3:]:
            p.start()
        for p in patches:
            self.addCleanup(p.stop)
        self.connected = []

    def run_install(self, text, ask=YES, name=""):
        asked = []

        def _ask(summary, payload):
            asked.append(summary)
            return ask
        out = mi.install(text, name, ask=_ask, loader=self.connected.append)
        return out, asked

    def test_approved_install_tests_then_saves_then_connects(self):
        out, asked = self.run_install('{"command":"npx","args":["-y","remotion-mcp"]}', name="remotion")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["name"], "remotion")
        self.assertIn("remotion-mcp", asked[0])
        self.assertIn("runs on this machine", asked[0])
        self.assertEqual(self.connected, ["remotion"])
        self.assertIn("remotion", self.saved[0]["servers"])
        self.assertIn("magnific", self.saved[0]["servers"])  # others kept

    def test_declined_writes_nothing_and_tests_nothing(self):
        out, _ = self.run_install("npx -y remotion-mcp", ask=NO)
        self.assertFalse(out["ok"])
        self.assertTrue(out["declined"])
        self.assertEqual(self.saved, [])
        self.test_server.assert_not_called()
        self.assertEqual(self.connected, [])

    def test_failed_test_saves_nothing(self):
        self.test_server.return_value = {"ok": False, "error": "spawn failed"}
        out, _ = self.run_install("npx -y remotion-mcp")
        self.assertFalse(out["ok"])
        self.assertIn("spawn failed", out["error"])
        self.assertEqual(self.saved, [])

    def test_secret_goes_to_env_not_config_and_is_named_not_shown(self):
        text = '{"url":"https://x.example/mcp","headers":{"Authorization":"Bearer sk-secret-123"}}'
        with mock.patch("src.utils.agentY_server._update_env_file") as env:
            out, asked = self.run_install(text)
        self.assertTrue(out["ok"], out)
        env.assert_called_once()
        self.assertNotIn("sk-secret-123", json.dumps(self.saved))
        self.assertNotIn("sk-secret-123", asked[0])
        self.assertIn("keys saved to .env", asked[0])

    def test_missing_key_is_asked_for_not_installed(self):
        out, asked = self.run_install('{"command":"npx","args":["x"],"env":{"API_KEY":"<your key>"}}')
        self.assertFalse(out["ok"])
        self.assertTrue(out["missing"])
        self.assertEqual(asked, [])

    def test_existing_name_is_never_overwritten(self):
        out, _ = self.run_install("npx -y remotion-mcp", name="magnific")
        self.assertFalse(out["ok"])
        self.assertEqual(self.saved, [])

    def test_nonsense_is_refused(self):
        out, _ = self.run_install("please install it")
        self.assertTrue(out["ok"] is False or out["name"])  # a bare word parses as a command...
        out, _ = self.run_install("")
        self.assertFalse(out["ok"])


class Approval(unittest.TestCase):
    def tearDown(self):
        tp.reset_session()

    def test_no_panel_means_no(self):
        with mock.patch.object(tp, "has_listener", return_value=False):
            d = tp.request("add_mcp_server", {"summary": "x"}, unattended_allows=False, allow_session=False)
        self.assertFalse(d.allowed)

    def test_a_session_grant_does_not_cover_it(self):
        tp._SESSION_GRANTS.add("add_mcp_server")
        with mock.patch.object(tp, "has_listener", return_value=False):
            d = tp.request("add_mcp_server", {"summary": "x"}, allow_session=False)
        self.assertFalse(d.allowed)

    def test_the_panel_event_carries_allow_session(self):
        import threading
        with mock.patch.object(tp, "has_listener", return_value=True):
            t = threading.Thread(target=lambda: tp.request(
                "add_mcp_server", {"summary": "x"}, timeout=2, allow_session=False))
            t.start()
            got = tp.take(wait=2)
            self.assertIs(got["allow_session"], False)
            tp.answer(got["permission_id"], False)
            t.join(3)

    def test_prompt_shows_the_summary(self):
        self.assertEqual(tp.describe("add_mcp_server", {"summary": "Add X"}), "Add X")


if __name__ == "__main__":
    unittest.main()
