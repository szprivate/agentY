"""The MCP preset catalog: the merge (scripts/mcp_presets.py) and the two routes
Settings uses (GET /agentY/mcp/presets, POST /agentY/mcp/presets/install).

    uv run --with pytest python -m pytest -q tests/test_mcp_presets.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import mcp_presets as mp
from src.utils import agentY_server as srv
from src.utils import mcp_preset_install as mpi

HTTP = {"enabled": True, "transport": "http", "url": "https://a.example/mcp", "auth": "header",
        "headers": {"Authorization": "Bearer ${ALPHA_KEY}"}}
LOCAL = {"enabled": True, "transport": "stdio", "command": "npx", "args": ["-y", "pkg@1"], "auth": "none"}
BUNDLE = {"enabled": True, "transport": "stdio", "command": "uv", "args": ["run", "x"], "auth": "none",
          "cwd": "${AGENTY_ROOT}/config/mcp_bundles/bun",
          "bundle": {"name": "Bun", "version": "1", "dir": "config/mcp_bundles/bun", "signed": False}}


def preset(pid, server, keys=(), steps="", default=False):
    return {"id": pid, "name": pid.title(), "category": "Test", "description": "d", "server": server,
            "env_keys": [{"key": k, "label": k, "help_url": "", "optional": False} for k in keys],
            "requires": [], "setup_steps": steps, "default_selected": default}


CATALOG = {"presets": [preset("alpha", HTTP, ["ALPHA_KEY"]), preset("beta", LOCAL),
                       preset("gamma", BUNDLE, steps="install the bundle"),
                       preset("delta", LOCAL, steps="do a thing")]}


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cat = self.tmp / "mcp_presets.json"
        self.cat.write_text(json.dumps(CATALOG), encoding="utf-8")
        self.cfg = self.tmp / "mcp.json"


class Merge(Scratch):
    def test_existing_servers_are_never_dropped_or_reordered(self):
        self.cfg.write_text(json.dumps({"servers": {"mine": {"url": "x"}, "other": {"url": "y"}}}))
        res = mp.apply(["alpha", "beta"], self.cat, self.cfg, root=self.tmp)
        self.assertEqual(res, {"alpha": "added", "beta": "added"})
        self.assertEqual(list(json.loads(self.cfg.read_text())["servers"]), ["mine", "other", "alpha", "beta"])

    def test_a_preset_whose_id_exists_is_skipped_not_overwritten(self):
        self.cfg.write_text(json.dumps({"servers": {"alpha": {"url": "mine"}}}))
        res = mp.apply(["alpha"], self.cat, self.cfg, root=self.tmp)
        self.assertEqual(res, {"alpha": "skipped"})
        self.assertEqual(json.loads(self.cfg.read_text())["servers"]["alpha"], {"url": "mine"})

    def test_placeholders_stay(self):
        mp.apply(["alpha"], self.cat, self.cfg, root=self.tmp)
        sc = json.loads(self.cfg.read_text())["servers"]["alpha"]
        self.assertEqual(sc["headers"]["Authorization"], "Bearer ${ALPHA_KEY}")

    def test_missing_file_is_created_and_unknown_ids_reported(self):
        res = mp.apply(["beta", "nope"], self.cat, self.cfg, root=self.tmp)
        self.assertEqual(res, {"nope": "unknown", "beta": "added"})
        self.assertIn("beta", json.loads(self.cfg.read_text())["servers"])

    def test_nothing_added_writes_nothing(self):
        mp.apply(["nope"], self.cat, self.cfg, root=self.tmp)
        self.assertFalse(self.cfg.exists())

    def test_npx_becomes_npx_cmd_on_windows_only(self):
        p = mp.load_catalog(self.cat)[1]
        self.assertEqual(mp.server_entry(p, "win32")["command"], "npx.cmd")
        self.assertEqual(mp.server_entry(p, "linux")["command"], "npx")

    def test_agenty_root_is_filled_in(self):
        p = mp.load_catalog(self.cat)[2]
        cwd = mp.server_entry(p, "linux", root=Path("/opt/agentY"))["cwd"]
        self.assertEqual(cwd, "/opt/agentY/config/mcp_bundles/bun")

    def test_a_bundle_preset_without_its_bundle_is_not_written(self):
        res = mp.apply(["gamma"], self.cat, self.cfg, root=self.tmp)
        self.assertEqual(res, {"gamma": "needs_bundle"})
        self.assertFalse(self.cfg.exists())
        (self.tmp / "config" / "mcp_bundles" / "bun").mkdir(parents=True)
        self.assertEqual(mp.apply(["gamma"], self.cat, self.cfg, root=self.tmp), {"gamma": "added"})

    def test_any_number_of_presets_and_no_hard_coded_ids(self):
        many = {"presets": [preset(f"p{i}", LOCAL) for i in range(40)]}
        self.cat.write_text(json.dumps(many))
        self.assertEqual(len(mp.load_catalog(self.cat)), 40)

    def test_the_shipped_catalog_is_well_formed(self):
        presets = mp.load_catalog()
        self.assertTrue(presets)
        self.assertEqual(len({p["id"] for p in presets}), len(presets))
        for p in presets:
            self.assertIn(p["server"].get("transport"), ("stdio", "http", "sse"), p["id"])
            for k in p["env_keys"]:   # a key is referenced as ${KEY}, never written out
                self.assertIn("${" + k["key"] + "}", json.dumps(p["server"]), p["id"])


class Routes(Scratch):
    def setUp(self):
        super().setUp()
        self.env, self.turns, self.saved = {}, [], []
        self.servers = {"mine": {"url": "x"}}
        for target, repl in [
            (mp, {"CATALOG": self.cat, "ROOT": self.tmp}),
        ]:
            for name, value in repl.items():
                self.enterContext(mock.patch.object(target, name, value))
        # the route imports these from src.tools.mcp_tools at call time
        from src.tools import mcp_tools as mt
        self.enterContext(mock.patch.object(mt, "load_mcp_config", lambda: {"servers": dict(self.servers)}))
        self.enterContext(mock.patch.object(mt, "save_mcp_config", self._save))
        self.test_result = {"ok": True, "tools": 3, "names": ["a"]}
        self.enterContext(mock.patch.object(mt, "test_server", lambda n, sc, s=None: self.test_result))
        self.enterContext(mock.patch.object(srv, "_update_env_file", lambda u: self.env.update(u)))
        self.enterContext(mock.patch.object(srv, "_api_start_turn", self._turn))
        from tests.route_client import authorised_client
        self.client = authorised_client()

    def _save(self, cfg):
        self.servers = dict(cfg["servers"])
        self.saved.append(cfg)

    def _turn(self, text, paths, tid="", title=""):
        self.turns.append((text, title))
        return "rid1", "thread1"

    def post(self, **body):
        return self.client.post("/agentY/mcp/presets/install", json=body)

    def test_get_lists_the_catalog_with_installed_flags(self):
        self.servers["beta"] = {"url": "z"}
        body = self.client.get("/agentY/mcp/presets").get_json()
        by_id = {p["id"]: p for p in body["presets"]}
        self.assertEqual(set(by_id), {"alpha", "beta", "gamma", "delta"})
        self.assertTrue(by_id["beta"]["installed"])
        self.assertFalse(by_id["alpha"]["installed"])
        self.assertTrue(by_id["gamma"]["needs_bundle"])

    def test_install_saves_key_merges_tests_and_starts_no_turn_when_all_is_well(self):
        r = self.post(id="alpha", secrets={"ALPHA_KEY": "k123"})
        j = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual((j["ok"], j["state"], j["turn_id"]), (True, "added", None))
        self.assertTrue(j["test"]["ok"])
        self.assertEqual(self.env, {"ALPHA_KEY": "k123"})
        self.assertEqual(list(self.servers), ["mine", "alpha"])      # mine kept
        self.assertEqual(self.servers["alpha"]["headers"]["Authorization"], "Bearer ${ALPHA_KEY}")
        self.assertNotIn("k123", json.dumps(self.saved))             # the key is in .env only
        self.assertEqual(self.turns, [])

    def test_a_missing_required_key_is_refused_before_anything_is_written(self):
        r = self.post(id="alpha", secrets={})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.saved, [])

    def test_a_key_the_preset_does_not_have_is_refused(self):
        r = self.post(id="alpha", secrets={"ALPHA_KEY": "k", "PATH": "x"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.env, {})

    def test_unknown_and_already_configured(self):
        self.assertEqual(self.post(id="nope").status_code, 404)
        self.assertEqual(self.post(id="mine").status_code, 404)
        self.servers["beta"] = {}
        self.assertEqual(self.post(id="beta").status_code, 409)

    def test_setup_steps_start_an_agent_turn(self):
        j = self.post(id="delta").get_json()
        self.assertEqual((j["turn_id"], j["conversation"]), ("rid1", "thread1"))
        text, title = self.turns[0]
        self.assertIn("do a thing", text)
        self.assertIn("Delta", text)
        self.assertIn("MCP preset", title)

    def test_a_failed_test_starts_a_turn_with_the_error(self):
        self.test_result = {"ok": False, "error": "spawn uvx ENOENT", "needs_auth": False}
        j = self.post(id="beta").get_json()
        self.assertEqual(j["turn_id"], "rid1")
        self.assertIn("spawn uvx ENOENT", self.turns[0][0])

    def test_a_missing_sign_in_is_the_users_not_the_agents(self):
        self.test_result = {"ok": False, "error": "sign in", "needs_auth": True}
        j = self.post(id="beta").get_json()
        self.assertIsNone(j["turn_id"])
        self.assertFalse(j["test"]["ok"])

    def test_a_missing_bundle_writes_nothing_and_hands_over_to_the_agent(self):
        j = self.post(id="gamma").get_json()
        self.assertEqual((j["state"], j["turn_id"], j["test"]), ("needs_bundle", "rid1", None))
        self.assertEqual(self.saved, [])
        self.assertIn("install the bundle", self.turns[0][0])

    def test_a_busy_conversation_is_reported_not_raised(self):
        def busy(*a, **k):
            raise ValueError("conversation is already running a turn")
        with mock.patch.object(srv, "_api_start_turn", busy):
            j = self.post(id="delta").get_json()
        self.assertTrue(j["ok"])
        self.assertIsNone(j["turn_id"])
        self.assertIn("already running", j["turn_error"])

    def test_the_message_is_built_from_the_partial(self):
        msg = mpi.turn_message(mp.find("delta"), "entry written", {"ok": False, "error": "boom"})
        self.assertIn("boom", msg)
        self.assertNotIn("{steps}", msg)


if __name__ == "__main__":
    unittest.main()
