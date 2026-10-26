"""MCP preset setup: the recipe runner, its steps, the approval gate, local presets, the agent tools.

Everything runs in temp folders with stand-ins for the parts that would reach the user's machine (processes, the
registry, user environment variables, the network). No real app preference, registry key or config/mcp.json is touched.

    .venv/Scripts/python.exe -m unittest tests.test_mcp_setup
"""
import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts import mcp_draft as md
from scripts import mcp_presets as mp
from scripts import mcp_setup as ms
from src.tools import mcp_on_demand as od
from src.tools import mcp_setup_tools as tools
from src.utils import mcp_preset_install as mpi

FIX = Path(__file__).parent / "fixtures"


class FakeEnv:
    def __init__(self):
        self.vars = {}

    def get(self, n):
        return self.vars.get(n)

    def set(self, n, v):
        self.vars[n] = v


class FakeReg:
    def __init__(self):
        self.vals = {}

    def get(self, k, n):
        return self.vals.get((k, n))

    def set(self, k, n, v):
        self.vals[(k, n)] = v


class Runner:
    """A process stand-in: records the commands, answers from a table of (substring -> (rc, output))."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, cmd, cwd=None, env=None, timeout=0):
        self.calls.append(list(cmd))
        text = " ".join(cmd)
        for key, ans in self.answers.items():
            if key in text:
                return ans(cmd) if callable(ans) else ans
        return 0, ""


def preset(setup, pid="demo", params=None, server=None):
    return {"id": pid, "name": pid.title(), "category": "T", "description": "d",
            "server": server or {"enabled": True, "transport": "stdio", "command": "uvx", "args": [pid + "==1.0"], "auth": "none"},
            "env_keys": [], "requires": [], "setup_steps": "", "setup": setup, "params": params or {}, "notes": "",
            "default_selected": False}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cfg = self.tmp / "mcp.json"

    def run_recipe(self, setup, **kw):
        p = kw.pop("preset", None) or preset(setup)
        kw.setdefault("root", self.tmp / "root")
        kw.setdefault("config_path", self.cfg)
        kw.setdefault("home", None)
        kw.pop("home")
        kw.setdefault("env_store", FakeEnv())
        kw.setdefault("registry", FakeReg())
        kw.setdefault("run", Runner())
        kw.setdefault("which", lambda n: f"/bin/{n}")
        return ms.run_setup(p["id"], preset=p, **kw)

    def status(self, rep):
        return {s["id"]: s["status"] for s in rep["steps"]}


class Catalog(unittest.TestCase):
    def test_all_nine_presets_have_a_valid_recipe(self):
        want = {"blender", "houdini", "maya", "unity", "nuke", "after-effects", "premiere-pro", "davinci-resolve", "cavalry"}
        have = {p["id"]: p for p in mp.load_catalog() if p.get("setup")}
        # The npx-based presets carry a one-step recipe that only checks for / installs Node.js (issue 78).
        node_only = {pid for pid, p in have.items() if [(x["type"], x.get("tool")) for x in p["setup"]] == [("ensure_tool", "node")]}
        self.assertEqual(want, set(have) - node_only)
        for pid, p in have.items():
            self.assertEqual(ms.validate_recipe(p), [], pid)

    def test_pins_are_present(self):
        for p in mp.load_catalog():
            for st in p.get("setup", []):
                if st["type"] == "git_clone":
                    self.assertRegex(st["commit"], r"^[0-9a-f]{40}$")
                if st["type"] == "download":
                    self.assertRegex(st["sha256"], r"^[0-9a-f]{64}$")

    def test_schema_lists_every_step_type(self):
        sch = ms.recipe_schema()
        kinds = {v["properties"]["type"]["const"] for v in sch["properties"]["setup"]["items"]["oneOf"]}
        self.assertEqual(kinds, set(ms.STEP_SPECS))


class Validation(unittest.TestCase):
    def ok(self):
        return preset([{"id": "uv", "type": "ensure_tool", "tool": "uv"}, {"id": "entry", "type": "ensure_entry"},
                       {"id": "test", "type": "test", "needs": ["uv", "entry"]}])

    def test_valid(self):
        self.assertEqual(ms.validate_recipe(self.ok()), [])

    def test_errors_are_plain_sentences(self):
        p = self.ok()
        p["setup"] += [{"type": "download", "url": "u", "dest": "d"},
                       {"type": "git_clone", "url": "u", "commit": "main", "dest": "d"},
                       {"type": "run_script", "interpreter": "perl", "script": "x"},
                       {"type": "wat"}, {"type": "run", "cmd": "uv sync"},
                       {"type": "test", "needs": ["ghost"]}]
        text = "\n".join(ms.validate_recipe(p))
        for frag in ("no sha256", "40-character commit", "interpreter must be", "description", "type must be one of",
                     "list of arguments", "not an earlier step"):
            self.assertIn(frag, text)

    def test_unpinned_package_and_bad_id(self):
        p = self.ok()
        p["server"] = {"command": "npx", "args": ["-y", "some-mcp"]}
        p["id"] = "Bad Id"
        text = "\n".join(ms.validate_recipe(p))
        self.assertIn("not pinned", text)
        self.assertIn("id must be", text)

    def test_unknown_field(self):
        p = self.ok()
        p["setup"][0]["bogus"] = 1
        self.assertIn("unknown field bogus", "\n".join(ms.validate_recipe(p)))


class Runner_(Tmp):
    def test_needs_gating_and_report_shape(self):
        steps = [{"id": "app", "type": "manual", "message": "Do the thing."},
                 {"id": "after", "type": "manual", "message": "second", "needs": ["app"]},
                 {"id": "free", "type": "ensure_entry"}]
        rep = self.run_recipe(steps)
        self.assertEqual(self.status(rep), {"app": "needs_user", "after": "skipped", "free": "done"})
        self.assertEqual(rep["needs_user"], ["Do the thing."])
        self.assertTrue(rep["ok"])
        self.assertFalse(rep["complete"])
        self.assertIn("waiting for app", rep["steps"][1]["message"])

    def test_idempotent_second_run(self):
        steps = [{"id": "entry", "type": "ensure_entry"}, {"id": "uv", "type": "ensure_tool", "tool": "uv"}]
        first = self.run_recipe(steps)
        second = self.run_recipe(steps)
        self.assertEqual(self.status(first)["entry"], "done")
        self.assertEqual(self.status(second)["entry"], "skipped")
        self.assertEqual(list(json.loads(self.cfg.read_text())["servers"]), ["demo"])

    def test_dry_run_changes_nothing(self):
        rep = self.run_recipe([{"id": "entry", "type": "ensure_entry"}, {"id": "t", "type": "test"}], dry_run=True)
        self.assertEqual(self.status(rep), {"entry": "planned", "t": "planned"})
        self.assertFalse(self.cfg.exists())

    def test_only_steps(self):
        rep = self.run_recipe([{"id": "a", "type": "ensure_entry"}, {"id": "b", "type": "manual", "message": "x"}], only_steps=["a"])
        self.assertEqual(self.status(rep), {"a": "done", "b": "skipped"})

    def test_a_crashing_step_is_a_failed_step(self):
        with mock.patch.dict(ms.HANDLERS, {"manual": mock.Mock(side_effect=RuntimeError("boom"))}):
            rep = self.run_recipe([{"id": "a", "type": "manual", "message": "x"}, {"id": "b", "type": "ensure_entry"}])
        self.assertEqual(self.status(rep), {"a": "failed", "b": "done"})
        self.assertFalse(rep["ok"])
        self.assertIn("RuntimeError: boom", rep["failed"][0])

    def test_unresolved_variable_fails_the_step(self):
        rep = self.run_recipe([{"id": "a", "type": "run", "cmd": ["x", "${nothing.here}"]}])
        self.assertEqual(self.status(rep)["a"], "failed")
        self.assertIn("unresolved", rep["steps"][0]["message"])

    def test_only_on(self):
        rep = self.run_recipe([{"id": "a", "type": "manual", "message": "x", "only_on": "mac"}], platform="win32")
        self.assertEqual(self.status(rep)["a"], "skipped")

    def test_unknown_preset(self):
        rep = ms.run_setup("nope-nope", catalog_path=self.tmp / "none.json")
        self.assertFalse(rep["ok"])

    def test_test_step(self):
        steps = [{"id": "entry", "type": "ensure_entry"}, {"id": "test", "type": "test", "timeout": 1}]
        rep = self.run_recipe(steps, test_server=lambda n, sc, s: {"ok": True, "tools": 3, "names": ["a"]})
        self.assertEqual(rep["steps"][1]["detail"]["tools"], 3)
        rep = self.run_recipe(steps, test_server=lambda n, sc, s: {"ok": False, "error": "no key", "needs_auth": True})
        self.assertEqual(self.status(rep)["test"], "needs_user")
        rep = self.run_recipe(steps, test_server=lambda n, sc, s: {"ok": False, "error": "refused"})
        self.assertEqual(self.status(rep)["test"], "failed")


class Edits(Tmp):
    def test_marker_block_is_idempotent_and_backs_up_once(self):
        f = self.tmp / "menu.py"
        f.write_text("print(1)\n", encoding="utf-8")
        self.assertTrue(ms.edit_marker_block(f, "# a", "# z", ["x = 1"]))
        self.assertFalse(ms.edit_marker_block(f, "# a", "# z", ["x = 1"]))
        self.assertTrue(ms.edit_marker_block(f, "# a", "# z", ["x = 2"]))
        text = f.read_text(encoding="utf-8")
        self.assertEqual(text.count("# a"), 1)
        self.assertIn("x = 2", text)
        self.assertEqual((self.tmp / "menu.py.agenty-bak").read_text(encoding="utf-8"), "print(1)\n")

    def test_json_merge(self):
        f = self.tmp / "m.json"
        f.write_text(json.dumps({"dependencies": {"a": "1"}, "x": 1}), encoding="utf-8")
        self.assertTrue(ms.edit_json_merge(f, {"dependencies": {"b": "2"}}))
        self.assertFalse(ms.edit_json_merge(f, {"dependencies": {"b": "2"}}))
        self.assertEqual(json.loads(f.read_text())["dependencies"], {"a": "1", "b": "2"})
        self.assertTrue((self.tmp / "m.json.agenty-bak").is_file())

    def test_json_merge_refuses_broken_json(self):
        f = self.tmp / "m.json"
        f.write_text("{oops", encoding="utf-8")
        with self.assertRaises(ms.StepError):
            ms.edit_json_merge(f, {"a": 1})
        self.assertEqual(f.read_text(), "{oops")

    def test_regex_set(self):
        f = self.tmp / "p.txt"
        f.write_text('"Pref_X" = "0"\nother\n', encoding="utf-8")
        self.assertEqual(ms.edit_regex_set(f, r'^("Pref_X" = )"\d"', r'\g<1>"1"'), (True, True))
        self.assertEqual(ms.edit_regex_set(f, r'^("Pref_X" = )"\d"', r'\g<1>"1"'), (True, False))
        self.assertEqual(ms.edit_regex_set(f, r"^nothing", "x"), (False, False))

    def test_set_env_step(self):
        env = FakeEnv()
        steps = [{"id": "e", "type": "set_env", "name": "FOO", "value": "bar"}]
        self.assertEqual(self.status(self.run_recipe(steps, env_store=env))["e"], "done")
        self.assertEqual(self.status(self.run_recipe(steps, env_store=env))["e"], "skipped")
        self.assertEqual(env.vars, {"FOO": "bar"})
        self.assertEqual(self.status(self.run_recipe(steps, env_store=FakeEnv(), dry_run=True))["e"], "planned")


class Tools(Tmp):
    def test_ensure_tool_found_missing_old(self):
        steps = [{"id": "t", "type": "ensure_tool", "tool": "uv"}]
        self.assertEqual(self.status(self.run_recipe(steps, which=lambda n: "/x/uv"))["t"], "skipped")
        rep = self.run_recipe(steps, which=lambda n: None)
        self.assertEqual(self.status(rep)["t"], "needs_user")
        self.assertIn("winget install -e --id astral-sh.uv", rep["needs_user"][0])
        steps2 = [{"id": "n", "type": "ensure_tool", "tool": "node", "min_version": "24"}]
        rep = self.run_recipe(steps2, run=Runner({"--version": (0, "v22.1.0")}))
        self.assertEqual(self.status(rep)["n"], "needs_user")
        self.assertIn("24 or newer", rep["needs_user"][0])

    def test_ensure_tool_installs_with_winget_only_when_allowed(self):
        state = {"have": False}
        runner = Runner({"winget": lambda c: (state.update(have=True) or (0, "ok"))})
        which = lambda n: "/x/uv" if state["have"] else None  # noqa: E731
        steps = [{"id": "t", "type": "ensure_tool", "tool": "uv"}]
        rep = self.run_recipe(steps, which=which, run=runner, allow_system=True, platform="win32")
        self.assertEqual(self.status(rep)["t"], "done")
        self.assertEqual(runner.calls[0][:2], ["winget", "install"])
        # not allowed: nothing is run
        runner2 = Runner()
        state["have"] = False
        rep = self.run_recipe(steps, which=which, run=runner2, platform="win32")
        self.assertEqual(self.status(rep)["t"], "needs_user")
        self.assertEqual(runner2.calls, [])
        # dry run with permission says what it would do and does nothing
        rep = self.run_recipe(steps, which=which, run=runner2, allow_system=True, dry_run=True)
        self.assertEqual(self.status(rep)["t"], "planned")
        self.assertEqual(runner2.calls, [])

    def test_find_app_picks_newest_that_meets_min_and_reports_others(self):
        for v in ("3.5", "5.2"):
            (self.tmp / f"Blender {v}").mkdir()
            (self.tmp / f"Blender {v}" / "blender.exe").write_bytes(b"")
        ov = {"system_apps": False, "app_globs": {"blender": [str(self.tmp / "Blender */blender.exe")]}}
        rep = self.run_recipe([{"id": "blender", "type": "find_app", "app": "blender", "min_version": "5.1"}], overrides=ov)
        self.assertEqual(rep["steps"][0]["detail"]["version"], "5.2")
        self.assertIn("also found 3.5", rep["steps"][0]["message"])
        rep = self.run_recipe([{"id": "blender", "type": "find_app", "app": "blender", "min_version": "9"}], overrides=ov)
        self.assertEqual(self.status(rep)["blender"], "needs_user")
        self.assertIn("9 or newer", rep["needs_user"][0])
        rep = self.run_recipe([{"id": "x", "type": "find_app", "app": "unity"}], overrides={"system_apps": False})
        self.assertIn("not installed", rep["needs_user"][0])

    def test_download_file_url_sha_and_skip(self):
        src = self.tmp / "src.bin"
        src.write_bytes(b"hello")
        sha = hashlib.sha256(b"hello").hexdigest()
        steps = [{"id": "d", "type": "download", "url": src.as_uri(), "sha256": sha, "dest": "${DOWNLOADS}/a/src.bin"}]
        rep = self.run_recipe(steps)
        self.assertEqual(self.status(rep)["d"], "done")
        self.assertEqual(self.status(self.run_recipe(steps))["d"], "skipped")
        bad = [{**steps[0], "id": "d2", "sha256": "0" * 64, "dest": "${DOWNLOADS}/b/src.bin"}]
        rep = self.run_recipe(bad)
        self.assertEqual(self.status(rep)["d2"], "failed")
        self.assertFalse((self.tmp / "root" / "config" / "mcp_bundles" / ".downloads" / "b" / "src.bin").exists())

    def test_download_refuses_a_big_file_on_c(self):
        with self.assertRaises(ms.StepError):
            ms.download_file("http://x/y", Path("C:/agenty_test_never/x.bin"), size_mb=500)

    def test_git_clone_and_checkout_pinned(self):
        commit = "a" * 40
        dest = self.tmp / "root" / "config" / "mcp_bundles" / "repo"

        def git(cmd):
            if cmd[1] == "clone":
                (Path(cmd[3]) / ".git").mkdir(parents=True)
            if cmd[1] == "rev-parse":
                return 0, commit + "\n"
            return 0, ""
        runner = Runner({"git": git})
        steps = [{"id": "c", "type": "git_clone", "url": "https://x/y.git", "commit": commit, "dest": "${BUNDLES}/repo"}]
        rep = self.run_recipe(steps, run=runner)
        self.assertEqual(self.status(rep)["c"], "done")
        self.assertTrue(any(c[1:3] == ["checkout", "--quiet"] and c[3] == commit for c in runner.calls))
        self.assertEqual(self.status(self.run_recipe(steps, run=runner))["c"], "skipped")
        self.assertTrue(dest.is_dir())

    def test_run_step_skip_if_exists_and_failure(self):
        (self.tmp / "root").mkdir()
        runner = Runner({"fail": (3, "bad things")})
        steps = [{"id": "ok", "type": "run", "cmd": ["uv", "sync"], "skip_if_exists": str(self.tmp)},
                 {"id": "bad", "type": "run", "cmd": ["fail"]}]
        rep = self.run_recipe(steps, run=runner)
        self.assertEqual(self.status(rep), {"ok": "skipped", "bad": "failed"})
        self.assertIn("exited 3", rep["steps"][1]["message"])

    def test_run_script_runs_and_is_flagged_in_summaries(self):
        p = preset([{"id": "s", "type": "run_script", "interpreter": "python", "script": "print(1)", "description": "prints one"}])
        runner = Runner()
        rep = self.run_recipe([], preset=p, run=runner)
        self.assertEqual(self.status(rep)["s"], "done")
        self.assertEqual(runner.calls[0][1], "-I")
        self.assertIn("RUNS A SCRIPT", ms.describe_recipe(p)[0])
        self.assertIn("print(1)", ms.describe_recipe(p)[0])
        dry = self.run_recipe([], preset=p, run=Runner(), dry_run=True)
        self.assertEqual(self.status(dry)["s"], "planned")


class Addons(Tmp):
    def vars(self, **kw):
        return kw

    def test_nuke_addon_into_a_temp_nuke_dir(self):
        src = self.tmp / "clone" / "nuke_mcp_addon.py"
        src.parent.mkdir()
        src.write_text("def start(): pass\n", encoding="utf-8")
        nuke = self.tmp / "dotnuke"
        steps = [{"id": "a", "type": "install_addon", "app": "nuke", "source": str(src)}]
        ov = {"prefs": {"nuke": str(nuke)}}
        self.assertEqual(self.status(self.run_recipe(steps, overrides=ov))["a"], "done")
        self.assertEqual(self.status(self.run_recipe(steps, overrides=ov))["a"], "skipped")
        menu = (nuke / "menu.py").read_text(encoding="utf-8")
        self.assertEqual(menu.count("# --- nuke-mcp (agentY) ---"), 1)
        self.assertIn("nuke_mcp_addon.start()", menu)
        self.assertTrue((nuke / "nuke_mcp_addon.py").is_file())
        # a hand-edited menu.py keeps its own lines and the block is not duplicated
        (nuke / "menu.py").write_text("mine = 1\n" + menu, encoding="utf-8")
        self.run_recipe(steps, overrides=ov)
        self.assertEqual((nuke / "menu.py").read_text(encoding="utf-8").count("# --- nuke-mcp (agentY) ---"), 1)

    def test_maya_userSetup_and_script(self):
        src = self.tmp / "ec.py"
        src.write_text("def enable_commandport(port=7001): pass\n", encoding="utf-8")
        steps = [{"id": "maya", "type": "install_addon", "app": "maya", "source": str(src)}]
        ov = {"prefs": {"maya": str(self.tmp / "mayaapp")}}
        ctx = ms.Ctx(preset(steps, params={"port": 7001}), root=self.tmp / "r", config_path=self.cfg, overrides=ov)
        ctx.vars["maya.version"] = "2026.2.0"
        r1 = ms.addon_maya(ctx, {**steps[0]})
        r2 = ms.addon_maya(ctx, {**steps[0]})
        self.assertEqual((r1["status"], r2["status"]), ("done", "skipped"))
        scripts = self.tmp / "mayaapp" / "2026" / "scripts"
        self.assertIn("enable_commandport(7001)", (scripts / "userSetup.py").read_text(encoding="utf-8"))
        self.assertTrue((scripts / "agenty_enable_commandport.py").is_file())

    def test_cavalry_bridge_script(self):
        src = self.tmp / "bridge.js"
        src.write_text("// bridge", encoding="utf-8")
        ctx = ms.Ctx(preset([]), root=self.tmp / "r", config_path=self.cfg, overrides={"prefs": {"cavalry": str(self.tmp / "Scripts")}})
        st = {"source": str(src), "app": "cavalry", "id": "b"}
        self.assertEqual(ms.addon_cavalry(ctx, st)["status"], "done")
        self.assertEqual(ms.addon_cavalry(ctx, st)["status"], "skipped")
        self.assertEqual((self.tmp / "Scripts" / "cavalry-mcp-bridge.js").read_text(), "// bridge")
        ctx.platform = "win32"
        ctx.overrides = {}
        ctx.appdata = self.tmp / "AppData"
        self.assertEqual(ms._cavalry_scripts(ctx), self.tmp / "AppData" / "Cavalry" / "Scripts")

    def test_premiere_panel_and_cep_flag_use_the_fake_registry(self):
        panel = self.tmp / "cep-plugin"
        (panel / "CSXS").mkdir(parents=True)
        (panel / "index.html").write_text("<html>", encoding="utf-8")
        (panel / "CSXS" / "manifest.xml").write_text("<x/>", encoding="utf-8")
        reg = FakeReg()
        ctx = ms.Ctx(preset([]), root=self.tmp / "r", config_path=self.cfg, platform="win32", registry=reg,
                     overrides={"prefs": {"premiere-pro": str(self.tmp / "CEP")}})
        st = {"source": str(panel), "app": "premiere-pro", "id": "p", "debug_mode": True}
        self.assertEqual(ms.addon_premiere(ctx, st)["status"], "done")
        self.assertEqual(ms.addon_premiere(ctx, st)["status"], "skipped")
        self.assertTrue((self.tmp / "CEP" / "MCPBridgeCEP" / "CSXS" / "manifest.xml").is_file())
        self.assertEqual(reg.vals[(r"Software\Adobe\CSXS.12", "PlayerDebugMode")], "1")
        self.assertEqual(len(reg.vals), 7)
        ctx.dry_run = True
        reg.vals.clear()
        self.assertEqual(ms.addon_premiere(ctx, st)["status"], "planned")
        self.assertEqual(reg.vals, {})

    def test_after_effects_pref_edit_and_missing_cases(self):
        root = self.tmp / "AE"
        ctx = ms.Ctx(preset([]), root=self.tmp / "r", config_path=self.cfg, overrides={"prefs": {"after-effects": str(root)}})
        with self.assertRaises(ms.NeedsUser):
            ms.addon_after_effects(ctx, {})
        d = root / "25.0"
        d.mkdir(parents=True)
        f = d / "Adobe After Effects 25.0 Prefs-indep-general.txt"
        f.write_text('"Pref_OTHER" = "1"\n', encoding="utf-8")
        with self.assertRaises(ms.NeedsUser):
            ms.addon_after_effects(ctx, {})
        f.write_text('"Pref_SCRIPTING_FILE_NETWORK_SECURITY" = "0"\n"Pref_OTHER" = "1"\n', encoding="utf-8")
        self.assertEqual(ms.addon_after_effects(ctx, {})["status"], "done")
        self.assertIn('"Pref_SCRIPTING_FILE_NETWORK_SECURITY" = "1"', f.read_text(encoding="utf-8"))
        self.assertIn('"Pref_OTHER" = "1"', f.read_text(encoding="utf-8"))
        self.assertEqual(ms.addon_after_effects(ctx, {})["status"], "skipped")

    def test_unity_needs_a_project_then_edits_the_manifest(self):
        ctx = ms.Ctx(preset([]), root=self.tmp / "r", config_path=self.cfg)
        st = {"dependency": "com.x.y", "dependency_url": "https://g/x.git#v1"}
        with self.assertRaises(ms.NeedsUser):
            ms.addon_unity(ctx, st)
        proj = self.tmp / "Proj"
        (proj / "Packages").mkdir(parents=True)
        (proj / "Packages" / "manifest.json").write_text(json.dumps({"dependencies": {"a": "1"}}), encoding="utf-8")
        ctx.params["project"] = str(proj)
        self.assertEqual(ms.addon_unity(ctx, st)["status"], "done")
        self.assertEqual(ms.addon_unity(ctx, st)["status"], "skipped")
        deps = json.loads((proj / "Packages" / "manifest.json").read_text())["dependencies"]
        self.assertEqual(deps, {"a": "1", "com.x.y": "https://g/x.git#v1"})

    def test_houdini_runs_the_package_installer_into_the_prefs_dir(self):
        prefs = self.tmp / "hprefs"

        def installer(cmd):
            pdir = Path(cmd[cmd.index("--prefs-dir") + 1])
            (pdir / "packages").mkdir(parents=True)
            (pdir / "houdinimcp").mkdir()
            (pdir / "packages" / "houdinimcp.json").write_text(json.dumps({"hpath": str(pdir / "houdinimcp")}), encoding="utf-8")
            return 0, "{}"
        runner = Runner({"houdinimcp-install": installer})
        ctx = ms.Ctx(preset([]), root=self.tmp / "r", config_path=self.cfg, run=runner, overrides={"prefs": {"houdini": str(prefs)}})
        ctx.vars["houdini.version"] = "21.0.631"
        st = {"dependency": "houdini-mcp-server==0.7.7"}
        self.assertEqual(ms.addon_houdini(ctx, st)["status"], "done")
        self.assertEqual(ms.addon_houdini(ctx, st)["status"], "skipped")
        self.assertTrue((prefs / "houdini21.0" / "packages" / "houdinimcp.json").is_file())
        self.assertIn("--harness", runner.calls[0])
        self.assertIn("none", runner.calls[0])

    def test_blender_addon_with_a_fake_blender(self):
        zp = self.tmp / "mcp.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("blender_manifest.toml", 'id = "mcp"\nversion = "1.0.3"\n')
        state = {"enabled": False, "version": "", "port": 9876, "autostart": True, "online": False}

        def blender(cmd):
            text = " ".join(cmd)
            if "install-file" in text:
                state.update(enabled=True, version="1.0.3")
                return 0, 'STATUS Installed "mcp"'
            if "AGENTY_SAVED" in text or "save_userpref" in text:
                state.update(autostart=True, port=9999, online=True)
                return 0, "AGENTY_SAVED"
            return 0, "AGENTY_STATE " + json.dumps(state)
        runner = Runner({"blender": blender, "-b": blender})
        ctx = ms.Ctx(preset([], params={"port": 9876}), root=self.tmp / "r", config_path=self.cfg, run=runner,
                     overrides={"params": {"port": 9999}, "prefs": {"blender": str(self.tmp / "res")}})
        ctx.vars["blender.path"] = "blender"
        st = {"zip": str(zp), "addon_id": "mcp", "app": "blender", "id": "addon"}
        r = ms.addon_blender(ctx, st)
        self.assertEqual(r["status"], "done")
        self.assertIn("port 9999", r["message"])
        self.assertIn("online access", r["message"])
        self.assertEqual(ms.addon_blender(ctx, st)["status"], "skipped")
        # the user's own prefs are never named: the override is the only prefs location passed to Blender
        envs = [c for c in runner.calls]
        self.assertTrue(envs)
        ctx.dry_run = True
        state.update(enabled=False)
        self.assertEqual(ms.addon_blender(ctx, st)["status"], "planned")


class Launch(Tmp):
    def test_not_started_unless_asked(self):
        steps = [{"id": "l", "type": "launch_and_wait", "app": "blender", "port": 59999}]
        rep = self.run_recipe(steps)
        self.assertEqual(self.status(rep)["l"], "needs_user")
        self.assertIn("Start Blender", rep["needs_user"][0])

    def test_an_open_port_means_running(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        self.addCleanup(s.close)
        port = s.getsockname()[1]
        rep = self.run_recipe([{"id": "l", "type": "launch_and_wait", "app": "blender", "port": port}])
        self.assertEqual(self.status(rep)["l"], "skipped")


class Approval(Tmp):
    def recipe(self, **kw):
        p = preset([{"id": "uv", "type": "ensure_tool", "tool": "uv"}, {"id": "entry", "type": "ensure_entry"}], pid="newone")
        p.update(kw)
        return p

    def test_hash_covers_everything_that_runs(self):
        a = ms.recipe_hash(self.recipe())
        self.assertEqual(a, ms.recipe_hash(self.recipe()))
        b = self.recipe()
        b["setup"][0]["tool"] = "git"
        self.assertNotEqual(a, ms.recipe_hash(b))
        c = self.recipe(server={"command": "uvx", "args": ["other==1"]})
        self.assertNotEqual(a, ms.recipe_hash(c))
        d = self.recipe()
        d["description"] = "different words"   # description does not run anything
        self.assertEqual(a, ms.recipe_hash(d))

    def test_gate_shows_first_approves_second_and_a_change_needs_a_new_yes(self):
        store = self.tmp / "pending.json"
        r = self.recipe()
        first = ms.approval_gate(r, "", store=store)
        self.assertEqual(first["status"], "waiting_for_user")
        self.assertTrue(first["summary"])
        # words without having been shown first do not approve
        other = self.recipe()
        other["setup"].append({"id": "t", "type": "test", "needs": ["entry"]})
        self.assertEqual(ms.approval_gate(other, "yes", store=store)["status"], "waiting_for_user")
        # the shown one, with the user's words
        self.assertEqual(ms.approval_gate(r, "yes please", store=store)["status"], "approved")
        # empty words never approve
        self.assertEqual(ms.approval_gate(r, "   ", store=store)["status"], "waiting_for_user")
        # a changed recipe after approval: new hash, shown again
        r["setup"][0]["tool"] = "git"
        self.assertEqual(ms.approval_gate(r, "yes", store=store)["status"], "waiting_for_user")

    def test_gate_rejects_an_invalid_recipe_before_showing_it(self):
        bad = self.recipe()
        bad["setup"].append({"type": "wat"})
        out = ms.approval_gate(bad, "", store=self.tmp / "p.json")
        self.assertEqual(out["status"], "invalid")
        self.assertFalse((self.tmp / "p.json").exists())

    def test_run_script_is_flagged_in_the_gate(self):
        r = self.recipe()
        r["setup"].insert(1, {"id": "s", "type": "run_script", "interpreter": "powershell", "script": "Write-Host hi", "description": "says hi"})
        out = ms.approval_gate(r, "", store=self.tmp / "p.json")
        self.assertEqual(out["run_script_steps"], [2])
        self.assertIn("Write-Host hi", "\n".join(out["summary"]))


class LocalPresets(Tmp):
    def setUp(self):
        super().setUp()
        self.cat = self.tmp / "cat.json"
        self.cat.write_text(json.dumps({"presets": [preset([], pid="official")]}), encoding="utf-8")
        self.local = self.tmp / "local.json"

    def draft(self, pid="mine"):
        return preset([{"id": "entry", "type": "ensure_entry"}], pid=pid)

    def test_merge_after_official(self):
        h = ms.recipe_hash(ms.normalise_draft(self.draft()))
        ms.save_local_preset(self.draft(), h, self.local)
        ids = [p["id"] for p in mp.load_catalog(self.cat, self.local)]
        self.assertEqual(ids, ["official", "mine"])
        mine = mp.find("mine", self.cat, self.local)
        self.assertTrue(mine["local"])
        self.assertTrue(ms.is_approved(mine))

    def test_official_wins_a_clash(self):
        ms.save_local_preset(self.draft("official"), "x" * 64, self.local)
        with self.assertLogs("scripts.mcp_presets", level="WARNING") as cm:
            cat = mp.load_catalog(self.cat, self.local)
        self.assertEqual([p["id"] for p in cat], ["official"])
        self.assertFalse(cat[0].get("local"))
        self.assertIn("official", cm.output[0])

    def test_edited_local_recipe_is_no_longer_approved_and_does_not_run(self):
        h = ms.recipe_hash(ms.normalise_draft(self.draft()))
        ms.save_local_preset(self.draft(), h, self.local)
        data = json.loads(self.local.read_text())
        data["presets"][0]["setup"].append({"id": "m", "type": "manual", "message": "sneaky"})
        self.local.write_text(json.dumps(data), encoding="utf-8")
        mine = mp.find("mine", self.cat, self.local)
        self.assertFalse(ms.is_approved(mine))
        rep = ms.run_setup("mine", preset=mine, root=self.tmp / "r", config_path=self.cfg)
        self.assertFalse(rep["ok"])
        self.assertIn("not approved", rep["failed"][0])
        self.assertFalse(self.cfg.exists())

    def test_the_local_file_is_gitignored(self):
        text = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("config/mcp_presets.local.json", text)


class InstallPreset(Tmp):
    def host(self, **kw):
        cfg = self.cfg
        return dict(load_config=lambda: mp.read_config(cfg),
                    save_config=lambda c: cfg.write_text(json.dumps(c), encoding="utf-8"),
                    update_env=mock.Mock(), test_server=lambda n, sc, s: {"ok": True, "tools": 2, "names": ["a", "b"]},
                    get_env=lambda k: "", **kw)

    def test_missing_key_is_needs_user_and_writes_nothing(self):
        p = preset([{"id": "entry", "type": "ensure_entry"}])
        p["env_keys"] = [{"key": "DEMO_KEY", "label": "Demo key", "help_url": "https://x/keys"}]
        host = self.host()
        out, status = mpi.install_preset("demo", {}, preset=p, **host)
        self.assertEqual(status, 400)
        self.assertEqual(out["status"], "needs_user")
        self.assertIn("DEMO_KEY", out["needs_user"][0])
        self.assertFalse(self.cfg.exists())
        host["update_env"].assert_not_called()

    def test_recipe_install_then_recheck_then_conflict(self):
        p = preset([{"id": "entry", "type": "ensure_entry"}, {"id": "test", "type": "test", "needs": ["entry"]}])
        out, status = mpi.install_preset("demo", {}, preset=p, root=self.tmp / "r", config_path=self.cfg, **self.host())
        self.assertEqual((status, out["status"], out["state"]), (200, "done", "added"))
        self.assertEqual(out["test"]["tools"], 2)
        self.assertIn("demo", json.loads(self.cfg.read_text())["servers"])
        out, status = mpi.install_preset("demo", {}, preset=p, **self.host())
        self.assertEqual(status, 409)
        out, status = mpi.install_preset("demo", {}, preset=p, recheck=True, root=self.tmp / "r", config_path=self.cfg, **self.host())
        self.assertEqual((status, out["status"]), (200, "done"))
        self.assertEqual(out["report"]["steps"][0]["status"], "skipped")

    def test_dry_run_writes_neither_keys_nor_config(self):
        p = preset([{"id": "entry", "type": "ensure_entry"}])
        p["env_keys"] = [{"key": "DEMO_KEY", "label": "k", "help_url": ""}]
        host = self.host()
        out, status = mpi.install_preset("demo", {"DEMO_KEY": "secret"}, preset=p, dry_run=True, root=self.tmp / "r", config_path=self.cfg, **host)
        self.assertEqual(out["status"], "dry_run")
        host["update_env"].assert_not_called()
        self.assertFalse(self.cfg.exists())

    def test_unapproved_local_preset_is_refused(self):
        p = {**preset([{"id": "entry", "type": "ensure_entry"}]), "local": True, "approved_hash": "bad"}
        out, status = mpi.install_preset("demo", {}, preset=p, **self.host())
        self.assertEqual(status, 403)

    def test_failed_step_starts_a_fallback_turn_with_the_report(self):
        p = preset([{"id": "x", "type": "run", "cmd": ["fail"]}, {"id": "entry", "type": "ensure_entry"}])
        turns = []
        with mock.patch.object(mp, "find", return_value=p):
            out, status = mpi.install("demo", {}, root=self.tmp / "r", config_path=self.cfg,
                                      start_turn=lambda *a: (turns.append(a) or ("rid", "tid")),
                                      run=Runner({"fail": (1, "nope")}), which=lambda n: n, **self.host())
        self.assertEqual(out["turn_id"], "rid")
        self.assertIn("FAIL", turns[0][0])
        self.assertIn("exited 1", turns[0][0])

    def test_clean_run_starts_no_turn(self):
        p = preset([{"id": "entry", "type": "ensure_entry"}])
        with mock.patch.object(mp, "find", return_value=p):
            out, status = mpi.install("demo", {}, root=self.tmp / "r", config_path=self.cfg,
                                      start_turn=lambda *a: self.fail("no turn expected"), **self.host())
        self.assertIsNone(out["turn_id"])

    def test_jobs_report_steps_and_result(self):
        import time as _t

        def run(progress):
            progress({"id": "a", "status": "done"})
            return {"ok": True}, 200
        jid = mpi.start_job(run, label="t")
        for _ in range(100):
            rec = mpi.job(jid)
            if not rec["running"]:
                break
            _t.sleep(0.02)
        self.assertEqual(rec["steps"], [{"id": "a", "status": "done"}])
        self.assertEqual(rec["result"], {"ok": True})
        self.assertIsNone(mpi.job("nope"))


class AgentTools(Tmp):
    def setUp(self):
        super().setUp()
        self.local = self.tmp / "local.json"
        self.pending = self.tmp / "pending.json"
        self.cat = self.tmp / "cat.json"
        self.cat.write_text(json.dumps({"presets": [preset([{"id": "entry", "type": "ensure_entry"}], pid="official")]}), encoding="utf-8")

    def host(self):
        cfg = self.cfg
        return dict(load_config=lambda: mp.read_config(cfg), save_config=lambda c: cfg.write_text(json.dumps(c), encoding="utf-8"),
                    update_env=mock.Mock(), test_server=lambda n, sc, s: {"ok": True, "tools": 1, "names": ["t"]},
                    get_env=lambda k: "")

    def recipe(self, **kw):
        r = preset([{"id": "entry", "type": "ensure_entry"}, {"id": "test", "type": "test", "needs": ["entry"]}], pid="drafted")
        r.update(kw)
        return json.dumps(r)

    def call(self, **kw):
        kw.setdefault("host", self.host())
        kw.setdefault("pending_store", self.pending)
        kw.setdefault("local_path", self.local)
        kw.setdefault("config_path", self.cfg)
        kw.setdefault("root", self.tmp / "r")
        return tools.run_install(**kw)

    def test_recipe_flow_dry_run_then_gate_then_approved_install(self):
        r = self.recipe()
        dry = self.call(recipe=r, dry_run=True)
        self.assertEqual(dry["status"], "dry_run")
        self.assertTrue(dry["summary"])
        self.assertFalse(self.cfg.exists())
        shown = self.call(recipe=r)
        self.assertEqual(shown["status"], "waiting_for_user")
        self.assertFalse(self.cfg.exists())
        self.assertFalse(self.local.exists())
        done = self.call(recipe=r, user_message="yes, go")
        self.assertEqual(done["status"], "done")
        self.assertIn("drafted", json.loads(self.cfg.read_text())["servers"])
        saved = json.loads(self.local.read_text())["presets"][0]
        self.assertEqual(saved["id"], "drafted")
        self.assertEqual(saved["approved_hash"], ms.recipe_hash(ms.normalise_draft(json.loads(r))))

    def test_changed_recipe_after_approval_is_shown_again(self):
        r = self.recipe()
        self.call(recipe=r)
        self.call(recipe=r, user_message="yes")
        changed = json.loads(r)
        changed["setup"].append({"id": "m", "type": "manual", "message": "new step"})
        out = self.call(recipe=json.dumps(changed), user_message="yes")
        self.assertEqual(out["status"], "waiting_for_user")

    def test_a_catalog_id_cannot_be_overridden_by_a_recipe(self):
        with mock.patch.object(mp, "CATALOG", self.cat):
            out = self.call(recipe=self.recipe(id="official"))
        self.assertEqual(out["status"], "invalid")
        self.assertIn("catalog preset", out["errors"][0])

    def test_invalid_json_and_missing_arguments(self):
        self.assertEqual(self.call(recipe="{nope")["status"], "invalid")
        self.assertFalse(self.call()["ok"])

    def test_preset_by_id_uses_the_catalog(self):
        with mock.patch.object(mp, "CATALOG", self.cat), mock.patch.object(mp, "LOCAL_CATALOG", self.local):
            out = self.call(preset_id="official")
        self.assertEqual(out["status"], "done")
        self.assertIn("report_text", out)

    def test_after_loads_the_tools_into_the_running_agent(self):
        agent = mock.Mock()
        ctx = mock.Mock(agent=agent)
        with mock.patch.object(od, "load", return_value=["demo__a", "demo__b"]) as load:
            out = tools._after({"ok": True, "test": {"ok": True}}, "demo", ctx)
        load.assert_called_once_with(agent, "demo")
        self.assertEqual(out["tools_loaded"], 2)
        out = tools._after({"ok": False, "needs_user": ["Start it."]}, "demo", ctx)
        self.assertEqual(out["tell_the_user"], ["Start it."])

    def test_the_conversation_keeps_a_server_an_install_tool_loaded(self):
        msgs = [{"role": "assistant", "content": [{"toolUse": {"name": "install_mcp_preset", "input": {"preset_id": "blender"}}}]},
                {"role": "assistant", "content": [{"toolUse": {"name": "install_mcp_preset", "input": {"recipe": '{"id": "drafted"}'}}}]}]
        self.assertEqual(od.servers_in(msgs, {"blender", "drafted", "other"}), {"blender", "drafted"})

    def test_preset_list(self):
        with mock.patch.object(mp, "CATALOG", self.cat), mock.patch.object(mp, "LOCAL_CATALOG", self.local):
            rows = tools.preset_list(include_recipe=True, configured={})
        self.assertEqual(rows[0]["id"], "official")
        self.assertTrue(rows[0]["has_setup"])
        self.assertIn("recipe", rows[0])

    def test_tools_are_registered_where_the_orchestrator_finds_them(self):
        import src.tools as T
        from src.tools import tool_packs as packs
        for n in ("install_mcp_preset", "setup_mcp_preset", "list_mcp_presets", "draft_mcp_recipe", "install_mcp_bundle", "find_app_installs"):
            self.assertIn(n, [packs.name_of(t) for t in T.ORCHESTRATOR_TOOLS], n)
            self.assertEqual(packs._PACK_OF[n], "mcp")


class Draft(Tmp):
    def fetch(self):
        rec = json.loads((FIX / "mcp_draft_time_responses.json").read_text(encoding="utf-8"))

        def f(url):
            if url not in rec["responses"]:
                raise OSError("not recorded: " + url)
            return rec["responses"][url]
        return rec["source"], f

    def test_parse_source(self):
        self.assertEqual(md.parse_source("https://github.com/o/r")["repo"], "r")
        self.assertEqual(md.parse_source("o/r")["owner"], "o")
        self.assertEqual(md.parse_source("https://github.com/o/r/tree/main/src/x")["subdir"], "src/x")
        self.assertEqual(md.parse_source("npm:@a/b")["package"], "@a/b")
        self.assertEqual(md.parse_source("pypi:foo")["kind"], "pypi")

    def test_draft_from_a_real_public_readme_and_dry_run_only(self):
        """mcp-server-time (modelcontextprotocol/servers, src/time): recorded README, pyproject, commit, PyPI version."""
        source, fetch = self.fetch()
        out = md.draft(source, fetch)
        d = out["draft"]
        self.assertTrue(out["validation"]["ok"], out["validation"])
        self.assertEqual(d["server"]["command"], "uvx")
        self.assertEqual(d["server"]["args"], ["mcp-server-time==2026.10.19"])
        self.assertEqual([s["type"] for s in d["setup"]], ["ensure_tool", "ensure_entry", "test"])
        self.assertIn("timezone", d["description"].lower())
        self.assertIn("Time MCP Server", out["facts"]["readme"])
        self.assertEqual(out["hash"], ms.recipe_hash(ms.normalise_draft(d)))
        # through the agent tool: validate (dry run), and nothing is written anywhere
        res = tools.run_install(recipe=json.dumps(d), dry_run=True, host=None or {"load_config": lambda: {"servers": {}}},
                                config_path=self.cfg, root=self.tmp / "r")
        self.assertEqual(res["status"], "dry_run")
        self.assertTrue(res["summary"])
        self.assertFalse(self.cfg.exists())
        # and the runner in dry run on it
        rep = ms.run_setup(d["id"], preset=ms.normalise_draft(d) | {"local": True, "approved_hash": out["hash"]},
                           dry_run=True, root=self.tmp / "r", config_path=self.cfg, which=lambda n: "/x/" + n)
        self.assertEqual({s["status"] for s in rep["steps"]}, {"skipped", "planned"})
        self.assertFalse(self.cfg.exists())

    def test_unknown_source_gives_a_skeleton_the_validator_rejects(self):
        out = md.draft("some random words", fetch=lambda u: (_ for _ in ()).throw(OSError("offline")))
        self.assertFalse(out["validation"]["ok"])
        self.assertIn("TODO", " ".join(out["validation"]["errors"]))

    def test_repo_without_a_release_is_cloned_at_its_commit(self):
        commit = "b" * 40
        facts = {"source": "o/r", "parsed": {"kind": "github", "owner": "o", "repo": "r"}, "commit": commit,
                 "repo_url": "https://github.com/o/r.git", "scripts": ["r-mcp"], "readme": "# R\n\nA server."}
        d = md.build_draft(facts)
        self.assertEqual([s["type"] for s in d["setup"]], ["ensure_tool", "ensure_tool", "git_clone", "run", "ensure_entry", "test"])
        self.assertEqual(d["setup"][2]["commit"], commit)
        self.assertEqual(ms.validate_recipe(d), [])


class InstallerCli(unittest.TestCase):
    def test_hasrecipe_and_setup_dry_run(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(mp._main(["hasrecipe", "blender"]), 0)
        self.assertEqual(buf.getvalue().strip(), "1")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            mp._main(["--config", str(tmp / "mcp.json"), "setup", "unity", "--dry-run"])
        self.assertIn("unity", buf.getvalue())
        self.assertIn("Left for you", buf.getvalue())
        self.assertFalse((tmp / "mcp.json").exists())


if __name__ == "__main__":
    unittest.main()
