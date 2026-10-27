"""node / npm / uv / git on macOS and Linux: where they are looked for, how they are installed, what the
catalog text says (GitHub issue 78 B).

    .venv/Scripts/python.exe -m unittest tests.test_tool_hints

The platform is a parameter (``Ctx(platform=...)``) or ``sys.platform`` patched; the home folder is a temp tree.
Nothing is installed and no real PATH matters: ``shutil.which`` is patched where a "not on PATH" is needed.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import mcp_presets as mp
from scripts import mcp_setup as ms
from scripts import tool_hints as th
from tests import test_mcp_setup as tms


def make_exe(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)
    return str(path)


class FakeHome(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = self.tmp / "home"
        self.home.mkdir()

    def isdir_with(self, *absolute):
        """os.path.isdir that also says yes to system folders that only exist on a Mac."""
        real = os.path.isdir
        return mock.patch("os.path.isdir", lambda p: str(p).replace("\\", "/") in absolute or real(p))


class ExtraPathDirs(FakeHome):
    def test_macos_looks_in_homebrew_nvm_volta_and_local_bin(self):
        (self.home / ".volta" / "bin").mkdir(parents=True)
        (self.home / ".local" / "bin").mkdir(parents=True)
        (self.home / ".nvm" / "versions" / "node" / "v18.20.0" / "bin").mkdir(parents=True)
        (self.home / ".nvm" / "versions" / "node" / "v20.11.1" / "bin").mkdir(parents=True)
        with self.isdir_with("/opt/homebrew/bin", "/usr/local/bin"):
            dirs = [d.replace("\\", "/") for d in th.extra_path_dirs("darwin", self.home)]
        h = str(self.home).replace("\\", "/")
        self.assertIn("/opt/homebrew/bin", dirs)
        self.assertIn("/usr/local/bin", dirs)
        self.assertIn(h + "/.volta/bin", dirs)
        self.assertIn(h + "/.local/bin", dirs)
        nvm = [d for d in dirs if "/.nvm/" in d]
        self.assertEqual([Path(d).parts[-2] for d in nvm], ["v20.11.1", "v18.20.0"], "the newest Node comes first")

    def test_only_folders_that_exist_are_added(self):
        self.assertEqual([d for d in th.extra_path_dirs("darwin", self.home) if str(self.home) in d], [])

    def test_linux_has_no_homebrew(self):
        with self.isdir_with("/opt/homebrew/bin"):
            self.assertNotIn("/opt/homebrew/bin", [d.replace("\\", "/") for d in th.extra_path_dirs("linux", self.home)])

    def test_the_path_is_extended_not_replaced(self):
        (self.home / ".volta" / "bin").mkdir(parents=True)
        out = th.augmented_path(os.pathsep.join(["/usr/bin", "/bin"]), "darwin", self.home).split(os.pathsep)
        self.assertEqual(out[:2], ["/usr/bin", "/bin"])
        self.assertIn(str(self.home / ".volta" / "bin"), out)
        again = th.augmented_path(os.pathsep.join(out), "darwin", self.home).split(os.pathsep)
        self.assertEqual(again, out, "a folder already on PATH is not added twice")

    def test_stdio_servers_get_a_path_that_can_see_node(self):
        (self.home / ".volta" / "bin").mkdir(parents=True)
        env = th.stdio_env({"FOO": "1", "PATH": "/usr/bin"}, "darwin", self.home)
        self.assertEqual(env["FOO"], "1")
        self.assertTrue(env["PATH"].startswith("/usr/bin"))
        self.assertIn(str(self.home / ".volta" / "bin"), env["PATH"])

    def test_the_host_applies_it_when_it_spawns_a_stdio_server(self):
        from src.tools import mcp_tools
        fake = self.tmp / "fakebin"
        fake.mkdir()
        make_exe(fake / "npx-fake-xyz.cmd" if sys.platform == "win32" else fake / "npx-fake-xyz")
        with mock.patch.object(th, "extra_path_dirs", return_value=[str(fake)]):
            env = mcp_tools._stdio_env({"A": "b"})
            self.assertIn(str(fake), env["PATH"].split(os.pathsep))
            self.assertEqual(env["A"], "b")
            self.assertIsNotNone(mcp_tools._which_tool("npx-fake-xyz"))
        self.assertIsNone(mcp_tools._which_tool("npx-fake-xyz"), "not on the plain PATH")


class InstallHints(unittest.TestCase):
    def test_windows_uses_winget(self):
        cmd = th.install_command("node", "win32")
        self.assertEqual(cmd[:4], ["winget", "install", "-e", "--id"])
        self.assertEqual(cmd[4], "OpenJS.NodeJS.LTS")
        self.assertEqual(th.install_hint("uv", "win32"), ("winget install astral-sh.uv", True))

    def test_macos_installs_with_brew_when_there_is_one(self):
        self.assertEqual(th.install_command("node", "darwin", has_brew=True), ["brew", "install", "node"])
        self.assertEqual(th.install_command("npx", "darwin", has_brew=True), ["brew", "install", "node"])
        self.assertEqual(th.install_command("uv", "darwin", has_brew=True), ["brew", "install", "uv"])
        self.assertEqual(th.install_hint("node", "darwin", has_brew=True), ("brew install node", True))

    def test_macos_without_brew_points_to_the_nodejs_pkg(self):
        self.assertIsNone(th.install_command("node", "darwin", has_brew=False))
        text, is_cmd = th.install_hint("node", "darwin", has_brew=False)
        self.assertFalse(is_cmd)
        self.assertIn("https://nodejs.org", text)
        self.assertIn(".pkg", text)
        self.assertNotIn("winget", text)
        self.assertEqual(th.install_hint("git", "darwin", has_brew=False), ("xcode-select --install", True))

    def test_linux_names_a_package_manager_and_runs_nothing(self):
        self.assertIsNone(th.install_command("node", "linux"))
        text, is_cmd = th.install_hint("node", "linux")
        self.assertTrue(is_cmd)
        self.assertTrue(text.startswith("sudo "), text)
        self.assertNotIn("winget", text)
        with mock.patch.object(shutil, "which", side_effect=lambda n: "/x/dnf" if n == "dnf" else None):
            self.assertEqual(th.install_hint("node", "linux")[0], "sudo dnf install nodejs npm")

    def test_hints_follow_sys_platform_when_none_is_given(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(th.install_hint("node", has_brew=True)[0], "brew install node")
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(th.install_hint("node")[0], "winget install OpenJS.NodeJS.LTS")


class LocalizedCatalogText(unittest.TestCase):
    WIN = "Install Node LTS (`winget install OpenJS.NodeJS.LTS`). Then `winget install astral-sh.uv` and `winget install Foo.Bar`."

    def test_windows_text_is_unchanged(self):
        self.assertEqual(th.localize_hints(self.WIN, "win32"), self.WIN)

    def test_macos_shows_brew_and_leaves_unknown_ids_alone(self):
        out = th.localize_hints(self.WIN, "darwin", has_brew=True)
        self.assertIn("`brew install node`", out)
        self.assertIn("`brew install uv`", out)
        self.assertIn("`winget install Foo.Bar`", out)          # not one of ours: untouched
        self.assertNotIn("OpenJS", out)

    def test_macos_without_brew_never_shows_a_command_that_cannot_run(self):
        out = th.localize_hints(self.WIN, "darwin", has_brew=False)
        self.assertIn("nodejs.org", out)
        self.assertNotIn("winget install OpenJS", out)

    def test_linux_shows_the_package_manager(self):
        out = th.localize_hints(self.WIN, "linux")
        self.assertIn("`sudo ", out)
        self.assertNotIn("OpenJS", out)

    def test_the_shipped_catalog_never_tells_a_mac_or_linux_user_to_use_winget(self):
        for plat in ("darwin", "linux"):
            with mock.patch.object(sys, "platform", plat), mock.patch.object(th, "which", return_value=None):
                for p in mp.load_catalog():
                    for field in ("setup_steps", "notes"):
                        text = p[field]
                        for m in __import__("re").finditer(r"winget install (\S+)", text):
                            self.assertNotIn(m.group(1).strip("`.,)").lower(),
                                             ("openjs.nodejs.lts", "astral-sh.uv", "git.git"), (plat, p["id"], field))

    def test_the_node_presets_get_a_recipe_that_checks_for_node(self):
        by_id = {p["id"]: p for p in mp.load_catalog()}
        for pid in ("remotion", "playwright", "context7", "figma", "meshy"):
            with self.subTest(preset=pid):
                steps = by_id[pid]["setup"]
                self.assertEqual([(s["type"], s.get("tool")) for s in steps],
                                 [("ensure_tool", "node"), ("ensure_entry", None), ("test", None)])
        with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(th, "which", return_value=None):
            text = {p["id"]: p for p in mp.load_catalog()}["remotion"]["notes"]
        self.assertNotIn("winget", text)

    def test_the_finish_later_text_comes_from_the_printing_script_on_this_os(self):
        """scripts/mcp_presets.py steps <id> is what both installers print under 'Finish later'."""
        out = subprocess.run([sys.executable, "-c",
                              "import sys; sys.platform = 'darwin'; sys.path.insert(0, 'scripts'); "
                              "import tool_hints as th; th.which = lambda *a, **k: None; "
                              "import mcp_presets as mp; sys.exit(mp._main(['steps', 'playwright']))"],
                             capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("winget", out.stdout)
        self.assertIn("nodejs.org", out.stdout)


class EnsureToolPerOs(tms.Tmp):
    """The recipe step itself: find it, install it the way this OS does (asking first), or say what to run."""
    NODE = [{"id": "node", "type": "ensure_tool", "tool": "node"}]

    def setUp(self):
        super().setUp()
        self.home = self.tmp / "home"
        self.home.mkdir()

    def installed(self, state, only=("node",)):
        return lambda n: ("/x/" + n) if (n in only and state["have"]) or n == "brew" and state.get("brew") else None

    def test_macos_without_brew_says_where_to_get_the_pkg_and_runs_nothing(self):
        runner = tms.Runner()
        rep = self.run_recipe(self.NODE, which=lambda n: None, run=runner, platform="darwin", allow_system=True)
        self.assertEqual(self.status(rep)["node"], "needs_user")
        msg = rep["needs_user"][0]
        self.assertIn("https://nodejs.org", msg)
        self.assertNotIn("winget", msg)
        self.assertEqual(runner.calls, [])

    def test_macos_with_brew_installs_after_a_yes_and_not_before(self):
        state = {"have": False, "brew": True}
        runner = tms.Runner({"brew": lambda c: (state.update(have=True) or (0, "ok"))})
        which = self.installed(state, only=("node", "npm", "npx"))
        # allowed up front (--allow-system)
        rep = self.run_recipe(self.NODE, which=which, run=runner, platform="darwin", allow_system=True)
        self.assertEqual(self.status(rep)["node"], "done")
        self.assertEqual(runner.calls[0], ["brew", "install", "node"])   # then it is probed: node, npm, npx --version
        self.assertTrue(all(c[-1] == "--version" for c in runner.calls[1:]))
        # asked, and the answer is no: nothing runs
        state["have"] = False
        asked = []
        runner2 = tms.Runner()
        rep = self.run_recipe(self.NODE, which=which, run=runner2, platform="darwin", interactive=True,
                              ask=lambda q: asked.append(q) or False)
        self.assertEqual(self.status(rep)["node"], "needs_user")
        self.assertEqual(runner2.calls, [])
        self.assertIn("brew install node", asked[0])
        self.assertIn("Run: brew install node", rep["needs_user"][0])
        # asked, and the answer is yes
        runner3 = tms.Runner({"brew": lambda c: (state.update(have=True) or (0, "ok"))})
        rep = self.run_recipe(self.NODE, which=which, run=runner3, platform="darwin", interactive=True, ask=lambda q: True)
        self.assertEqual(self.status(rep)["node"], "done")
        self.assertEqual(runner3.calls[0], ["brew", "install", "node"])

    def test_without_permission_a_dry_run_only_plans(self):
        runner = tms.Runner()
        rep = self.run_recipe(self.NODE, which=self.installed({"have": False, "brew": True}), run=runner,
                              platform="darwin", dry_run=True)
        self.assertEqual(self.status(rep)["node"], "planned")
        self.assertIn("brew install node", rep["steps"][0]["message"])
        self.assertEqual(runner.calls, [])

    def test_linux_gives_the_package_manager_hint_even_when_allowed(self):
        runner = tms.Runner()
        rep = self.run_recipe(self.NODE, which=lambda n: None, run=runner, platform="linux", allow_system=True)
        self.assertEqual(self.status(rep)["node"], "needs_user")
        self.assertIn("sudo ", rep["needs_user"][0])
        self.assertNotIn("winget", rep["needs_user"][0])
        self.assertEqual(runner.calls, [])

    def test_windows_still_runs_winget(self):
        state = {"have": False}
        runner = tms.Runner({"winget": lambda c: (state.update(have=True) or (0, "ok"))})
        rep = self.run_recipe(self.NODE, which=lambda n: "/x/node" if state["have"] else None, run=runner,
                              platform="win32", allow_system=True)
        self.assertEqual(self.status(rep)["node"], "done")
        self.assertEqual(runner.calls[0][:2], ["winget", "install"])

    def test_a_node_in_volta_or_nvm_is_found_although_it_is_not_on_path(self):
        for rel in (".volta/bin/node", ".nvm/versions/node/v20.1.0/bin/node", ".local/bin/node"):
            with self.subTest(where=rel):
                home = self.tmp / rel.replace("/", "_")
                make_exe(home / rel)
                for sib in ("npm", "npx"):
                    make_exe((home / rel).with_name(sib))
                with mock.patch("shutil.which", return_value=None):
                    rep = self.run_recipe(self.NODE, which=None, platform="darwin", overrides={"home": str(home)})
                self.assertEqual(self.status(rep)["node"], "skipped", rep)
                self.assertIn(rel.split("/")[-2], rep["steps"][0]["message"].replace("\\", "/"))

    def test_npm_and_npx_are_found_next_to_node(self):
        home = self.tmp / "h2"
        make_exe(home / ".volta" / "bin" / "npx")
        steps = [{"id": "x", "type": "ensure_tool", "tool": "npx"}]
        with mock.patch("shutil.which", return_value=None):
            rep = self.run_recipe(steps, which=None, platform="darwin", overrides={"home": str(home)})
        self.assertEqual(self.status(rep)["x"], "skipped")

    def test_processes_the_runner_starts_see_the_same_folders(self):
        home = self.tmp / "h3"
        make_exe(home / ".volta" / "bin" / "node")
        ctx = ms.Ctx(tms.preset([]), root=self.tmp, platform="darwin", overrides={"home": str(home)},
                     which=None)
        seen = {}

        def fake_run(cmd, **kw):
            seen.update(kw["env"])
            return mock.Mock(returncode=0, stdout="v20.0.0", stderr="")

        with mock.patch("shutil.which", return_value=None), mock.patch.object(ms.subprocess, "run", side_effect=fake_run):
            rc, _ = ctx.run(["node", "--version"])
        self.assertEqual(rc, 0)
        self.assertIn(str(home / ".volta" / "bin"), seen["PATH"].split(os.pathsep))


if __name__ == "__main__":
    unittest.main()
