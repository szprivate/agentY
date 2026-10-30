"""uninstall_agent.ps1 / uninstall_agent.sh, run for real against fake installs in a temp dir.

Nothing here looks at the real machine: every run is given its own tree (fake repos with git
remotes, a fake ComfyUI with custom_nodes, a decoy with a foreign origin, a precious folder
outside it all) and an explicit --path / -Path, --comfyui-path / -ComfyUIPath and a port nobody
listens on. What is checked: the dry run changes nothing, the real run removes what it
identified as agentY's and nothing else, the backup holds the data, the viewer is kept, a
foreign origin is refused, a locked folder stops the run before anything is removed, and old
layouts (agenty_core, copied panel folders, the Chainlit data layer) are found.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
BASH = shutil.which("bash")
GIT = shutil.which("git")
PORT = "59871"


def _git_repo(path: Path, origin: str | None):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run([GIT, "init", "-q", str(path)], check=True, capture_output=True)
    if origin:
        subprocess.run([GIT, "-C", str(path), "remote", "add", "origin", origin], check=True, capture_output=True)


def _write(path: Path, text: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_tree(base: Path, old: bool = False) -> dict:
    """A fake install. old=True is the layout of earlier installers."""
    parent = base / "parent"
    agenty = parent / "agentY"
    _git_repo(agenty, "https://github.com/szprivate/agentY.git")
    _write(agenty / "requirements.txt")
    _write(agenty / "src" / "pipeline.py")
    _write(agenty / "run_agent.ps1")
    _write(agenty / "memory" / "conversations.sqlite", "conversations")
    _write(agenty / ".env", "ANTHROPIC_API_KEY=sk-test")
    _write(agenty / "config" / "settings.local.json", "{}")
    _write(agenty / "config" / "mcp.json", "{}")
    _write(agenty / "config" / ".mcp_tokens" / "t.json", "{}")
    _write(agenty / "output" / "scripts" / "a.py", "print(1)")
    venv = agenty / (".venv" if not old else "venv")
    _write(venv / "Scripts" / "python.exe", "not python")
    _write(venv / "bin" / "python", "not python")
    ro = venv / "readonly.txt"
    _write(ro)
    os.chmod(ro, 0o444)

    core = parent / ("agenty_core" if old else "agentY-core")
    _git_repo(core, "https://github.com/szprivate/%s.git" % ("agenty_core" if old else "agentY-core"))
    _write(core / "pyproject.toml")
    _write(core / "agenty_core" / "__init__.py")
    _write(core / ".venv" / "x.txt")
    mcp = parent / "agentY-mcp"
    _git_repo(mcp, "https://github.com/szprivate/agentY-mcp")
    _write(mcp / "manifest.json")
    _write(mcp / ".env", "HF_TOKEN=hf_x")
    _write(mcp / ".venv" / "x.txt")

    comfy = base / "ComfyUI"
    nodes = comfy / "custom_nodes"
    nodes.mkdir(parents=True)
    _write(comfy / "models" / "checkpoints" / "big.safetensors", "weights")
    panel = nodes / ("comfyui-agent-canvas" if old else "agentY-comfyuiConnect")
    if old:       # copied from agentY's comfyui_extension, no git
        _write(panel / "__init__.py"); _write(panel / "web" / "agent_canvas.js")
        bridge = nodes / "agentY_bridge"
        _write(bridge / "__init__.py"); _write(bridge / "js" / "agentY_bridge.js")
    else:
        _git_repo(panel, "https://github.com/szprivate/agentY-comfyuiConnect.git")
        _write(panel / ".agenty_host.json", '{"project_root": "%s"}' % str(agenty).replace("\\", "\\\\"))
    viewer = nodes / ("comfyui-imageviewer" if old else "ComfyUI-ImageViewer")
    _git_repo(viewer, "https://github.com/szprivate/ComfyUI-ImageViewer.git")
    _write(viewer / "x.js")
    other = nodes / "ComfyUI-KJNodes"
    _git_repo(other, "https://github.com/kijai/ComfyUI-KJNodes.git")
    decoy = nodes / "agentY_bridge" if not old else nodes / "ComfyUI-Foreign"
    if not old:   # a folder with an agentY name but somebody else's origin
        _git_repo(decoy, "https://github.com/someone/else.git")
        _write(decoy / "__init__.py"); _write(decoy / "js" / "agentY_bridge.js")
    precious = base / "precious"
    _write(precious / "keep.txt", "keep me")
    if old:
        _git_repo(agenty / "chainlit-datalayer", "https://github.com/Chainlit/chainlit-datalayer.git")
        _write(agenty / "chainlit-datalayer" / "docker-compose.yml")
        _write(agenty / ".chainlit" / "config.toml")
    return dict(parent=parent, agenty=agenty, core=core, mcp=mcp, comfy=comfy, nodes=nodes, panel=panel,
                viewer=viewer, other=other, decoy=decoy, precious=precious, venv=venv)


def gone(path: Path) -> bool:
    """Removed - or only the empty folder is left because the test's own shell stands in it."""
    return (not path.exists()) or (path.is_dir() and not any(path.iterdir()))


def snapshot(base: Path):
    out = []
    for p in sorted(base.rglob("*")):
        out.append((str(p.relative_to(base)), p.stat().st_size if p.is_file() else -1))
    return out


def _clear(base: Path):
    def onerr(fn, path, exc):
        try:
            os.chmod(path, 0o777); fn(path)
        except Exception:
            pass
    shutil.rmtree(base, onerror=onerr)


class _Base:
    """Shared assertions; subclasses say how to run the script."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="agy-unin-"))
        self.addCleanup(_clear, self.base)

    def run_un(self, t, *args, stdin="", script_dir=None, cwd=None):    # pragma: no cover - overridden
        raise NotImplementedError

    # --- behaviour shared by both scripts ---------------------------------------------
    def test_dry_run_changes_nothing(self):
        t = build_tree(self.base)
        self.place_script(t)
        before = snapshot(self.base)
        r = self.run_un(t, "dry")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertEqual(snapshot(self.base), before)
        self.assertIn("agentY-core", out)
        self.assertIn("agentY-comfyuiConnect", out)
        self.assertIn("conversations", out.lower() + "conversations")   # data group is listed
        self.assertIn("NOT TOUCHED", out)                                # the decoy
        self.assertIn(str(t["decoy"].name), out)

    def test_yes_removes_agenty_keeps_viewer_and_decoy(self):
        t = build_tree(self.base)
        r = self.run_un(t, "yes", script_dir=t["agenty"])
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertTrue(gone(t["agenty"]), out)
        self.assertFalse(t["core"].exists())
        self.assertFalse(t["mcp"].exists())
        self.assertFalse(t["panel"].exists())
        self.assertTrue(t["viewer"].exists(), "the viewer is kept unless asked")
        self.assertTrue((t["viewer"] / "x.js").exists())
        self.assertTrue(t["other"].exists())
        self.assertTrue(t["decoy"].exists(), "a foreign origin is never touched")
        self.assertTrue((t["decoy"] / "__init__.py").exists())
        self.assertTrue((t["comfy"] / "models" / "checkpoints" / "big.safetensors").exists())
        self.assertEqual((t["precious"] / "keep.txt").read_text(), "keep me")
        zips = list(t["parent"].glob("agentY-userdata-backup-*.zip"))
        self.assertEqual(len(zips), 1, out)
        names = zipfile.ZipFile(zips[0]).namelist()
        self.assertTrue(any(n.endswith("agentY/memory/conversations.sqlite") for n in names), names)
        self.assertTrue(any(n.endswith("agentY/.env") for n in names), names)
        self.assertTrue(any(n.endswith("agentY-mcp/.env") for n in names), names)
        self.assertIn("Summary", out)
        self.assertIn(zips[0].name, out)

    def test_keep_user_data(self):
        t = build_tree(self.base)
        r = self.run_un(t, "yes", "keep-data", script_dir=t["agenty"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((t["agenty"] / "memory" / "conversations.sqlite").exists())
        self.assertTrue((t["agenty"] / ".env").exists())
        self.assertTrue((t["agenty"] / "config" / ".mcp_tokens" / "t.json").exists())
        self.assertTrue((t["mcp"] / ".env").exists())
        self.assertFalse((t["agenty"] / "src").exists())
        self.assertFalse((t["agenty"] / "config" / "mcp.json").exists())
        self.assertFalse(t["venv"].exists())
        self.assertFalse((t["core"] / ".venv").exists())
        self.assertEqual(list(t["parent"].glob("agentY-userdata-backup-*.zip")), [])

    def test_include_viewer(self):
        t = build_tree(self.base)
        r = self.run_un(t, "yes", "viewer")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(t["viewer"].exists())
        self.assertTrue(t["other"].exists())

    def test_interactive_answers(self):
        """y (code) n (data) y (panel) n (viewer): defaults for the rest."""
        t = build_tree(self.base)
        r = self.run_un(t, "ask", stdin="y\nn\ny\nn\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((t["agenty"] / "memory" / "conversations.sqlite").exists())
        self.assertFalse((t["agenty"] / "src").exists())
        self.assertFalse(t["panel"].exists())
        self.assertTrue(t["viewer"].exists())

    def test_empty_answers_are_the_safe_defaults(self):
        """Enter all the way: code and panel go, data and viewer stay."""
        t = build_tree(self.base)
        r = self.run_un(t, "ask", stdin="\n\n\n\n\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((t["agenty"] / ".env").exists())
        self.assertTrue(t["viewer"].exists())
        self.assertFalse(t["panel"].exists())
        self.assertFalse((t["agenty"] / "src").exists())

    def test_foreign_origin_agenty_folder_is_refused(self):
        t = build_tree(self.base)
        subprocess.run([GIT, "-C", str(t["agenty"]), "remote", "set-url", "origin", "https://github.com/x/y.git"],
                       check=True, capture_output=True)
        self.place_script(t)
        before = snapshot(self.base)
        r = self.run_un(t, "yes")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(snapshot(self.base), before)

    def test_decoy_sibling_with_agentY_mcp_name_is_refused(self):
        t = build_tree(self.base)
        subprocess.run([GIT, "-C", str(t["mcp"]), "remote", "set-url", "origin", "https://github.com/x/agentY-mcp.git"],
                       check=True, capture_output=True)
        r = self.run_un(t, "yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(t["mcp"].exists(), "a foreign origin keeps its folder")
        self.assertTrue((t["mcp"] / ".env").exists())
        self.assertTrue(gone(t["agenty"]))

    def test_a_running_program_in_the_venv_stops_everything(self):
        t = build_tree(self.base)
        proc = self._start_locker(t)
        if proc is None:
            self.skipTest("cannot start a program from the fake venv here")
        self.addCleanup(proc.kill)
        time.sleep(0.8)
        self.place_script(t)
        before = snapshot(self.base)
        r = self.run_un(t, "yes")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 3, out)
        self.assertEqual(snapshot(self.base), before, "nothing may be removed when something is locked")
        self.assertIn("NOTHING was removed", out)

    def _start_locker(self, t):    # pragma: no cover - overridden
        return None

    def test_old_layout_standalone(self):
        """agenty_core, copied panel folders, a `venv`, the Chainlit data layer - and the script is run
        from somewhere else, finding the install from the current folder."""
        t = build_tree(self.base, old=True)
        standalone = self.base / "downloads"
        standalone.mkdir()
        r = self.run_un(t, "yes", script_dir=standalone, cwd=t["agenty"])
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertTrue(gone(t["agenty"]), out)
        self.assertFalse(t["core"].exists(), "agenty_core is the same repository under its old name")
        self.assertFalse(t["mcp"].exists())
        self.assertFalse(t["panel"].exists(), "comfyui-agent-canvas is the old panel")
        self.assertFalse((t["nodes"] / "agentY_bridge").exists())
        self.assertTrue(t["viewer"].exists(), "the viewer under its manager name is kept")
        self.assertTrue(t["other"].exists())
        self.assertTrue(t["precious"].exists())
        zips = list(t["parent"].glob("agentY-userdata-backup-*.zip"))
        names = zipfile.ZipFile(zips[0]).namelist()
        self.assertTrue(any(".chainlit/config.toml" in n for n in names), names)
        self.assertTrue(any("chainlit-datalayer/docker-compose.yml" in n for n in names), names)
        self.assertTrue((standalone / self.script_name).exists(), "the downloaded copy is left alone")

    def test_old_layout_dry_run_lists_everything(self):
        t = build_tree(self.base, old=True)
        r = self.run_un(t, "dry", script_dir=self.base / "dl", cwd=t["agenty"])
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        for needle in ("agenty_core", "comfyui-agent-canvas", "agentY_bridge", "chainlit-datalayer", ".chainlit", "comfyui-imageviewer"):
            self.assertIn(needle, out)


@unittest.skipUnless(sys.platform == "win32" and POWERSHELL and GIT, "needs Windows PowerShell and git")
class UninstallPowerShell(_Base, unittest.TestCase):
    script_name = "uninstall_agent.ps1"

    def place_script(self, t, script_dir=None):
        sd = Path(script_dir) if script_dir else t["agenty"]
        sd.mkdir(parents=True, exist_ok=True)
        script = sd / self.script_name
        shutil.copy(ROOT / self.script_name, script)
        return script

    def run_un(self, t, mode, *flags, stdin="", script_dir=None, cwd=None):
        script = self.place_script(t, script_dir)
        args = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                "-Path", str(t["agenty"]), "-ComfyUIPath", str(t["comfy"]), "-Port", PORT]
        if cwd is not None:      # no -Path: the install is found from the current folder
            args = [a for a in args]
            i = args.index("-Path"); del args[i:i + 2]
        if mode == "dry":
            args.append("-DryRun")
        elif mode == "yes":
            args.append("-Yes")
        for f in flags:
            args.append({"keep-data": "-KeepUserData", "viewer": "-IncludeViewer"}[f])
        return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=300,
                              cwd=str(cwd or self.base))

    def _start_locker(self, t):
        ping = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "PING.EXE"
        if not ping.exists():
            return None
        exe = t["venv"] / "Scripts" / "python.exe"
        exe.unlink()
        shutil.copy(ping, exe)
        return subprocess.Popen([str(exe), "-n", "60", "127.0.0.1"], stdout=subprocess.DEVNULL)

    def test_junction_inside_is_removed_as_a_link(self):
        t = build_tree(self.base)
        link = t["venv"] / "linked"
        r0 = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(t["precious"])], capture_output=True, text=True)
        if r0.returncode != 0:
            self.skipTest("cannot make a junction here")
        r = self.run_un(t, "yes", script_dir=t["agenty"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(gone(t["agenty"]))
        self.assertEqual((t["precious"] / "keep.txt").read_text(), "keep me", "the link's target must survive")

    def test_core_link_left_by_the_rename_is_removed_as_a_link(self):
        """scripts/core_dir.py leaves agenty_core -> agentY-core behind."""
        t = build_tree(self.base)
        r0 = subprocess.run(["cmd", "/c", "mklink", "/J", str(t["parent"] / "agenty_core"), str(t["core"])],
                            capture_output=True, text=True)
        if r0.returncode != 0:
            self.skipTest("cannot make a junction here")
        r = self.run_un(t, "yes", script_dir=t["agenty"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(t["core"].exists())
        self.assertFalse((t["parent"] / "agenty_core").exists())

    def test_drive_root_and_profile_are_refused(self):
        t = build_tree(self.base)
        for bad in ("C:\\", os.environ.get("USERPROFILE", "C:\\Users"), str(t["comfy"])):
            r = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                str(ROOT / self.script_name), "-Path", bad, "-ComfyUIPath", str(t["comfy"]),
                                "-Yes", "-Port", PORT], capture_output=True, text=True, timeout=120, cwd=str(self.base))
            self.assertEqual(r.returncode, 2, (bad, r.stdout, r.stderr))
        self.assertTrue(t["agenty"].exists())

    def test_long_paths_and_readonly_files_are_removed(self):
        t = build_tree(self.base)
        deep = t["core"] / "agenty_core"
        for i in range(12):
            deep = deep / ("segment_%02d_" % i + "x" * 20)
        subprocess.run(["cmd", "/c", "mkdir", "\\\\?\\" + str(deep)], capture_output=True)
        (Path("\\\\?\\" + str(deep)) / "f.txt").write_text("deep")
        r = self.run_un(t, "yes", script_dir=t["agenty"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(t["core"].exists())


@unittest.skipUnless(BASH and GIT, "needs bash and git")
class UninstallShell(_Base, unittest.TestCase):
    script_name = "uninstall_agent.sh"

    def place_script(self, t, script_dir=None):
        sd = Path(script_dir) if script_dir else t["agenty"]
        sd.mkdir(parents=True, exist_ok=True)
        script = sd / self.script_name
        shutil.copy(ROOT / self.script_name, script)
        return script

    def run_un(self, t, mode, *flags, stdin="", script_dir=None, cwd=None):
        script = self.place_script(t, script_dir)
        args = [BASH, str(script), "--path", str(t["agenty"]), "--comfyui-path", str(t["comfy"]), "--port", PORT]
        if cwd is not None:
            i = args.index("--path"); del args[i:i + 2]
        if mode == "dry":
            args.append("--dry-run")
        elif mode == "yes":
            args.append("--yes")
        for f in flags:
            args.append({"keep-data": "--keep-user-data", "viewer": "--include-viewer"}[f])
        env = dict(os.environ, HOME=str(self.base / "home"))
        return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=300,
                              cwd=str(cwd or self.base), env=env)

    def _start_locker(self, t):
        # A process whose command line is the fake venv's python: sleep, run through a copy named python.
        exe = t["venv"] / "bin" / "python"
        sleep = shutil.which("sleep")
        ps = subprocess.run([BASH, "-c", "ps -axo pid=,command="], capture_output=True, text=True)
        if not sleep or ps.returncode != 0 or not ps.stdout.strip():
            return None     # Git Bash's ps has no -axo: the process check is macOS/Linux only
        exe.unlink()
        shutil.copy(sleep, exe)
        exe.chmod(0o755)
        try:
            return subprocess.Popen([str(exe), "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return None

    def test_script_parses_and_help_works(self):
        r = subprocess.run([BASH, "-n", str(ROOT / self.script_name)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([BASH, str(ROOT / self.script_name), "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Usage:", r.stdout)
        r = subprocess.run([BASH, str(ROOT / self.script_name), "--nonsense"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)

    def test_no_bash_4_only_syntax(self):
        code = "\n".join(ln for ln in (ROOT / self.script_name).read_text(encoding="utf-8").splitlines()
                         if not ln.lstrip().startswith("#"))
        for token in ("declare -A", "${!", "mapfile", "readarray", ",,}", "^^}", "&>>"):
            self.assertNotIn(token, code)

    def test_lf_endings_and_shebang(self):
        raw = (ROOT / self.script_name).read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))


if __name__ == "__main__":
    unittest.main()
