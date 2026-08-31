"""agentY runs on Windows and on a Mac, and neither may quietly break the other.

Every test here pins BOTH answers, not just the new one. A platform switch is the
easiest kind of code to half-fix: the machine you are sitting at keeps working, so
nothing tells you the other branch now returns the wrong thing. These assertions
fail on the Windows box that made the change.
"""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _fake_torch(*, cuda: bool, mps: bool, has_mps_attr: bool = True):
    """A stand-in for torch reporting a given pair of accelerators."""
    t = types.SimpleNamespace()
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.backends = types.SimpleNamespace()
    if has_mps_attr:
        t.backends.mps = types.SimpleNamespace(is_available=lambda: mps)
    return t


class DeviceSelection(unittest.TestCase):
    """Which accelerator SAM3 grounding runs on."""

    def setUp(self):
        from src.utils import image_locate
        self.mod = image_locate
        self._real_cfg = image_locate._cfg
        image_locate._cfg = lambda *a, **k: "auto"

    def tearDown(self):
        self.mod._cfg = self._real_cfg

    def _device_with(self, torch_mod):
        real = __import__

        def fake(name, *a, **k):
            if name == "torch":
                return torch_mod
            return real(name, *a, **k)

        import builtins
        builtins.__import__ = fake
        try:
            return self.mod._device()
        finally:
            builtins.__import__ = real

    def test_cuda_wins_when_present(self):
        # The Windows answer, and the one a Mac change is most likely to break.
        self.assertEqual(self._device_with(_fake_torch(cuda=True, mps=False)), "cuda")

    def test_cuda_wins_even_if_mps_also_reports_available(self):
        # Not a real machine today, but the preference must be stated rather than
        # left to whichever branch happens to come first.
        self.assertEqual(self._device_with(_fake_torch(cuda=True, mps=True)), "cuda")

    def test_mps_when_only_mps(self):
        self.assertEqual(self._device_with(_fake_torch(cuda=False, mps=True)), "mps")

    def test_cpu_when_neither(self):
        self.assertEqual(self._device_with(_fake_torch(cuda=False, mps=False)), "cpu")

    def test_cpu_on_torch_without_mps_backend(self):
        # torch.backends.mps arrived in 1.12; an older build must not AttributeError.
        self.assertEqual(
            self._device_with(_fake_torch(cuda=False, mps=False, has_mps_attr=False)),
            "cpu")

    def test_cpu_when_torch_is_missing_entirely(self):
        broken = types.SimpleNamespace()
        self.assertEqual(self._device_with(broken), "cpu")

    def test_explicit_setting_is_honoured(self):
        for want in ("cpu", "cuda", "mps"):
            self.mod._cfg = lambda *a, _w=want, **k: _w
            self.assertEqual(self.mod._device(), want)

    def test_unknown_setting_falls_through_to_detection(self):
        # A typo must not pin the run to a device that does not exist.
        self.mod._cfg = lambda *a, **k: "metal"
        self.assertEqual(self._device_with(_fake_torch(cuda=False, mps=False)), "cpu")


class RequirementMarkers(unittest.TestCase):
    """requirements.txt must be installable on each platform it claims to support."""

    @staticmethod
    def _env(platform):
        return {
            "sys_platform": platform,
            "os_name": "nt" if platform == "win32" else "posix",
            "platform_system": {"win32": "Windows", "darwin": "Darwin",
                                "linux": "Linux"}[platform],
            "platform_machine": "arm64" if platform == "darwin" else "x86_64",
            "python_version": "3.12", "python_full_version": "3.12.0",
            "implementation_name": "cpython", "implementation_version": "3.12.0",
            "platform_release": "", "platform_version": "", "extra": "",
        }

    def _resolve(self, platform):
        """The distribution names requirements.txt asks for on *platform*."""
        from packaging.requirements import Requirement
        names = set()
        for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-e "):
                continue
            req = Requirement(line)
            if req.marker is None or req.marker.evaluate(self._env(platform)):
                names.add(req.name.lower())
        return names

    def test_triton_is_linux_only(self):
        # Upstream triton publishes manylinux wheels and nothing else — no macOS
        # wheel, no sdist. Marked "!= win32" it made the whole install fail on a Mac.
        self.assertIn("triton", self._resolve("linux"))
        self.assertNotIn("triton", self._resolve("darwin"))
        self.assertNotIn("triton", self._resolve("win32"))

    def test_triton_windows_is_windows_only(self):
        self.assertIn("triton-windows", self._resolve("win32"))
        self.assertNotIn("triton-windows", self._resolve("darwin"))
        self.assertNotIn("triton-windows", self._resolve("linux"))

    def test_exactly_one_triton_per_platform(self):
        # Two of them at once would fight over the same `triton` module name.
        for platform in ("win32", "linux"):
            got = {n for n in self._resolve(platform) if "triton" in n}
            self.assertEqual(len(got), 1, f"{platform}: {got}")
        self.assertEqual({n for n in self._resolve("darwin") if "triton" in n}, set())

    def test_every_other_requirement_is_asked_for_on_mac(self):
        """Only triton may be Windows/Linux-only.

        A marker that silently drops a package on one platform costs a feature
        there, and the loss shows up as behaviour rather than an install error —
        which is exactly how the triton line went unnoticed in reverse.
        """
        win, mac = self._resolve("win32"), self._resolve("darwin")
        self.assertEqual(win - mac, {"triton-windows"})
        self.assertEqual(mac - win, set())


class GpuAdvice(unittest.TestCase):
    """`check_env.py --gpu` must not send a Mac after a wheel that doesn't exist."""

    @staticmethod
    def _gpu_line(torch_mod, platform):
        sys.path.insert(0, str(ROOT / "scripts"))
        import check_env
        return check_env.gpu_line(torch_mod, platform)

    @staticmethod
    def _torch(*, cuda, mps, has_mps_attr=True):
        t = _fake_torch(cuda=cuda, mps=mps, has_mps_attr=has_mps_attr)
        t.__version__ = "2.11.0"
        t.cuda.get_device_name = lambda _i: "NVIDIA GeForce RTX 5090"
        return t

    def test_cuda_is_reported_on_windows(self):
        line = self._gpu_line(self._torch(cuda=True, mps=False), "win32")
        self.assertIn("sees CUDA", line)
        self.assertIn("RTX 5090", line)

    def test_mps_is_reported_on_a_mac(self):
        line = self._gpu_line(self._torch(cuda=False, mps=True), "darwin")
        # "sees", not just "Metal (MPS)": the failure message on the same platform
        # contains that phrase too ("reports no Metal (MPS) support"), so the loose
        # substring passed even when a working GPU was reported as broken.
        self.assertIn("sees Metal (MPS)", line)
        self.assertNotIn("reports no", line)
        self.assertNotIn("reinstall", line.lower())
        self.assertNotIn("CUDA", line)

    def test_a_mac_without_mps_is_not_sent_to_the_cuda_index(self):
        # There is no macOS build of torch on the CUDA index. The old text told
        # people to install one, which is advice that cannot be followed.
        line = self._gpu_line(self._torch(cuda=False, mps=False), "darwin")
        self.assertNotIn("cu128", line)
        self.assertNotIn("Scripts", line)
        self.assertIn("force-reinstall", line)

    def test_windows_without_cuda_still_gets_the_cuda_index(self):
        line = self._gpu_line(self._torch(cuda=False, mps=False), "win32")
        self.assertIn("cu128", line)
        self.assertIn("CPU-only", line)

    def test_an_old_torch_without_the_mps_backend_does_not_raise(self):
        line = self._gpu_line(
            self._torch(cuda=False, mps=False, has_mps_attr=False), "win32")
        self.assertIn("CPU-only", line)


class AgentServerPort(unittest.TestCase):
    """Which port the chat host serves on, per platform.

    macOS does not leave 5000 free: ControlCenter's AirPlay Receiver holds *:5000
    on a stock Mac. What makes it worth a platform switch rather than a note in
    the README is that AirPlay ANSWERS — a 403 from `Server: AirTunes/...`, not a
    refused connection — so the sidebar reports the host as down while the host is
    running perfectly well, and nothing in the obvious places looks wrong.

    Both answers are pinned here, so moving either is a deliberate act.
    """

    @staticmethod
    def _settings():
        from src.utils import settings
        return settings

    def test_the_default_port_differs_only_on_macos(self):
        st = self._settings()
        self.assertEqual(st.default_agent_port("darwin"), 5001)
        self.assertEqual(st.default_agent_port("win32"), 5000)
        self.assertEqual(st.default_agent_port("linux"), 5000)

    def test_the_shipped_defaults_state_both_numbers(self):
        """Read from the committed file, not from the constants: the TOML is what
        an install actually gets, and the two could drift apart silently."""
        st = self._settings()
        defaults = st.load_defaults()
        self.assertEqual(defaults.get("agent_server_url"), "http://127.0.0.1:5000")
        self.assertEqual(defaults.get("agent_server_url_macos"), "http://127.0.0.1:5001")

    def test_macos_takes_the_macos_default(self):
        st = self._settings()
        url = st.agent_server_url("darwin",
                                  defaults={"agent_server_url": "http://127.0.0.1:5000",
                                            "agent_server_url_macos": "http://127.0.0.1:5001"},
                                  local={})
        self.assertEqual(url, "http://127.0.0.1:5001")

    def test_windows_ignores_the_macos_default(self):
        st = self._settings()
        url = st.agent_server_url("win32",
                                  defaults={"agent_server_url": "http://127.0.0.1:5000",
                                            "agent_server_url_macos": "http://127.0.0.1:5001"},
                                  local={})
        self.assertEqual(url, "http://127.0.0.1:5000")

    def test_a_local_override_wins_on_a_mac_too(self):
        """The trap this avoids: a Mac user picks a port in the settings UI, which
        writes agent_server_url to settings.local.json, and a platform default that
        outranked it would silently ignore the choice."""
        st = self._settings()
        url = st.agent_server_url("darwin",
                                  defaults={"agent_server_url": "http://127.0.0.1:5000",
                                            "agent_server_url_macos": "http://127.0.0.1:5001"},
                                  local={"agent_server_url": "http://127.0.0.1:6000"})
        self.assertEqual(url, "http://127.0.0.1:6000")

    def test_a_local_override_wins_on_windows(self):
        st = self._settings()
        url = st.agent_server_url("win32",
                                  defaults={"agent_server_url": "http://127.0.0.1:5000"},
                                  local={"agent_server_url": "http://127.0.0.1:6000"})
        self.assertEqual(url, "http://127.0.0.1:6000")

    def test_empty_files_still_give_a_usable_address(self):
        st = self._settings()
        self.assertEqual(st.agent_server_url("darwin", defaults={}, local={}),
                         "http://127.0.0.1:5001")
        self.assertEqual(st.agent_server_url("win32", defaults={}, local={}),
                         "http://127.0.0.1:5000")

    def test_a_blank_setting_is_not_an_address(self):
        """A key present but empty is the shape a half-edited config takes."""
        st = self._settings()
        self.assertEqual(
            st.agent_server_url("darwin", defaults={"agent_server_url_macos": ""},
                                local={"agent_server_url": "   "}),
            "http://127.0.0.1:5001")

    def test_the_server_derives_host_and_port_from_it(self):
        from src.agenty_ui_server import _agent_server_url_defaults
        host, port = _agent_server_url_defaults()
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, self._settings().default_agent_port())


class TheSidebarIsToldThePort(unittest.TestCase):
    """The panel is a browser tab. It cannot read a settings file, a --port flag or
    an environment variable, so it has to be told which port to call — and what it
    used to do instead was assume 5000, the one number a Mac cannot use.

    The host registers the port it ACTUALLY BOUND rather than one re-read from
    settings, because `--port` on the launcher appears in no file at all: reading
    the config back would confidently report the wrong number.
    """

    def _capture_registration(self, port):
        """Run _register_with_comfyui against a stub standing in for ComfyUI, and
        return the JSON body it posted."""
        import http.server
        import json as _json
        import socketserver
        import threading

        received = {}

        class Stub(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                received.update(_json.loads(self.rfile.read(n) or b"{}"))
                received["_path"] = self.path
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *a):
                pass

        srv = socketserver.TCPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        from src.utils import settings as st
        from src.utils import agentY_server as A

        previous = st._cache
        st._cache = dict(st.load_settings())
        st._cache["comfyui_url"] = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            A._register_with_comfyui(port)
            for t in threading.enumerate():
                if t.name == "agentY-register-host":
                    t.join(timeout=10)
        finally:
            st._cache = previous
            srv.shutdown()
            srv.server_close()
        return received

    def test_the_bound_port_reaches_the_extension(self):
        body = self._capture_registration(5001)
        self.assertEqual(body.get("_path"), "/agent/register_host")
        self.assertEqual(body.get("agent_server_port"), 5001)
        self.assertTrue(body.get("project_root"))
        self.assertTrue(body.get("run_script"))

    def test_an_unusual_port_is_reported_as_it_is(self):
        """A --port nobody could have predicted is exactly the case this exists
        for, so it must not be normalised into anything tidier."""
        self.assertEqual(self._capture_registration(6123).get("agent_server_port"), 6123)

    def test_no_port_is_sent_rather_than_a_wrong_one(self):
        """A caller that does not know the port must leave the key out entirely.
        Sending 0, or a guess, would overwrite a good recorded value on the other
        side with something that cannot be dialled."""
        self.assertNotIn("agent_server_port", self._capture_registration(0))

    def test_the_call_site_passes_what_was_bound(self):
        """Read from the source because the alternative — actually starting the
        host — is not a unit test. `port` here is start_agentY_server's own
        argument; if this ever became a fresh settings read, --port would stop
        reaching the panel and nothing else would notice.
        """
        src = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        body = src[src.index("def start_agentY_server("):][:3000]
        self.assertIn("_register_with_comfyui(port)", body)

    def test_the_signature_no_longer_hardcodes_5000(self):
        src = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        self.assertIn("port: int | None = None", src)
        self.assertNotIn('port: int = 5000', src)


class HiddenPthFiles(unittest.TestCase):
    """The macOS file flag that makes a correct install unimportable.

    Since 3.11, site.addpackage() skips any .pth file carrying UF_HIDDEN and says
    nothing. agenty_core is installed editable, so it reaches the interpreter
    through exactly one .pth file: flag that file and the shared tool layer is
    gone at import time while the package, its dist-info and its finder all sit
    correctly on disk. It cost an afternoon once. It should cost a line now.
    """

    @staticmethod
    def _check_env():
        sys.path.insert(0, str(ROOT / "scripts"))
        import check_env
        return check_env

    def _venv_with_pth(self, hidden):
        """A directory holding one .pth file, flagged or not. Returns its path."""
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        pth = Path(d) / "__editable__.agenty_core-0.1.0.pth"
        pth.write_text("import nothing\n", encoding="utf-8")
        if hidden:
            import subprocess
            if subprocess.run(["chflags", "hidden", str(pth)]).returncode != 0:
                self.skipTest("chflags could not set the flag")
        return d

    def test_no_flag_means_nothing_to_report(self):
        d = self._venv_with_pth(hidden=False)
        self.assertEqual(self._check_env().hidden_pth_files([d], "darwin"), [])

    @unittest.skipIf(sys.platform != "darwin", "UF_HIDDEN needs a real chflags")
    def test_a_flagged_pth_is_found(self):
        d = self._venv_with_pth(hidden=True)
        found = self._check_env().hidden_pth_files([d], "darwin")
        self.assertEqual([Path(f).name for f in found],
                         ["__editable__.agenty_core-0.1.0.pth"])

    @unittest.skipIf(sys.platform != "darwin", "UF_HIDDEN needs a real chflags")
    def test_windows_never_reports_a_flag_it_cannot_have(self):
        # The same flagged file, asked about as Windows. st_flags does not exist
        # there, so the honest answer is "none" - anything else is a fault
        # invented for a platform that cannot suffer it.
        d = self._venv_with_pth(hidden=True)
        self.assertEqual(self._check_env().hidden_pth_files([d], "win32"), [])

    def test_a_missing_directory_is_not_an_error(self):
        self.assertEqual(
            self._check_env().hidden_pth_files(["/nonexistent/site-packages"], "darwin"), [])

    def test_silence_when_there_is_nothing_to_say(self):
        self.assertEqual(self._check_env().hidden_pth_advice([]), "")

    def test_the_advice_names_the_fix_that_works(self):
        text = self._check_env().hidden_pth_advice(
            ["/x/lib/python3.12/site-packages/__editable__.agenty_core-0.1.0.pth"])
        self.assertIn("chflags", text)
        self.assertIn("nohidden", text)
        self.assertIn("__editable__.agenty_core-0.1.0.pth", text)

    def test_the_advice_does_not_send_you_to_reinstall(self):
        """The whole point. `uv pip install` finds the requirement satisfied, does
        nothing, and leaves the package visible on disk and unimportable - so the
        one instruction that must NOT appear here is the one you would try first.
        """
        text = self._check_env().hidden_pth_advice(["/x/site-packages/a.pth"])
        self.assertNotIn("uv pip install", text)
        self.assertNotIn("requirements.txt", text)


class HiddenPthIsClearedByTheLaunchers(unittest.TestCase):
    """Detecting it is second best. Both entry points must clear it unasked."""

    def test_the_installer_clears_it_after_installing(self):
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        self.assertIn("unhide_pth()", sh)
        # Called, not merely defined - and after the install that writes the .pth.
        self.assertIn('unhide_pth "$venv"', sh)
        self.assertLess(sh.index("uv pip install --python"), sh.index('unhide_pth "$venv"'))

    def test_the_runner_clears_it_before_activating(self):
        """The flag can arrive after the install, so the runner cannot assume the
        installer already dealt with it."""
        sh = (ROOT / "run_agent.sh").read_text(encoding="utf-8")
        self.assertIn("chflags nohidden", sh)
        self.assertLess(sh.index("chflags nohidden"),
                        sh.index('. "$PROJECT_ROOT/.venv/bin/activate"'))

    def test_both_guard_on_chflags_existing(self):
        """Linux has no chflags. An unguarded call is a command-not-found on every
        start of the platform this repo also runs on."""
        for name in ("install_agent.sh", "run_agent.sh"):
            with self.subTest(script=name):
                sh = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("command -v chflags", sh)


class LauncherName(unittest.TestCase):
    """The host tells the sidebar which script restarts it."""

    def test_named_by_platform(self):
        from src.utils.agentY_server import _launcher_name
        self.assertEqual(_launcher_name("win32"), "run_agent.ps1")
        self.assertEqual(_launcher_name("darwin"), "run_agent.sh")
        self.assertEqual(_launcher_name("linux"), "run_agent.sh")

    def test_defaults_to_this_platform(self):
        from src.utils.agentY_server import _launcher_name
        expected = "run_agent.ps1" if sys.platform == "win32" else "run_agent.sh"
        self.assertEqual(_launcher_name(), expected)

    def test_both_launchers_are_shipped(self):
        for name in ("run_agent.ps1", "run_agent.sh",
                     "install_agent.ps1", "install_agent.sh"):
            self.assertTrue((ROOT / name).is_file(), f"{name} is missing")


class LineEndings(unittest.TestCase):
    """A CRLF shell script is not a style problem; it does not run at all.

    The kernel reads the shebang literally, so `#!/usr/bin/env bash\\r` sends it
    looking for an interpreter named "bash\\r" — and the error names the script,
    not the carriage return. This repo is developed with core.autocrlf=true, so
    without .gitattributes that is what a Mac would check out.
    """

    def test_gitattributes_pins_both_families(self):
        text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", text)
        self.assertIn("*.ps1 text eol=crlf", text)

    def test_shell_scripts_have_no_carriage_returns(self):
        for name in ("run_agent.sh", "install_agent.sh"):
            data = (ROOT / name).read_bytes()
            self.assertNotIn(b"\r", data, f"{name} contains CR bytes")

    def test_shell_scripts_start_with_a_shebang(self):
        for name in ("run_agent.sh", "install_agent.sh"):
            first = (ROOT / name).read_bytes().split(b"\n", 1)[0]
            self.assertEqual(first, b"#!/usr/bin/env bash", name)

    def test_shell_scripts_are_executable_in_git(self):
        """Mode 100755 in the index, or `./run_agent.sh` is Permission denied.

        Windows git reports every file as 0644 (core.filemode=false), so the bit has
        to be set in the index deliberately — `git update-index --chmod=+x`. It
        cannot be observed from the working tree on this platform, which is why it
        is checked against git itself.
        """
        import subprocess
        out = subprocess.run(["git", "ls-files", "-s", "run_agent.sh", "install_agent.sh"],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.strip():
            self.skipTest("not a git checkout")
        for line in out.stdout.strip().splitlines():
            mode, _rest = line.split(" ", 1)
            self.assertEqual(mode, "100755", f"not executable in git: {line}")


if __name__ == "__main__":
    unittest.main()
