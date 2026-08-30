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
