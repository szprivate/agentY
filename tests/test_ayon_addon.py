"""The agentY AYON add-on (ayon_addon/): its logic against fake ComfyUI / agentY servers, its launch hook and
tray entry against stand-ins for the AYON launcher modules, and the package build."""

from __future__ import annotations

import http.server
import importlib
import json
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "ayon_addon"
sys.path.insert(0, str(ADDON / "client"))




def _ayon_stubs():
    """Stand-ins for the AYON launcher modules the add-on imports."""
    core = types.ModuleType("ayon_core")
    core_addon = types.ModuleType("ayon_core.addon")

    class AYONAddon:
        def __init__(self, settings=None):
            self.log = mock.Mock()
            self.initialize(settings or {})

    class ITrayAction:
        def execute_in_main_thread(self, cb):
            cb()

        def show_tray_message(self, title, message):
            self.tray_messages = getattr(self, "tray_messages", []) + [message]
    core_addon.AYONAddon, core_addon.ITrayAction = AYONAddon, ITrayAction
    apps = types.ModuleType("ayon_applications")

    class PreLaunchHook:
        def __init__(self, launch_context):
            self.launch_context = launch_context
            self.log = mock.Mock()

        @property
        def data(self):
            return self.launch_context.data

    class ApplicationLaunchFailed(Exception):
        pass
    apps.PreLaunchHook, apps.ApplicationLaunchFailed = PreLaunchHook, ApplicationLaunchFailed
    apps.LaunchTypes = types.SimpleNamespace(local="local")
    return {"ayon_core": core, "ayon_core.addon": core_addon, "ayon_applications": apps}


# The package imports ayon_core; import it once against the stand-ins (the launcher side tests patch them
# in again for the hook, which imports ayon_applications when it is loaded).
_stubs = _ayon_stubs()
sys.modules.update(_stubs)
try:
    from ayon_agenty import addon as addon_mod, lib  # noqa: E402
finally:
    for _name in _stubs:
        sys.modules.pop(_name, None)

AgentYAddon = addon_mod.AgentYAddon


class _Fake(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):  # noqa: N802
        body = self.routes.get(self.path.split("?")[0])
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _serve(routes: dict) -> tuple[http.server.HTTPServer, int]:
    handler = type("H", (_Fake,), {"routes": routes})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def _free_port() -> int:
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
    port = srv.server_address[1]
    srv.server_close()
    return port


def _comfy(ports, static_port=None, remotes=()):
    return {"local_settings": {"http_server_port": static_port or _free_port(),
                               "local_setting_list": [{"name": f"p{p}", "comfy_launch_port": p} for p in ports]},
            "remote_settings": {"remote_setting_list": [{"comfy_web_adress": r} for r in remotes]}}


STATS = {"system": {"comfyui_version": "0.3.1"}, "devices": []}


class Settings(unittest.TestCase):
    def test_normalize_fills_defaults(self):
        s = lib.normalize({"agenty_root": {"windows": "D:\\AI\\agentY"}, "host_args": ["", "-NoUpdate"]})
        self.assertEqual(s["agent_server_url"], "http://127.0.0.1:5000")
        self.assertEqual(s["agenty_root"], {"windows": "D:\\AI\\agentY", "linux": "", "darwin": ""})
        self.assertEqual(s["host_args"], ["-NoUpdate"])
        self.assertEqual(lib.normalize(None)["comfyui_port_busy"], "block")

    def test_server_defaults_match_the_client(self):
        # server/settings.py imports ayon_server; read its DEFAULT_VALUES with a stand-in.
        stub = types.ModuleType("ayon_server.settings")

        class Base:
            def __init_subclass__(cls, **kw):
                pass
        stub.BaseSettingsModel = Base
        stub.SettingsField = lambda *a, **k: None
        with mock.patch.dict(sys.modules, {"ayon_server": types.ModuleType("ayon_server"),
                                           "ayon_server.settings": stub}):
            spec = importlib.util.spec_from_file_location("agenty_server_settings", ADDON / "server" / "settings.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        self.assertEqual(mod.DEFAULT_VALUES, lib.DEFAULTS)

    def test_local_ports_skip_remote_profiles_on_this_machine(self):
        comfy = _comfy([8188, 8189, 8188], remotes=["http://127.0.0.1:8189", "http://farm01:8188"])
        self.assertEqual(lib.comfyui_local_ports(comfy), [8188])
        self.assertEqual(lib.comfyui_local_ports({}), [])

    def test_root_per_platform(self):
        s = lib.normalize({"agenty_root": {"windows": "D:\\AI\\agentY", "linux": "/opt/agentY"}})
        self.assertEqual(str(lib.agenty_root(s, "linux")), "/opt/agentY")
        self.assertIsNone(lib.agenty_root(s, "darwin"))


class PortClash(unittest.TestCase):
    def test_free_port_passes(self):
        self.assertEqual(lib.check_comfyui_ports(_comfy([_free_port()]), "block"), [])

    def test_running_comfyui_blocks(self):
        srv, port = _serve({"/system_stats": STATS})
        self.addCleanup(_stop, srv)
        with self.assertRaises(lib.PortBusy) as cm:
            lib.check_comfyui_ports(_comfy([port]), "block")
        self.assertIn(f"port {port}", str(cm.exception))

    def test_ignore_lets_it_pass(self):
        srv, port = _serve({"/system_stats": STATS})
        self.addCleanup(_stop, srv)
        self.assertEqual(lib.check_comfyui_ports(_comfy([port]), "ignore"), [])

    def test_open_session_always_blocks(self):
        srv, page = _serve({"/": {}})
        self.addCleanup(_stop, srv)
        for mode in ("block", "stop_if_idle"):
            with self.assertRaises(lib.PortBusy) as cm:
                lib.check_comfyui_ports(_comfy([_free_port()], static_port=page), mode)
            self.assertIn("session is already open", str(cm.exception))

    def test_busy_queue_is_never_closed(self):
        srv, port = _serve({"/system_stats": STATS, "/queue": {"queue_running": [[1]], "queue_pending": []}})
        self.addCleanup(_stop, srv)
        with mock.patch.object(lib, "stop_process_tree") as stop:
            with self.assertRaises(lib.PortBusy) as cm:
                lib.check_comfyui_ports(_comfy([port]), "stop_if_idle")
        stop.assert_not_called()
        self.assertIn("1 prompt(s)", str(cm.exception))

    def test_idle_comfyui_is_closed(self):
        srv, port = _serve({"/system_stats": STATS, "/queue": {"queue_running": [], "queue_pending": []}})
        stopped = []

        def stop(pid, timeout=10.0):
            stopped.append(pid)
            _stop(srv)
            return True
        pids = iter([[4242], []])
        with mock.patch.object(lib, "listening_pids", lambda p: next(pids, [])), \
                mock.patch.object(lib, "stop_process_tree", stop):
            notes = lib.check_comfyui_ports(_comfy([port]), "stop_if_idle", log=lambda m: None)
        self.assertEqual(stopped, [4242])
        self.assertEqual(notes, [f"closed the idle ComfyUI on port {port}"])

    def test_other_program_on_the_port_blocks(self):
        port = _free_port()
        with mock.patch.object(lib, "listening_pids", return_value=[99]):
            with self.assertRaises(lib.PortBusy) as cm:
                lib.check_comfyui_ports(_comfy([port]), "block")
        self.assertIn("another program", str(cm.exception))


class Host(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "run_agent.ps1").write_text("")
        (self.tmp / "run_agent.sh").write_text("")
        self.url = f"http://127.0.0.1:{_free_port()}"

    def _settings(self, **kw):
        return lib.normalize({"agenty_root": {"windows": str(self.tmp), "linux": str(self.tmp)},
                              "agent_server_url": self.url, **kw})

    def test_running_host_is_left_alone(self):
        srv, port = _serve({"/agentY/health": {"status": "ok"}})
        self.addCleanup(_stop, srv)
        with mock.patch.object(lib, "start_host") as start:
            state = lib.ensure_host(self._settings(agent_server_url=f"http://127.0.0.1:{port}"))
        self.assertEqual(state, "running")
        start.assert_not_called()

    def test_down_host_is_started_with_a_clean_env(self):
        comfy = {"local_settings": {"http_server_port": 6000}}
        with mock.patch.object(lib, "start_host") as start, \
                mock.patch.dict("os.environ", {"PYTHONPATH": "x", "AYON_API_KEY": "secret", "QT_API": "pyside6"}):
            state = lib.ensure_host(self._settings(host_args=["-NoUpdate"]), comfy, log=lambda m: None,
                                    platform="win32")
        self.assertEqual(state, "started")
        root, url, args, env, plat = start.call_args.args
        self.assertEqual((root, url, args), (self.tmp, self.url, ["-NoUpdate"]))
        self.assertEqual(env["AYON_COMFYUI_STATIC_URL"], "http://127.0.0.1:6000")
        for k in ("PYTHONPATH", "AYON_API_KEY", "QT_API"):
            self.assertNotIn(k, env)

    def test_not_started_when_off_remote_or_unset(self):
        logs = []
        with mock.patch.object(lib, "start_host") as start:
            self.assertEqual(lib.ensure_host(self._settings(start_host="never")), "off")
            self.assertEqual(lib.ensure_host(self._settings(agent_server_url="http://render07:5000"),
                                             log=logs.append), "remote")
            self.assertEqual(lib.ensure_host(lib.normalize({"agent_server_url": self.url}), log=logs.append),
                             "no_root")
            (self.tmp / "run_agent.sh").unlink()
            self.assertEqual(lib.ensure_host(self._settings(), log=logs.append, platform="linux"), "missing")
        start.assert_not_called()
        self.assertEqual(len(logs), 3)

    def test_commands(self):
        root = Path("D:/AI/agentY")
        self.assertEqual(lib.host_command(root, "http://127.0.0.1:5001", ["-NoUpdate"], "win32")[-3:],
                         ["-Port", "5001", "-NoUpdate"])
        self.assertEqual(lib.host_command(root, "http://127.0.0.1:5000", (), "darwin")[2:], ["--port", "5000"])

    def test_start_host_on_posix_detaches_and_logs(self):
        (self.tmp / "run_agent.sh").write_text("echo started $@\n")
        proc = lib.start_host(self.tmp, "http://127.0.0.1:5123", ["--no-update"], env={"PATH": "/usr/bin:/bin"},
                              platform="linux")
        proc.wait(timeout=10)
        self.assertIn("started --port 5123 --no-update",
                      (self.tmp / ".logs" / "ayon_host_start.log").read_text())


class LauncherSide(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, _ayon_stubs())
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location(
            "agenty_prelaunch", ADDON / "client" / "ayon_agenty" / "hooks" / "pre_launch_agenty.py")
        self.hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hooks)

    def _hook(self, settings):
        ctx = types.SimpleNamespace(data={"project_name": "p", "project_settings": settings})
        return self.hooks.AgentYComfyUIPrelaunch(ctx)

    def test_hook_is_for_comfyui_after_its_own_hook(self):
        cls = self.hooks.AgentYComfyUIPrelaunch
        self.assertEqual(cls.app_groups, {"comfyui"})
        self.assertGreater(cls.order, 20)

    def test_hook_turns_a_clash_into_a_launch_failure(self):
        with mock.patch.object(lib, "check_comfyui_ports", side_effect=lib.PortBusy("taken")), \
                mock.patch.object(lib, "ensure_host") as ensure:
            with self.assertRaises(sys.modules["ayon_applications"].ApplicationLaunchFailed) as cm:
                self._hook({"agenty": {}, "comfyui": {}}).execute()
        self.assertEqual(str(cm.exception), "taken")
        ensure.assert_not_called()

    def test_hook_never_fails_the_launch_on_the_host(self):
        with mock.patch.object(lib, "check_comfyui_ports", return_value=[]), \
                mock.patch.object(lib, "ensure_host", side_effect=OSError("no powershell")):
            self._hook({"agenty": {}, "comfyui": {}}).execute()

    def test_disabled_does_nothing(self):
        with mock.patch.object(lib, "check_comfyui_ports") as check:
            self._hook({"agenty": {"enabled": False}}).execute()
        check.assert_not_called()

    def test_addon_and_tray(self):
        addon = AgentYAddon({"agenty": {"enabled": True, "start_host": "never"}})
        self.assertEqual((addon.name, addon.label), ("agenty", "agentY"))
        self.assertTrue(addon.get_launch_hook_paths(None)[0].endswith("hooks"))
        with mock.patch.object(lib, "host_healthy", return_value=False), \
                mock.patch.object(lib, "ensure_host", return_value="started") as ensure:
            addon._tray_action()
        self.assertEqual(ensure.call_args.args[0]["start_host"], "if_not_running")
        self.assertIn("Starting the agentY host", addon.tray_messages[0])
        with mock.patch.object(lib, "host_healthy", return_value=True), \
                mock.patch("webbrowser.open") as browse:
            addon._tray_action()
        browse.assert_called_once_with("http://127.0.0.1:5000/agentY/log_viewer")


class Package(unittest.TestCase):
    def test_create_package_builds_the_zip(self):
        out = Path(tempfile.mkdtemp())
        res = subprocess.run([sys.executable, "create_package.py", "--output", str(out)], cwd=ADDON,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        zips = list(out.glob("agenty-*.zip"))
        self.assertEqual(len(zips), 1, list(out.iterdir()))
        with zipfile.ZipFile(zips[0]) as z:
            names = set(z.namelist())
            self.assertIn("package.py", names)
            self.assertIn("server/__init__.py", names)
            self.assertIn("server/settings.py", names)
            private = [n for n in names if n.startswith("private/") and n.endswith(".zip")]
            self.assertEqual(private, ["private/client.zip"])
            with z.open("private/client.zip") as f, zipfile.ZipFile(f) as client:
                cnames = set(client.namelist())
        for n in ("ayon_agenty/__init__.py", "ayon_agenty/addon.py", "ayon_agenty/lib.py",
                  "ayon_agenty/hooks/pre_launch_agenty.py", "ayon_agenty/version.py"):
            self.assertIn(n, cnames)


if __name__ == "__main__":
    unittest.main()
