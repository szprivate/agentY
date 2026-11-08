"""The add-on's logic, free of AYON imports so it can be tested without an AYON launcher.

Two jobs, both for a ComfyUI session started from the AYON launcher (docs/dev/ayon.md in agentY):

* **The port clash.** The ComfyUI add-on starts ComfyUI on the port of the chosen local launch profile (8188 by
  default), the same port the ComfyUI agentY normally drives. If a ComfyUI already holds it, the new one fails
  to bind, the session's heartbeat attaches to the old one and the session is broken. ``check_comfyui_ports``
  finds that before the launch and blocks it, closes an idle old ComfyUI, or lets it pass, as the settings say.
* **The agentY host.** The agentY panel inside ComfyUI talks to the agentY host (``run_agent.ps1``, port
  5000). ``ensure_host`` starts it when it does not answer, so the artist does not have to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "agenty_root": {"windows": "", "linux": "", "darwin": ""},
    "agent_server_url": "http://127.0.0.1:5000",
    "start_host": "if_not_running",
    "host_args": [],
    "comfyui_port_busy": "block",
}
DEFAULT_COMFYUI_PORT = 8188          # ComfyUI add-on: comfy_launch_port
DEFAULT_STATIC_PORT = 5454           # ComfyUI add-on: http_server_port (the page the session tab shows)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

# Variables of the AYON launcher's own Python and Qt. The agentY host runs in its own venv: inheriting these
# would load the launcher's packages into it.
_LAUNCHER_ENV_PREFIXES = ("PYTHON", "QT_", "AYON_", "OPENPYPE_", "PYBLISH", "VIRTUAL_ENV", "CONDA_")

Log = Callable[[str], None]


class PortBusy(Exception):
    """A launch that cannot work; the message is shown to the artist as it is."""


# ---------------------------------------------------------------------------------------------- settings

def normalize(settings: Optional[dict]) -> dict:
    """The add-on's settings with defaults filled in (an older server may not have every key yet)."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (settings or {}).items() if v is not None})
    out["agenty_root"] = {**DEFAULTS["agenty_root"], **(out.get("agenty_root") or {})}
    out["host_args"] = [str(a) for a in (out.get("host_args") or []) if str(a).strip()]
    return out


def platform_key(platform: str = "") -> str:
    p = platform or sys.platform
    if p.startswith("win"):
        return "windows"
    if p == "darwin":
        return "darwin"
    return "linux"


def agenty_root(settings: dict, platform: str = "") -> Optional[Path]:
    raw = str((settings.get("agenty_root") or {}).get(platform_key(platform)) or "").strip()
    return Path(os.path.expandvars(os.path.expanduser(raw))) if raw else None


def _port_of(url: str, default: int) -> int:
    try:
        return urllib.parse.urlsplit(url).port or default
    except ValueError:
        return default


def _is_local(url: str) -> bool:
    return (urllib.parse.urlsplit(url).hostname or "").lower() in LOCAL_HOSTS


def comfyui_local_ports(comfy_settings: Optional[dict]) -> list[int]:
    """The ports the ComfyUI add-on's local launch profiles start ComfyUI on, minus the ones a remote profile
    points at on this machine (a remote profile WANTS the ComfyUI there to be running)."""
    comfy_settings = comfy_settings or {}
    profiles = ((comfy_settings.get("local_settings") or {}).get("local_setting_list")) or []
    ports = []
    for p in profiles:
        try:
            port = int(p.get("comfy_launch_port") or DEFAULT_COMFYUI_PORT)
        except (TypeError, ValueError):
            port = DEFAULT_COMFYUI_PORT
        if port not in ports:
            ports.append(port)
    remotes = ((comfy_settings.get("remote_settings") or {}).get("remote_setting_list")) or []
    attached = {_port_of(str(r.get("comfy_web_adress") or ""), DEFAULT_COMFYUI_PORT)
                for r in remotes if _is_local(str(r.get("comfy_web_adress") or ""))}
    return [p for p in ports if p not in attached]


def static_url(comfy_settings: Optional[dict]) -> str:
    port = ((comfy_settings or {}).get("local_settings") or {}).get("http_server_port") or DEFAULT_STATIC_PORT
    return f"http://127.0.0.1:{int(port)}"


# ---------------------------------------------------------------------------------------------- probes

def _get_json(url: str, timeout: float = 2.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    # A studio proxy must never see 127.0.0.1 requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "null")


def _answers(url: str, timeout: float = 1.5) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True             # something answers there
    except (OSError, ValueError):
        return False


def comfyui_on_port(port: int) -> Optional[dict]:
    """``/system_stats`` of the ComfyUI on *port*, or None when none answers."""
    try:
        data = _get_json(f"http://127.0.0.1:{port}/system_stats")
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "system" in data else None


def comfyui_queue_size(port: int) -> Optional[int]:
    """Running + pending prompts of the ComfyUI on *port*; None when it cannot be read."""
    try:
        q = _get_json(f"http://127.0.0.1:{port}/queue")
    except (OSError, ValueError):
        return None
    if not isinstance(q, dict):
        return None
    return len(q.get("queue_running") or []) + len(q.get("queue_pending") or [])


def host_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        data = _get_json(url.rstrip("/") + "/agentY/health", timeout=timeout)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("status") == "ok"


# ---------------------------------------------------------------------------------------------- processes

def listening_pids(port: int) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    pids = []
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid and c.pid not in pids:
                pids.append(c.pid)
    except (psutil.Error, OSError):
        return []
    return pids


def stop_process_tree(pid: int, timeout: float = 10.0) -> bool:
    """Terminate *pid* and its children; kill what is left after *timeout*. True when all are gone."""
    import psutil
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return True
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    return not alive


def wait_until(predicate: Callable[[], bool], timeout: float, step: float = 0.5) -> bool:
    end = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(step)


# ---------------------------------------------------------------------------------------------- the port clash

def check_comfyui_ports(comfy_settings: Optional[dict], mode: str, log: Log = print) -> list[str]:
    """Make sure the ComfyUI add-on can start its ComfyUI. Raises PortBusy when it cannot and *mode* says to
    block; returns what was done, for the log."""
    if mode == "ignore":
        return []
    notes: list[str] = []
    page = static_url(comfy_settings)
    if _answers(page):
        # An AYON ComfyUI session is still open: its tab owns the page and the RPC ports. Never closed from
        # here - closing its browser tab is how a session ends cleanly.
        raise PortBusy(
            f"An AYON ComfyUI session is already open ({page}). Use its browser tab, or close that tab to end "
            "it, then launch again.")
    for port in comfyui_local_ports(comfy_settings):
        stats = comfyui_on_port(port)
        if stats is None:
            if listening_pids(port):
                raise PortBusy(f"Port {port} is taken by another program, so AYON cannot start ComfyUI on it. "
                               "Free the port or give the launch profile another one.")
            continue
        version = ((stats.get("system") or {}).get("comfyui_version")) or "?"
        what = f"A ComfyUI (version {version}) already runs on port {port}"
        if mode != "stop_if_idle":
            raise PortBusy(
                f"{what}. AYON would attach to it and the session would break. Close that ComfyUI first "
                "(or set agentY > 'When a ComfyUI already runs on the launch port' to close it when idle).")
        queued = comfyui_queue_size(port)
        if queued is None or queued > 0:
            busy = "its queue could not be read" if queued is None else f"it has {queued} prompt(s) queued or running"
            raise PortBusy(f"{what} and {busy}. Let it finish or close it yourself, then launch again.")
        pids = listening_pids(port)
        if not pids:
            raise PortBusy(f"{what} but its process could not be found (psutil missing or no access). "
                           "Close it yourself, then launch again.")
        for pid in pids:
            log(f"agentY: closing the idle ComfyUI on port {port} (pid {pid}) for the AYON session")
            stop_process_tree(pid)
        if not wait_until(lambda p=port: comfyui_on_port(p) is None and not listening_pids(p), timeout=20):
            raise PortBusy(f"{what}; it was asked to close but port {port} is still held. Close it yourself.")
        notes.append(f"closed the idle ComfyUI on port {port}")
    return notes


# ---------------------------------------------------------------------------------------------- the agentY host

def host_command(root: Path, url: str, extra_args: Iterable[str] = (), platform: str = "") -> list[str]:
    port = str(_port_of(url, 5000))
    if platform_key(platform) == "windows":
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(root / "run_agent.ps1"), "-Port", port, *extra_args]
    return ["bash", str(root / "run_agent.sh"), "--port", port, *extra_args]


def host_env(base: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    """*base* (default: this process's environment) without the launcher's Python / Qt / AYON variables."""
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not k.upper().startswith(_LAUNCHER_ENV_PREFIXES)}
    env.update({k: str(v) for k, v in (extra or {}).items() if v})
    return env


def start_host(root: Path, url: str, extra_args: Iterable[str] = (), env: Optional[dict] = None,
               platform: str = "") -> subprocess.Popen:
    """Start ``run_agent`` detached: in a console of its own on Windows (where it prints its log), in its
    own session with the output in ``.logs/ayon_host_start.log`` elsewhere."""
    cmd = host_command(root, url, extra_args, platform)
    kwargs: dict[str, Any] = {"cwd": str(root), "env": env if env is not None else host_env(),
                              "stdin": subprocess.DEVNULL, "close_fds": True}
    if platform_key(platform) == "windows":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        logs = root / ".logs"
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / "ayon_host_start.log", "ab") as out:    # the child keeps its own copy
            return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, **kwargs)
    return subprocess.Popen(cmd, **kwargs)


def ensure_host(settings: dict, comfy_settings: Optional[dict] = None, log: Log = print,
                platform: str = "") -> str:
    """Start the agentY host when the settings say so and it does not answer. Returns one of
    ``running`` / ``started`` / ``off`` / ``remote`` / ``no_root`` / ``missing`` (for the log and tests)."""
    url = str(settings.get("agent_server_url") or DEFAULTS["agent_server_url"])
    if host_healthy(url):
        return "running"
    if settings.get("start_host") == "never":
        return "off"
    if not _is_local(url):
        log(f"agentY: the host at {url} does not answer; it is not on this machine, so it is not started here")
        return "remote"
    root = agenty_root(settings, platform)
    if root is None:
        log("agentY: the host does not answer and no agentY folder is set (agentY add-on settings), "
            "so it is not started")
        return "no_root"
    script = "run_agent.ps1" if platform_key(platform) == "windows" else "run_agent.sh"
    if not (root / script).is_file():
        log(f"agentY: {root / script} not found; check the agentY folder in the add-on settings")
        return "missing"
    extra = {}
    page = static_url(comfy_settings)
    if page != f"http://127.0.0.1:{DEFAULT_STATIC_PORT}":
        extra["AYON_COMFYUI_STATIC_URL"] = page        # read by agenty_core.tools.ayon
    start_host(root, url, settings.get("host_args") or [], host_env(extra=extra), platform)
    log(f"agentY: started the agentY host from {root} ({url})")
    return "started"
