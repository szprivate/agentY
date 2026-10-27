"""Set an MCP preset up the same way every time: one runner for typed, idempotent steps.

A preset in ``config/mcp_presets.json`` (or a local one in ``config/mcp_presets.local.json``) can carry a
``setup`` recipe: a list of steps like ``find_app``, ``download``, ``install_addon``. ``run_setup`` runs them.
The installers, the Settings Install / Re-check buttons and the agent's tools all call it, so a preset is set
up the same way whichever door it came through. ``setup_steps`` (the human text) stays next to it.

    python scripts/mcp_setup.py run <id> [--dry-run] [--allow-system] [--launch] [--json]
    python scripts/mcp_setup.py schema        the recipe schema (JSON)

Why here and not in ``agenty_core``: the runner works on agentY's own files (``config/mcp.json``, the
bundle folder, the preset catalog) and on agentY's connection test. What is general lives in core
(``agenty_core.tools.app_installs``: finding the apps). Both are plain Python, so the installers run this
before the host exists.

Every step reports one of ``done`` / ``skipped`` (already in place) / ``needs_user`` (a human has to do it;
the message says exactly what) / ``failed`` (the error) and, in a dry run, ``planned``. A step never fails
because of an earlier ``needs_user``: it is ``skipped`` with the reason when its ``needs`` are not met.
Steps are idempotent: they look first and do nothing when it is already so. Edits to a user's files keep a
one-time ``.agenty-bak`` copy.

Nothing system-wide happens without ``allow_system`` (winget installs of uv, Node, git), and no app is
started without ``launch``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
# Run as a script, ``scripts`` must be THIS checkout's folder: another folder called ``scripts`` earlier
# on the path (a dependency's stray top-level folder, another checkout) would otherwise win the import.
_ROOT_DIR = Path(__file__).resolve().parent.parent
if not sys.path or sys.path[0] != str(_ROOT_DIR):
    sys.path.insert(0, str(_ROOT_DIR))
_pkg = sys.modules.get("scripts")
if _pkg is not None and str(_ROOT_DIR / "scripts") not in [str(Path(p)) for p in (getattr(_pkg, "__path__", None) or [])]:
    del sys.modules["scripts"]

from scripts import mcp_presets as mp  # noqa: E402
from scripts import tool_hints as th  # noqa: E402

logger = logging.getLogger("agentY.mcp_setup")

DONE, SKIPPED, NEEDS_USER, FAILED, PLANNED = "done", "skipped", "needs_user", "failed", "planned"
ADDON_APPS = ("blender", "houdini", "maya", "nuke", "cavalry", "premiere-pro", "after-effects", "unity")

# type -> (required, optional). Every step may also carry id, needs, note.
STEP_SPECS: dict[str, tuple[set, set]] = {
    "ensure_tool": ({"tool"}, {"winget", "min_version", "hint"}),
    "find_app": ({"app"}, {"min_version", "var"}),
    "download": ({"url", "dest"}, {"sha256", "size_mb"}),
    "git_clone": ({"url", "commit", "dest"}, set()),
    "npm_fetch": ({"package", "dest"}, set()),
    "install_bundle": ({"url", "name"}, {"sha256", "version"}),
    "ensure_entry": (set(), {"env", "env_if_param_changed"}),
    "install_addon": ({"app"}, {"zip", "source", "addon_id", "dependency", "dependency_url", "debug_mode",
                                "project", "port"}),
    "edit_file": ({"path"}, {"marker_block", "json_merge", "regex_set"}),
    "set_env": ({"name", "value"}, set()),
    "run": ({"cmd"}, {"cwd", "env", "skip_if_exists", "skip_if_ok", "timeout", "ok_codes"}),
    "run_script": ({"interpreter", "script", "description"}, {"cwd", "timeout", "skip_if_exists"}),
    "manual": ({"message"}, {"unless_port"}),
    "launch_and_wait": (set(), {"app", "port", "host", "timeout", "args"}),
    "test": (set(), {"timeout"}),
}
_COMMON = {"type", "id", "needs", "note", "only_on"}
_INTERPRETERS = ("powershell", "python", "bash", "cmd", "node")


def recipe_schema() -> dict:
    """The recipe as a JSON schema: what an agent drafting a recipe for an unknown server fills in."""
    variants = []
    for name, (req, opt) in STEP_SPECS.items():
        props = {k: {} for k in sorted(req | opt)}
        props.update({"type": {"const": name}, "id": {"type": "string"}, "note": {"type": "string"},
                      "only_on": {"enum": ["win", "mac", "linux"]},
                      "needs": {"type": "array", "items": {"type": "string"}}})
        variants.append({"type": "object", "required": ["type", *sorted(req)], "properties": props,
                         "additionalProperties": False})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "agentY MCP preset",
        "type": "object", "required": ["id", "name", "server", "setup"],
        "properties": {
            "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"}, "name": {"type": "string"},
            "category": {"type": "string"}, "description": {"type": "string"},
            "server": {"type": "object", "description": "a config/mcp.json server entry"},
            "env_keys": {"type": "array", "items": {"type": "object", "required": ["key"]}},
            "requires": {"type": "array", "items": {"type": "string"}},
            "params": {"type": "object", "description": "name -> default; steps read ${params.name}"},
            "setup_steps": {"type": "string", "description": "the same, for a human"},
            "setup": {"type": "array", "items": {"oneOf": variants}},
        },
    }


def validate_recipe(preset: dict) -> list[str]:
    """Everything wrong with a drafted preset, as plain sentences. Empty means it can be shown to the user."""
    errs: list[str] = []
    if not isinstance(preset, dict):
        return ["the recipe must be a JSON object"]
    pid = preset.get("id")
    if not isinstance(pid, str) or not re.match(r"^[a-z0-9][a-z0-9_-]*$", pid):
        errs.append("id must be lowercase letters, digits, - or _")
    if not preset.get("name"):
        errs.append("name is missing")
    server = preset.get("server")
    if not isinstance(server, dict) or not (server.get("command") or server.get("url")):
        errs.append("server must be a config/mcp.json entry with a command or a url")
    elif server.get("command") and re.search(r"(^|[\\/])(npx|uvx)(\.cmd|\.exe)?$", str(server["command"])):
        pinned = any(re.search(r"(==|@)\d", str(a)) or re.search(r"[A-Za-z0-9_.-]@\d", str(a))
                     for a in server.get("args") or [])
        if not pinned:
            errs.append("the npx/uvx package is not pinned to a version (pkg@1.2.3 / pkg==1.2.3)")
    setup = preset.get("setup")
    if not isinstance(setup, list) or not setup:
        errs.append("setup must be a non-empty list of steps")
        return errs
    ids: list[str] = []
    for i, st in enumerate(setup):
        where = f"step {i + 1}"
        if not isinstance(st, dict) or st.get("type") not in STEP_SPECS:
            errs.append(f"{where}: type must be one of {', '.join(STEP_SPECS)}")
            continue
        req, opt = STEP_SPECS[st["type"]]
        sid = str(st.get("id") or f"{st['type']}{i + 1}")
        where = f"step {i + 1} ({sid})"
        if sid in ids:
            errs.append(f"{where}: duplicate id")
        ids.append(sid)
        for k in sorted(req):
            if st.get(k) in (None, "", [], {}):
                errs.append(f"{where}: {k} is required for {st['type']}")
        if "only_on" in st and st["only_on"] not in ("win", "mac", "linux"):
            errs.append(f"{where}: only_on must be win, mac or linux")
        for k in st:
            if k not in req | opt | _COMMON:
                errs.append(f"{where}: unknown field {k}")
        for n in st.get("needs") or []:
            if n not in ids[:-1]:
                errs.append(f"{where}: needs '{n}', which is not an earlier step")
        t = st["type"]
        if t == "download" and not st.get("sha256"):
            errs.append(f"{where}: download has no sha256 (pin it where the source allows it)")
        if t == "git_clone" and not re.fullmatch(r"[0-9a-f]{40}", str(st.get("commit") or "")):
            errs.append(f"{where}: git_clone needs a full 40-character commit, not a branch")
        if t == "run_script":
            if st.get("interpreter") not in _INTERPRETERS:
                errs.append(f"{where}: interpreter must be one of {', '.join(_INTERPRETERS)}")
            if not str(st.get("description") or "").strip():
                errs.append(f"{where}: run_script needs a description of what the script does")
        if t == "install_addon" and st.get("app") not in ADDON_APPS:
            errs.append(f"{where}: install_addon app must be one of {', '.join(ADDON_APPS)}")
        if t == "edit_file" and sum(bool(st.get(k)) for k in ("marker_block", "json_merge", "regex_set")) != 1:
            errs.append(f"{where}: edit_file needs exactly one of marker_block, json_merge, regex_set")
        if t == "run" and not isinstance(st.get("cmd"), list):
            errs.append(f"{where}: run.cmd must be a list of arguments (no shell string)")
    return errs


def recipe_hash(preset: dict) -> str:
    """A hash over everything an approved recipe can do: the server entry, its keys and the steps."""
    body = {k: preset.get(k) for k in ("id", "server", "env_keys", "requires", "params", "setup")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def describe_recipe(preset: dict) -> list[str]:
    """A short step list for the user; run_script steps are shown in full and flagged."""
    lines = []
    for i, st in enumerate(preset.get("setup") or [], 1):
        t = st.get("type")
        if t == "run_script":
            lines.append(f"{i}. [RUNS A SCRIPT - {st.get('interpreter')}] {st.get('description')}\n"
                         + "\n".join("      | " + ln for ln in str(st.get("script") or "").splitlines()))
            continue
        bits = {k: v for k, v in st.items() if k not in ("type", "id", "needs", "note")}
        lines.append(f"{i}. {t}: " + ", ".join(f"{k}={json.dumps(v) if not isinstance(v, str) else v}"
                                               for k, v in bits.items()) + (f"  ({st['note']})" if st.get("note") else ""))
    return lines


# ── drafts, approval, local presets ─────────────────────────────────────────

PENDING = ROOT / "config" / ".mcp_recipe_pending.json"


def normalise_draft(draft: dict) -> dict:
    """A drafted preset in the shape the catalog loader gives (and the hash is taken over)."""
    n = mp._normalise(draft) or {}
    return n


def is_approved(preset: dict) -> bool:
    """An official preset always is; a local one only while its recipe is the one the user approved."""
    if not preset.get("local"):
        return True
    return bool(preset.get("approved_hash")) and preset["approved_hash"] == recipe_hash(preset)


def approval_gate(draft: dict, user_message: str = "", *, store: Path | None = None) -> dict:
    """The user's gate before an agent-written recipe runs. Two calls, like the LoRA dataset review:

    1. no ``user_message`` (or a recipe never shown before): returns ``{"status": "waiting_for_user", "hash",
       "summary"}``. The hash is remembered as *shown*. The agent shows the summary and ENDS ITS TURN.
    2. the user answers; the agent calls again with the SAME recipe and their words in ``user_message``:
       ``{"status": "approved", "hash"}`` when the hash was shown before, so a recipe changed after it was shown
       is a new hash that has to be shown (and approved) again.

    Errors in the recipe come back as ``{"status": "invalid", "errors": [...]}`` before anything is shown.
    """
    store = Path(store or PENDING)
    preset = normalise_draft(draft)
    errs = validate_recipe(preset)
    if errs:
        return {"status": "invalid", "errors": errs}
    h = recipe_hash(preset)
    try:
        shown = json.loads(store.read_text(encoding="utf-8")) if store.is_file() else {}
    except ValueError:
        shown = {}
    msg = str(user_message or "").strip()
    if msg and h in shown:
        return {"status": "approved", "hash": h, "approved_by_words": msg}
    shown[h] = {"id": preset["id"], "shown": time.strftime("%Y-%m-%dT%H:%M:%S")}
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(shown, indent=1), encoding="utf-8")
    flagged = [i + 1 for i, st in enumerate(preset["setup"]) if st.get("type") == "run_script"]
    return {"status": "waiting_for_user", "hash": h, "id": preset["id"],
            "summary": describe_recipe(preset), "run_script_steps": flagged,
            "server": preset["server"], "env_keys": [k["key"] for k in preset["env_keys"]],
            "next": "Show the user this step list (every run_script in full), say what the server entry is, then END YOUR "
                    "TURN and wait. When they approve in their own words, call again with the same recipe and "
                    "user_message=<their words>. Never approve on their behalf."}


def save_local_preset(draft: dict, approved_hash: str, path: Path | None = None) -> Path:
    """Write an approved recipe to ``config/mcp_presets.local.json`` (replacing a local preset of the same id)."""
    path = Path(path or mp.LOCAL_CATALOG)
    preset = normalise_draft(draft)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        data = {}
    items = [p for p in (data.get("presets") or []) if isinstance(p, dict) and p.get("id") != preset["id"]]
    items.append({**preset, "approved_hash": approved_hash})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"presets": items}, indent=2), encoding="utf-8")
    return path


# ── results, context ────────────────────────────────────────────────────────

class StepError(Exception):
    """A step cannot go on; the message is the report line."""


class NeedsUser(Exception):
    """A human has to do this; the message says exactly what."""


def R(status: str, message: str = "", **detail) -> dict:
    return {"status": status, "message": message, **({"detail": detail} if detail else {})}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def port_open(port: int, host: str = "localhost", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class WinUserEnv:
    """User-scope environment variables (HKCU\\Environment on Windows, a marked block in ~/.profile elsewhere)."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home or Path.home())

    def get(self, name: str) -> Optional[str]:
        if sys.platform.startswith("win"):
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                    return str(winreg.QueryValueEx(k, name)[0])
            except OSError:
                return None
        prof = self.home / ".profile"
        if prof.is_file():
            m = re.search(rf'^export {re.escape(name)}="(.*)"$', prof.read_text(encoding="utf-8"), re.M)
            return m.group(1) if m else None
        return None

    def set(self, name: str, value: str) -> None:
        if sys.platform.startswith("win"):
            import winreg
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
            try:   # tell running programs to re-read it
                import ctypes
                ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 2, 3000, None)
            except Exception:  # noqa: BLE001
                pass
            return
        edit_marker_block(self.home / ".profile", f"# --- agentY env {name} ---", f"# --- end agentY env {name} ---",
                          [f'export {name}="{value}"'])


class WinRegistry:
    """The few HKCU values the recipes write (Adobe CEP debug mode)."""

    def get(self, key: str, name: str) -> Optional[str]:
        if not sys.platform.startswith("win"):
            return None
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                return str(winreg.QueryValueEx(k, name)[0])
        except OSError:
            return None

    def set(self, key: str, name: str, value: str) -> None:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)


# ── idempotent edits of a user's files ──────────────────────────────────────

def backup_once(path: Path) -> None:
    """Keep the first version of a file we are about to change, next to it, once."""
    bak = path.with_name(path.name + ".agenty-bak")
    if path.is_file() and not bak.exists():
        shutil.copy2(path, bak)


def edit_marker_block(path: Path, begin: str, end: str, lines: list[str]) -> bool:
    """Make *path* hold exactly one ``begin ... end`` block with *lines*. True when the file changed."""
    path = Path(path)
    block = "\n".join([begin, *lines, end])
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pat.search(text):
        new = pat.sub(lambda _m: block, text, count=1)
    else:
        new = text + ("" if not text or text.endswith("\n") else "\n") + block + "\n"
    if new == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_once(path)
    path.write_text(new, encoding="utf-8")
    return True


def _merge(dst: dict, patch: dict) -> bool:
    changed = False
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            changed |= _merge(dst[k], v)
        elif dst.get(k) != v:
            dst[k] = v
            changed = True
    return changed


def edit_json_merge(path: Path, patch: dict) -> bool:
    """Deep-merge *patch* into the JSON object in *path* (created when missing). True when it changed."""
    path = Path(path)
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except ValueError as exc:
            raise StepError(f"{path} is not valid JSON ({exc}); left untouched") from None
        if not isinstance(data, dict):
            raise StepError(f"{path} is not a JSON object; left untouched")
    if not _merge(data, patch):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_once(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def edit_regex_set(path: Path, pattern: str, repl: str) -> tuple[bool, bool]:
    """``(found, changed)``: replace the first match of *pattern* in *path* with *repl*."""
    path = Path(path)
    if not path.is_file():
        return False, False
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(pattern, text, re.M)
    if not m:
        return False, False
    new = text[:m.start()] + re.sub(pattern, repl, m.group(0), count=1, flags=re.M) + text[m.end():]
    if new == text:
        return True, False
    backup_once(path)
    path.write_text(new, encoding="utf-8")
    return True, True


class Ctx:
    """Everything a step touches. Tests pass fakes for the parts that would reach outside the temp folder."""

    def __init__(self, preset: dict, *, root: Path | None = None, config_path: Path | None = None,
                 catalog_path: Path | None = None, interactive: bool = False, dry_run: bool = False,
                 allow_system: bool = False, launch: bool = False, overrides: dict | None = None,
                 load_config: Callable | None = None, save_config: Callable | None = None,
                 test_server: Callable | None = None, run: Callable | None = None,
                 which: Callable | None = None, env_store=None, registry=None,
                 ask: Callable | None = None, progress: Callable | None = None, platform: str | None = None):
        self.preset = preset
        self.root = Path(root or ROOT)
        self.bundles = self.root / "config" / "mcp_bundles"
        self.downloads = self.bundles / ".downloads"
        self.config_path = Path(config_path) if config_path else self.root / "config" / "mcp.json"
        self.catalog_path = catalog_path
        self.interactive, self.dry_run = interactive, dry_run
        self.allow_system, self.launch = allow_system, launch
        self.overrides = overrides or {}
        self.platform = platform or sys.platform
        self.home = Path(self.overrides.get("home") or Path.home())
        self.appdata = Path(self.overrides.get("appdata") or os.environ.get("APPDATA") or self.home / "AppData" / "Roaming")
        self.params = {**(preset.get("params") or {}), **(self.overrides.get("params") or {})}
        self.vars: dict[str, str] = {}
        self.launched: list[subprocess.Popen] = []
        self._load, self._save = load_config, save_config
        self._test, self._run, self._which = test_server, run, which
        self.env_store = env_store or WinUserEnv(self.home)
        self.registry = registry or WinRegistry()
        self.ask = ask
        self.progress = progress or (lambda *_a, **_k: None)

    # config -------------------------------------------------------------
    def load_config(self) -> dict:
        if self._load:
            return self._load()
        return mp.read_config(self.config_path)

    def save_config(self, cfg: dict) -> None:
        if self._save:
            self._save({"servers": cfg["servers"]})
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps({"servers": cfg["servers"]}, indent=2), encoding="utf-8")

    # expansion ----------------------------------------------------------
    def prefs(self, app: str) -> Optional[Path]:
        v = (self.overrides.get("prefs") or {}).get(app)
        return Path(v) if v else None

    def expand(self, value):
        if isinstance(value, str):
            def sub(m):
                key = m.group(1)
                table = {"ROOT": str(self.root), "BUNDLES": str(self.bundles), "DOWNLOADS": str(self.downloads),
                         "HOME": str(self.home), "APPDATA": str(self.appdata)}
                if key in table:
                    return table[key]
                if key.startswith("params."):
                    if key[7:] in self.params:
                        return str(self.params[key[7:]])
                elif key in self.vars:
                    return self.vars[key]
                raise StepError(f"unresolved ${{{key}}} (an earlier step has to set it)")
            return re.sub(r"\$\{([A-Za-z_][\w.]*)\}", sub, value)
        if isinstance(value, list):
            return [self.expand(v) for v in value]
        if isinstance(value, dict):
            return {k: self.expand(v) for k, v in value.items()}
        return value

    # processes ----------------------------------------------------------
    def tool_path(self, name: str) -> Optional[str]:
        if self._which:
            return self._which(name)
        hit = shutil.which(name)
        if hit:
            return hit
        # Not on the PATH this process got (a GUI-started process on macOS has no login-shell PATH):
        # the usual folders - Homebrew, nvm, Volta, ~/.local/bin - and on Windows the installer's folders.
        for d in th.extra_path_dirs(self.platform, self.home):
            for ext in ("", ".exe", ".cmd"):
                p = Path(d) / (name + ext)
                if p.is_file() and os.access(p, os.X_OK):
                    return str(p)
        return None

    def run(self, cmd: list[str], *, cwd: str | None = None, env: dict | None = None, timeout: int = 600) -> tuple[int, str]:
        if self._run:
            return self._run(cmd, cwd=cwd, env=env, timeout=timeout)
        exe = self.tool_path(cmd[0]) or cmd[0]
        full = {**os.environ, **(env or {})}
        full["PATH"] = th.augmented_path(full.get("PATH", ""), self.platform, self.home)   # npm finds node, etc.
        try:
            p = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, cwd=cwd,
                               env=full, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout}s"
        except OSError as exc:
            return 127, str(exc)
        return p.returncode, (p.stdout or "") + (p.stderr or "")


def _drive_of(p: Path) -> str:
    return os.path.splitdrive(str(p.resolve()))[0].upper()


def download_file(url: str, dest: Path, sha256: str = "", size_mb: float = 0, *, progress=None) -> bool:
    """Download *url* to *dest* unless it is already there with the right hash. True when it fetched."""
    import urllib.request
    dest = Path(dest)
    if dest.is_file() and (not sha256 or _hash_file(dest).lower() == sha256.lower()):
        return False
    if size_mb >= 200 and _drive_of(dest) == "C:":
        raise StepError(f"{dest} is on C:, and {size_mb:.0f} MB is too much for the system drive; choose another drive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if size_mb and free < size_mb * 1e6 + 2e9:
        raise StepError(f"not enough free space on {_drive_of(dest) or dest.parent} ({free / 1e9:.1f} GB free)")
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "agentY-setup"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:   # noqa: S310
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001
        part.unlink(missing_ok=True)
        raise StepError(f"download of {url} failed: {exc}") from None
    if sha256 and _hash_file(part).lower() != sha256.lower():
        part.unlink(missing_ok=True)
        raise StepError(f"{url}: the downloaded file does not match the pinned sha256")
    part.replace(dest)
    return True


# ── step handlers ───────────────────────────────────────────────────────────

def _version_of_tool(ctx: Ctx, tool: str) -> str:
    rc, out = ctx.run([tool, "--version"], timeout=30)
    m = re.search(r"(\d+(?:\.\d+)+)", out or "")
    return m.group(1) if rc == 0 and m else ""


def step_ensure_tool(ctx: Ctx, st: dict) -> dict:
    """Find uv / node / npm / git, or install it the way THIS OS does (winget, brew), or say exactly what to run."""
    tool = st["tool"]
    from agenty_core.tools.app_installs import version_tuple
    path = ctx.tool_path(tool)
    if path:
        have = _version_of_tool(ctx, tool) if st.get("min_version") else ""
        if st.get("min_version") and have and version_tuple(have) < version_tuple(st["min_version"]):
            raise NeedsUser(f"{tool} {have} is installed, {st['min_version']} or newer is needed. {st.get('hint', '')}".strip())
        return R(SKIPPED, f"{tool} {have} found at {path}".replace("  ", " "), path=path)
    has_brew = bool(ctx.tool_path("brew")) if not ctx.platform.startswith("win") else False
    cmd = th.install_command(tool, ctx.platform, has_brew=has_brew, winget_id=st.get("winget") or "")
    text, is_cmd = th.install_hint(tool, ctx.platform, has_brew=has_brew, winget_id=st.get("winget") or "")
    how = " ".join(cmd) if cmd else text
    todo = (f"Run: {text}" if is_cmd else f"Install {text}") if not cmd else f"Run: {how}"
    if st.get("hint") and not cmd:
        todo += f" ({st['hint']})"
    if ctx.dry_run:
        return R(PLANNED, f"{tool} is missing; would run: {how}" if ctx.allow_system and cmd else f"{tool} is missing; needs: {how}")
    permitted = ctx.allow_system or bool(ctx.interactive and ctx.ask and ctx.ask(f"{tool} is not installed. Install it system-wide with: {how} ?"))
    if not (permitted and cmd):
        raise NeedsUser(f"{tool} is not installed. {todo}" + (" (then start agentY's setup again)" if cmd else ""))
    rc, out = ctx.run(cmd, timeout=900)
    if rc != 0 or not ctx.tool_path(tool):
        if rc == 0:
            raise NeedsUser(f"{tool} was installed; open a new terminal so it is on PATH, then set up again.")
        raise StepError(f"{cmd[0]} install failed (exit {rc}): {(out or '').strip().splitlines()[-1:] or ['']}")
    return R(DONE, f"installed {tool}", path=ctx.tool_path(tool))


def step_find_app(ctx: Ctx, st: dict) -> dict:
    from agenty_core.tools import app_installs as ai
    app = ai.canonical(st["app"])
    var = st.get("var") or st.get("id") or app
    forced = (ctx.overrides.get("apps") or {}).get(app)
    if forced:
        import glob as _glob
        installs = ai.find_installs(app, system=False, extra_globs=[_glob.escape(str(forced))])
    else:
        installs = ai.find_installs(app, system=ctx.overrides.get("system_apps", True),
                                    extra_globs=ctx.overrides.get("app_globs", {}).get(app, ()))
    chosen, others, why = ai.pick(installs, st.get("min_version", ""))
    if not chosen:
        label = app.replace("-", " ").title()
        raise NeedsUser(f"{label}: {why}. Install {label}" + (f" {st['min_version']} or newer" if st.get("min_version") else "")
                        + " and set the preset up again.")
    ctx.vars.update({f"{var}.path": chosen["path"], f"{var}.version": chosen["version"], f"{var}.root": chosen["root"]})
    msg = f"{chosen['version'] or 'version unknown'} at {chosen['path']} ({why})"
    if others:
        msg += "; also found " + ", ".join(f"{o['version'] or 'version unknown'}" for o in others)
    return R(SKIPPED, msg, **chosen)


def step_download(ctx: Ctx, st: dict) -> dict:
    dest = Path(ctx.expand(st["dest"]))
    sha = st.get("sha256", "")
    if dest.is_file() and (not sha or _hash_file(dest).lower() == sha.lower()):
        ctx.vars[f"{st.get('id', 'download')}.path"] = str(dest)
        return R(SKIPPED, f"{dest.name} is already here")
    if ctx.dry_run:
        ctx.vars[f"{st.get('id', 'download')}.path"] = str(dest)
        return R(PLANNED, f"would download {st['url']} to {dest}")
    fetched = download_file(ctx.expand(st["url"]), dest, sha, float(st.get("size_mb") or 0))
    ctx.vars[f"{st.get('id', 'download')}.path"] = str(dest)
    return R(DONE if fetched else SKIPPED, f"downloaded {dest.name}")


def _git(ctx: Ctx, *args: str, cwd: str | None = None, timeout: int = 600) -> tuple[int, str]:
    return ctx.run(["git", *args], cwd=cwd, timeout=timeout)


def step_git_clone(ctx: Ctx, st: dict) -> dict:
    dest = Path(ctx.expand(st["dest"]))
    commit = st["commit"]
    if (dest / ".git").exists():
        rc, head = _git(ctx, "rev-parse", "HEAD", cwd=str(dest))
        if rc == 0 and head.strip() == commit:
            return R(SKIPPED, f"{dest.name} is at {commit[:8]}")
    if ctx.dry_run:
        return R(PLANNED, f"would clone {st['url']} into {dest} and check out {commit[:8]}")
    if dest.exists() and not (dest / ".git").exists() and any(dest.iterdir()):
        raise StepError(f"{dest} exists and is not a git clone; not touching it")
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc, out = _git(ctx, "clone", ctx.expand(st["url"]), str(dest), timeout=900)
        if rc != 0:
            raise StepError(f"git clone failed: {out.strip().splitlines()[-1:]}")
    rc, out = _git(ctx, "checkout", "--quiet", commit, cwd=str(dest))
    if rc != 0:
        _git(ctx, "fetch", "--quiet", "origin", commit, cwd=str(dest), timeout=900)
        rc, out = _git(ctx, "checkout", "--quiet", commit, cwd=str(dest))
    if rc != 0:
        raise StepError(f"git checkout {commit[:8]} failed: {out.strip().splitlines()[-1:]}")
    return R(DONE, f"cloned {dest.name} at {commit[:8]}")


def step_npm_fetch(ctx: Ctx, st: dict) -> dict:
    """The package's files in a folder of our own (``npm pack`` + unpack): no ``npm -g``."""
    dest = Path(ctx.expand(st["dest"]))
    pkg = ctx.expand(st["package"])
    marker = dest / "package" / "package.json"
    want = pkg.rsplit("@", 1)[-1] if re.search(r"@\d", pkg) else ""
    if marker.is_file():
        try:
            if not want or json.loads(marker.read_text(encoding="utf-8")).get("version") == want:
                return R(SKIPPED, f"{pkg} is already unpacked")
        except ValueError:
            pass
    if ctx.dry_run:
        return R(PLANNED, f"would npm pack {pkg} into {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run(["npm", "pack", pkg, "--pack-destination", str(dest), "--silent"], cwd=str(dest), timeout=300)
    if rc != 0:
        raise StepError(f"npm pack {pkg} failed: {out.strip().splitlines()[-1:]}")
    tgz = dest / (out.strip().splitlines()[-1] if out.strip() else "")
    if not tgz.is_file():
        found = sorted(dest.glob("*.tgz"))
        if not found:
            raise StepError("npm pack produced no archive")
        tgz = found[-1]
    import tarfile
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise StepError("the package archive has an unsafe path")
        tf.extractall(dest)   # noqa: S202 - paths checked above
    return R(DONE, f"unpacked {pkg}")


def step_install_bundle(ctx: Ctx, st: dict) -> dict:
    from src.utils import mcp_bundle as mb
    name = st["name"]
    target = ctx.bundles / name
    marker = target / mb.MARKER
    want = str(st.get("version") or "")
    if marker.is_file():
        try:
            have = json.loads(marker.read_text(encoding="utf-8")).get("version", "")
        except ValueError:
            have = ""
        if not want or have == want:
            return R(SKIPPED, f"bundle {name} {have} is installed")
    if ctx.dry_run:
        return R(PLANNED, f"would download {st['url']} and unpack it into {target}")
    mcpb = ctx.downloads / f"{name}-{want or 'bundle'}.mcpb"
    download_file(ctx.expand(st["url"]), mcpb, st.get("sha256", ""))
    old = mb.BUNDLES_ROOT
    mb.BUNDLES_ROOT = ctx.bundles
    try:
        with open(mcpb, "rb") as f:
            token = mb.stage_upload(f, mcpb.name)
        info = mb.inspect(token, {}, ())
        if info.get("problems"):
            mb.discard(token)
            raise NeedsUser("this machine cannot run the bundle: " + " ".join(info["problems"]))
        res = mb.install(token, name, {})
    except mb.BundleError as exc:
        raise StepError(str(exc)) from None
    finally:
        mb.BUNDLES_ROOT = old
    return R(DONE, f"unpacked bundle {name} {want}", warnings=res.get("warnings", []))


def step_ensure_entry(ctx: Ctx, st: dict) -> dict:
    pid = ctx.preset["id"]
    cfg = ctx.load_config()
    entry = mp.server_entry(ctx.preset, ctx.platform, ctx.root)
    env = dict(ctx.expand(st.get("env") or {}))
    for var, param in (st.get("env_if_param_changed") or {}).items():
        default = (ctx.preset.get("params") or {}).get(param)
        if str(ctx.params.get(param)) != str(default):
            env[var] = str(ctx.params[param])
    if env:
        entry["env"] = {**(entry.get("env") or {}), **env}
    existing = (cfg.get("servers") or {}).get(pid)
    if existing is not None:
        missing = {k: v for k, v in env.items() if (existing.get("env") or {}).get(k) != v}
        if not missing:
            return R(SKIPPED, f"'{pid}' is already in the MCP config")
        if ctx.dry_run:
            return R(PLANNED, f"would set {', '.join(missing)} in the '{pid}' entry")
        existing["env"] = {**(existing.get("env") or {}), **missing}
        ctx.save_config(cfg)
        return R(DONE, f"updated env of '{pid}': {', '.join(missing)}")
    if mp.needs_bundle(entry, ctx.root):
        if ctx.dry_run:
            return R(PLANNED, f"would write '{pid}' once its bundle is unpacked")
        raise StepError("the server's bundle is not unpacked, so the entry would not start")
    if ctx.dry_run:
        return R(PLANNED, f"would add '{pid}' to the MCP config")
    servers = dict(cfg.get("servers") or {})
    servers[pid] = entry
    ctx.save_config({**cfg, "servers": servers})
    return R(DONE, f"added '{pid}' to the MCP config")


def step_edit_file(ctx: Ctx, st: dict) -> dict:
    path = Path(ctx.expand(st["path"]))
    if st.get("marker_block"):
        mb = ctx.expand(st["marker_block"])
        lines = mb.get("lines") or []
        begin, end = mb.get("begin", "# --- agentY ---"), mb.get("end", "# --- end agentY ---")
        if ctx.dry_run:
            cur = path.read_text(encoding="utf-8") if path.is_file() else ""
            return R(SKIPPED if "\n".join([begin, *lines, end]) in cur else PLANNED, f"{path.name}: block")
        return R(DONE if edit_marker_block(path, begin, end, lines) else SKIPPED, f"{path.name}: block")
    if st.get("json_merge"):
        patch = ctx.expand(st["json_merge"])
        if ctx.dry_run:
            probe = Path(tempfile.mkdtemp()) / "p.json"
            if path.is_file():
                shutil.copyfile(path, probe)
            try:
                changed = edit_json_merge(probe, patch)
            finally:
                shutil.rmtree(probe.parent, ignore_errors=True)
            return R(PLANNED if changed else SKIPPED, f"{path.name}: json merge")
        return R(DONE if edit_json_merge(path, patch) else SKIPPED, f"{path.name}: json merge")
    rs = ctx.expand(st["regex_set"])
    found, changed = edit_regex_set(path, rs["pattern"], rs["replace"]) if not ctx.dry_run else (path.is_file(), False)
    if not found:
        raise NeedsUser(f"{path} has no setting matching {rs['pattern']}")
    return R(DONE if changed else SKIPPED, f"{path.name}: setting")


def step_set_env(ctx: Ctx, st: dict) -> dict:
    name, value = st["name"], ctx.expand(st["value"])
    if ctx.env_store.get(name) == value:
        return R(SKIPPED, f"{name} is already set")
    if ctx.dry_run:
        return R(PLANNED, f"would set user variable {name}={value}")
    ctx.env_store.set(name, value)
    return R(DONE, f"set user variable {name}")


def step_run(ctx: Ctx, st: dict) -> dict:
    cmd = ctx.expand(st["cmd"])
    if st.get("skip_if_ok") and not ctx.dry_run:
        rc, _out = ctx.run(ctx.expand(st["skip_if_ok"]), cwd=ctx.expand(st.get("cwd")) if st.get("cwd") else None, timeout=120)
        if rc == 0:
            return R(SKIPPED, "its check already passes: " + " ".join(st["skip_if_ok"][:4]))
    skip = [Path(ctx.expand(p)) for p in ([st["skip_if_exists"]] if isinstance(st.get("skip_if_exists"), str) else st.get("skip_if_exists") or [])]
    if skip and all(p.exists() for p in skip):
        return R(SKIPPED, f"{skip[0].name} exists")
    if ctx.dry_run:
        return R(PLANNED, "would run: " + " ".join(cmd))
    rc, out = ctx.run(cmd, cwd=ctx.expand(st.get("cwd")) if st.get("cwd") else None,
                      env=ctx.expand(st.get("env") or {}), timeout=int(st.get("timeout") or 900))
    if rc not in (st.get("ok_codes") or [0]):
        raise StepError(f"{' '.join(cmd[:3])} exited {rc}: {(out or '').strip().splitlines()[-1:]}")
    return R(DONE, "ran: " + " ".join(cmd[:4]), output=(out or "")[-1500:])


def step_run_script(ctx: Ctx, st: dict) -> dict:
    skip = st.get("skip_if_exists")
    if skip and Path(ctx.expand(skip)).exists():
        return R(SKIPPED, f"{skip} exists")
    if ctx.dry_run:
        return R(PLANNED, f"would run a {st['interpreter']} script: {st['description']}")
    interp = st["interpreter"]
    ext = {"powershell": ".ps1", "python": ".py", "bash": ".sh", "cmd": ".cmd", "node": ".js"}[interp]
    tmp = Path(tempfile.mkdtemp(prefix="agenty-recipe-"))
    script = tmp / ("step" + ext)
    script.write_text(ctx.expand(st["script"]), encoding="utf-8")
    cmd = {"powershell": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
           "python": [sys.executable, "-I", str(script)], "bash": ["bash", str(script)],
           "cmd": ["cmd", "/c", str(script)], "node": ["node", str(script)]}[interp]
    try:
        rc, out = ctx.run(cmd, cwd=ctx.expand(st.get("cwd")) if st.get("cwd") else None,
                          timeout=int(st.get("timeout") or 900))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0:
        raise StepError(f"the script exited {rc}: {(out or '').strip().splitlines()[-1:]}")
    return R(DONE, st["description"], output=(out or "")[-1500:])


def step_manual(ctx: Ctx, st: dict) -> dict:
    port = st.get("unless_port") and int(ctx.expand(str(st["unless_port"])))
    if port and port_open(port):
        return R(SKIPPED, f"port {port} is open already")
    raise NeedsUser(ctx.expand(st["message"]))


def step_launch_and_wait(ctx: Ctx, st: dict) -> dict:
    app = st.get("app") or ""
    port = int(ctx.expand(str(st.get("port") or ctx.params.get("port") or 0)))
    host = st.get("host", "localhost")
    label = app.replace("-", " ").title() or "the app"
    if port and port_open(port, host):
        return R(SKIPPED, f"{label} is running, port {port} is open")
    if not ctx.launch:
        raise NeedsUser(f"Start {label}; its add-on opens port {port}." if port else f"Start {label}.")
    exe = ctx.vars.get(f"{app}.path")
    if not exe:
        raise StepError(f"no {label} path known (find_app has to run before)")
    if ctx.dry_run:
        return R(PLANNED, f"would start {exe} and wait for port {port}")
    env = {**os.environ}
    res = ctx.prefs(app)
    if app == "blender" and res:
        env["BLENDER_USER_RESOURCES"] = str(res)
    args = [exe, *ctx.expand(st.get("args") or [])]
    proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ctx.launched.append(proc)
    deadline = time.time() + int(st.get("timeout") or 120)
    while time.time() < deadline:
        if port_open(port, host):
            return R(DONE, f"{label} started (pid {proc.pid}), port {port} is open", pid=proc.pid)
        if proc.poll() is not None:
            raise StepError(f"{label} exited with {proc.returncode} before port {port} opened")
        time.sleep(1)
    raise StepError(f"port {port} did not open within the timeout")


def step_test(ctx: Ctx, st: dict) -> dict:
    pid = ctx.preset["id"]
    entry = (ctx.load_config().get("servers") or {}).get(pid)
    if entry is None and ctx.dry_run:
        return R(PLANNED, f"would connect to '{pid}' once it is in the MCP config")
    if entry is None:
        raise StepError(f"'{pid}' is not in the MCP config, nothing to test")
    if ctx.dry_run:
        return R(PLANNED, f"would connect to '{pid}' and list its tools")
    fn = ctx._test
    if fn is None:
        from src.tools.mcp_tools import test_server as fn   # noqa: N813
    deadline = time.time() + int(st.get("timeout") or 60)
    last: dict = {}
    while True:
        last = fn(pid, entry, {})
        if last.get("ok") or last.get("needs_auth") or time.time() >= deadline:
            break
        time.sleep(3)
    if last.get("ok"):
        return R(DONE, f"connected, {last.get('tools', 0)} tools", tools=last.get("tools"), names=last.get("names", [])[:60])
    if last.get("needs_auth"):
        raise NeedsUser("The server wants a key or a sign-in: " + str(last.get("error") or ""))
    raise StepError(str(last.get("error") or "the connection test failed"))


# ── add-on handlers (per app) ───────────────────────────────────────────────

def _copy_if_different(src: Path, dst: Path, dry: bool) -> bool:
    if dst.is_file() and _hash_file(src) == _hash_file(dst):
        return False
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        backup_once(dst)
        shutil.copyfile(src, dst)
    return True


def _copy_tree_if_different(src: Path, dst: Path, dry: bool) -> bool:
    changed = False
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        changed |= _copy_if_different(f, dst / f.relative_to(src), dry)
    return changed


_BL_STATE = """
import bpy, json, os
mod = {mod!r}
a = bpy.context.preferences.addons.get(mod)
ext = bpy.utils.user_resource('EXTENSIONS')
manifest = os.path.join(ext, 'user_default', {addon!r}, 'blender_manifest.toml')
ver = ''
if os.path.isfile(manifest):
    import re
    m = re.search(r'^version\\s*=\\s*"([^"]+)"', open(manifest, encoding='utf-8').read(), re.M)
    ver = m.group(1) if m else ''
p = a.preferences if a else None
print('AGENTY_STATE ' + json.dumps({{'enabled': bool(a), 'version': ver,
    'port': getattr(p, 'port', None), 'autostart': getattr(p, 'use_autostart', None),
    'online': bool(bpy.context.preferences.system.use_online_access)}}))
"""
_BL_PREFS = """
import bpy
a = bpy.context.preferences.addons[{mod!r}]
a.preferences.use_autostart = True
a.preferences.port = {port}
# the add-on refuses to start its socket while Blender's online access is off
bpy.context.preferences.system.use_online_access = True
bpy.ops.wm.save_userpref()
print('AGENTY_SAVED')
"""


def _blender_env(ctx: Ctx) -> dict:
    res = ctx.prefs("blender")
    return {"BLENDER_USER_RESOURCES": str(res)} if res else {}


def addon_blender(ctx: Ctx, st: dict) -> dict:
    exe = ctx.vars.get("blender.path")
    if not exe:
        raise StepError("find_app blender has to run before")
    addon = st.get("addon_id", "mcp")
    mod = f"bl_ext.user_default.{addon}"
    port = int(ctx.params.get("port", 9876))
    zpath = Path(ctx.expand(st["zip"]))
    if not zpath.is_file():
        if ctx.dry_run:
            return R(PLANNED, f"would install {zpath.name} into Blender, enable it and set autostart, port {port}")
        raise StepError(f"{zpath} is missing")
    with zipfile.ZipFile(zpath) as z:
        want = re.search(r'^version\s*=\s*"([^"]+)"', z.read("blender_manifest.toml").decode("utf-8"), re.M).group(1)

    def state() -> dict:
        rc, out = ctx.run([exe, "-b", "--python-expr", _BL_STATE.format(mod=mod, addon=addon)], env=_blender_env(ctx), timeout=180)
        m = re.search(r"AGENTY_STATE (\{.*\})", out or "")
        if rc != 0 or not m:
            raise StepError("could not read Blender's add-on state: " + ((out or "").strip().splitlines() or [""])[-1])
        return json.loads(m.group(1))

    cur = state()
    installed = cur["enabled"] and cur["version"] == want
    prefs_ok = installed and cur["autostart"] is True and cur["port"] == port and cur["online"] is True
    if prefs_ok:
        return R(SKIPPED, f"add-on {want} is enabled, autostart on, port {port}, online access on")
    if ctx.dry_run:
        return R(PLANNED, ("" if installed else f"would install the add-on {want} and enable it; ") + f"would set autostart and port {port}")
    did = []
    if not installed:
        rc, out = ctx.run([exe, "-b", "--factory-startup", "-c", "extension", "install-file", "-r", "user_default",
                           "--enable", str(zpath)], env=_blender_env(ctx), timeout=300)
        if rc != 0 or "Installed" not in (out or ""):
            raise StepError("blender extension install-file failed: " + ((out or "").strip().splitlines() or [""])[-1])
        did.append(f"installed {want}")
    rc, out = ctx.run([exe, "-b", "--python-expr", _BL_PREFS.format(mod=mod, port=port)], env=_blender_env(ctx), timeout=180)
    if rc != 0 or "AGENTY_SAVED" not in (out or ""):
        raise StepError("could not save the add-on preferences: " + ((out or "").strip().splitlines() or [""])[-1])
    did.append(f"autostart on, port {port}, Blender online access on (the add-on needs it), preferences saved")
    final = state()
    if not (final["enabled"] and final["autostart"] and final["port"] == port and final["online"]):
        raise StepError(f"the add-on state after install is {final}")
    return R(DONE, "; ".join(did), **final)


def addon_houdini(ctx: Ctx, st: dict) -> dict:
    ver = ".".join(ctx.vars.get("houdini.version", "").split(".")[:2])
    if not ver:
        raise StepError("find_app houdini has to run before")
    base = ctx.prefs("houdini")
    envdir = os.environ.get("HOUDINI_USER_PREF_DIR", "")
    prefs = (base / f"houdini{ver}") if base else (Path(envdir.replace("__HVER__", ver)) if envdir else ctx.home / "Documents" / f"houdini{ver}")
    pkg = prefs / "packages" / "houdinimcp.json"
    try:
        hpath = Path(json.loads(pkg.read_text(encoding="utf-8")).get("hpath", "")) if pkg.is_file() else None
    except ValueError:
        hpath = None
    if hpath and hpath.is_dir():
        return R(SKIPPED, f"plugin package is in {prefs}")
    if ctx.dry_run:
        return R(PLANNED, f"would run houdinimcp-install for Houdini {ver} into {prefs}")
    dep = ctx.expand(st.get("dependency", "houdini-mcp-server"))
    prefs.mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run(["uv", "tool", "run", "--from", dep, "houdinimcp-install", "--prefs-dir", str(prefs),
                       "--harness", "none", "--yes", "--skip-deps", "--json"], timeout=600)
    if rc != 0 or not pkg.is_file():
        raise StepError("houdinimcp-install did not write the package: " + ((out or "").strip().splitlines() or [""])[-1])
    return R(DONE, f"installed the plugin package in {prefs}")


_MAYA_BEGIN, _MAYA_END = "# --- agentY maya-mcp ---", "# --- end agentY maya-mcp ---"


def addon_maya(ctx: Ctx, st: dict) -> dict:
    year = ctx.vars.get("maya.version", "")
    if not year:
        raise StepError("find_app maya has to run before")
    year = year.split(".")[0]
    appdir = ctx.prefs("maya") or (Path(os.environ["MAYA_APP_DIR"]) if os.environ.get("MAYA_APP_DIR") else ctx.home / "Documents" / "maya")
    scripts = appdir / year / "scripts"
    src = Path(ctx.expand(st["source"]))
    port = int(ctx.params.get("port", 7001))
    if not src.is_file() and not ctx.dry_run:
        raise StepError(f"{src} is missing")
    lines = ["import maya.utils", "def _agenty_open_commandport():",
             "    import agenty_enable_commandport as _m", f"    _m.enable_commandport({port})",
             "maya.utils.executeDeferred(_agenty_open_commandport)"]
    us = scripts / "userSetup.py"
    cur = us.read_text(encoding="utf-8") if us.is_file() else ""
    copy_needed = not (scripts / "agenty_enable_commandport.py").is_file() or (src.is_file() and _hash_file(src) != _hash_file(scripts / "agenty_enable_commandport.py"))
    block_ok = "\n".join([_MAYA_BEGIN, *lines, _MAYA_END]) in cur
    if not copy_needed and block_ok:
        return R(SKIPPED, f"commandPort script is in {scripts}")
    if ctx.dry_run:
        return R(PLANNED, f"would put the script and a userSetup block into {scripts}")
    if copy_needed:
        _copy_if_different(src, scripts / "agenty_enable_commandport.py", False)
    edit_marker_block(us, _MAYA_BEGIN, _MAYA_END, lines)
    return R(DONE, f"commandPort opens on port {port} each time Maya {year} starts ({scripts})")


_NUKE_BEGIN, _NUKE_END = "# --- nuke-mcp (agentY) ---", "# --- end nuke-mcp ---"


def addon_nuke(ctx: Ctx, st: dict) -> dict:
    nukedir = ctx.prefs("nuke") or ctx.home / ".nuke"
    src = Path(ctx.expand(st["source"]))
    if not src.is_file() and not ctx.dry_run:
        raise StepError(f"{src} is missing (the clone has to run before)")
    dst = nukedir / "nuke_mcp_addon.py"
    menu = nukedir / "menu.py"
    lines = ["import nuke_mcp_addon", "nuke_mcp_addon.start()"]
    cur = menu.read_text(encoding="utf-8") if menu.is_file() else ""
    block_ok = "\n".join([_NUKE_BEGIN, *lines, _NUKE_END]) in cur
    same = dst.is_file() and src.is_file() and _hash_file(src) == _hash_file(dst)
    if same and block_ok:
        return R(SKIPPED, f"addon and menu.py block are in {nukedir}")
    if ctx.dry_run:
        return R(PLANNED, f"would copy nuke_mcp_addon.py and add the block to {menu}")
    _copy_if_different(src, dst, False)
    edit_marker_block(menu, _NUKE_BEGIN, _NUKE_END, lines)
    return R(DONE, f"addon copied and menu.py starts it ({nukedir})")


def _cavalry_scripts(ctx: Ctx) -> Path:
    if ctx.prefs("cavalry"):
        return ctx.prefs("cavalry")
    if ctx.platform == "darwin":
        return ctx.home / "Library" / "Application Support" / "Cavalry" / "Scripts"
    if ctx.platform.startswith("win"):
        return ctx.appdata / "Cavalry" / "Scripts"
    return ctx.home / ".local" / "share" / "Cavalry" / "Scripts"


def addon_cavalry(ctx: Ctx, st: dict) -> dict:
    src = Path(ctx.expand(st["source"]))
    if not src.is_file() and not ctx.dry_run:
        raise StepError(f"{src} is missing (the clone has to run before)")
    dst = _cavalry_scripts(ctx) / "cavalry-mcp-bridge.js"
    if dst.is_file() and src.is_file() and _hash_file(src) == _hash_file(dst):
        return R(SKIPPED, f"bridge script is in {dst.parent}")
    if ctx.dry_run:
        return R(PLANNED, f"would copy the bridge script to {dst.parent}")
    _copy_if_different(src, dst, False)
    return R(DONE, f"bridge script copied to {dst.parent}")


def addon_premiere(ctx: Ctx, st: dict) -> dict:
    src = Path(ctx.expand(st["source"]))
    if ctx.prefs("premiere-pro"):
        cep = ctx.prefs("premiere-pro")
    elif ctx.platform == "darwin":
        cep = ctx.home / "Library" / "Application Support" / "Adobe" / "CEP" / "extensions"
    else:
        cep = ctx.appdata / "Adobe" / "CEP" / "extensions"
    dst = cep / "MCPBridgeCEP"
    if not src.is_dir() and not ctx.dry_run:
        raise StepError(f"{src} is missing (npm_fetch has to run before)")
    reg_todo = []
    if st.get("debug_mode", True) and ctx.platform.startswith("win"):
        reg_todo = [n for n in range(9, 16) if ctx.registry.get(rf"Software\Adobe\CSXS.{n}", "PlayerDebugMode") != "1"]
    files_todo = src.is_dir() and _copy_tree_if_different(src, dst, True)
    if not files_todo and not reg_todo:
        return R(SKIPPED, f"CEP panel is in {dst} and the debug flag is set")
    if ctx.dry_run:
        return R(PLANNED, f"would copy the CEP panel to {dst}" + (f" and set PlayerDebugMode for CSXS.{reg_todo[0]}-{reg_todo[-1]}" if reg_todo else ""))
    did = []
    if files_todo:
        _copy_tree_if_different(src, dst, False)
        did.append(f"panel copied to {dst}")
    for n in reg_todo:
        ctx.registry.set(rf"Software\Adobe\CSXS.{n}", "PlayerDebugMode", "1")
    if reg_todo:
        did.append("CEP PlayerDebugMode set (current user)")
    return R(DONE, "; ".join(did))


def addon_after_effects(ctx: Ctx, st: dict) -> dict:
    root = (ctx.prefs("after-effects") or ctx.appdata / "Adobe" / "After Effects")
    files = sorted(root.glob("*/Adobe After Effects * Prefs-indep-general.txt")) if root.is_dir() else []
    key = "Pref_SCRIPTING_FILE_NETWORK_SECURITY"
    if not files:
        raise NeedsUser("After Effects has not written its preferences yet. Start it once, close it, set up again; or turn on "
                        "Edit > Preferences > Scripting & Expressions > 'Allow Scripts to Write Files and Access Network'.")
    target = files[-1]
    pat = rf'^(\s*\[?"?{key}"?\]?\s*=\s*)"?(\d)"?'
    text = target.read_text(encoding="utf-8", errors="replace")
    m = re.search(pat, text, re.M)
    if not m:
        raise NeedsUser(f"{target.name} has no '{key}' line. In After Effects turn on Edit > Preferences > Scripting & Expressions > "
                        "'Allow Scripts to Write Files and Access Network'.")
    if m.group(2) == "1":
        return R(SKIPPED, "scripting network access is already on")
    if ctx.prefs("after-effects") is None and _process_running("AfterFX.exe", ctx):
        raise NeedsUser("After Effects is running and would overwrite its preferences on quit. Close it and set up again, "
                        "or turn on Edit > Preferences > Scripting & Expressions > 'Allow Scripts to Write Files and Access Network'.")
    if ctx.dry_run:
        return R(PLANNED, f"would turn scripting network access on in {target.name}")
    edit_regex_set(target, pat, r'\g<1>"1"')
    return R(DONE, f"scripting network access turned on in {target}")


def _process_running(name: str, ctx: Ctx) -> bool:
    if not ctx.platform.startswith("win"):
        return False
    rc, out = ctx.run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"], timeout=20)
    return name.lower() in (out or "").lower()


def addon_unity(ctx: Ctx, st: dict) -> dict:
    project = ctx.params.get("project") or ""
    if not project:
        raise NeedsUser("Tell me which Unity project to add the package to (the folder with Assets and Packages in it). "
                        "Then set the preset up again with that project.")
    manifest = Path(project) / "Packages" / "manifest.json"
    if not manifest.is_file():
        raise NeedsUser(f"{manifest} does not exist; give the folder of a Unity project.")
    dep = ctx.expand(st["dependency"])
    url = ctx.expand(st["dependency_url"])
    changed = edit_json_merge(manifest, {"dependencies": {dep: url}}) if not ctx.dry_run else \
        (json.loads(manifest.read_text(encoding="utf-8")).get("dependencies", {}).get(dep) != url)
    if not changed:
        return R(SKIPPED, f"{dep} is already in {manifest.name}")
    return R(PLANNED if ctx.dry_run else DONE, f"{dep} added to {manifest}; open the project in Unity once to import it")


ADDON_HANDLERS: dict[str, Callable] = {
    "blender": addon_blender, "houdini": addon_houdini, "maya": addon_maya, "nuke": addon_nuke,
    "cavalry": addon_cavalry, "premiere-pro": addon_premiere, "after-effects": addon_after_effects, "unity": addon_unity,
}


def step_install_addon(ctx: Ctx, st: dict) -> dict:
    return ADDON_HANDLERS[st["app"]](ctx, st)


HANDLERS: dict[str, Callable] = {
    "ensure_tool": step_ensure_tool, "find_app": step_find_app, "download": step_download,
    "git_clone": step_git_clone, "npm_fetch": step_npm_fetch, "install_bundle": step_install_bundle,
    "ensure_entry": step_ensure_entry, "install_addon": step_install_addon, "edit_file": step_edit_file,
    "set_env": step_set_env, "run": step_run, "run_script": step_run_script, "manual": step_manual,
    "launch_and_wait": step_launch_and_wait, "test": step_test,
}


# ── the runner ──────────────────────────────────────────────────────────────

def run_setup(preset_id: str, *, interactive: bool = False, dry_run: bool = False,
              only_steps: list[str] | None = None, preset: dict | None = None, **ctx_kwargs) -> dict:
    """Run the recipe of *preset_id* and return the report::

        {preset, ok, complete, dry_run, steps: [{id, type, status, message, detail?}], needs_user: [..], failed: [..]}

    ``ok``: nothing failed. ``complete``: nothing failed and nothing waits for a human. ``only_steps``: step ids
    to run (the rest are reported ``skipped``). ``preset`` lets a caller pass a preset that is not in the
    catalog (an approved draft). Extra keyword arguments go to :class:`Ctx` (config_path, allow_system, launch,
    overrides, load_config, ...).
    """
    preset = preset or mp.find(preset_id, ctx_kwargs.get("catalog_path"))
    if preset is None:
        return {"preset": preset_id, "ok": False, "complete": False, "dry_run": dry_run, "steps": [],
                "needs_user": [], "failed": [f"unknown preset '{preset_id}'"], "error": f"unknown preset '{preset_id}'"}
    recipe = preset.get("setup") or []
    report = {"preset": preset["id"], "ok": True, "complete": True, "dry_run": dry_run, "has_recipe": bool(recipe),
              "steps": [], "needs_user": [], "failed": []}
    if not recipe:
        return report
    errs = validate_recipe(preset) if preset.get("local") else []
    if preset.get("local") and not errs and not is_approved(preset):
        errs = ["this recipe was not approved, or it was changed after it was approved; it has to be shown to the user again"]
    if errs:
        report.update(ok=False, complete=False, failed=errs, error="the recipe is not valid: " + "; ".join(errs))
        return report
    ctx = Ctx(preset, interactive=interactive, dry_run=dry_run, **ctx_kwargs)
    status: dict[str, str] = {}
    for i, st in enumerate(recipe, 1):
        sid = str(st.get("id") or f"{st['type']}{i}")
        if only_steps and sid not in only_steps:
            res = R(SKIPPED, "not asked for")
        elif st.get("only_on") and not ctx.platform.startswith({"win": "win", "mac": "darwin", "linux": "linux"}.get(st["only_on"], st["only_on"])):
            res = R(SKIPPED, f"only on {st['only_on']}")
        elif any(status.get(n) in (NEEDS_USER, FAILED) for n in st.get("needs") or []):
            res = R(SKIPPED, "waiting for " + ", ".join(n for n in st["needs"] if status.get(n) in (NEEDS_USER, FAILED)))
        else:
            try:
                st = {**st, "id": sid}
                res = HANDLERS[st["type"]](ctx, st)
            except NeedsUser as exc:
                res = R(NEEDS_USER, str(exc))
            except StepError as exc:
                res = R(FAILED, str(exc))
            except Exception as exc:  # noqa: BLE001 - a step must never take the runner down
                logger.warning("setup step %s crashed", sid, exc_info=True)
                res = R(FAILED, f"{type(exc).__name__}: {exc}")
        status[sid] = res["status"]
        row = {"id": sid, "type": st["type"], **res}
        report["steps"].append(row)
        ctx.progress(row)
        if res["status"] == NEEDS_USER:
            report["needs_user"].append(res["message"])
        elif res["status"] == FAILED:
            report["failed"].append(f"{sid}: {res['message']}")
    report["ok"] = not report["failed"]
    report["complete"] = report["ok"] and not report["needs_user"]
    report["launched"] = [p.pid for p in ctx.launched]
    report["_ctx"] = ctx
    return report


def public(report: dict) -> dict:
    """The report without the live context object (for JSON)."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def format_report(report: dict) -> str:
    icon = {DONE: "ok  ", SKIPPED: "--  ", NEEDS_USER: "TODO", FAILED: "FAIL", PLANNED: "plan"}
    lines = [f"{report['preset']}: " + ("complete" if report.get("complete") else "needs you" if report.get("ok") else "FAILED")
             + (" (dry run)" if report.get("dry_run") else "")]
    for s in report.get("steps", []):
        lines.append(f"  [{icon.get(s['status'], s['status'])}] {s['id']}: {s['message']}")
    if report.get("error"):
        lines.append("  " + report["error"])
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("ids", nargs="+")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--allow-system", action="store_true", help="allow winget installs of uv / Node / git")
    run.add_argument("--launch", action="store_true", help="allow starting an app and waiting for its port")
    run.add_argument("--json", action="store_true")
    run.add_argument("--config", default=None)
    sub.add_parser("schema")
    a = ap.parse_args(argv)
    if a.cmd == "schema":
        print(json.dumps(recipe_schema(), indent=2))
        return 0
    worst = 0
    for pid in a.ids:
        rep = run_setup(pid, dry_run=a.dry_run, allow_system=a.allow_system, launch=a.launch,
                        config_path=Path(a.config) if a.config else None)
        pub = public(rep)
        print(json.dumps(pub, indent=2) if a.json else format_report(pub))
        worst = max(worst, 0 if rep["ok"] else 1)
    return worst


if __name__ == "__main__":
    sys.exit(_main())
