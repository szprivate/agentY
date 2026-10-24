"""Optional Python packages the installer offers, and the record of the answer.

requirements.txt holds what agentY cannot run without. A package that only adds
a capability is offered by the installer instead, and the answer is kept in
``config/optional_deps.json`` (gitignored, machine state; outside the venv so a
rebuilt venv still knows what was asked for):

    {"reverse_image_search": true}

* true  - wanted: scripts/sync_deps.py installs it whenever it cannot be imported;
* false - declined: nothing is ever installed;
* absent - never asked: nothing is installed.

An entry with ``"kind": "action"`` is not a package but a one-time setup the
installer runs when it is ticked (``claude_code_plugin``). It starts UNticked,
is never run by ``ensure`` (the launcher), and ``install`` is idempotent.

Stdlib only; the installers call it with the venv's interpreter:

    python scripts/optional_deps.py list                    # name|label|hint|choice
    python scripts/optional_deps.py set <name> yes|no       # record the answer
    python scripts/optional_deps.py install <name>          # install the package
    python scripts/optional_deps.py ensure [--quiet]        # install what is wanted and missing
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHOICE_FILE = ROOT / "config" / "optional_deps.json"

# name -> what to import to know it is there, what to install, how to offer it.
OPTIONAL = {
    "reverse_image_search": {
        "module": "playwright",
        "spec": "playwright>=1.40",
        "label": "Reverse image search",
        "hint": ("finds more photos of the same object on the web; uses your installed "
                 "Chrome or Edge, ~40 MB Python package, no browser download"),
    },
    "claude_code_plugin": {
        "kind": "action",
        "label": "Set up agentY for Claude Code",
        "hint": ("adds agentY's tools, skills and agents to Claude Code through the agentY-mcp plugin; "
                 "needs the claude CLI, else the two commands are printed"),
    },
}

PLUGIN_MARKETPLACE = "agentY"
PLUGIN_ID = "agenty@" + PLUGIN_MARKETPLACE


def read_choices(path: Path = None) -> dict:
    try:
        data = json.loads((path or CHOICE_FILE).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, bool)} if isinstance(data, dict) else {}


def write_choice(name: str, wanted: bool, path: Path = None) -> None:
    path = path or CHOICE_FILE
    choices = read_choices(path)
    choices[name] = bool(wanted)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(choices, indent=2) + "\n", encoding="utf-8")


def plugin_commands(mcp_path, claude: str = "claude") -> list:
    """The two commands that set up the plugin from the local agentY-mcp checkout (read in place,
    so a later `git pull` of agentY-mcp is also the plugin update)."""
    return [[claude, "plugin", "marketplace", "add", str(mcp_path)],
            [claude, "plugin", "install", PLUGIN_ID]]


def install_plugin() -> int:
    mcp = ROOT.parent / "agentY-mcp"
    claude = shutil.which("claude")
    shown = plugin_commands(mcp)
    if not (mcp / ".claude-plugin" / "plugin.json").is_file():
        print(f"[optional] agentY-mcp with its Claude Code plugin was not found at {mcp}; update it "
              "(re-run the installer), then run:", flush=True)
    elif claude is None:
        print("[optional] The claude CLI is not on PATH. When Claude Code is installed, run:", flush=True)
    else:
        for cmd in plugin_commands(mcp, claude):
            try:
                done = subprocess.run(cmd, cwd=str(ROOT))
            except OSError as exc:
                print(f"[optional] Could not run {' '.join(shown[0][:3])}: {exc}", flush=True)
                break
            if done.returncode != 0:
                print("[optional] That did not work; run these yourself:", flush=True)
                break
        else:
            print("[optional] agentY is set up in Claude Code (restart it, or /reload-plugins). "
                  "Updating agentY-mcp updates the plugin.", flush=True)
            return 0
        print(chr(10).join("    " + " ".join(c) for c in shown), flush=True)
        return 1
    print(chr(10).join("    " + " ".join(c) for c in shown), flush=True)
    return 0


def installed(name: str) -> bool:
    if OPTIONAL[name].get("kind") == "action":
        return False
    try:
        return importlib.util.find_spec(OPTIONAL[name]["module"]) is not None
    except (ImportError, ValueError):
        return False


def install_command(name: str, python: str = None) -> list:
    python = python or sys.executable
    spec = OPTIONAL[name]["spec"]
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", python, spec]
    return [python, "-m", "pip", "install", spec]


def install(name: str) -> int:
    if name == "claude_code_plugin":
        return install_plugin()
    try:
        return subprocess.run(install_command(name), cwd=str(ROOT)).returncode
    except OSError as exc:
        print(f"[optional] Could not run the installer for {name}: {exc}", flush=True)
        return 1


def wanted_and_missing(choices: dict = None) -> list:
    choices = read_choices() if choices is None else choices
    return [n for n in OPTIONAL if OPTIONAL[n].get("kind") != "action"
            and choices.get(n) is True and not installed(n)]


def ensure() -> int:
    """Install what was chosen and is not importable. No pip call otherwise."""
    code = 0
    for name in wanted_and_missing():
        print(f"[optional] Installing {OPTIONAL[name]['label']} ({OPTIONAL[name]['spec']})", flush=True)
        rc = install(name)
        importlib.invalidate_caches()
        if rc != 0:
            print(f"[optional] {OPTIONAL[name]['label']} could not be installed (code {rc}); agentY "
                  "starts without it. Retry: re-run the installer.", flush=True)
            code = rc
    return code


def main(argv: list) -> int:
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        choices = read_choices()
        for name, o in OPTIONAL.items():
            c = choices.get(name)
            if c is None and o.get("kind") == "action":
                c = False                      # an action starts unticked
            print("|".join([name, o["label"], o["hint"], "" if c is None else ("1" if c else "0")]))
        return 0
    if cmd == "set" and len(argv) == 3 and argv[1] in OPTIONAL:
        write_choice(argv[1], argv[2].lower() in ("yes", "y", "1", "true"))
        return 0
    if cmd == "install" and len(argv) == 2 and argv[1] in OPTIONAL:
        return install(argv[1])
    if cmd == "ensure":
        return ensure()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
