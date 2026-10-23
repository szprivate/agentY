"""Optional Python packages the installer offers, and the record of the answer.

requirements.txt holds what agentY cannot run without. A package that only adds
a capability is offered by the installer instead, and the answer is kept in
``config/optional_deps.json`` (gitignored, machine state; outside the venv so a
rebuilt venv still knows what was asked for):

    {"reverse_image_search": true}

* true  - wanted: scripts/sync_deps.py installs it whenever it cannot be imported;
* false - declined: nothing is ever installed;
* absent - never asked: nothing is installed.

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
}


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


def installed(name: str) -> bool:
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
    try:
        return subprocess.run(install_command(name), cwd=str(ROOT)).returncode
    except OSError as exc:
        print(f"[optional] Could not run the installer for {name}: {exc}", flush=True)
        return 1


def wanted_and_missing(choices: dict = None) -> list:
    choices = read_choices() if choices is None else choices
    return [n for n in OPTIONAL if choices.get(n) is True and not installed(n)]


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
