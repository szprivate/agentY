"""Keep the venv in line with requirements.txt, on every start.

The launchers used to reinstall dependencies only when their own update step had
just pulled a change to requirements.txt or agenty_core's pyproject.toml. A
`git pull` done by hand before starting never told them: the launcher found
nothing new to pull, installed nothing, and the host came up without packages it
imports (OpenCV and scenedetect, on a machine set up before either was added),
with only a warning to show for it.

So the decision is made here, from the environment itself. The install runs when
any of these hold:

* the launcher's update step changed a dependency file (``--changed=<repos>``);
* the dependency files differ from what this venv was last installed from — a
  fingerprint kept inside the venv (``.agenty-deps``), so a version bump pulled by
  hand counts too;
* a package agentY requires cannot be imported (scripts/check_env.py's list).

A venv seen for the first time with nothing missing is only fingerprinted, not
reinstalled. With ``--quiet`` this says nothing unless it installs something or
something is still missing afterwards; it ends with check_env's own quiet report,
so its advice (the macOS hidden-.pth case among it) still reaches the console.

Run it with the venv's interpreter, as the launchers do:

    .venv/Scripts/python.exe scripts/sync_deps.py --quiet
"""
from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEP_FILES = (ROOT / "requirements.txt", ROOT.parent / "agenty_core" / "pyproject.toml")
STAMP_NAME = ".agenty-deps"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_env  # noqa: E402  (a sibling script, not a package)


def fingerprint(files=DEP_FILES) -> str:
    """One hash over the dependency files. Line endings are normalised, so a
    checkout that flips between LF and CRLF is not a reason to reinstall."""
    digest = hashlib.sha256()
    for path in files:
        path = Path(path)
        digest.update(path.name.encode("utf-8"))
        try:
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            digest.update(b"<absent>")
    return digest.hexdigest()


def stamp_path() -> Path:
    """Inside the venv: it records what THIS environment was installed from."""
    return Path(sys.prefix) / STAMP_NAME


def read_stamp():
    try:
        return stamp_path().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_stamp(value: str) -> None:
    try:
        stamp_path().write_text(value + "\n", encoding="utf-8")
    except OSError:
        pass   # an unwritable venv only means the check runs again next start


def missing_required() -> list:
    """Required modules that cannot be imported, as the host would see them."""
    if not check_env._importable("agenty_core"):
        # The host puts the sibling checkout on sys.path when the editable install's
        # .pth cannot be read (macOS hides it); reinstalling cannot fix that, so it
        # must not count as missing here.
        check_env._bootstrap_agenty_core()
    return [module for module, _dist, _what in check_env.REQUIRED
            if not check_env._importable(module)]


def reasons(stamp, current: str, missing: list, changed: str) -> list:
    """Why an install is needed now; empty when it is not."""
    why = []
    if changed:
        why.append(f"the update changed dependencies in {changed}")
    if stamp is not None and stamp != current:
        why.append("requirements changed since this environment was last installed")
    if missing:
        why.append("missing: " + ", ".join(missing))
    return why


def install_command(python: str) -> list:
    """uv when it is there, pip otherwise; always naming this venv's interpreter
    (with a conda env active, an unnamed target is conda's, not this venv)."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", python, "-r", "requirements.txt"]
    return [python, "-m", "pip", "install", "-r", "requirements.txt"]


def main(argv: list) -> int:
    quiet = "--quiet" in argv
    changed = ""
    for arg in argv:
        if arg.startswith("--changed="):
            changed = arg.split("=", 1)[1].strip(" ,")

    current = fingerprint()
    stamp = read_stamp()
    why = reasons(stamp, current, missing_required(), changed)

    if not why:
        if stamp != current:
            write_stamp(current)   # first sight and nothing missing: remember it
        if not quiet:
            print("[deps] The environment is in line with requirements.txt.")
        return check_env.main(["--quiet"])

    print("[deps] Installing from requirements.txt - " + "; ".join(why), flush=True)
    command = install_command(sys.executable)
    try:
        # requirements.txt names agenty_core as `-e ../agenty_core`, relative to here.
        code = subprocess.run(command, cwd=str(ROOT)).returncode
    except OSError as exc:
        print(f"[deps] Could not run {command[0]}: {exc}")
        code = 1
    if code == 0:
        write_stamp(current)
    else:
        print(f"[deps] The install returned {code} - see the output above. agentY will "
              "still start, but may be missing features until it succeeds.")
    importlib.invalidate_caches()   # so the check below sees what was just installed
    return check_env.main(["--quiet"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
