"""Where the agentY-core checkout is, and moving an old install onto its new name.

The shared tool layer's repository was called ``agenty_core`` - the one name in
the stack that did not follow ``agentY-<part>``. It is now ``agentY-core``. Only
the repository and its folder changed: the Python package inside is still
``agenty_core`` (an import name cannot carry a hyphen).

A machine installed before the rename has ``<parent>/agenty_core``; everything
from here on expects ``<parent>/agentY-core``. :func:`migrate` closes that gap on
the machine itself, and is safe to run on every start:

* the old folder is **renamed**, and a link is left under the old name, so every
  absolute path that still says ``agenty_core`` - an editable install's finder, a
  Claude Desktop config, a script of the user's - keeps resolving;
* when the folder cannot be renamed right now (Windows refuses while a process
  has a file in it open), a link is made the other way round instead - the new
  name pointing at the old folder - and the rename is tried again next time;
* nothing is ever copied or deleted except a link this module made itself.

Run by scripts/sync_deps.py before it installs anything (requirements.txt names
``-e ../agentY-core``), and by the installers. Standard library only: it has to
work in a venv that has nothing installed yet.

    .venv/Scripts/python.exe scripts/core_dir.py            # migrate, say what happened
    .venv/Scripts/python.exe scripts/core_dir.py --where    # print the checkout's path
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "agentY-core"
LEGACY_NAME = "agenty_core"
PACKAGE = "agenty_core"          # the import name; unchanged by the rename


def is_link(path) -> bool:
    """A symlink, or a Windows junction (which ``os.path.islink`` does not see
    before Python 3.12)."""
    path = str(path)
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None:
        try:
            return bool(isjunction(path))
        except OSError:
            return False
    try:
        os.readlink(path)
        return True
    except (OSError, ValueError, NotImplementedError):
        return False


def is_checkout(path) -> bool:
    """The real thing: a folder (not a link) holding the ``agenty_core`` package."""
    path = Path(path)
    return path.is_dir() and not is_link(path) and (path / PACKAGE / "__init__.py").is_file()


def make_link(link, target) -> None:
    """*link* -> *target*, a directory. A junction on Windows (no admin rights or
    developer mode needed, unlike a symlink), a relative symlink elsewhere."""
    link, target = Path(link), Path(target)
    if os.name == "nt":
        try:
            import _winapi  # noqa: PLC0415
            _winapi.CreateJunction(str(target), str(link))
            return
        except (ImportError, AttributeError, OSError):
            pass
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise OSError(done.stderr.strip() or done.stdout.strip() or "mklink failed")
        return
    os.symlink(target.name if link.parent == target.parent else str(target), str(link),
               target_is_directory=True)


def remove_link(link) -> None:
    """Remove a link this module made - never what it points at."""
    link = str(link)
    if not is_link(link):
        raise OSError(f"{link} is not a link; refusing to remove it")
    if os.name == "nt":
        os.rmdir(link)       # on a junction this removes the junction only
    else:
        os.unlink(link)


def find(parent=None, env=None):
    """The checkout to use, or None: ``AGENTY_CORE_DIR``, then the new name, then
    the old one. A link under either name counts - it leads to the same files."""
    env = os.environ if env is None else env
    parent = Path(parent) if parent is not None else ROOT.parent
    named = str(env.get("AGENTY_CORE_DIR", "") or "").strip()
    candidates = ([Path(named).expanduser()] if named else []) + [parent / NAME, parent / LEGACY_NAME]
    for cand in candidates:
        if (cand / PACKAGE / "__init__.py").is_file():
            return cand
    return None


def migrate(parent=None) -> str:
    """Bring *parent* to the new layout. Returns what was done:

    ``"current"``  - already ``agentY-core`` (or nothing installed yet);
    ``"renamed"``  - the old folder was renamed and a link left under its old name;
    ``"linked"``   - the rename was refused; ``agentY-core`` is a link to the old
                     folder for now;
    ``"failed"``   - neither worked (the old folder is untouched).
    """
    parent = Path(parent) if parent is not None else ROOT.parent
    new, old = parent / NAME, parent / LEGACY_NAME

    if is_checkout(new):
        return "current"
    if not is_checkout(old):
        return "current"        # a fresh machine, or a layout that is not ours to touch

    had_link = is_link(new)
    if os.path.lexists(str(new)) and not had_link:
        return "failed"         # something else is in the way under the new name

    try:
        if had_link:
            remove_link(new)
        os.rename(str(old), str(new))
    except OSError:
        # In use right now. Make sure the new name resolves anyway.
        if not os.path.lexists(str(new)):
            try:
                make_link(new, old)
            except OSError:
                return "failed"
        return "linked"

    try:
        make_link(old, new)
    except OSError:
        pass                    # the rename stands; only the courtesy link is missing
    return "renamed"


MESSAGES = {
    "renamed": f"[core] The tool layer's folder is now {NAME} (it was {LEGACY_NAME}). A link "
               f"under the old name keeps existing paths working.",
    "linked": f"[core] {LEGACY_NAME} could not be renamed to {NAME} right now (a program has it "
              f"open), so {NAME} is a link to it for the moment. It is renamed on a later start.",
    "failed": f"[core] {LEGACY_NAME} could not be moved to its new name, {NAME}, and no link could "
              f"be made either. Close programs using that folder and start again, or rename it "
              f"by hand.",
}


def main(argv: list) -> int:
    if "--where" in argv:
        found = find()
        if found is None:
            return 1
        print(found)
        return 0
    result = migrate()
    if result in MESSAGES:
        print(MESSAGES[result], flush=True)
    return 1 if result == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
