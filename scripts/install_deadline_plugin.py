"""Install (or update) the agentY ComfyUI plugin into a Deadline 10 repository.

YOU run this - agentY never does. It only copies files; it does not create pools, assign
workers, or submit anything.

    python scripts/install_deadline_plugin.py --dry-run
    python scripts/install_deadline_plugin.py
    python scripts/install_deadline_plugin.py --repository \\\\bepic-storage01\\farm\\deadline
    python scripts/install_deadline_plugin.py --uninstall

The repository is, in order: --repository, the DEADLINE_REPOSITORY setting (.env), then what
`deadlinecommand -GetRepositoryRoot` answers. The plugin lands in
`<repository>/custom/plugins/ComfyUI`. An existing copy that differs is moved aside first, to
`<repository>/custom/_backup/ComfyUI-<time>` (outside the plugins folder, so Deadline does not
list it as a plugin; delete it when you are happy). Workers pick the new files up on their
next job.

Afterwards, in Deadline Monitor: Tools > Configure Plugins > ComfyUI, set the ComfyUI folder for
your workers; and create the pool (see the commands printed at the end, or docs/deadline.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "deadline" / "plugins" / "ComfyUI"
PLUGIN_FILES = ("ComfyUI.py", "ComfyUI.param", "ComfyUI.options")
IGNORED = ("__pycache__", ".pyc")


def _env(key: str) -> str:
    """A value from the process environment or the checkout's .env."""
    if os.environ.get(key):
        return os.environ[key].strip().strip('"')
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key and v.strip():
                return v.strip().strip('"')
    except OSError:
        pass
    return ""


def _deadlinecommand() -> str:
    names = ["deadlinecommand.exe"] if os.name == "nt" else ["deadlinecommand"]
    folders = [_env("DEADLINE_PATH"), r"C:\Program Files\Thinkbox\Deadline10\bin",
               "/opt/Thinkbox/Deadline10/bin", "/Applications/Thinkbox/Deadline10/Resources"]
    for f in folders:
        for n in names:
            if f and (Path(f) / n).is_file():
                return str(Path(f) / n)
    return shutil.which("deadlinecommand") or ""


def find_repository(explicit: str = "") -> tuple:
    """(path, where it came from) or ("", reason)."""
    if explicit:
        return explicit, "--repository"
    env = _env("DEADLINE_REPOSITORY")
    if env:
        return env, "DEADLINE_REPOSITORY"
    exe = _deadlinecommand()
    if exe:
        try:
            out = subprocess.run([exe, "-GetRepositoryRoot"], capture_output=True, text=True,
                                 timeout=60).stdout.strip()
            if out and not out.lower().startswith("error"):
                return out.splitlines()[-1].strip(), "deadlinecommand -GetRepositoryRoot"
        except (OSError, subprocess.SubprocessError):
            pass
    return "", ("no repository given: pass --repository, set DEADLINE_REPOSITORY in .env, or make "
                "deadlinecommand available (DEADLINE_PATH)")


def _files(folder: Path) -> dict:
    out = {}
    if folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.is_file() and not any(i in p.parts or p.name.endswith(i) for i in IGNORED):
                out[str(p.relative_to(folder)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def plan(repository: Path) -> dict:
    """What an install would do, without doing it."""
    target = repository / "custom" / "plugins" / "ComfyUI"
    src, dst = _files(SOURCE), _files(target)
    missing = [f for f in PLUGIN_FILES if f not in src]
    return {"source": SOURCE, "target": target, "missing_in_source": missing,
            "new": sorted(set(src) - set(dst)), "changed": sorted(k for k in src if k in dst and src[k] != dst[k]),
            "removed": sorted(set(dst) - set(src)), "same": sorted(k for k in src if k in dst and src[k] == dst[k]),
            "installed": target.is_dir()}


def install(repository: Path, *, dry_run: bool = False, log=print) -> dict:
    """Copy the plugin; returns the plan plus ``backup`` (path or "")."""
    p = plan(repository)
    if p["missing_in_source"]:
        raise SystemExit(f"the plugin source is incomplete, missing: {', '.join(p['missing_in_source'])} "
                         f"(in {SOURCE})")
    if not repository.is_dir():
        raise SystemExit(f"repository not reachable: {repository}")
    p["backup"] = ""
    if not (p["new"] or p["changed"] or p["removed"]):
        log(f"already up to date: {p['target']}")
        return p
    for label in ("new", "changed", "removed"):
        for f in p[label]:
            log(f"  {label:8} {f}")
    if dry_run:
        log(f"dry run: nothing written. Would install into {p['target']}")
        return p
    target: Path = p["target"]
    if p["installed"] and (p["changed"] or p["removed"]):
        backup = repository / "custom" / "_backup" / f"ComfyUI-{time.strftime('%Y%m%d-%H%M%S')}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(backup))
        p["backup"] = str(backup)
        log(f"existing plugin moved to {backup}")
    target.mkdir(parents=True, exist_ok=True)
    for rel in _files(SOURCE):
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / rel, dst)
    # verify what landed, byte for byte
    if _files(SOURCE) != _files(target):
        raise SystemExit(f"the copy in {target} does not match the source after installing - check permissions")
    log(f"installed {len(_files(SOURCE))} file(s) into {target}")
    return p


def uninstall(repository: Path, *, dry_run: bool = False, log=print) -> bool:
    target = repository / "custom" / "plugins" / "ComfyUI"
    if not target.is_dir():
        log("not installed")
        return False
    if dry_run:
        log(f"dry run: would remove {target}")
        return True
    backup = repository / "custom" / "_backup" / f"ComfyUI-removed-{time.strftime('%Y%m%d-%H%M%S')}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(backup))
    log(f"plugin moved to {backup} (delete it when you are sure)")
    return True


def next_steps(pool: str = "comfyui") -> str:
    return f"""
Next, in Deadline (not done by this script):
  1. Monitor > Tools > Configure Plugins > ComfyUI: set the ComfyUI folder per OS (and the Python,
     extra arguments, the Comfy.org key variable if you use partner nodes).
  2. Create the pool and put the machines that have ComfyUI in it - and only those:
       deadlinecommand -AddPool {pool}
       deadlinecommand -AddPoolToSlave <worker1,worker2> {pool}
       deadlinecommand -GetSlaveNamesInPool {pool}
  3. In agentY's .env: DEADLINE_OUTPUT_DIR (a folder the workers can write), and DEADLINE_PATH or
     DEADLINE_WEBSERVICE_URL. Then ask agentY "is the farm ready?" (deadline_farm_info).
  See docs/deadline.md.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repository", default="", help="the Deadline repository (UNC path or folder)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    ap.add_argument("--uninstall", action="store_true", help="move the plugin out of the repository")
    args = ap.parse_args(argv)

    repo, how = find_repository(args.repository)
    if not repo:
        print(how, file=sys.stderr)
        return 2
    print(f"Deadline repository: {repo}  ({how})")
    repository = Path(repo)
    if not repository.is_dir():
        print(f"not reachable from this machine: {repo}", file=sys.stderr)
        return 2
    if args.uninstall:
        uninstall(repository, dry_run=args.dry_run)
        return 0
    install(repository, dry_run=args.dry_run)
    if not args.dry_run:
        print(next_steps(_env("DEADLINE_COMFYUI_POOL") or "comfyui"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
