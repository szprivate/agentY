"""Make a release of the whole agentY stack, and read what a release is made of.

agentY is five repositories - agentY, agentY-core, agentY-comfyuiConnect,
agentY-mcp and ComfyUI-ImageViewer (the bEpic viewer) - that only work as a set. A release pins that set:

* ``release.toml`` (in agentY) records the version and the commit of each
  sibling it was made with;
* ``requirements.lock`` records the Python package versions it was tested with
  (used as pip constraints by the installers and the launcher);
* before anything is tagged, the Claude Code plugin in agentY-mcp is brought up to the
  release: its skills are mirrored from agentY (scripts/sync_plugin_skills.py), its plugin,
  marketplace and .mcpb versions are set to the release version, the manifest is validated,
  and that is committed and pushed on agentY-mcp's branch (--dry-run reports it);
* every repository gets the tag ``v<version>``, a GitHub release, and its
  ``stable`` branch moved to that commit.

Machines on the ``stable`` update channel (the default) follow the ``stable``
branches, so they move from one tested set to the next and never sit on a
half-updated mix. ``update_channel = "dev"`` follows every commit instead.

    .venv/Scripts/python.exe scripts/make_release.py 1.1.0 --dry-run   # what it would do
    .venv/Scripts/python.exe scripts/make_release.py 1.1.0             # do it
    .venv/Scripts/python.exe scripts/make_release.py --show            # the current release
    .venv/Scripts/python.exe scripts/make_release.py --check           # is this machine on it?

New work goes to each repository's `dev` branch first. A release takes the
tested `dev` commits and moves the default branch (`main`; `master` in
agentY-core and ComfyUI-ImageViewer) and `stable` to them, in all five repositories under one version
number. Every repository is released at the commit it has checked out, which
must already be pushed. Run the tests first; this does not.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_dir  # noqa: E402

MANIFEST = ROOT / "release.toml"
LOCK = ROOT / "requirements.lock"
SIBLINGS = ("agentY-core", "agentY-comfyuiConnect", "agentY-mcp", "ComfyUI-ImageViewer")
STABLE = "stable"


# ── reading ──────────────────────────────────────────────────────────────────
def parse_manifest(text: str) -> dict:
    """``{"version": "1.0.0", "date": "...", "repos": {name: {"tag":…, "commit":…}}}``.

    A tiny reader for the tiny file ``render_manifest`` writes, so the launcher's
    Python needs no TOML library (3.10 has none)."""
    out: dict = {"version": "", "date": "", "repos": {}}
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = re.fullmatch(r"\[repos\.\"?([A-Za-z0-9_.\-]+)\"?\]", line)
        if head:
            section = out["repos"].setdefault(head.group(1), {})
            continue
        pair = re.fullmatch(r"([A-Za-z_]+)\s*=\s*\"([^\"]*)\"", line)
        if not pair:
            continue
        if section is None:
            out[pair.group(1)] = pair.group(2)
        else:
            section[pair.group(1)] = pair.group(2)
    return out


def read_manifest(path: Path = MANIFEST) -> dict | None:
    try:
        return parse_manifest(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None


def render_manifest(version: str, date: str, repos: dict) -> str:
    lines = [
        "# The set of versions that make up this release of agentY. Written by",
        "# scripts/make_release.py - do not edit by hand.",
        "#",
        "# Each sibling repository carries the same tag, and its `stable` branch points",
        "# at the commit named here.",
        f'version = "{version}"',
        f'date = "{date}"',
        "",
    ]
    for name in SIBLINGS:
        if name in repos:
            lines += [f'[repos."{name}"]', f'tag = "{repos[name]["tag"]}"', f'commit = "{repos[name]["commit"]}"', ""]
    return "\n".join(lines)


def repo_dirs(parent: Path | None = None) -> dict:
    """Where each repository is on this machine (missing ones are left out)."""
    parent = Path(parent) if parent is not None else ROOT.parent
    found = {"agentY": ROOT}
    core = core_dir.find(parent)
    if core is not None:
        found["agentY-core"] = Path(core)
    for name in ("agentY-comfyuiConnect", "agentY-mcp", "ComfyUI-ImageViewer"):
        if (parent / name / ".git").exists():
            found[name] = parent / name
    return found


def channel(env=None, settings: Path | None = None) -> str:
    """``stable`` unless the environment or settings.local.json says ``dev``."""
    env = os.environ if env is None else env
    value = str(env.get("AGENTY_UPDATE_CHANNEL", "") or "").strip()
    if not value:
        try:
            path = settings or (ROOT / "config" / "settings.local.json")
            value = str(json.loads(Path(path).read_text(encoding="utf-8-sig")).get("update_channel") or "")
        except (OSError, ValueError, AttributeError):
            value = ""
    return "dev" if value.strip().lower() == "dev" else STABLE


def _git(cwd, *args, check=True) -> str:
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {cwd}: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


def default_branch(path) -> str:
    """The repository's default branch on the remote: main, or master."""
    head = _git(path, "symbolic-ref", "-q", "--short", "refs/remotes/origin/HEAD", check=False)
    return head.split("/", 1)[1] if "/" in head else "main"


def drift(manifest: dict | None = None, dirs: dict | None = None) -> list:
    """Siblings that are NOT at the commit this release was made with, as
    ``(name, have, want)``. Empty when the machine runs the released set."""
    manifest = read_manifest() if manifest is None else manifest
    if not manifest:
        return []
    dirs = repo_dirs() if dirs is None else dirs
    off = []
    for name, info in (manifest.get("repos") or {}).items():
        want = info.get("commit") or ""
        if not want or name not in dirs:
            continue
        have = _git(dirs[name], "rev-parse", "HEAD", check=False)
        if have and have != want:
            off.append((name, have[:7], want[:7]))
    return off


def drift_note() -> str:
    """One line for the launcher, or "" - only on the stable channel, where the
    promise is that the five repositories are a released set."""
    try:
        if channel() != STABLE:
            return ""
        manifest = read_manifest()
        off = drift(manifest)
        if not off:
            return ""
        parts = ", ".join(f"{name} is at {have} (the release has {want})" for name, have, want in off)
        return (f"[release] Not the set agentY {manifest.get('version')} was released with: {parts}. "
                "It updates on the next start; if this stays, see docs/reference.md > Releases.")
    except Exception:  # noqa: BLE001 - a note, never a reason to fail a start
        return ""


# ── the Claude Code plugin (agentY-mcp) ──────────────────────────────────────
# Paths in agentY-mcp that a release rewrites, and nothing else.
PLUGIN_PATHS = ("skills", "skills.sync.json", "manifest.json", ".claude-plugin")


def _mcp_python(mcp: Path) -> str | None:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (Path(mcp) / rel).is_file():
            return str(Path(mcp) / rel)
    return None


def _update_manifest(mcp: Path, *args) -> subprocess.CompletedProcess:
    py = _mcp_python(mcp)
    if py is None:
        raise RuntimeError(f"{mcp} has no .venv - run the installer (or: uv venv; uv pip install -r requirements.txt) there first")
    return subprocess.run([py, "scripts/update_manifest.py", *args], cwd=str(mcp), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def plugin_prepare(version: str, mcp: Path, dry_run: bool) -> tuple:
    """Make the Claude Code plugin in agentY-mcp match this release: the skills mirrored from
    agentY, the plugin / marketplace / .mcpb versions set to ``version`` and the .mcpb tool list
    current, the plugin manifest validated - committed to agentY-mcp on its branch and pushed,
    BEFORE the repositories are tagged. Returns ``(ok, lines)``. With ``dry_run`` nothing is written.

    Refuses (rather than guessing) when those paths already hold uncommitted changes: the
    commit would sweep somebody's work into a release."""
    import sync_plugin_skills as sps

    lines: list = []
    cfg = sps.load_config()
    try:
        before = sps.changes(ROOT / "skills", mcp, cfg["exclude"])
    except ValueError as e:
        return False, [f"plugin skills cannot be synced: {e}"]
    dirty = _git(mcp, "status", "--porcelain", "--", *PLUGIN_PATHS, check=False)
    if dirty:
        return False, ["agentY-mcp has uncommitted changes in the plugin files - commit or discard them first:",
                       *["  " + l for l in dirty.splitlines()[:10]]]
    lines.append("  plugin skills: " + ("in sync" if sps.is_clean(before) else "would be synced (" + sps.describe(before) + ")"))
    try:
        if dry_run:
            off = _update_manifest(mcp, "--check", "--version", version)
            lines.append("  plugin versions / tool list: " + ("current" if off.returncode == 0 else
                         "would be updated to " + version + " (" + off.stdout.strip().splitlines()[0] + ")"))
            ok = _update_manifest(mcp, "--validate")
            lines.append("  plugin manifest: " + (ok.stdout.strip().splitlines() or ["?"])[-1])
            return ok.returncode == 0, lines + ["  would commit and push these in agentY-mcp before tagging"]
        sps.sync(ROOT / "skills", mcp, cfg["exclude"])
        done = _update_manifest(mcp, "--version", version, "--validate")
    except RuntimeError as e:
        return False, lines + [str(e)]
    lines.append("  " + " | ".join(l.strip() for l in done.stdout.strip().splitlines()) if done.stdout.strip() else "  versions set")
    if done.returncode != 0:
        # the paths were clean before, so put them back: nothing is committed, a rerun starts clean
        _git(mcp, "checkout", "-q", "--", *PLUGIN_PATHS, check=False)
        _git(mcp, "clean", "-fdq", "--", "skills", check=False)
        return False, lines + ["the plugin manifest is not valid (claude plugin validate) - fix it in agentY-mcp, then run again"]
    # refresh the index of skills/ first: on Windows/macOS git keeps the OLD case of a renamed folder
    _git(mcp, "rm", "-r", "-q", "--cached", "--ignore-unmatch", "--", "skills", check=False)
    _git(mcp, "add", "--", *PLUGIN_PATHS)
    if _git(mcp, "status", "--porcelain", "--", *PLUGIN_PATHS, check=False):
        _git(mcp, "commit", "-q", "-m", f"Plugin for release v{version}: skills mirrored from agentY, versions set\n\n"
             "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        branch = _git(mcp, "branch", "--show-current")
        _git(mcp, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
        lines.append(f"  agentY-mcp: plugin committed and pushed ({_git(mcp, 'rev-parse', '--short', 'HEAD')})")
    else:
        lines.append("  agentY-mcp: plugin already current, nothing to commit")
    return True, lines


# ── making ───────────────────────────────────────────────────────────────────
def build_lock(python: str = sys.executable) -> str:
    """requirements.lock: every package pinned, for Windows, macOS and Linux, at
    the versions installed in this environment where it has them."""
    frozen = subprocess.run(["uv", "pip", "freeze", "--python", python], capture_output=True, text=True, check=True).stdout
    keep = []
    for line in frozen.splitlines():
        if not line or line.startswith("-e ") or "@ file:" in line or line.lower().startswith("agenty-core"):
            continue
        keep.append(re.sub(r"\+[A-Za-z0-9.]+$", "", line))       # 2.11.0+cu128 -> 2.11.0
    core = core_dir.find()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "installed.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
        plain = [l for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if not l.strip().startswith("-e ")]
        (tmp / "requirements.txt").write_text("\n".join(plain) + "\n", encoding="utf-8")
        sources = [str(tmp / "requirements.txt")] + ([str(Path(core) / "pyproject.toml")] if core else [])
        out = subprocess.run(
            ["uv", "pip", "compile", *sources, "--universal", "--python-version", "3.12",
             "-c", str(tmp / "installed.txt"), "--no-header", "--no-annotate", "-q"],
            capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError("uv pip compile failed:\n" + out.stderr.strip())
    header = (
        "# The package versions this release of agentY was tested with, for every\n"
        "# platform. Used as constraints: `uv pip install -r requirements.txt -c\n"
        "# requirements.lock` (the installers and the launcher do this). Written by\n"
        "# scripts/make_release.py - do not edit by hand.\n"
        "#\n"
        "# torch is pinned by version only: on Windows with an NVIDIA GPU the installer\n"
        "# takes that version from the CUDA wheel index instead of PyPI.\n"
    )
    return header + out.stdout.strip() + "\n"


def release(version: str, dry_run: bool = False, lock: bool = True, notes: str = "") -> int:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print(f"'{version}' is not a version like 1.2.0"); return 2
    tag = f"v{version}"
    dirs = repo_dirs()
    missing = [n for n in ("agentY", *SIBLINGS) if n not in dirs]
    if missing:
        print("Not found next to agentY: " + ", ".join(missing)); return 2

    problems = []
    for name, path in dirs.items():
        _git(path, "fetch", "--quiet", "--tags", check=False)
        if _git(path, "tag", "--list", tag):
            problems.append(f"{name} already has the tag {tag}")
        if name != "agentY" and not _git(path, "branch", "-r", "--contains", "HEAD", check=False):
            problems.append(f"{name}: the checked-out commit is not pushed")
    if problems:
        print("Cannot release:\n  " + "\n  ".join(problems)); return 2

    print(f"Claude Code plugin (agentY-mcp):")
    ok, plugin_lines = plugin_prepare(version, dirs["agentY-mcp"], dry_run)
    print("\n".join(plugin_lines))
    if not ok:
        print("Cannot release: the plugin is not ready (see above)."); return 2

    repos = {n: {"tag": tag, "commit": _git(dirs[n], "rev-parse", "HEAD")} for n in SIBLINGS}
    manifest = render_manifest(version, datetime.date.today().isoformat(), repos)
    print(f"Release {tag}")
    for name in SIBLINGS:
        print(f"  {name:24} {repos[name]['commit'][:7]}  ({_git(dirs[name], 'branch', '--show-current') or 'detached'})")
    if dry_run:
        print("\n--dry-run: nothing written, tagged or pushed.\n\n" + manifest); return 0

    if lock:
        print("Writing requirements.lock …")
        LOCK.write_text(build_lock(), encoding="utf-8", newline="\n")
    MANIFEST.write_text(manifest, encoding="utf-8", newline="\n")
    agent = dirs["agentY"]
    _git(agent, "add", "release.toml", *(["requirements.lock"] if LOCK.is_file() else []))
    _git(agent, "commit", "-q", "-m", f"Release {tag}")
    branch = _git(agent, "branch", "--show-current")
    _git(agent, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
    print(f"  agentY                   {_git(agent, 'rev-parse', 'HEAD')[:7]}  ({branch})")
    if branch != "dev":
        print(f"  note: agentY is on '{branch}', not 'dev' - releases are meant to be cut from dev.")

    body = notes or f"agentY {version}."
    for name, path in dirs.items():
        _git(path, "tag", "-a", tag, "-m", f"agentY {version}")
        _git(path, "push", "-q", "origin", tag)
        # The default branch and `stable` both become the release. They only ever
        # move forward; a push that would rewind one is refused by git, and that
        # is the right answer.
        for line in dict.fromkeys((default_branch(path), STABLE)):
            moved = subprocess.run(["git", "-C", str(path), "push", "-q", "origin",
                                    f"{tag}^{{commit}}:refs/heads/{line}"], capture_output=True, text=True)
            if moved.returncode != 0:
                print(f"  ! {name}: could not move `{line}` to {tag}: {moved.stderr.strip()}")
        made = subprocess.run(["gh", "release", "create", tag, "--title", f"agentY {version}", "--notes", body,
                               "--verify-tag"], cwd=str(path), capture_output=True, text=True)
        print(f"  {name}: tagged, {default_branch(path)} and stable moved" + (", GitHub release made" if made.returncode == 0
                                                    else f" - GitHub release FAILED: {made.stderr.strip()}"))
    return 0


def main(argv: list) -> int:
    if "--show" in argv:
        manifest = read_manifest()
        if not manifest:
            print("No release.toml - this checkout is not a release."); return 1
        print(f"agentY {manifest['version']} ({manifest['date']}), channel: {channel()}")
        for name, info in manifest["repos"].items():
            print(f"  {name:24} {info.get('tag', ''):8} {info.get('commit', '')[:7]}")
        return 0
    if "--check" in argv:
        off = drift()
        for name, have, want in off:
            print(f"{name}: at {have}, the release has {want}")
        if not off:
            print("Every repository is at the released commit.")
        return 1 if off else 0
    versions = [a for a in argv if not a.startswith("--")]
    if len(versions) != 1:
        print(__doc__); return 2
    notes = ""
    for a in argv:
        if a.startswith("--notes-file="):
            notes = Path(a.split("=", 1)[1]).read_text(encoding="utf-8")
    return release(versions[0], dry_run="--dry-run" in argv, lock="--no-lock" not in argv, notes=notes)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
