"""Check what is COMMITTED, not what is on this disk.

The working tree of a developer machine can hold edits that make committed code
work which, checked out on its own, does not: a hunk of src/pipeline.py that was
never committed, a folder whose name differs from the index only in case (this
disk does not care, a clean checkout does). Tests run against the working tree
and pass. So a release once shipped an agentY whose src/pipeline.py did not even
parse (b8a2820, in v1.8.7 to v1.8.10).

This exports a commit with ``git archive`` - exactly the committed files, with
their committed names - and checks the export:

* every .py compiles;
* every .js / .mjs passes ``node --check`` (when node is installed);
* agentY: every skill folder is named as its SKILL.md frontmatter ``name`` says;
* agentY: ``src.pipeline`` and ``src.agenty_ui_server`` import, against the
  exported agentY-core of the same check when there is one.

    .venv/Scripts/python.exe scripts/check_committed_tree.py           # agentY HEAD
    .venv/Scripts/python.exe scripts/check_committed_tree.py --stack   # all five repos
    .venv/Scripts/python.exe scripts/check_committed_tree.py --ref v1.8.10

scripts/make_release.py runs the --stack check before it tags anything.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"node_modules", "vendor", "__pycache__", ".venv"}
ENTRY_MODULES = ("src.pipeline", "src.agenty_ui_server")


def export(repo, ref: str, dest) -> Path:
    """Write the files of ``ref`` in ``repo`` under ``dest`` (created). Returns ``dest``."""
    done = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", ref], capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"git archive {ref} in {repo}: {done.stderr.decode(errors='replace').strip()}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def _files(root: Path, suffixes: tuple):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(suffixes):
                yield Path(dirpath) / name


def compile_errors(root) -> list:
    """``path: error`` for every .py under ``root`` that does not compile."""
    root = Path(root)
    bad = []
    for path in sorted(_files(root, (".py",))):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            line = getattr(exc, "lineno", None)
            bad.append(f"{path.relative_to(root).as_posix()}{f':{line}' if line else ''}: "
                       f"{type(exc).__name__}: {getattr(exc, 'msg', None) or exc}")
    return bad


def js_errors(root) -> list:
    """``path: error`` for every .js/.mjs that ``node --check`` rejects ([] without node)."""
    node = shutil.which("node")
    if not node:
        return []
    root = Path(root)
    bad = []
    for path in sorted(_files(root, (".js", ".mjs"))):
        done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if done.returncode != 0:
            first = next((l for l in done.stderr.splitlines() if "Error" in l), done.stderr.strip()[:200])
            bad.append(f"{path.relative_to(root).as_posix()}: {first}")
    return bad


def skill_name_errors(root) -> list:
    """Skill folders whose name is not their SKILL.md frontmatter name (case counts)."""
    bad = []
    skills = Path(root) / "skills"
    if not skills.is_dir():
        return bad
    for md in sorted(skills.glob("*/SKILL.md")):
        m = re.search(r"^name:\s*['\"]?([^'\"\r\n]+?)['\"]?\s*$", md.read_text(encoding="utf-8", errors="replace"), re.M)
        if m and m.group(1) != md.parent.name:
            bad.append(f"skills/{md.parent.name}: frontmatter name is '{m.group(1)}'")
    return bad


def import_errors(root, core=None, modules=ENTRY_MODULES, python: str = sys.executable) -> list:
    """Import ``modules`` from the exported agentY at ``root`` in a fresh interpreter."""
    paths = [str(root)] + ([str(core)] if core else [])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "PYTHONIOENCODING": "utf-8"}
    code = "import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\n"
    done = subprocess.run([python, "-c", code, *modules], cwd=str(root), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    if done.returncode == 0:
        return []
    tail = [l for l in done.stderr.strip().splitlines() if l.strip()][-3:]
    return ["import " + ", ".join(modules) + " failed: " + " | ".join(t.strip() for t in tail)]


def check_repo(name: str, repo, ref: str = "HEAD", work=None, core=None, imports: bool = True) -> list:
    """Problems of ``ref`` in ``repo`` as committed, each prefixed with ``name``."""
    with tempfile.TemporaryDirectory(prefix="agenty_committed_") as tmp:
        root = export(repo, ref, Path(work or tmp) / name)
        found = compile_errors(root) + js_errors(root)
        if name == "agentY":
            found += skill_name_errors(root)
            if imports and not found:
                found += import_errors(root, core)
        return [f"{name} {ref}: {p}" for p in found]


def check_stack(dirs: dict, ref: str = "HEAD", imports: bool = True) -> list:
    """Every repository in ``dirs`` ({name: path}) at ``ref``; agentY imported against agentY-core's export."""
    problems = []
    with tempfile.TemporaryDirectory(prefix="agenty_committed_") as tmp:
        core = None
        for name, path in dirs.items():
            if name == "agentY":
                continue
            root = export(path, ref, Path(tmp) / name)
            problems += [f"{name} {ref}: {p}" for p in compile_errors(root) + js_errors(root)]
            if name == "agentY-core":
                core = root
        if "agentY" in dirs:
            problems += check_repo("agentY", dirs["agentY"], ref, work=tmp, core=core, imports=imports)
    return problems


def main(argv: list) -> int:
    ref = "HEAD"
    for i, a in enumerate(argv):
        if a == "--ref" and i + 1 < len(argv):
            ref = argv[i + 1]
        elif a.startswith("--ref="):
            ref = a.split("=", 1)[1]
    if "--stack" in argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_release import repo_dirs
        problems = check_stack(repo_dirs(), ref)
    else:
        problems = check_repo("agentY", ROOT, ref)
    for p in problems:
        print(p)
    if not problems:
        print(f"{ref}: the committed tree compiles and imports.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
