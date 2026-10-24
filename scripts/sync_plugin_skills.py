"""Keep the agentY-mcp Claude Code plugin's skills in step with agentY's.

agentY/skills/ is the source of truth. The plugin (agentY-mcp/skills/) is a mirror
of it: this script adds new skills, updates changed files and removes skills that
are gone, leaving out the host-only ones listed in config/plugin_skills.json (each
with its reason). It also writes agentY-mcp/skills.sync.json - the file count and
hash of every mirrored skill - so a test in agentY-mcp can tell the folder was
edited by hand or went stale.

    .venv/Scripts/python.exe scripts/sync_plugin_skills.py            # mirror
    .venv/Scripts/python.exe scripts/sync_plugin_skills.py --check    # exit 1 if it would change anything
    .venv/Scripts/python.exe scripts/sync_plugin_skills.py --report --tools=<file>   # shipped skills naming non-MCP tools

Never edit agentY-mcp/skills/ by hand; the next sync would undo it.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "plugin_skills.json"
SYNC_MANIFEST = "skills.sync.json"
_SKIP_PARTS = {"__pycache__", ".git"}


def load_config(path: Path = CONFIG) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.setdefault("exclude", {})
    data.setdefault("notes", {})
    return data


def mcp_repo(parent: Path | None = None) -> Path | None:
    p = (Path(parent) if parent else ROOT.parent) / "agentY-mcp"
    return p if (p / ".git").exists() else None


def _frontmatter_name(skill_md: Path) -> str | None:
    text = skill_md.read_text(encoding="utf-8-sig")
    m = re.match(r"---\s*\r?\n(.*?)\r?\n---", text, re.S)
    if not m:
        return None
    n = re.search(r"^name:\s*(.+?)\s*$", m.group(1), re.M)
    return n.group(1).strip().strip("\"'") if n else None


def _files(folder: Path) -> list:
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and not (_SKIP_PARTS & set(p.relative_to(folder).parts)) and p.suffix != ".pyc")


def _norm(data: bytes) -> bytes:
    """Line endings do not count as a difference (git autocrlf differs per machine)."""
    return data.replace(b"\r\n", b"\n")


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for f in _files(folder):
        h.update(f.relative_to(folder).as_posix().encode() + b"\0" + _norm(f.read_bytes()) + b"\0")
    return h.hexdigest()


def source_skills(src: Path, exclude: dict) -> dict:
    """{name: folder} for every skill that is shipped. Problems raise ValueError."""
    out: dict = {}
    for d in sorted(p for p in Path(src).iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if d.name in exclude:
            continue
        md = d / "SKILL.md"
        if not md.is_file():
            raise ValueError(f"skills/{d.name} has no SKILL.md - add it, or list the folder under 'exclude' in config/plugin_skills.json with a reason")
        fm = _frontmatter_name(md)
        if fm != d.name:
            raise ValueError(f"skills/{d.name}: frontmatter name is {fm!r}; the folder and the name must match (Claude Code names the skill after the folder)")
        out[d.name] = d
    for name in exclude:
        if not (Path(src) / name).exists():
            raise ValueError(f"config/plugin_skills.json excludes '{name}', which no longer exists in skills/ - remove the entry")
    return out


def plan(src: Path, dst: Path, exclude: dict) -> dict:
    """What a sync would do to the skills folder: names to add, update, remove."""
    skills = source_skills(src, exclude)
    have = {p.name for p in Path(dst).iterdir() if p.is_dir()} if Path(dst).is_dir() else set()
    return {
        "add": sorted(set(skills) - have),
        "update": sorted(n for n in set(skills) & have if _digest(skills[n]) != _digest(Path(dst) / n)),
        "remove": sorted(have - set(skills)),
    }


def manifest_for(src: Path, exclude: dict) -> dict:
    skills = source_skills(src, exclude)
    return {
        "_comment": "Written by agentY/scripts/sync_plugin_skills.py - do not edit. skills/ is a mirror of agentY/skills/.",
        "excluded": sorted(exclude),
        "skills": {n: {"files": len(_files(d)), "sha256": _digest(d)} for n, d in skills.items()},
    }


def _manifest_text(m: dict) -> str:
    return json.dumps(m, indent=2, ensure_ascii=False) + "\n"


def changes(src: Path, mcp: Path, exclude: dict) -> dict:
    """plan() plus whether skills.sync.json would change."""
    p = plan(src, mcp / "skills", exclude)
    mf = mcp / SYNC_MANIFEST
    want = _manifest_text(manifest_for(src, exclude))
    cur = mf.read_text(encoding="utf-8") if mf.is_file() else ""
    p["manifest"] = _norm(cur.encode()) != _norm(want.encode())
    return p


def is_clean(p: dict) -> bool:
    return not (p["add"] or p["update"] or p["remove"] or p["manifest"])


def sync(src: Path, mcp: Path, exclude: dict) -> dict:
    p = changes(src, mcp, exclude)
    dst = mcp / "skills"
    dst.mkdir(exist_ok=True)
    skills = source_skills(src, exclude)
    for name in p["remove"]:
        shutil.rmtree(dst / name)
    for name in p["add"] + p["update"]:
        target = dst / name
        if target.exists():
            shutil.rmtree(target)
        for f in _files(skills[name]):
            out = target / f.relative_to(skills[name])
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)
    (mcp / SYNC_MANIFEST).write_text(_manifest_text(manifest_for(src, exclude)), encoding="utf-8", newline="\n")
    return p


def describe(p: dict) -> str:
    parts = [f"{k}: {', '.join(p[k])}" for k in ("add", "update", "remove") if p[k]]
    if p["manifest"] and not parts:
        parts.append("skills.sync.json out of date")
    return "; ".join(parts) or "in sync"


def report(src: Path, exclude: dict, mcp_tools: set) -> list:
    """Shipped skills whose allowed-tools name something the MCP server does not offer."""
    rows = []
    for name, d in source_skills(src, exclude).items():
        text = (d / "SKILL.md").read_text(encoding="utf-8-sig")
        m = re.search(r"^allowed-tools:[ \t]*(.*)$", text, re.M)
        names = [t for t in re.split(r"[,\s]+", m.group(1)) if t] if m else []
        extra = [t for t in names if t not in mcp_tools and "__" not in t]
        if extra:
            rows.append((name, extra))
    return rows


def main(argv: list) -> int:
    cfg = load_config()
    mcp = mcp_repo()
    if mcp is None:
        print("agentY-mcp is not next to agentY.")
        return 2
    src = ROOT / "skills"
    try:
        if "--report" in argv:
            tools = [a.split("=", 1)[1] for a in argv if a.startswith("--tools=")]
            if not tools:
                print("--report needs --tools=<file with the MCP tool names, one per line>")
                return 2
            for n, extra in report(src, cfg["exclude"], set(Path(tools[0]).read_text().split())):
                print(f"{n}: {', '.join(extra)}  -> {cfg['notes'].get(n, 'NOT NOTED in config/plugin_skills.json')}")
            return 0
        if "--check" in argv:
            p = changes(src, mcp, cfg["exclude"])
            if is_clean(p):
                print("Plugin skills are in sync.")
                return 0
            print("Plugin skills are OUT OF DATE - " + describe(p)
                  + "\nRun: .venv/Scripts/python.exe scripts/sync_plugin_skills.py")
            return 1
        p = sync(src, mcp, cfg["exclude"])
    except ValueError as e:
        print(f"Cannot sync: {e}")
        return 2
    print("Synced the plugin's skills: " + describe(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
