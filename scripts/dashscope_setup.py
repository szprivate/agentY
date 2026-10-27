"""Set the DashScope endpoint that matches the key in .env (called by install_agent.ps1 / .sh).

Alibaba Model Studio sells pay-as-you-go, Token Plan and Coding Plan access; each has its own
key and its own base URL, and a mismatch answers 401. The installer used to ask for the key
and never for the URL. One place for the logic (the table is agenty_core.dashscope_endpoints):

    python scripts/dashscope_setup.py --env .env [--non-interactive] [--embedder]

* Detects the plan: a key starting ``sk-sp-`` is a Token/Coding Plan key; then one free
  GET <base>/models per documented endpoint, the first to answer 200 is taken. It says which.
* Not detected and interactive: asks the plan and the region.
* Writes ``DASHSCOPE_BASE_URL``. An existing one is kept.
* A plan key: quotes Alibaba's restriction to interactive coding tools, says generation and
  embeddings need a pay-as-you-go key, and offers to enter one as ``DASHSCOPE_PAYG_API_KEY``
  (+ ``DASHSCOPE_PAYG_BASE_URL`` for its region).
* ``--non-interactive`` asks nothing: detects, else leaves the default.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from agenty_core import dashscope_endpoints as ep  # noqa: E402

KEY, URL, PAYG_KEY, PAYG_URL = ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL",
                                "DASHSCOPE_PAYG_API_KEY", "DASHSCOPE_PAYG_BASE_URL")


def get_env(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(rf"\s*{re.escape(key)}\s*=\s*(.*)$", line)
        if m:
            return m.group(1).strip()
    return ""


def set_env(path: Path, key: str, value: str) -> None:
    """Replace the line of ``key`` (or append one); every other line stays byte for byte."""
    text = path.read_bytes().decode("utf-8", errors="replace") if path.is_file() else ""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"\s*{re.escape(key)}\s*=", line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((nl.join(lines) + nl).encode("utf-8"))


def _is_set(v: str) -> bool:
    return bool(v) and not re.fullmatch(r"sk-\.\.\.|", v)


def _choose(ask, say, title: str, options: list[tuple[str, str]], default: int) -> str | None:
    say(title)
    for i, (_v, label) in enumerate(options, 1):
        say(f"   {'*' if i == default else ' '}{i}) {label}")
    for _ in range(3):
        a = (ask(f"    [1-{len(options)}, Enter = {default}]: ") or "").strip()
        if not a:
            return options[default - 1][0]
        if a.isdigit() and 1 <= int(a) <= len(options):
            return options[int(a) - 1][0]
        say("  Type a number from the list.")
    return None


def _ask_endpoint(ask, say, plan_default: str, region_default: str = "singapore") -> tuple[str, str, str] | None:
    plans = [(ep.PAYG, "Pay-as-you-go (standard API key)"),
             (ep.TOKEN_PLAN, "Token Plan (key starts sk-sp-)"),
             (ep.CODING_PLAN, "Coding Plan (key starts sk-sp-)")]
    plan = _choose(ask, say, "  Which kind of DashScope access is this key from?", plans,
                   [p for p, _ in plans].index(plan_default) + 1)
    if not plan:
        return None
    regions = ep.regions_for(plan)
    region = _choose(ask, say, "  Which region?", [(r, ep.REGION_LABELS.get(r, r)) for r in regions],
                     regions.index(region_default) + 1 if region_default in regions else 1)
    return (plan, region, ep.endpoint_for(plan, region)) if region else None


def run(env_path: Path, interactive: bool, embedder: bool = False, *, ask=input, say=print,
        probe_fn=ep.probe) -> int:
    key = get_env(env_path, KEY)
    if not _is_set(key):
        return 0
    base = get_env(env_path, URL)
    plan = ep.plan_of_url(base) if base else None
    if base:
        say(f"  DashScope endpoint kept: {base}" + (f" ({ep.PLAN_LABELS[plan]})" if plan else ""))
    elif not embedder:
        hit = ep.detect(key, probe_fn)
        if hit:
            plan, region, base = hit
            say(f"  DashScope: this key works on {ep.PLAN_LABELS[plan]}, {ep.REGION_LABELS.get(region, region)} "
                f"({base})")
        elif interactive:
            say("  Could not tell which DashScope endpoint this key belongs to (offline, or a key none of "
                "the documented endpoints accepted).")
            got = _ask_endpoint(ask, say, ep.TOKEN_PLAN if ep.is_plan_key(key) else ep.PAYG)
            if got:
                plan, _region, base = got
        else:
            say(f"  DashScope: endpoint not detected; keeping the default {ep.DEFAULT_BASE_URL}")
            plan = ep.TOKEN_PLAN if ep.is_plan_key(key) else None
        if base:
            set_env(env_path, URL, base)
            say(f"  [ok] {URL} set")
    plan = plan or (ep.TOKEN_PLAN if ep.is_plan_key(key) else None)
    if plan in (ep.TOKEN_PLAN, ep.CODING_PLAN):
        say("")
        say("  NOTE: " + ep.TERMS_WARNING)
        say("  " + ep.PAYG_NEEDED)
        if interactive and not _is_set(get_env(env_path, PAYG_KEY)):
            pkey = (ask(f"    {PAYG_KEY} (pay-as-you-go key) [Enter = skip]: ") or "").strip()
            if pkey:
                set_env(env_path, PAYG_KEY, pkey)
                say(f"  [ok] {PAYG_KEY} set")
                got = ep.detect(pkey, probe_fn)
                if got:
                    say(f"  DashScope pay-as-you-go: this key works on {ep.REGION_LABELS.get(got[1], got[1])}")
                    purl = got[2]
                else:
                    got = _ask_endpoint(ask, say, ep.PAYG)
                    purl = got[2] if got else ""
                if purl:
                    set_env(env_path, PAYG_URL, purl)
                    say(f"  [ok] {PAYG_URL} set")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", required=True)
    ap.add_argument("--non-interactive", action="store_true")
    ap.add_argument("--embedder", action="store_true",
                    help="only warn about a plan key (the endpoint was set when the key was entered)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return run(Path(a.env), not a.non_interactive, a.embedder)


if __name__ == "__main__":
    raise SystemExit(main())
