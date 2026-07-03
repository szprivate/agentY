"""Python-only brain assembly pass over researcher_ok briefings.

Takes the ``researcher_ok`` briefings from a recipe-reliability report and runs
the *deterministic* brain happy-path in code — ``get_workflow_template`` ->
``apply_brainbriefing`` — with NO LLM. Measures how many briefings assemble
cleanly one-shot (apply returns ``status: ok``) versus needing an LLM brain
fix-up, and records the exact ``problems`` for the ones that fail so they can be
fixed at the source in ``apply_brainbriefing``.

Run:
    python -m scripts.recipe_reliability.brain_assembly
    python -m scripts.recipe_reliability.brain_assembly --report scripts/recipe_reliability/report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# Match the sweep's sub-HD latent clamp so assembly conditions are identical.
os.environ.setdefault("AGENTY_MAX_DIM", "768")

from agenty_core.tools.comfyui import get_workflow_template, apply_brainbriefing  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORT = os.path.join(_HERE, "report.json")
_OUT = os.path.join(_HERE, "brain_assembly_report.json")


def assemble(briefing_json: str) -> dict:
    """Run the deterministic assembly for one briefing. Returns a result dict
    with ``ok`` and, on failure, a ``reason`` and ``problems``."""
    try:
        bb = json.loads(briefing_json)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"briefing not JSON: {e}"}
    name = (bb.get("template") or {}).get("name")
    if not name:
        return {"ok": False, "reason": "no template name"}
    try:
        tinfo = json.loads(get_workflow_template(name))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"get_workflow_template failed: {str(e)[:200]}", "template": name}
    path = tinfo.get("workflow_path")
    if not path:
        return {"ok": False, "reason": "template returned no workflow_path", "template": name}
    try:
        res = json.loads(apply_brainbriefing(path, briefing_json))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"apply_brainbriefing exception: {str(e)[:200]}", "template": name}
    if res.get("status") == "ok":
        return {"ok": True, "template": name}
    probs = (res.get("problems") or res.get("server_errors")
             or res.get("node_errors") or res.get("local_errors") or [])
    return {"ok": False, "reason": "apply_brainbriefing error", "template": name, "problems": probs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=_REPORT, help="recipe-reliability report.json to read briefings from")
    args = ap.parse_args()

    data = json.load(open(args.report, encoding="utf-8"))
    oks = [r for r in data["results"]
           if r.get("outcome") == "researcher_ok" and r.get("briefing")]
    print(f"[assembly] {len(oks)} researcher_ok briefings to assemble (Python-only, no LLM)\n")

    results = []
    for r in oks:
        a = assemble(r["briefing"])
        a["id"] = r["id"]
        results.append(a)
        tag = "OK  " if a["ok"] else "FAIL"
        extra = "" if a["ok"] else f"  {a.get('reason','')}  {str(a.get('problems',''))[:220]}"
        print(f"  [{tag}] {r['id']:44s} tmpl={a.get('template')}{extra}")

    n_ok = sum(1 for x in results if x["ok"])
    print(f"\n[assembly] {n_ok}/{len(results)} assemble deterministically (no LLM brain needed)")
    # Aggregate failure reasons to guide fixes.
    fails = [x for x in results if not x["ok"]]
    if fails:
        print("[assembly] failure reasons:")
        for reason, n in Counter(x["reason"] for x in fails).most_common():
            print(f"   {n:3d}x  {reason}")
    json.dump({"total": len(results), "ok": n_ok, "results": results},
              open(_OUT, "w", encoding="utf-8"), indent=2)
    print(f"[assembly] report -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
