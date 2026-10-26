"""Smoke test of the ayon tools against a LIVE Ayon session.

Run it only against a session the USER started from the Ayon launcher (project i002_ai_rnd, folder
agentRND/agentRND_0010, task ai; see docs/dev/ayon.md section 9). It never starts anything.

    .venv\\Scripts\\python.exe scripts\\ayon_smoke.py                       # context, ping, canvas read
    .venv\\Scripts\\python.exe scripts\\ayon_smoke.py --path W:\\x\\imageMain_v003.png --product imageMain   # plus a dry-run load

It changes nothing: ayon_context only reads, ayon_load runs with dry_run=True (no upload, no node).
Exit code 0 when the session answers, 1 when it does not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECT = {"project": "i002_ai_rnd", "folder_path": "agentRND/agentRND_0010", "task": "ai"}


def show(title: str, raw: str) -> dict:
    data = json.loads(raw)
    print(f"\n== {title}\n{json.dumps(data, indent=2)}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default="", help="local file of a published representation: also run a dry-run ayon_load")
    ap.add_argument("--product", default="imageMain", help="its product name (default imageMain)")
    ap.add_argument("--folder", default="", help="folder path of that product (default: the session's folder)")
    args = ap.parse_args()

    from agenty_core.tools import ayon
    from src.tools import ayon_host
    # Outside the host there is no panel: the provider falls back to the cached snapshot, then to the selection.
    ayon_host.install()

    ctx = show("ayon_context", ayon.ayon_context())
    if ctx.get("status") != "ok":
        print("\nFAIL: no Ayon session. Start ComfyUI from the Ayon launcher first (docs/dev/ayon.md section 9).")
        return 1
    session = ctx["session"]
    ok = True
    print(f"\nsession: ComfyUI {session['comfy_url']}, RPC port {session['rpc_port']}, ping {'ok' if session['rpc_ok'] else 'FAILED'}")
    ok &= bool(session["rpc_ok"])
    launch = ctx.get("launch_context") or {}
    for key, want in EXPECT.items():
        got = launch.get(key)
        flag = "ok" if got == want else "DIFFERENT"
        print(f"launch {key}: {got!r} (expected {want!r}) {flag}")
    if not session["same_comfyui_as_agenty"]:
        print("note: agentY's COMFYUI_URL is not this session's ComfyUI (see docs/dev/ayon.md section 2)")
    print("canvas:", ctx["canvas"]["source"], "- AYON Context node", ctx["canvas"]["context_node"])

    if args.path:
        res = show("ayon_load (dry run)", ayon.ayon_load(args.path, args.product, folder_path=args.folder, dry_run=True))
        ok &= res.get("status") == "dry_run"
    print("\nDONE" if ok else "\nDONE with problems")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
