#!/usr/bin/env python3
"""
agentY – headless chat host launcher.

Replaces the Chainlit GUI entry point. It builds the pipeline once and serves the
bridge + chat host (:mod:`src.utils.agentY_server`) that the ComfyUI-native chat
sidebar (``comfyui_extension/agentY-comfyuiConnect``) talks to over HTTP/SSE. There
is no web GUI here — the UI lives inside ComfyUI.

Launch:
    python -m src.agenty_ui_server                 # host 127.0.0.1, port 5000
    python -m src.agenty_ui_server --port 5001
    python -m src.agenty_ui_server --host 0.0.0.0  # expose to the LAN

Per-stage LLM overrides are read from the same env vars / settings.json as the
CLI (QUERYTEMPLATES_LLM, ASSEMBLEWORKFLOW_LLM, …).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")


def _unload_ollama_models() -> None:
    """Best-effort: free VRAM by unloading every Ollama model named in settings.json."""
    import json
    import urllib.request

    try:
        cfg_path = _project_root / "config" / "settings.json"
        cfg = json.loads(
            "".join(ln for ln in cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)
                    if not ln.lstrip().startswith("//"))
        )
        llm = cfg.get("llm", {})
        host = llm.get("ollama", {}).get("host", "http://localhost:11434")
        models: set[str] = set()
        for val in llm.get("pipeline", {}).values():
            if not isinstance(val, str):
                continue
            if val.startswith("ollama,"):
                models.add(val.split(",", 1)[1])
            elif "," not in val and val:
                models.add(val)
        for model in sorted(models):
            try:
                req = urllib.request.Request(
                    f"{host.rstrip('/')}/api/generate",
                    data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                    method="POST", headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5):
                    pass
                print(f"[agenty-ui] Unloaded Ollama model: {model}")
            except Exception:
                pass
    except Exception as exc:
        print(f"[agenty-ui] Ollama unload skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="agentY headless chat host (ComfyUI sidebar backend)")
    parser.add_argument("--host", default=os.environ.get("AGENTY_UI_HOST", "127.0.0.1"),
                        help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTY_UI_PORT", "5000")),
                        help="Port (default: 5000).")
    parser.add_argument("--no-unload", action="store_true",
                        help="Skip unloading Ollama models before startup.")
    args = parser.parse_args()

    if not args.no_unload:
        _unload_ollama_models()

    from src.pipeline import create_pipeline
    from src.utils.agentY_server import start_agentY_server

    print("[agenty-ui] Building pipeline …")
    pipeline = create_pipeline()
    print("[agenty-ui] Pipeline ready.")

    ok = start_agentY_server(pipeline, host=args.host, port=args.port)
    if not ok:
        print("[agenty-ui] ERROR: could not start the chat host (is Flask installed?).", file=sys.stderr)
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 64)
    print("  agentY chat host is running.")
    print(f"  Backend:  {url}   (health: {url}/agentY/health)")
    print("  UI:       open ComfyUI and click the agentY tab in the left sidebar.")
    print("            (install comfyui_extension/agentY-comfyuiConnect into")
    print("             <ComfyUI>/custom_nodes/ and restart ComfyUI once).")
    print("  Stop:     Ctrl+C, or type /stop in the chat.")
    print("=" * 64 + "\n")

    # Block the main thread until interrupted; the server runs in a daemon thread.
    stop = threading.Event()

    def _handle(_sig, _frm):  # noqa: ANN001
        print("\n[agenty-ui] Shutting down.")
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    try:
        signal.signal(signal.SIGTERM, _handle)
    except (ValueError, AttributeError):
        pass  # SIGTERM not settable on some platforms/threads
    stop.wait()


if __name__ == "__main__":
    main()
