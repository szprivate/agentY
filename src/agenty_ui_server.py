#!/usr/bin/env python3
"""
agentY – headless chat host launcher.

Replaces the Chainlit GUI entry point. It builds the pipeline once and serves the
bridge + chat host (:mod:`src.utils.agentY_server`) that the ComfyUI-native chat
sidebar (the separate ``agentY-comfyuiConnect`` repo) talks to over HTTP/SSE. There
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


def _agent_server_url_defaults() -> tuple[str, int]:
    """Default (host, port) for the chat host, read from config/settings.json.

    The single ``agent_server_url`` setting (e.g. ``http://127.0.0.1:5000``) is
    the source of truth; host + port are derived from it. Env vars
    (AGENTY_UI_HOST / AGENTY_UI_PORT) still override, and a CLI flag overrides
    everything. Falls back to 127.0.0.1:5000 if the file/URL is absent or
    unparseable.
    """
    host, port = "127.0.0.1", 5000
    try:
        import json
        from urllib.parse import urlsplit

        cfg_path = _project_root / "config" / "settings.json"
        if cfg_path.exists():
            cfg = json.loads(
                "".join(ln for ln in cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)
                        if not ln.lstrip().startswith("//"))
            )
            url = str(cfg.get("agent_server_url", "")).strip()
            if url:
                if "//" not in url:
                    url = "//" + url  # allow a bare host:port
                parts = urlsplit(url)
                if parts.hostname:
                    host = parts.hostname
                if parts.port:
                    port = parts.port
    except Exception:
        pass
    return host, port


def _port_in_use(host: str, port: int) -> bool:
    """True if a TCP listener is already accepting connections on host:port.

    Guards against launching a second host on top of a leftover one: on Windows
    SO_REUSEADDR lets the second bind "succeed" silently, so the stale instance
    keeps answering with old code. A quick connect probe catches that regardless
    of platform.
    """
    import socket

    target = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((target, port)) == 0
        except OSError:
            return False


def main() -> None:
    # Defaults come from settings.json (agent_server_url); env vars override; a CLI flag wins.
    _def_host, _def_port = _agent_server_url_defaults()
    parser = argparse.ArgumentParser(description="agentY headless chat host (ComfyUI sidebar backend)")
    parser.add_argument("--host", default=os.environ.get("AGENTY_UI_HOST", _def_host),
                        help=f"Bind address (default from settings.json agent_server_url: {_def_host}).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTY_UI_PORT", str(_def_port))),
                        help=f"Port (default from settings.json agent_server_url: {_def_port}).")
    parser.add_argument("--no-unload", action="store_true",
                        help="Skip unloading Ollama models before startup.")
    args = parser.parse_args()

    # Fail fast if a host is already serving this port — otherwise a leftover
    # instance would keep answering with stale code while this one silently does
    # nothing (see _port_in_use). run_agent.ps1 frees the port before launching;
    # this covers bare `python -m src.agenty_ui_server` launches too.
    if _port_in_use(args.host, args.port):
        print(f"[agenty-ui] ERROR: port {args.port} is already in use — another agentY "
              f"host appears to be running. Stop it first (run_agent.ps1 frees the port "
              f"automatically on launch), or use --port <other>.", file=sys.stderr)
        sys.exit(1)

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
    print("            (install the separate agentY-comfyuiConnect repo into")
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
