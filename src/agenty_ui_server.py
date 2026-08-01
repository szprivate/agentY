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
import time
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
        from src.utils.settings import load_settings
        cfg = load_settings()
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
        from urllib.parse import urlsplit
        from src.utils.settings import load_settings

        cfg = load_settings()
        if cfg:
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


def _refresh_workflow_corpus() -> None:
    """Pick up templates added to the custom folder by hand, before anything reads them.

    Adding or removing a workflow through the agent already re-indexes and
    regenerates, so that path stays in sync on its own. A template *copied into
    the folder by hand* does not: nothing indexes it, and the symptom is the
    researcher insisting a template the user can plainly see on disk does not
    exist.

    Only missing entries are indexed. Re-deriving every entry is lossy for some
    graphs — an output node the parser cannot trace becomes no outputs at all —
    so that stays a maintenance command (``scripts/update_all_workflows.ps1``)
    rather than something that fires unattended on every launch. When nothing is
    new this does no work and writes no files, so it neither dirties the corpus
    repo (which would block the auto-updater) nor costs startup time.

    Never fatal: on failure the existing index and database stay in place, which
    is the state the host would have started in regardless.
    """
    try:
        from src.utils.workflow_admin import index_missing_templates
        started = time.perf_counter()
        res = index_missing_templates()
    except Exception as exc:  # noqa: BLE001 — a bad corpus must not block startup
        print(f"[agenty-ui] WARNING: could not check the workflow corpus ({exc}); "
              f"continuing with the existing recipe database.", file=sys.stderr)
        return
    added, failed = res.get("added") or [], res.get("failed") or []
    if not added and not failed:
        return                      # nothing new — say nothing
    # Reporting sits outside the guard above, and stays ASCII: a console that
    # cannot encode the summary (cp1252 under a bare `python -m`) must not be
    # reported as a corpus that failed to rebuild.
    counts = res.get("recipes") or {}
    print(f"[agenty-ui] Workflow corpus: indexed {len(added)} new template(s) -> "
          f"{counts.get('recipe_count', '?')} recipes "
          f"({time.perf_counter() - started:.1f}s): {', '.join(added[:5])}"
          + (f" - {len(failed)} unreadable: {', '.join(failed[:5])}" if failed else ""))


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
    parser.add_argument("--no-reindex", action="store_true",
                        default=os.environ.get("AGENTY_NO_REINDEX", "").strip().lower()
                        in ("1", "true", "yes", "on"),
                        help="Skip the startup template re-index / recipe rebuild (~1s). "
                             "Env: AGENTY_NO_REINDEX=1.")
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

    # Before the pipeline caches the recipe tree it reads at scope time.
    if not args.no_reindex:
        _refresh_workflow_corpus()

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
