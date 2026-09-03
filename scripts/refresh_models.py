#!/usr/bin/env python3
"""
Refresh ComfyUI model caches before agent startup.

Writes config/models.json → adds/overwrites an "available" key with all model
files grouped by folder (custom_nodes excluded).

Run once at startup via run_agent.ps1.  Exits silently if ComfyUI is offline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Imported for its side effect: src/__init__.py makes agenty_core importable even
# when the editable install's .pth has been flagged hidden (iCloud does this to
# everything under .venv). Without it this script reports "Could not import
# utils" and the model cache silently goes stale.
import src  # noqa: E402,F401

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    # Guard against double execution (e.g. uvicorn reload spawning a child process).
    if os.environ.get("COMFYUI_MODELS_REFRESHED") == "1":
        return

    try:
        from agenty_core.utils.comfyui_retrieve_models_customnodes import fetch_available_models
    except Exception as exc:
        print(f"[refresh_models] Could not import utils: {exc}")
        return

    try:
        available = fetch_available_models()
    except Exception as exc:
        print(f"[refresh_models] ComfyUI offline or unreachable – skipping model cache refresh: {exc}")
        return

    # ── Write models.json ─────────────────────────────────────────────────────
    models_path = PROJECT_ROOT / "config" / "models.json"
    if models_path.exists():
        with open(models_path, "r", encoding="utf-8") as f:
            raw = "".join(ln for ln in f if not ln.lstrip().startswith("//"))
        models_data = json.loads(raw) if raw.strip() else {}
    else:
        models_data = {}

    models_data["available"] = available

    with open(models_path, "w", encoding="utf-8") as f:
        json.dump(models_data, f, indent=2)

    total = sum(len(v) for v in available.values() if isinstance(v, list))
    print(f"[refresh_models] models.json refreshed – {len(available)} folders, {total} files")


if __name__ == "__main__":
    main()

