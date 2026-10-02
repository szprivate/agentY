"""Settings while the host is not running.

The settings page lives in ComfyUI, and everything it shows is read from files in
this checkout (.env, config/settings*.json, config/mcp.json) — yet it could only
be opened while the host answered, which is exactly when a wrong setting (a dead
API key, a model that no longer exists, a port in use) cannot be fixed.

So the ComfyUI extension runs this module, with this checkout's own Python, when
the host does not answer:

    python -m src.utils.settings_offline get         -> the page's data, as JSON
    python -m src.utils.settings_offline save        <- a save, as JSON on stdin

It calls the same functions the host's routes call (``settings_payload`` /
``settings_save``), so the two cannot drift, and keys are masked exactly as they
are over HTTP. One JSON object is printed on the last line of stdout; anything a
library prints goes before it.
"""

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """The host loads .env at startup; the model lists below read the keys from
    the environment, so this has to as well."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass


def read() -> dict:
    from src.utils import agentY_server as srv
    out = {"ok": True, "offline": True, "settings": srv.settings_payload()}
    try:
        from src.tools.mcp_tools import load_mcp_config, mcp_status
        out["mcp"] = {"ok": True, "config": load_mcp_config(), "status": mcp_status()}
    except Exception as exc:  # noqa: BLE001 — the page simply leaves the section out
        out["mcp"] = {"ok": False, "error": str(exc)}
    return out


def save(body: dict) -> dict:
    from src.utils import agentY_server as srv
    result = {"ok": True, "offline": True}
    result.update(srv.settings_save(body if isinstance(body, dict) else {}))
    cfg = (body or {}).get("mcp_config")
    if isinstance(cfg, dict):
        from src.tools.mcp_tools import save_mcp_config
        save_mcp_config(cfg)
        result["mcp_saved"] = True
    return result


def main(argv: list[str]) -> int:
    os.chdir(_ROOT)
    _load_env()
    action = argv[1] if len(argv) > 1 else ""
    try:
        if action == "get":
            out = read()
        elif action == "save":
            out = save(json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}"))
        else:
            out = {"ok": False, "error": "usage: settings_offline get | save"}
    except Exception as exc:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
