"""End-to-end: set the Blender preset up against a PORTABLE Blender, with its own preferences and port.

Not part of the unit suite (it downloads, starts Blender and a server). Run it by hand:

    .venv/Scripts/python.exe tests/e2e_blender_portable.py D:/ai/_test/blender_portable/blender-5.1.2-windows-x64/blender.exe

It never touches the installed Blender, its preferences, or config/mcp.json: the agentY "root" (bundles, downloads,
mcp.json) is a fresh folder under D:/ai/_test/e2e, the Blender preferences are BLENDER_USER_RESOURCES under it, and the
add-on port is 9877 (the real one is 9876).
"""
from __future__ import annotations

import json
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import mcp_setup as ms   # noqa: E402

PORT = 9877


def roundtrip(code: str, port: int = PORT) -> str:
    with socket.create_connection(("localhost", port), timeout=10) as s:
        s.sendall(json.dumps({"type": "execute", "code": code, "strict_json": False}).encode() + b"\0")
        buf = b""
        while not buf.endswith(b"\0"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return buf.rstrip(b"\0").decode("utf-8", "replace")


def main(exe: str) -> int:
    base = Path("D:/ai/_test/e2e")
    shutil.rmtree(base, ignore_errors=True)
    root, res = base / "root", base / "res"
    root.mkdir(parents=True)
    cfg = root / "config" / "mcp.json"
    if ms.port_open(PORT):
        print(f"port {PORT} is in use; pick another")
        return 2
    kw = dict(root=root, config_path=cfg, launch=True,
              overrides={"apps": {"blender": exe}, "prefs": {"blender": str(res)}, "params": {"port": PORT}})
    rep = None
    procs = []
    try:
        for n in (1, 2):
            rep = ms.run_setup("blender", **kw)
            procs += rep["_ctx"].launched
            print(f"--- run {n}")
            print(ms.format_report(ms.public(rep)))
            if n == 1:
                if rep["ok"]:
                    print("tools reply:", roundtrip("import bpy; print(bpy.app.version_string, bpy.context.preferences.system.use_online_access)"))
        # the entry that was written
        print(json.dumps(json.loads(cfg.read_text())["servers"]["blender"], indent=1))
    finally:
        for proc in procs:
            proc.terminate()
    return 0 if rep and rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
