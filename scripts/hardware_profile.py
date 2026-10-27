"""What can this machine run, and which execution mode fits it? (docs/execution_modes.md)

Called by both installers (install_agent.ps1 / install_agent.sh) so the recommendation is made in ONE place, and
usable on its own. Standard library only: it runs before anything is installed.

    python scripts/hardware_profile.py                       # JSON: the hardware and the recommendation
    python scripts/hardware_profile.py --field mode          # just "local", "hybrid" or "cloud"
    python scripts/hardware_profile.py --field summary       # one line for a person
    python scripts/hardware_profile.py --set-mode hybrid --settings config/settings.local.json

The rules (``recommend``):

====================================================  ========
NVIDIA GPU with 12 GB VRAM or more                    local    (cloud stays available as overflow)
NVIDIA GPU with 6 to under 12 GB                      hybrid
NVIDIA GPU under 6 GB, no GPU, or a non-NVIDIA GPU    cloud    (Windows / Linux)
Apple Silicon Mac (unified memory)                    hybrid
Intel Mac                                             cloud
====================================================  ========

Partner (API) nodes run in the local ComfyUI in every mode; the mode only moves open-weight models.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys

MODES = ("local", "hybrid", "cloud")
# nvidia-smi reports MiB, and a "12 GB" card shows 12282-12288 of them: thresholds sit a little below the label.
LOCAL_MIN_MIB = int(11.5 * 1024)
HYBRID_MIN_MIB = int(5.5 * 1024)


def _run(cmd: list, timeout: float = 8.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - a missing tool is an answer, not an error
        return ""


def parse_nvidia_smi(text: str) -> list:
    """``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`` -> [{"name", "vram_mib"}]."""
    gpus = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.rsplit(",", 1)]
        if len(parts) == 2 and re.fullmatch(r"\d+", parts[1]):
            gpus.append({"name": parts[0], "vram_mib": int(parts[1])})
    return gpus


def detect(system: str | None = None, machine: str | None = None, runner=_run) -> dict:
    """The hardware, as a dict. *runner* takes a command list and returns its stdout ("" when it fails)."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    prof: dict = {"system": system, "machine": machine, "nvidia": [], "apple_silicon": False, "chip": "",
                  "memory_gb": None, "other_gpu": ""}
    prof["nvidia"] = parse_nvidia_smi(runner(["nvidia-smi", "--query-gpu=name,memory.total",
                                              "--format=csv,noheader,nounits"]))
    if system == "darwin":
        prof["apple_silicon"] = machine in ("arm64", "aarch64")
        prof["chip"] = runner(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        mem = runner(["sysctl", "-n", "hw.memsize"]).strip()
        if mem.isdigit():
            prof["memory_gb"] = round(int(mem) / 1024 ** 3)
    elif system == "windows" and not prof["nvidia"]:
        names = runner(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_VideoController).Name -join ';'"]).strip()
        prof["other_gpu"] = names
    return prof


def recommend(prof: dict) -> dict:
    """``{"mode", "reason", "summary"}`` for a hardware dict from :func:`detect`."""
    gpus = sorted(prof.get("nvidia") or [], key=lambda g: g.get("vram_mib", 0), reverse=True)
    if gpus:
        best = gpus[0]
        gb = best["vram_mib"] / 1024
        label = f"{best['name']} ({gb:.0f} GB)"
        if best["vram_mib"] >= LOCAL_MIN_MIB:
            return _rec("local", f"{label}: enough for most open-weight models locally; Comfy Cloud stays "
                                 "available as optional overflow")
        if best["vram_mib"] >= HYBRID_MIN_MIB:
            return _rec("hybrid", f"{label}: runs smaller models locally, the big ones are better on Comfy Cloud")
        return _rec("cloud", f"{label}: too little VRAM for most current models")
    if prof.get("system") == "darwin":
        chip = prof.get("chip") or "this Mac"
        mem = f", {prof['memory_gb']} GB unified memory" if prof.get("memory_gb") else ""
        if prof.get("apple_silicon"):
            return _rec("hybrid", f"{chip}{mem}: lighter models run locally, heavy ones on Comfy Cloud")
        return _rec("cloud", f"{chip}: an Intel Mac has no usable GPU for ComfyUI")
    other = prof.get("other_gpu") or ""
    what = f"no NVIDIA GPU found ({other})" if other else "no NVIDIA GPU found"
    return _rec("cloud", f"{what}: open-weight models run on Comfy Cloud; partner (API) nodes still run locally")


def _rec(mode: str, reason: str) -> dict:
    return {"mode": mode, "reason": reason, "summary": f"{mode} - {reason}"}


def profile() -> dict:
    prof = detect()
    return {**prof, **recommend(prof)}


def set_mode(settings_path: str, mode: str) -> None:
    """Write ``execution.mode`` into settings.local.json (merged; other settings untouched)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    data: dict = {}
    if os.path.isfile(settings_path):
        try:
            with open(settings_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            data = loaded if isinstance(loaded, dict) else {}
        except Exception:  # noqa: BLE001
            raise SystemExit(f"{settings_path} is not valid JSON - left untouched")
    ex = data.get("execution")
    data["execution"] = {**(ex if isinstance(ex, dict) else {}), "mode": mode}
    os.makedirs(os.path.dirname(os.path.abspath(settings_path)), exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def current_mode(settings_path: str) -> str:
    try:
        with open(settings_path, encoding="utf-8") as fh:
            m = (json.load(fh).get("execution") or {}).get("mode")
        return m if m in MODES else ""
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", help="print one field (mode, reason, summary, chip, ...) instead of the JSON")
    ap.add_argument("--set-mode", choices=MODES)
    ap.add_argument("--get-mode", action="store_true", help="print execution.mode of --settings (blank if unset)")
    ap.add_argument("--settings", default=os.path.join("config", "settings.local.json"))
    args = ap.parse_args(argv)
    if args.set_mode:
        set_mode(args.settings, args.set_mode)
        return 0
    if args.get_mode:
        print(current_mode(args.settings))
        return 0
    info = profile()
    if args.field:
        v = info.get(args.field, "")
        print(v if not isinstance(v, (list, dict)) else json.dumps(v))
    else:
        print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
