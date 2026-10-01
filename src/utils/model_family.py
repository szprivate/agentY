"""model_family — the model family a request names, and whether a graph loads it.

"Build an SDXL workflow" once came back ``ready`` loading
``SD15\\v1-5-pruned-emaonly-fp16.safetensors`` inside the SDXL template: the
graph was well-formed, ComfyUI accepted it, and it would have rendered a broken
SD 1.5 image at 1024². Nothing checked that the model was the one asked for.
This is that check, plus the obvious repair: load a file of the right family.

Conservative on purpose. A loader is wrong only when its file is recognisably
ANOTHER family — a fine-tune with an unrecognisable name ("juggernaut_v9") is
not flagged, because a false alarm here would swap out the user's own model.
Pure: no I/O; the caller supplies the graph and each input's options.
"""

from __future__ import annotations

import re

# (name, how a request names it, how its files are named). Order matters only
# for reporting; a file is classified by the first family whose pattern fits.
FAMILIES: list[tuple[str, str, str]] = [
    ("SD 1.5", r"\b(sd[\s-]?1\.?5|stable diffusion 1\.5|sd15)\b",
     r"(^|[\\/])sd15[\\/]|v1-5|sd-?v?1[-_.]5|(^|[^a-z0-9])sd15([^a-z0-9]|$)"),
    ("SDXL", r"\b(sdxl|stable diffusion xl)\b", r"sdxl|sd_xl"),
    ("SD3.5", r"\bsd[\s-]?3\.5\b", r"sd3\.?5"),
    ("FLUX.2", r"\bflux[\s.-]?2\b", r"flux[\s._-]?2"),
    ("FLUX.1", r"\bflux(?![\s.-]?2)", r"flux[\s._-]?1|(^|[\\/])flux1[\\/]"),
    ("Qwen Image", r"\bqwen\b", r"qwen"),
    ("Wan 2.2", r"\bwan[\s-]?2\.2\b", r"wan[\s._-]?2[._]?2|wan22"),
    ("Wan 2.1", r"\bwan[\s-]?2\.1\b", r"wan[\s._-]?2[._]?1|wan21"),
    ("LTX-2", r"\bltx[\s-]?2\b", r"ltx[\s_-]?2"),
    ("HiDream", r"\bhidream\b", r"hidream"),
    ("Z-Image", r"\bz-?image\b", r"z[_-]?image"),
]

# The inputs that pick a graph's main model. VAEs, text encoders and LoRAs are
# chosen to match these, so the main model is where a family is decided.
MAIN_MODEL_INPUTS = ("ckpt_name", "unet_name")

# Files a plain request should not land on unless it asked for them.
_SPECIAL = re.compile(r"inpaint|refiner|turbo|lightning|distill|lora|fill|kontext|edit|depth|canny|"
                      r"controlnet|control|vace|fun|i2v|upscal", re.I)


def named(text: str) -> list[str]:
    """Families *text* asks for, in FAMILIES order."""
    t = (text or "").lower()
    return [name for name, ask, _ in FAMILIES if re.search(ask, t)]


def family_of(filename: str) -> str | None:
    f = str(filename or "")
    for name, _, pat in FAMILIES:
        if re.search(pat, f, re.I):
            return name
    return None


def _file_re(name: str) -> str:
    return next(pat for n, _, pat in FAMILIES if n == name)


def _rank(option: str, request: str) -> tuple:
    """Lower is better: plain over special-purpose files (unless the request says
    the purpose), 'base' / the reference release over the rest, shorter names."""
    base = option.replace("\\", "/").rsplit("/", 1)[-1].lower()
    special = [m.lower() for m in _SPECIAL.findall(base) if m.lower() not in request.lower()]
    reference = bool(re.search(r"base|v1-5-pruned|[-_]dev\b|[-_]dev[-_.]", base))
    return (len(special), not reference, len(base))


def check(workflow: dict, families: list[str]) -> list[dict]:
    """Main-model loaders whose file is recognisably not a requested family.

    Returns ``[{"node_id", "class_type", "input", "file", "is", "wanted"}]``.
    """
    if not families:
        return []
    out = []
    for nid, node in (workflow or {}).items():
        if not isinstance(node, dict):
            continue
        for name in MAIN_MODEL_INPUTS:
            val = (node.get("inputs") or {}).get(name)
            if not isinstance(val, str):
                continue
            fam = family_of(val)
            if fam is not None and fam not in families:
                out.append({"node_id": str(nid), "class_type": node.get("class_type"),
                            "input": name, "file": val, "is": fam, "wanted": families})
    return out


def repair(workflow: dict, violations: list[dict], options_for, request: str = "") -> tuple[list, list]:
    """Rebind each violating loader to an installed file of a wanted family.

    *options_for(class_type, input)* returns that input's installed options.
    Changes *workflow* in place. Returns ``(swaps, unresolved)``.
    """
    swaps, unresolved = [], []
    for v in violations:
        opts = [o for o in (options_for(v["class_type"], v["input"]) or []) if isinstance(o, str)]
        pick = None
        for fam in v["wanted"]:
            cands = [o for o in opts if re.search(_file_re(fam), o, re.I)]
            if cands:
                pick = sorted(cands, key=lambda o: _rank(o, request))[0]
                break
        if pick is None:
            unresolved.append(v)
            continue
        workflow[v["node_id"]]["inputs"][v["input"]] = pick
        swaps.append({**v, "now": pick})
    return swaps, unresolved
