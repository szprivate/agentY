"""Text animation presets: what an overlay clip on the timeline is made from.

A preset is a look and a movement for text, written as CSS. The model does not
write animation code for a title: it picks a preset and fills in the words, the
moment and a few settings (size, colour, position). That is what keeps a title
looking the same whoever wrote the cut.

The page a preset is drawn on is built HERE, never taken from a preset: the text
is split into units (``.u``, numbered ``--i`` of ``--n``) inside ``.text``, and
the settings arrive as CSS custom properties (``--size``, ``--color``, ``--font``,
``--weight``, ``--in``, ``--out``, ``--len`` in seconds, plus any the preset
declares). A preset is therefore only ever CSS, and :func:`problems` refuses one
that tries to be more - a script, an import, anything fetched from elsewhere.

Built-in presets live in ``config/text_presets.json``. Presets added at run time
(the agent's import tool, or by hand) are kept beside the conversations, one
file each, and win over a built-in of the same name.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

SPLITS = ("none", "lines", "words", "letters")
_NAME = re.compile(r"[a-z0-9][a-z0-9_]{1,39}$")
_PARAM = re.compile(r"[a-z][a-z0-9_]{0,23}$")
# What a preset's CSS may not contain: anything that runs, and anything that
# reaches outside the page.
_FORBIDDEN = (
    (re.compile(r"<\s*/?\s*(script|style|iframe|link|object|embed)", re.I), "an HTML tag"),
    (re.compile(r"@import", re.I), "an @import"),
    (re.compile(r"url\s*\(", re.I), "a url() - a preset fetches nothing"),
    (re.compile(r"expression\s*\(|javascript:|behavior\s*:", re.I), "something that runs"),
    (re.compile(r"</", re.I), "a closing tag"),
)
_BASE = {"size": 72, "color": "#ffffff", "font": "Arial, Helvetica, sans-serif", "weight": 700,
         "x": 50, "y": 50, "align": "center", "in": 0.5, "out": 0.4}
_SECONDS = ("in", "out", "stagger")
_ALIGN = {"left": ("0", "left"), "center": ("-50%", "center"), "right": ("-100%", "right")}


def _root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def builtin_file() -> Path:
    return _root() / "config" / "text_presets.json"


def user_dir(create: bool = False) -> Path:
    """Where presets added at run time are kept (``AGENTY_TEXT_PRESETS`` names another)."""
    named = os.environ.get("AGENTY_TEXT_PRESETS", "").strip()
    d = Path(named) if named else _root() / "memory" / "text_presets"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def slug(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")[:40]


def problems(preset: dict) -> list:
    """Why *preset* cannot be used, in words (empty = it can)."""
    out = []
    if not isinstance(preset, dict):
        return ["a preset is an object"]
    if not _NAME.match(str(preset.get("name") or "")):
        out.append("name must be lower-case letters, digits and underscores (2-40 characters)")
    css = preset.get("css")
    if not isinstance(css, str) or not css.strip():
        out.append("css is missing")
    else:
        for pattern, what in _FORBIDDEN:
            if pattern.search(css):
                out.append(f"the css contains {what}")
        if "@keyframes" not in css:
            out.append("the css animates nothing (no @keyframes)")
        if css.count("{") != css.count("}"):
            out.append("the css has unbalanced braces")
    if str(preset.get("split") or "none") not in SPLITS:
        out.append("split must be one of " + ", ".join(SPLITS))
    params = preset.get("params")
    if params is not None and not isinstance(params, dict):
        out.append("params must be an object")
    for key, value in (params or {}).items():
        if not _PARAM.match(str(key)):
            out.append(f"parameter name '{key}' is not usable")
        elif not isinstance(value, (int, float, str)) or isinstance(value, bool):
            out.append(f"parameter '{key}' must be a number or a string")
        elif isinstance(value, str) and re.search(r"[;{}<>]|url\s*\(|/\*", value, re.I):
            out.append(f"parameter '{key}' holds more than a value")
    return out


def _read(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    found = data.get("presets") if isinstance(data, dict) and "presets" in data else [data]
    return [p for p in found or [] if isinstance(p, dict)]


def load() -> dict:
    """``{name: preset}`` - the built-ins, with the ones added at run time over them."""
    out = {}
    for p in _read(builtin_file()):
        if not problems(p):
            out[p["name"]] = dict(p, builtin=True)
    d = user_dir()
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            for p in _read(f):
                if not problems(p):
                    out[p["name"]] = dict(p, builtin=False)
    return out


def get(name: str) -> dict | None:
    return load().get(slug(name))


def listing() -> list:
    """What the agent chooses from: one entry per preset, without the CSS."""
    return [{"name": p["name"], "title": p.get("title") or p["name"],
             "description": p.get("description") or "", "use_for": p.get("use_for") or "",
             "split": p.get("split") or "none", "params": dict(p.get("params") or {}),
             **({"source": p["source"]} if p.get("source") else {}),
             **({} if p.get("builtin") else {"added": True})}
            for p in sorted(load().values(), key=lambda p: (not p.get("builtin"), p["name"]))]


def save(preset: dict) -> Path:
    """Keep *preset* (already checked) for later runs; returns its file."""
    path = user_dir(create=True) / f"{preset['name']}.json"
    keep = {k: v for k, v in preset.items() if k != "builtin"}
    path.write_text(json.dumps(keep, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def remove(name: str) -> bool:
    path = user_dir() / f"{slug(name)}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


# ── the page ─────────────────────────────────────────────────────────────────

def units(text: str, split: str) -> list:
    """*text* cut the way *split* says. A line break in the text always breaks;
    spaces are kept with the unit before them so the spacing survives."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if split == "lines":
        return [ln for ln in text.split("\n")]
    if split == "words":
        return re.findall(r"\n|[^\s]+[ \t]*", text)
    if split == "letters":
        return [ch for ch in text]
    return [text]


def _value(key: str, value) -> str:
    if key in _SECONDS:
        return f"{float(value):g}s"
    if key == "size":
        return f"{float(value):g}px"
    return str(value)


def settings(preset: dict, params: dict | None = None) -> dict:
    """The preset's own settings with *params* over them; unknown ones are dropped
    (a setting the preset does not read would do nothing and say nothing)."""
    own = dict(_BASE, **(preset.get("params") or {}))
    for key, value in (params or {}).items():
        if key in own and isinstance(value, (int, float, str)) and not isinstance(value, bool) \
                and not (isinstance(value, str) and re.search(r"[;{}<>]|url\s*\(|/\*", value, re.I)):
            own[key] = value
    return own


def page(preset: dict, text: str, length: float, width: int, height: int,
         params: dict | None = None) -> str:
    """The page that draws *text* in *preset* for *length* seconds."""
    s = settings(preset, params)
    split = str(preset.get("split") or "none")
    parts = units(text, split)
    shift, align = _ALIGN.get(str(s.get("align")), _ALIGN["center"])
    spans = []
    for i, part in enumerate(parts):
        if part == "\n":
            spans.append("<br>")
            continue
        shown = html.escape(part).replace("\n", "<br>")
        if split == "letters" and part == " ":
            shown = "&nbsp;"
        elif split == "words":
            shown = shown.rstrip(" \t") + ("&nbsp;" if part[-1:] in " \t" else "")
        spans.append(f'<span class="u" style="--i:{i}">{shown}</span>')
    props = "".join(f"--{k}:{_value(k, v)};" for k, v in s.items()
                    if k not in ("x", "y", "align"))
    in_s, out_s = float(s.get("in") or 0), float(s.get("out") or 0)
    # An overlay too short for its own way in and out gives each its share.
    if in_s + out_s > float(length) > 0:
        scale = float(length) / (in_s + out_s)
        props += f"--in:{in_s * scale:g}s;--out:{out_s * scale:g}s;"
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        "html,body{margin:0;width:100%;height:100%;background:transparent;overflow:hidden}"
        f":root{{{props}--len:{float(length):g}s;--n:{len(parts)};}}"
        f".stage{{position:absolute;left:{float(s.get('x', 50)):g}%;top:{float(s.get('y', 50)):g}%;"
        f"transform:translate({shift},-50%);text-align:{align};max-width:90%;"
        f"width:max-content}}"
        ".text{font-family:var(--font);font-size:var(--size);font-weight:var(--weight);"
        "color:var(--color);line-height:1.15;white-space:pre-wrap;margin:0}"
        f"{preset['css']}"
        f'</style></head><body><div class="stage"><div class="text">{"".join(spans)}</div></div>'
        "</body></html>")
