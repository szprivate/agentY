"""Text presets from the web: reading a demo so it can be turned into a preset.

Pages like freefrontend.com/css-text-animations collect text effects - most of
them CodePen pens. They are web effects, not film titles: some run on scroll,
some follow the mouse, many loop for ever, a few need script libraries. So a
demo is never taken as it is. This module only READS: it lists the demos an
index page links to, and hands over one demo's markup and CSS. The agent writes
a preset from that (CSS only, on the clock: in, hold, out), the host checks it
(:func:`text_presets.problems`), draws a test clip, and it is kept only when the
user has seen the clip and said so.

Nothing fetched here is ever run, and nothing fetched ends up on a page: a
preset's page is built by :mod:`src.utils.text_presets` from CSS that passed its
check.
"""

from __future__ import annotations

import html as _html
import ipaddress
import os
import re
import socket
import subprocess
import urllib.parse
import urllib.request

_MAX_BYTES = 3_000_000
_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/126.0 Safari/537.36")
_PEN = re.compile(r"https?://(?:www\.)?codepen\.io/([A-Za-z0-9_-]+)/(?:pen|full|details|embed)/"
                  r"([A-Za-z0-9]+)", re.I)
# What CodePen says of its public pens; the tool reports it and the user checks.
PEN_LICENCE = "MIT (a public CodePen pen - CodePen licenses those MIT; check the pen's own page)"


class ImportError_(RuntimeError):
    """Reading a page failed; the message is for the agent."""


def _public(host: str) -> bool:
    """Is *host* somewhere on the internet, not this machine or its network?"""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return bool(infos)


def fetch(url: str, timeout: float = 25.0) -> str:
    """The text of a public web page. Local and private addresses are refused."""
    parts = urllib.parse.urlsplit(str(url or "").strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ImportError_("give a web address (http or https)")
    if not _public(parts.hostname):
        raise ImportError_(f"{parts.hostname} is not a public web address")
    req = urllib.request.Request(parts.geturl(), headers={"User-Agent": _AGENT,
                                                          "Accept": "text/html,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            final = urllib.parse.urlsplit(r.geturl())
            if final.hostname and not _public(final.hostname):
                raise ImportError_("the page redirected to an address that is not public")
            raw = r.read(_MAX_BYTES + 1)
            charset = r.headers.get_content_charset() or "utf-8"
    except ImportError_:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImportError_(f"could not read {parts.hostname}: {exc}") from None
    if len(raw) > _MAX_BYTES:
        raise ImportError_("the page is too large to read")
    return raw.decode(charset, "replace")


def pen_of(url: str) -> tuple | None:
    """``(user, id)`` when *url* is a CodePen pen."""
    m = _PEN.search(str(url or ""))
    return (m.group(1), m.group(2)) if m else None


def demo_links(page: str, limit: int = 80) -> list:
    """The demos an index page links to: ``[{"url", "author", "title"}]``, in page order."""
    out, seen = [], set()
    for m in _PEN.finditer(page or ""):
        user, pen = m.group(1), m.group(2)
        if pen in seen or user.lower() in ("pen", "anon"):
            continue
        seen.add(pen)
        # The words nearest the link, as a title: a heading or an alt text before it.
        before = page[max(0, m.start() - 4000):m.start()]
        titles = re.findall(r"<h[1-4][^>]*>(.*?)</h[1-4]>", before, re.S | re.I)
        title = ""
        if titles:
            title = _html.unescape(" ".join(re.sub(r"<[^>]+>", "", titles[-1]).split()))[:100]
        out.append({"url": f"https://codepen.io/{user}/pen/{pen}", "author": user, "title": title})
        if len(out) >= limit:
            break
    return out


def _cut(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"\n/* … cut, {len(text) - limit} more characters */"


# What CodePen's preview page adds around a pen: none of it is the demo.
_HOST_SCRIPT = ("window.console = window.console", "Capture all errors for this pen",
                "__animationDuration", "CodePen")


def inner_page(page: str) -> str:
    """The demo itself, when *page* is a preview that carries it in a frame."""
    m = re.search(r'<iframe[^>]*\bsrcdoc="([^"]*)"', page, re.S | re.I)
    return _html.unescape(m.group(1)) if m else page


def parts_of(page: str) -> dict:
    """A demo page taken apart: its markup, its CSS, and whether it needs script."""
    page = inner_page(page)
    css = "\n".join(s.strip() for s in re.findall(r"<style[^>]*>(.*?)</style>", page, re.S | re.I))
    scripts = [s.strip() for s in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", page,
                                             re.S | re.I)
               if s.strip() and not any(mark in s for mark in _HOST_SCRIPT)]
    libraries = re.findall(r"<script[^>]*\bsrc=[\"']([^\"']+)[\"']", page, re.I)
    body = re.search(r"<body[^>]*>(.*?)</body>", page, re.S | re.I)
    markup = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", "", body.group(1) if body else "",
                    flags=re.S | re.I)
    markup = re.sub(r"\n\s*\n+", "\n", markup)
    return {"markup": _cut(markup, 5000), "css": _cut(css, 14000),
            "script": _cut("\n".join(scripts), 3000),
            "libraries": [s for s in libraries if "codepen.io/assets" not in s and "cpwebassets" not in s],
            "uses_script": bool(scripts) or bool(libraries),
            "keyframes": len(re.findall(r"@keyframes", css)),
            "driven_by": [what for what, pattern in (
                ("scrolling", r"animation-timeline|scroll\s*\(|view\s*\(|ScrollTrigger|onscroll"),
                ("the mouse", r":hover|mousemove|pointermove|:active"),
                ("an endless loop", r"\binfinite\b"),
            ) if re.search(pattern, page, re.I)]}


def read_demo(url: str) -> dict:
    """One demo's source, ready to be turned into a preset.

    A CodePen pen is read from its preview host (the site itself turns automated
    requests away); any other page is read as it is.
    """
    pen = pen_of(url)
    if pen:
        user, pen_id = pen
        page = fetch(f"https://cdpn.io/{user}/fullcpgrid/{pen_id}")
        got = parts_of(page)
        # The demo's own title, not the preview page's.
        title = re.search(r"<title>(.*?)</title>", inner_page(page), re.S | re.I)
        return {"kind": "demo", "source": f"https://codepen.io/{user}/pen/{pen_id}", "author": user,
                "licence": PEN_LICENCE,
                "title": _html.unescape(" ".join((title.group(1) if title else "").split()))[:120],
                **got}
    page = fetch(url)
    return {"kind": "demo", "source": url, "author": "", "licence": "not stated - ask the user",
            **parts_of(page)}


def read(url: str) -> dict:
    """An index page as its list of demos, or a single demo as its source."""
    if pen_of(url):
        return read_demo(url)
    page = fetch(url)
    demos = demo_links(page)
    if len(demos) >= 3:
        return {"kind": "index", "source": url, "count": len(demos), "demos": demos}
    return {"kind": "demo", "source": url, "author": "", "licence": "not stated - ask the user",
            **parts_of(page)}


# ── a look at a preset before it is kept ─────────────────────────────────────

def test_clip(preset: dict, text: str, out_path: str, seconds: float = 3.5, width: int = 1280,
              height: int = 720, fps: float = 24.0, params: dict | None = None) -> str:
    """Draw *text* in *preset* over a plain dark frame; returns the film's path."""
    import shutil
    import tempfile
    from src.utils import text_presets as _tp
    from src.utils import text_render as _tr
    from src.utils.edit_render import RenderError, ffmpeg_path
    exe = ffmpeg_path()
    if not exe:
        raise RenderError("ffmpeg is not installed - a test clip cannot be made without it.")
    work = tempfile.mkdtemp(prefix="agenty_preset_try_")
    try:
        got = _tr.render_frames(_tp.page(preset, text, seconds, width, height, params), work,
                                width, height, seconds, fps)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        run = subprocess.run(
            [exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             f"color=c=0x2b313b:size={width}x{height}:rate={fps:g}:duration={seconds:g}",
             "-framerate", f"{fps:g}", "-start_number", "1", "-i", got["pattern"],
             "-filter_complex", "[0:v][1:v]overlay=0:0:eof_action=pass,format=yuv420p[v]",
             "-map", "[v]", "-t", f"{seconds:g}", "-c:v", "libx264", "-crf", "18",
             "-movflags", "+faststart", out_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        if run.returncode != 0 or not os.path.isfile(out_path):
            raise RenderError("the test clip could not be made: " + (run.stderr or "").strip()[-400:])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path
