"""Animated text as pictures: a page rendered frame by frame, with no background.

Text on a film is an overlay clip on the timeline. Each one is a small web page -
HTML and CSS from a preset (:mod:`src.utils.text_presets`) - and this module
turns it into a sequence of transparent PNGs that ffmpeg lays over the picture.

A browser does the drawing, because type set by a browser is type set properly.
It is a headless Chrome (or Edge) driven over its DevTools socket. Nothing is
played in real time: every animation on the page is paused and set to the moment
of each frame, so the frames are exact and the same on every machine, however
slow.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


class TextRenderError(RuntimeError):
    pass


_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
_NAMES = ("chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
          "msedge", "microsoft-edge")


def browser_path() -> str:
    """A Chromium-family browser on this machine ("" when there is none).

    ``AGENTY_BROWSER`` names one outright.
    """
    named = os.environ.get("AGENTY_BROWSER", "").strip()
    if named and os.path.isfile(named):
        return named
    for path in _CANDIDATES:
        if os.path.isfile(path):
            return path
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    return ""


def frame_count(seconds: float, fps: float) -> int:
    return max(1, int(round(float(seconds) * float(fps))))


# Set every animation on the page to one moment. Animations are paused first, so
# nothing moves on its own; an animation that repeats forever is held within its
# own cycle by the browser, which is what a loop should do.
_SEEK = """(ms) => {
  for (const a of document.getAnimations({subtree: true})) {
    try { a.pause(); a.currentTime = ms; } catch (e) {}
  }
  return document.getAnimations({subtree: true}).length;
}"""


class _Tab:
    """One page of a browser, spoken to over its DevTools socket."""

    def __init__(self, socket):
        self._ws = socket
        self._n = 0

    async def call(self, method: str, **params):
        self._n += 1
        mine = self._n
        await self._ws.send(json.dumps({"id": mine, "method": method, "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=60))
            if msg.get("id") == mine:
                if "error" in msg:
                    raise TextRenderError(f"the browser refused {method}: "
                                          f"{msg['error'].get('message')}")
                return msg.get("result") or {}

    async def wait_for(self, expression: str, seconds: float = 10.0) -> None:
        for _ in range(int(seconds * 20)):
            got = await self.call("Runtime.evaluate", expression=expression, returnByValue=True,
                                  awaitPromise=True)
            if (got.get("result") or {}).get("value"):
                return
            await asyncio.sleep(0.05)
        raise TextRenderError("the page did not finish loading")


def _debug_url(port_file: Path, proc, seconds: float = 20.0) -> str:
    """The socket of the browser's first page, once it has started."""
    import time
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise TextRenderError("the browser closed before it could be used")
        try:
            port = int(port_file.read_text(encoding="utf-8").splitlines()[0])
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                for target in json.loads(r.read()):
                    if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        return target["webSocketDebuggerUrl"]
        except (OSError, ValueError, IndexError):
            pass
        time.sleep(0.1)
    raise TextRenderError("the browser did not start in time")


async def _capture(html: str, out_dir: str, width: int, height: int, seconds: float,
                   fps: float, exe: str) -> list:
    import websockets
    profile = tempfile.mkdtemp(prefix="agenty_text_profile_")
    page = Path(profile) / "page.html"
    page.write_text(html, encoding="utf-8")
    # Not reachable from another machine, a profile of its own, and no network:
    # a preset is local HTML and CSS and has no business fetching anything.
    proc = subprocess.Popen(
        [exe, "--headless=new", "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
         f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
         "--disable-extensions", "--disable-background-networking", "--disable-sync",
         "--hide-scrollbars", "--mute-audio", "--host-resolver-rules=MAP * ~NOTFOUND",
         f"--window-size={int(width)},{int(height)}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = []
    try:
        url = await asyncio.to_thread(_debug_url, Path(profile) / "DevToolsActivePort", proc)
        async with websockets.connect(url, max_size=None, open_timeout=20) as socket:
            tab = _Tab(socket)
            await tab.call("Page.enable")
            await tab.call("Emulation.setDeviceMetricsOverride", width=int(width),
                           height=int(height), deviceScaleFactor=1, mobile=False)
            await tab.call("Emulation.setDefaultBackgroundColorOverride",
                           color={"r": 0, "g": 0, "b": 0, "a": 0})
            await tab.call("Page.navigate", url=page.as_uri())
            await tab.wait_for("document.readyState === 'complete'")
            await tab.wait_for("document.fonts.ready.then(() => true)")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(frame_count(seconds, fps)):
                await tab.call("Runtime.evaluate", returnByValue=True,
                               expression=f"({_SEEK})({i * 1000.0 / float(fps):.4f})")
                shot = await tab.call("Page.captureScreenshot", format="png",
                                      captureBeyondViewport=False)
                path = os.path.join(out_dir, f"f_{i + 1:05d}.png")
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(shot["data"]))
                files.append(path)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(profile, ignore_errors=True)
    return files


def render_frames(html: str, out_dir: str, width: int, height: int, seconds: float,
                  fps: float = 24.0) -> dict:
    """Render *html* as *seconds* of transparent frames into *out_dir*.

    Returns ``{"pattern", "frames", "fps"}`` - *pattern* is what ffmpeg reads the
    sequence by (``…/f_%05d.png``).
    """
    exe = browser_path()
    if not exe:
        raise TextRenderError("no Chrome or Edge was found on this machine - text is drawn by a "
                              "browser. Install one, or name it in AGENTY_BROWSER.")
    try:
        import websockets  # noqa: F401
    except ImportError:
        raise TextRenderError("the 'websockets' package is missing from agentY's environment") from None
    try:
        files = asyncio.run(_capture(html, out_dir, width, height, seconds, fps, exe))
    except TextRenderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TextRenderError(f"the text could not be rendered: {exc}") from None
    if not files:
        raise TextRenderError("the browser produced no frames")
    return {"pattern": os.path.join(out_dir, "f_%05d.png"), "frames": len(files), "fps": float(fps)}
