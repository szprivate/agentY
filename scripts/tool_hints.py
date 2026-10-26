"""Where node / npm / uv / git live on this OS, and how to install them.

One place for three things that used to be Windows-only guesses:

* ``extra_path_dirs`` / ``augmented_path`` - the folders a tool can be in although the process PATH
  does not show them. A process started from the Finder, from ComfyUI's sidebar or from a launcher does
  not get the login shell's PATH on macOS, so Homebrew (/opt/homebrew/bin), nvm, Volta and ~/.local/bin
  are invisible to it. The setup recipes and the host's stdio MCP servers both add these.
* ``install_command`` / ``install_hint`` - what installs a tool on THIS OS (winget on Windows, brew on
  macOS when brew exists, the vendor's page or the package manager otherwise).
* ``localize_hints`` - the preset catalog was written with winget commands; shown on another OS the text
  is rewritten with that OS's command, so a Mac is never told to run winget.

Nothing here installs or starts anything; the platform and the home folder are parameters so tests can
pretend to be another machine.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

# tool -> the program family it belongs to
_FAMILY = {"node": "node", "npm": "node", "npx": "node", "uv": "uv", "uvx": "uv", "git": "git"}
_WINGET = {"node": "OpenJS.NodeJS.LTS", "uv": "astral-sh.uv", "git": "Git.Git"}
_BREW = {"node": "node", "uv": "uv", "git": "git"}
_FROM_WINGET_ID = {v.lower(): k for k, v in _WINGET.items()}
NODE_PKG_URL = "https://nodejs.org/en/download"


def family(tool: str) -> str:
    return _FAMILY.get(str(tool or "").lower(), str(tool or "").lower())


def _kind(platform: str) -> str:
    p = str(platform or sys.platform).lower()
    return "win" if p.startswith("win") else "mac" if p.startswith("darwin") else "linux"


def _version_key(path: str) -> tuple:
    m = re.search(r"v?(\d+(?:\.\d+)*)", os.path.basename(os.path.dirname(path)))
    return tuple(int(n) for n in m.group(1).split(".")) if m else ()


def extra_path_dirs(platform: str | None = None, home: Path | str | None = None) -> list[str]:
    """Existing folders worth putting on PATH, most specific first."""
    kind = _kind(platform)
    h = Path(home) if home else Path.home()
    out: list[str] = []

    def add(p) -> None:
        s = str(p)
        if s not in out and os.path.isdir(s):
            out.append(s)

    if kind == "win":
        for d in ("C:/Program Files/nodejs", "C:/Program Files/Git/cmd"):
            add(d)
        add(h / ".local" / "bin")
        add(h / ".cargo" / "bin")
        return out
    nvm = sorted(glob.glob(str(h / ".nvm" / "versions" / "node" / "*" / "bin")), key=_version_key, reverse=True)
    for d in nvm:
        add(d)
    add(h / ".volta" / "bin")
    if kind == "mac":
        add("/opt/homebrew/bin")
        add("/opt/homebrew/sbin")
    add("/usr/local/bin")
    add(h / ".local" / "bin")
    add(h / ".cargo" / "bin")
    return out


def augmented_path(base: str | None = None, platform: str | None = None, home: Path | str | None = None) -> str:
    """*base* (default: the current PATH) with the extra folders appended when they are missing."""
    base = os.environ.get("PATH", "") if base is None else base
    have = [p for p in base.split(os.pathsep) if p]
    for d in extra_path_dirs(platform, home):
        if d not in have:
            have.append(d)
    return os.pathsep.join(have)


def which(name: str, platform: str | None = None, home: Path | str | None = None) -> Optional[str]:
    """``shutil.which`` over the augmented PATH."""
    return shutil.which(name, path=augmented_path(None, platform, home))


def stdio_env(env: dict | None, platform: str | None = None, home: Path | str | None = None) -> dict:
    """The environment for a stdio MCP server: what the config gave, with PATH able to see node/npx/uv."""
    out = dict(env or {})
    out["PATH"] = augmented_path(out.get("PATH") or os.environ.get("PATH", ""), platform, home)
    return out


def install_command(tool: str, platform: str | None = None, *, has_brew: Optional[bool] = None,
                    winget_id: str = "") -> Optional[list[str]]:
    """The command that installs *tool* here, or None when nothing can be run on the user's behalf."""
    kind, fam = _kind(platform), family(tool)
    if kind == "win":
        wid = winget_id or _WINGET.get(fam)
        return ["winget", "install", "-e", "--id", wid, "--accept-package-agreements",
                "--accept-source-agreements"] if wid else None
    if kind == "mac":
        brew = has_brew if has_brew is not None else bool(which("brew", "darwin"))
        if brew and fam in _BREW:
            return ["brew", "install", _BREW[fam]]
    return None


def _linux_pm() -> str:
    for pm, cmd in (("apt-get", "sudo apt install {pkg}"), ("dnf", "sudo dnf install {pkg}"),
                    ("pacman", "sudo pacman -S {pkg}"), ("zypper", "sudo zypper install {pkg}")):
        if shutil.which(pm):
            return cmd
    return "sudo apt install {pkg}"


def install_hint(tool: str, platform: str | None = None, *, has_brew: Optional[bool] = None,
                 winget_id: str = "") -> tuple[str, bool]:
    """``(text, is_command)``: what to tell the user for *tool* on this OS. A command is to be run; anything else is
    a noun phrase to follow "Install "."""
    kind, fam = _kind(platform), family(tool)
    cmd = install_command(tool, platform, has_brew=has_brew, winget_id=winget_id)
    if cmd:
        short = ["winget", "install", cmd[cmd.index("--id") + 1]] if kind == "win" else cmd
        return " ".join(short), True
    if kind == "win":
        return fam, False
    if kind == "mac":
        if fam == "node":
            return (f"the Node.js LTS .pkg from {NODE_PKG_URL} (or `brew install node` once Homebrew, "
                    "https://brew.sh, is installed)"), False
        if fam == "uv":
            return "curl -LsSf https://astral.sh/uv/install.sh | sh", True
        if fam == "git":
            return "xcode-select --install", True
    if fam == "node":
        return _linux_pm().format(pkg="nodejs npm"), True
    if fam == "uv":
        return "curl -LsSf https://astral.sh/uv/install.sh | sh", True
    if fam == "git":
        return _linux_pm().format(pkg="git"), True
    return fam, False


_WINGET_TEXT = re.compile(r"(`?)winget install ([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)(`?)")


def localize_hints(text: str, platform: str | None = None, *, has_brew: Optional[bool] = None) -> str:
    """Rewrite ``winget install <id>`` in catalog text with the command for this OS.

    Windows text is returned as it is. Only the three ids the catalog uses (Node, uv, git) are known;
    any other winget command is left alone.
    """
    if not text or _kind(platform) == "win" or "winget" not in text:
        return text

    def sub(m: "re.Match[str]") -> str:
        fam = _FROM_WINGET_ID.get(m.group(2).lower())
        if not fam:
            return m.group(0)
        hint, is_cmd = install_hint(fam, platform, has_brew=has_brew)
        return f"`{hint}`" if is_cmd else hint

    return _WINGET_TEXT.sub(sub, text)
