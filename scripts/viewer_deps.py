"""Install the ComfyUI Image Viewer's Python packages into ComfyUI's own Python.

The viewer is a ComfyUI custom node, so its packages (usd-core for USD scenes,
trimesh for 3D input, ...) belong in the interpreter ComfyUI runs, not in agentY's
venv. ComfyUI itself never installs a custom node's requirements.txt - only
ComfyUI-Manager does, and a node cloned by agentY's installer is not one it saw.
Without this, the viewer loads but every USD stage fails on ``import pxr``.

Stdlib only; the installers and launchers call it after the viewer is cloned or
updated:

    python scripts/viewer_deps.py <ComfyUI dir> [--quiet]

<ComfyUI dir> is the folder that holds custom_nodes/. Nothing is installed when
every requirement already imports, so it is cheap to run on every start.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

# requirement name -> the module that proves it is installed, where they differ.
IMPORT_NAMES = {"pillow": "PIL", "usd-core": "pxr", "opencv-python": "cv2", "opencv-python-headless": "cv2"}


def find_viewer(custom_nodes: Path) -> Path | None:
    """The viewer under any folder name (ComfyUI-Manager names it comfyui-imageviewer)."""
    if not custom_nodes.is_dir():
        return None
    for d in sorted(custom_nodes.iterdir()):
        if not (d / "requirements.txt").is_file():
            continue
        for marker in (d / ".git" / "config", d / "pyproject.toml"):
            try:
                if re.search(r"szprivate/ComfyUI-ImageViewer", marker.read_text(encoding="utf-8", errors="ignore")):
                    return d
            except OSError:
                pass
    return None


def find_comfy_python(comfy: Path) -> Path | None:
    """ComfyUI's interpreter, from the layouts agentY installs into or finds."""
    names = ("bin/python", "bin/python3", "Scripts/python.exe", "python.exe")
    envs = [comfy / ".venv", comfy / "venv",
            comfy.parent / "python_embeded",     # Windows portable
            comfy.parent / "standalone-env",     # ComfyUI Desktop / standalone on macOS
            comfy.parent / ".venv", comfy.parent / "venv"]
    for env in envs:
        for name in names:
            if (env / name).is_file():
                return env / name
    return None


def requirements(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(line)
    return out


def missing(python: Path, reqs: list[str]) -> list[str]:
    gone = []
    for req in reqs:
        name = re.split(r"[\s<>=!~;\[@]", req, 1)[0]
        module = IMPORT_NAMES.get(name.lower(), name.replace("-", "_"))
        probe = subprocess.run([str(python), "-c", f"import {module}"], capture_output=True)
        if probe.returncode != 0:
            gone.append(req)
    return gone


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    quiet = "--quiet" in argv
    if len(args) != 1:
        print(__doc__)
        return 2
    comfy = Path(args[0]).expanduser()
    viewer = find_viewer(comfy / "custom_nodes")
    if viewer is None:
        if not quiet:
            print(f"[viewer] No ComfyUI Image Viewer under {comfy / 'custom_nodes'} - nothing to install.")
        return 0
    python = find_comfy_python(comfy)
    req_file = viewer / "requirements.txt"
    if python is None:
        print(f"[viewer] Could not find ComfyUI's Python next to {comfy}. Install the viewer's packages")
        print(f"         with ComfyUI's interpreter:  python -m pip install -r \"{req_file}\"")
        return 0
    gone = missing(python, requirements(req_file))
    if not gone:
        if not quiet:
            print("[viewer] The ComfyUI Image Viewer's packages are installed in ComfyUI's Python.")
        return 0
    print(f"[viewer] Installing into ComfyUI's Python ({python}): {', '.join(gone)}", flush=True)
    has_pip = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True).returncode == 0
    if has_pip:
        cmd = [str(python), "-m", "pip", "install", "-r", str(req_file)]
    elif shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", str(python), "-r", str(req_file)]
    else:
        print(f"[viewer] ComfyUI's Python has no pip and uv is not on PATH. Install by hand:")
        print(f"         \"{python}\" -m ensurepip; \"{python}\" -m pip install -r \"{req_file}\"")
        return 0
    if subprocess.run(cmd).returncode != 0:
        print("[viewer] Installing the viewer's packages failed - USD scenes and 3D input will not load in the viewer.")
        return 0
    print("[viewer] Done - restart ComfyUI to load them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
