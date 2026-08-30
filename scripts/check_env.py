"""Report which of agentY's dependencies are actually importable in this venv.

Run it with the venv's own interpreter - the installer does this as its last step:

    .venv/Scripts/python.exe scripts/check_env.py          # fast, spec lookup only
    .venv/Scripts/python.exe scripts/check_env.py --gpu    # also: is torch on CUDA?
    .venv/Scripts/python.exe scripts/check_env.py --quiet  # silent unless broken

Why this exists: nearly everything here is a dependency of *something else* as
well, so a missing entry in requirements.txt stays invisible until the one
machine that resolved differently loses a feature - OpenCV went undeclared for
months and only ever arrived as a stray transitive dep, taking analyze_video with
it when it didn't. Checking imports against a written-down list is the cheap way
to notice.

Exit code is 1 when something REQUIRED is missing, 0 otherwise (missing optional
packages are reported but do not fail the run).
"""

from __future__ import annotations

import importlib.util
import sys

# (module, distribution, what stops working without it)
REQUIRED: list[tuple[str, str, str]] = [
    ("agenty_core", "-e ../agenty_core", "the shared ComfyUI/HF/web/file tool layer - nothing runs"),
    ("strands", "strands-agents", "the agent runtime"),
    ("strands_tools", "strands-agents-tools", "the built-in tool set"),
    ("requests", "requests", "every HTTP call to ComfyUI and the model APIs"),
    ("websockets", "websockets", "the ComfyUI progress/console socket"),
    ("httpx", "httpx", "src/utils/llm_functions.py - imported at import time, so the host won't start"),
    ("pydantic", "pydantic", "the tool/decision contracts in src/utils/models.py"),
    ("dotenv", "python-dotenv", "reading .env (API keys)"),
    ("flask", "flask", "the chat host on :5000"),
    ("werkzeug", "flask", "the chat host's request plumbing (ships with flask)"),
    ("PIL", "Pillow", "every image staged, resized or annotated"),
    ("anthropic", "anthropic", "the Claude provider"),
    ("openai", "openai", "the DashScope/Qwen provider (OpenAI-compatible endpoint)"),
    ("ollama", "ollama", "the local-model provider"),
    ("mcp", "mcp", "config/mcp.json servers (Magnific et al.) and their OAuth flow"),
    ("cv2", "opencv-python-headless", "analyze_video - sampling frames for the video agent"),
    ("imageio", "imageio[ffmpeg]", "Vision QA of generated video (falls back to OpenCV, then ffmpeg)"),
    ("imageio_ffmpeg", "imageio[ffmpeg]", "the bundled ffmpeg binary split_video_into_shots cuts shots with"),
    ("scenedetect", "scenedetect", "split_video_into_shots - finding where a video cuts"),
    ("yaml", "PyYAML", "skill frontmatter validation in scripts/build_skill.py"),
    ("ddgs", "ddgs", "web search"),
    ("tqdm", "tqdm", "download progress"),
    ("mem0", "mem0ai", "the memory layer"),
    ("faiss", "faiss-cpu", "the memory layer's vector index"),
]

# Missing these costs a feature; the host still starts and everything else works.
OPTIONAL: list[tuple[str, str, str]] = [
    ("torch", "torch (from the CUDA index - see requirements.txt)", "SAM3 grounding; without it annotation falls back to the vision model"),
    ("sam3", "sam3", "locating what to circle from a text prompt, with ComfyUI down"),
    ("safetensors", "safetensors", "loading ComfyUI's SAM3 checkpoint"),
    ("numpy", "numpy", "SAM3 boxes/masks, and mask handling in agenty_core's annotator"),
    ("psutil", "psutil", "an undeclared sam3 import"),
    ("insightface", "insightface", "the QA likeness check for faces - is this the same person?"),
    ("onnxruntime", "onnxruntime", "what insightface runs the face embedding on"),
    ("dreamsim", "dreamsim", "the QA likeness check for everything else - same place, product, grade?"),
    ("spacy", "spacy", "mem0's lemmatised keyword search (degrades, doesn't break)"),
    ("en_core_web_sm", "en_core_web_sm (URL-pinned in requirements.txt)", "the spaCy model that search uses"),
]


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        # A half-installed or namespace-shadowed package raises here - from the
        # caller's point of view that is just as broken as absent.
        return False


def _report(title: str, checks: list[tuple[str, str, str]], quiet: bool = False) -> list[tuple[str, str, str]]:
    if not quiet:
        print(f"\n{title}")
    missing = []
    for module, dist, what in checks:
        if _importable(module):
            if not quiet:
                print(f"  ok       {module}")
        else:
            missing.append((module, dist, what))
            if not quiet:
                print(f"  MISSING  {module:16} ({dist}) - {what}")
    return missing


def gpu_line(torch, platform: str) -> str:
    """What ``--gpu`` says about the accelerator torch found.

    A function so it can be tested on a platform you are not sitting at, because the
    advice differs per platform and the WRONG advice is worse than none: a Mac used
    to be told to reinstall torch "from the CUDA index", which publishes no macOS
    wheel at all, so following it faithfully got you nowhere.

    The pitfall it exists for: PyPI's win_amd64 wheel is CPU-only, and on CPU a
    single SAM3 grounding call goes from ~0.2s to about a minute.
    """
    mps = getattr(torch.backends, "mps", None)
    if torch.cuda.is_available():
        return f"GPU: torch {torch.__version__} sees CUDA ({torch.cuda.get_device_name(0)})"
    if mps is not None and mps.is_available():
        return f"GPU: torch {torch.__version__} sees Metal (MPS)"
    if platform == "darwin":
        # No CUDA remedy here. On Apple Silicon the ordinary PyPI wheel already
        # carries MPS, so a Mac reporting none has a broken install rather than a
        # missing download to go and find.
        return (f"GPU: torch {torch.__version__} reports no Metal (MPS) support."
                "\n     SAM3 grounding will take about a minute per call. On Apple"
                "\n     Silicon the ordinary PyPI wheel carries MPS, so reinstalling"
                "\n     torch is usually the fix:"
                "\n       .venv/bin/python -m pip install --force-reinstall torch torchvision")
    return (f"GPU: torch {torch.__version__} is CPU-only. SAM3 grounding will take about a"
            "\n     minute per call. Reinstall it from the CUDA index:"
            "\n       .venv/Scripts/python -m pip install torch torchvision --index-url"
            " https://download.pytorch.org/whl/cu128")


def main(argv: list[str]) -> int:
    # --quiet: say nothing at all unless something REQUIRED is missing. That is
    # the form run_agent.ps1 uses on every start, to catch a venv that has
    # drifted out of line with requirements.txt without adding startup noise.
    quiet = "--quiet" in argv
    if not quiet:
        print(f"agentY dependency check - {sys.executable}")
    missing_req = _report("Required:", REQUIRED, quiet)
    missing_opt = _report("Optional (a feature degrades):", OPTIONAL, quiet)

    if quiet:
        if not missing_req:
            return 0
        print("[check_env] Missing dependencies this install needs:")
        for module, dist, what in missing_req:
            print(f"             {module:16} ({dist}) - {what}")
        print("[check_env] Fix with:  uv pip install -r requirements.txt")
        return 1

    if "--gpu" in argv and _importable("torch"):
        try:
            import torch  # noqa: PLC0415  (deliberately deferred - importing torch is slow)

            print("\n" + gpu_line(torch, sys.platform))
        except Exception as exc:
            print(f"\nGPU: could not query torch ({exc})")

    print()
    if missing_req:
        print(f"{len(missing_req)} required package(s) missing. Install them with:")
        print("    uv pip install -r requirements.txt")
        return 1
    if missing_opt:
        print(f"All required packages present. {len(missing_opt)} optional one(s) missing "
              "(see above for what that costs).")
    else:
        print("All dependencies present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
