"""Strata: a local model server as an agentY provider, and one GPU for it and ComfyUI.

Strata (https://github.com/Niko1221/Strata) runs a GGUF model behind an
OpenAI-compatible API (``http://127.0.0.1:8080/v1``). A tier names it like any
provider - ``"strata,qwen3.6-35b-a3b-ud-q4_k_xl"`` - and the model is Strands'
OpenAI client pointed at it (:func:`build_model`), with two things of its own:

* ``chat_template_kwargs.enable_thinking`` is sent, off unless the role thinks.
  With thinking on, Qwen 3.6 invented a tool call in a measured run, took twice
  as long and lost the prompt cache from one call to the next.
* every call passes :class:`_Gate` first, which is the reason this module exists.

Strata takes the whole card (31 of 32 GB with a 35b model at a 128k window), and
so does a ComfyUI render. Both at once starve each other: the render crawls in
low-VRAM mode and the model runs from shared memory at a few tokens a second.
Ollama can be asked to unload a model and loads it again by itself; Strata cannot
- it is a process with the model in it. So the process is stopped and started:

* **before a render** - every graph agentY queues passes the prompt hook of
  agenty_core's ComfyUI client (:func:`_prompt_hook`). When a role runs on Strata,
  Strata is running and the graph needs the card, the hook waits for the Strata
  calls in flight to finish, holds new ones back, runs ``strata launch --stop``
  and waits until the card is free. Only then is the graph queued.
* **before the next call** - a Strata model call that finds Strata not running
  waits until ComfyUI's queue is empty, asks ComfyUI to let go of its models, runs
  the launcher and waits until ``/v1/models`` answers. "Starting local model...
  (Strata)" is shown in the panel meanwhile.

The orchestrator itself may be the Strata model, and the turn that queues a
render is then a Strata turn - which does not deadlock, because the stop happens
inside a tool call (the model's answer is complete by then and nothing of it is
in flight) and the start happens at the next model call. Agents that run side by
side (prepare_workflow's researchers) all wait at the gate for the one start, and
then queue on Strata's single slot: their client timeout
(``llm.strata.request_timeout``) is long for that reason.

None of it happens unless a role runs on Strata (a tier or a per-role setting, or
a model built in this process), and ``llm.strata.gpu_sharing = false`` turns the
stopping off for a machine with a second card. Nothing here raises into a render:
a stop that fails is logged and the graph is queued anyway.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("agentY.strata")

PROVIDER = "strata"
DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.6-35b-a3b-ud-q4_k_xl"
DEFAULT_LAUNCH_ARGS = "launch --family qwen36 --model UD-Q4_K_XL --context 128k --parallel 1 --no-tui --yes"
STOP_ARGS = ("launch", "--stop")
LABEL = "Strata (local)"

_PROBE_TIMEOUT = 3.0
_STOP_TIMEOUT = 120.0
_VRAM_SETTLE_TIMEOUT = 60.0
_QUEUE_POLL = 2.0


# ── settings ──────────────────────────────────────────────────────────────────

def _settings() -> dict:
    try:
        from src.utils.settings import load_settings
        return load_settings() or {}
    except Exception:  # noqa: BLE001
        return {}


def _flag(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _num(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def config(settings: dict | None = None) -> dict:
    """``llm.strata`` with its defaults filled in, and the .env overrides
    (``STRATA_BASE_URL``, ``STRATA_API_KEY``) applied."""
    cfg = settings if settings is not None else _settings()
    raw = ((cfg.get("llm") or {}).get("strata") or {}) if isinstance(cfg, dict) else {}
    url = (os.environ.get("STRATA_BASE_URL") or "").strip() or str(raw.get("base_url") or "").strip() or DEFAULT_URL
    extra = raw.get("extra_body")
    if isinstance(extra, str) and extra.strip():
        try:
            extra = json.loads(extra)
        except ValueError:
            extra = None
    return {
        "base_url": url.rstrip("/"),
        # The OpenAI client wants a key; Strata on localhost wants none.
        "api_key": (os.environ.get("STRATA_API_KEY") or "").strip() or "none",
        "model": str(raw.get("model") or "").strip() or DEFAULT_MODEL,
        "max_tokens": int(_num(raw.get("max_tokens"), 8192)) or 8192,
        "think": _flag(raw.get("think"), False),
        "vision": _flag(raw.get("vision"), True),
        "launcher": str(raw.get("launcher") or "").strip(),
        "launch_args": str(raw.get("launch_args") or "").strip() or DEFAULT_LAUNCH_ARGS,
        "autostart": _flag(raw.get("autostart"), True),
        "gpu_sharing": _flag(raw.get("gpu_sharing"), True),
        "start_timeout": max(10.0, _num(raw.get("start_timeout"), 600.0)),
        "request_timeout": max(60.0, _num(raw.get("request_timeout"), 1800.0)),
        "extra_body": extra if isinstance(extra, dict) else {},
    }


def extra_body(think: bool, settings: dict | None = None) -> dict:
    """What every request carries besides the OpenAI fields: thinking on or off
    (Qwen's chat template reads ``enable_thinking``), then ``llm.strata.extra_body``
    laid over it."""
    cfg = config(settings)
    body: dict = {"chat_template_kwargs": {"enable_thinking": bool(think or cfg["think"])}}
    for key, value in cfg["extra_body"].items():
        if key == "chat_template_kwargs" and isinstance(value, dict):
            body["chat_template_kwargs"] = {**body["chat_template_kwargs"], **value}
        else:
            body[key] = value
    return body


# Set once a Strata model is built in this process: a conversation switched to
# Strata (switch_model, the lead) uses it without any setting saying so.
_built = False


def in_use(settings: dict | None = None) -> bool:
    """Does any agent run on Strata - a tier, a per-role setting, or a model built here?"""
    if _built:
        return True
    try:
        cfg = settings if settings is not None else _settings()
        llm = cfg.get("llm") or {}
        specs = list((llm.get("tiers") or {}).values()) + list((llm.get("pipeline") or {}).values())
        return any(str(s or "").strip().lower().startswith(PROVIDER + ",") for s in specs)
    except Exception:  # noqa: BLE001
        return False


# ── talking to Strata ────────────────────────────────────────────────────────

def _open(url: str, body: dict | None = None, timeout: float = _PROBE_TIMEOUT, key: str = "") -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers)
    # Never through a proxy: these are servers on this machine.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace").strip()
    got = json.loads(raw) if raw else {}
    return got if isinstance(got, dict) else {"data": got}


def models(timeout: float = _PROBE_TIMEOUT, settings: dict | None = None) -> list[dict]:
    """The models Strata serves, ``[{id, n_ctx, vision}]``. Raises when it does not answer."""
    cfg = config(settings)
    body = _open(cfg["base_url"] + "/models", timeout=timeout, key=cfg["api_key"])
    out = []
    for m in body.get("data") or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        meta = m.get("meta") or {}
        modal = ((m.get("architecture") or {}).get("input_modalities")) or []
        out.append({"id": str(m["id"]), "n_ctx": meta.get("n_ctx"), "vision": "image" in modal})
    return out


def alive(timeout: float = _PROBE_TIMEOUT, settings: dict | None = None) -> bool:
    """Is Strata answering (and so holding the card)?"""
    try:
        models(timeout=timeout, settings=settings)
        return True
    except Exception:  # noqa: BLE001
        return False


def test(settings: dict | None = None) -> dict:
    """Settings > Strata > Test: ``{ok, url, models, error}``."""
    cfg = config(settings)
    try:
        found = models(timeout=8.0, settings=settings)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": cfg["base_url"], "models": [], "running": False,
                "busy": _control["busy"], "control_error": _control["error"],
                "error": f"Strata does not answer at {cfg['base_url']} ({exc})"}
    return {"ok": True, "url": cfg["base_url"], "models": found, "running": True,
            "busy": _control["busy"], "control_error": _control["error"], "error": ""}


def _command(args: str | tuple, settings: dict | None = None) -> list[str]:
    """The argv for ``strata <args>`` with the configured launcher."""
    cfg = config(settings)
    launcher = cfg["launcher"]
    if launcher and not Path(launcher).exists():
        found = shutil.which(launcher)
        if not found:
            raise RuntimeError(f"Strata launcher not found: {launcher} (Settings > Connections > Strata)")
        launcher = found
    if not launcher:
        launcher = shutil.which("strata") or ""
        if not launcher:
            raise RuntimeError("Strata is not installed or not on PATH - set its launcher in "
                               "Settings > Connections > Strata")
    if isinstance(args, str):
        # Not POSIX rules: a Windows path's backslashes must survive.
        parts = [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p
                 for p in shlex.split(args, posix=False)]
    else:
        parts = list(args)
    if os.name == "nt" and launcher.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", launcher, *parts]
    return [launcher, *parts]


def _log_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    path = root / ".logs" / "strata_launch.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tail(path: Path, lines: int = 6) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return ""


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def start(settings: dict | None = None, timeout: float | None = None) -> float:
    """Launch Strata and wait until ``/v1/models`` answers. Returns the seconds it
    took; raises RuntimeError (with the end of the launch log) when it does not come up."""
    cfg = config(settings)
    timeout = cfg["start_timeout"] if timeout is None else timeout
    argv = _command(cfg["launch_args"], settings)
    log = _log_path()
    t0 = time.monotonic()
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        fh.flush()
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=_NO_WINDOW)
    while time.monotonic() - t0 < timeout:
        if alive(timeout=2.0, settings=settings):
            return time.monotonic() - t0
        code = proc.poll()
        # The launcher may hand the engine off and exit 0; a non-zero exit with
        # nothing answering is a failed start.
        if code not in (None, 0) and not alive(timeout=2.0, settings=settings):
            raise RuntimeError(f"Strata did not start (exit {code}):\n{_tail(log)}")
        time.sleep(1.0)
    raise RuntimeError(f"Strata did not answer at {cfg['base_url']} within {int(timeout)} s:\n{_tail(log)}")


def vram_free_mib() -> int | None:
    """Free memory of the card with the most of it, from nvidia-smi; None without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW)
        vals = [int(v.strip()) for v in out.stdout.split() if v.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def stop(settings: dict | None = None) -> bool:
    """``strata launch --stop``, then wait until it no longer answers and the card's
    free memory has settled. True when it is down."""
    argv = _command(STOP_ARGS, settings)
    log = _log_path()
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        fh.flush()
        try:
            subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           timeout=_STOP_TIMEOUT, creationflags=_NO_WINDOW)
        except subprocess.TimeoutExpired:
            logger.warning("strata: the stop command did not return within %d s", _STOP_TIMEOUT)
    deadline = time.monotonic() + _VRAM_SETTLE_TIMEOUT
    while alive(timeout=1.5, settings=settings):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.5)
    # The port closes before the engine has handed its memory back: wait until
    # two readings a second apart agree (or there is no card to ask).
    last = vram_free_mib()
    while last is not None and time.monotonic() < deadline:
        time.sleep(1.0)
        now = vram_free_mib()
        if now is None or abs(now - last) < 256:
            break
        last = now
    return True


# ── ComfyUI, asked briefly ───────────────────────────────────────────────────

def _comfy_base() -> str:
    from agenty_core.utils.comfyui_client import get_client
    return str(get_client().base_url).rstrip("/")


def comfy_busy() -> bool:
    """Is ComfyUI rendering or has it something queued? A ComfyUI that does not
    answer is not busy."""
    try:
        q = _open(_comfy_base() + "/queue", timeout=4.0)
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:  # noqa: BLE001
        return False


def comfy_free() -> None:
    """Ask ComfyUI to unload its models: Strata needs the whole card."""
    try:
        _open(_comfy_base() + "/free", {"unload_models": True, "free_memory": True}, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        logger.info("strata: could not ask ComfyUI to free the card (%s)", exc)


def _status(text: str, level: str = "info") -> None:
    try:
        from src.utils import status_bus
        status_bus.notify(text, level=level)
    except Exception:  # noqa: BLE001 - a status line must never cost the call
        logger.info(text)


# ── the gate ─────────────────────────────────────────────────────────────────

class _Gate:
    """Who may use the card: Strata's calls, or a render.

    ``phase`` is ``idle``, ``waiting`` (a call waits for ComfyUI before starting
    Strata), ``launching`` or ``stopping``. A call passes only in ``idle`` with
    Strata answering, and is counted in ``inflight`` until its answer is read.
    ``submitting`` counts graphs between the prompt hook and the end of their POST:
    until they are in ComfyUI's queue, an empty queue does not mean "done".
    """

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.inflight = 0
        self.submitting = 0
        self.phase = "idle"

    # A Strata call, before it is sent. Blocking: run it off the event loop.
    def acquire(self, settings: dict | None = None, *, force_start: bool = False) -> None:
        while True:
            with self.cond:
                while self.phase != "idle":
                    self.cond.wait(1.0)
                if alive(settings=settings):
                    self.inflight += 1
                    return
                cfg = config(settings)
                if not (cfg["autostart"] or force_start):
                    raise RuntimeError(f"Strata is not running at {cfg['base_url']} - start it "
                                       "(Settings > Connections > Strata > Start) or turn on llm.strata.autostart")
                self.phase = "waiting"
            try:
                self._start(settings)
            finally:
                with self.cond:
                    self.phase = "idle"
                    self.cond.notify_all()

    def release(self) -> None:
        with self.cond:
            self.inflight = max(0, self.inflight - 1)
            self.cond.notify_all()

    def _start(self, settings: dict | None) -> bool:
        """Wait for ComfyUI to finish, then start Strata. False when a render
        arrived at the last moment (the caller tries again after it)."""
        cfg = config(settings)
        if cfg["gpu_sharing"]:
            told = False
            quiet = 0
            while quiet < 2:              # two empty readings in a row
                with self.cond:
                    busy = self.submitting > 0
                busy = busy or comfy_busy()
                quiet = 0 if busy else quiet + 1
                if busy and not told:
                    _status("⏳ Waiting for ComfyUI to finish before the local model starts (Strata)…")
                    told = True
                if quiet < 2:
                    time.sleep(_QUEUE_POLL)
            comfy_free()
        with self.cond:
            if self.submitting:
                return False
            self.phase = "launching"
        _status("⏳ Starting local model… (Strata)")
        try:
            secs = start(settings)
        except Exception as exc:
            _status(f"❌ The local model did not start (Strata): {exc}", level="error")
            raise
        _status(f"✅ Local model ready (Strata, {secs:.0f} s)")
        return True

    # A graph about to be queued. Blocking, on the caller's thread.
    def yield_gpu(self, settings: dict | None = None, *, wait_inflight: float = 900.0) -> bool:
        """Stop Strata so a render has the card. True when it was stopped here."""
        with self.cond:
            # A call waiting to start Strata: it is not running, and the waiter
            # sees this graph (submitting) and goes on waiting.
            while self.phase not in ("idle", "waiting"):
                self.cond.wait(1.0)
            if self.phase == "waiting" or not alive(settings=settings):
                return False
            self.phase = "stopping"
            deadline = time.monotonic() + wait_inflight
            told = False
            while self.inflight > 0 and time.monotonic() < deadline:
                if not told:
                    _status("⏳ Waiting for the local model's answer before the render (Strata)…")
                    told = True
                self.cond.wait(1.0)
            if self.inflight > 0:
                logger.warning("strata: %d call(s) still running after %.0f s - stopping anyway",
                               self.inflight, wait_inflight)
        try:
            _status("⏸ Stopping the local model so the render has the GPU (Strata)…")
            ok = stop(settings)
            if ok:
                _status("✅ Local model stopped for the render - it starts again with the next agent call (Strata)")
            else:
                _status("⚠️ Strata still answers after the stop - the render shares the GPU with it", level="warning")
            return ok
        except Exception as exc:  # noqa: BLE001 - the render goes ahead regardless
            logger.warning("strata: stop failed (%s)", exc)
            _status(f"⚠️ Could not stop the local model before the render (Strata): {exc}", level="warning")
            return False
        finally:
            with self.cond:
                self.phase = "idle"
                self.cond.notify_all()


GATE = _Gate()


def wanted(settings: dict | None = None) -> bool:
    """Should a render send Strata away? A role runs on it and gpu_sharing is on."""
    try:
        return in_use(settings) and config(settings)["gpu_sharing"]
    except Exception:  # noqa: BLE001
        return False


def yield_gpu(prompt=None, settings: dict | None = None) -> bool:
    """Before a graph is queued: stop Strata when it is used, running, and the graph
    needs the card. Never raises."""
    try:
        if not wanted(settings):
            return False
        if prompt is not None:
            from src.utils.ollama_unload import needs_card
            if not needs_card(prompt):
                return False
        return GATE.yield_gpu(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("strata: yield before the render failed (%s)", exc)
        return False


def _prompt_hook(prompt):
    """agenty_core's prompt hook: Strata steps aside, and the start of the next
    call waits until this graph's POST has landed in ComfyUI's queue."""
    if not wanted():
        return None
    with GATE.cond:
        GATE.submitting += 1
    try:
        yield_gpu(prompt)
    except Exception:  # noqa: BLE001
        pass

    def _done():
        with GATE.cond:
            GATE.submitting = max(0, GATE.submitting - 1)
            GATE.cond.notify_all()
    return _done


# Settings > Strata > Start / Stop, run in the background (a start can take
# minutes); the page polls :func:`test`, which carries this.
_control = {"busy": "", "error": ""}


def control(action: str) -> dict:
    """Start or stop Strata by hand, through the gate (a start still waits for
    ComfyUI's queue; a stop waits for calls in flight). Returns at once."""
    action = str(action or "").strip().lower()
    if action not in ("start", "stop"):
        return {"ok": False, "error": f"unknown action {action!r}"}
    if _control["busy"]:
        return {"ok": False, "error": f"already busy: {_control['busy']}"}

    def _run():
        _control.update(busy=action, error="")
        try:
            if action == "start":
                GATE.acquire(force_start=True)
                GATE.release()
            elif not GATE.yield_gpu():
                if alive():
                    _control["error"] = "Strata still answers after the stop"
        except Exception as exc:  # noqa: BLE001
            _control["error"] = str(exc)
        finally:
            _control["busy"] = ""

    threading.Thread(target=_run, name=f"strata-{action}", daemon=True).start()
    return {"ok": True, "pending": action}


def yield_for_browser_queue(hold: float = 15.0) -> dict:
    """POST /agentY/strata/yield: the panel is about to queue the canvas (the
    Queue button, the prompt loop) - a graph agentY never sees. Strata steps aside
    the same way; the start of the next call waits *hold* seconds for the graph to
    reach ComfyUI's queue. ``{used, stopped}``."""
    done = _prompt_hook(None)
    if done is None:
        return {"used": False, "stopped": False}
    timer = threading.Timer(hold, done)
    timer.daemon = True
    timer.start()
    return {"used": True, "stopped": not alive(timeout=1.0)}


_installed = False


def install() -> None:
    """Register the prompt hook with agenty_core's ComfyUI client (once)."""
    global _installed
    if _installed:
        return
    try:
        from agenty_core.utils import comfyui_client
        comfyui_client.add_prompt_hook(_prompt_hook)
        _installed = True
    except Exception as exc:  # noqa: BLE001 - an older agenty_core: no seam, no stopping
        logger.info("strata: prompt hook not installed (%s)", exc)


# ── the model ────────────────────────────────────────────────────────────────

async def _acquire_async() -> None:
    """GATE.acquire off the event loop (it can wait minutes for a render). A call
    cancelled while it waits still lets go of the slot it gets."""
    task = asyncio.ensure_future(asyncio.to_thread(GATE.acquire))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        def _drop(t):
            if not t.cancelled() and t.exception() is None:
                GATE.release()
        task.add_done_callback(_drop)
        raise


def build_model(model_id: str, *, max_tokens: int | None = None, think: bool = False,
                settings: dict | None = None):
    """Strands' OpenAI client for Strata, every call through the gate."""
    global _built
    from src.utils.usage_capture import capturing_openai_model  # noqa: PLC0415
    base = capturing_openai_model()
    cfg = config(settings)

    class StrataModel(base):
        """A Strata call waits at the gate (Strata started, no render holding the card)."""

        provider = PROVIDER

        async def stream(self, *args, **kwargs):  # type: ignore[override]
            await _acquire_async()
            try:
                async for event in super().stream(*args, **kwargs):
                    yield event
            finally:
                GATE.release()

        async def structured_output(self, *args, **kwargs):  # type: ignore[override]
            await _acquire_async()
            try:
                async for event in super().structured_output(*args, **kwargs):
                    yield event
            finally:
                GATE.release()

    model = StrataModel(
        client_args={"api_key": cfg["api_key"], "base_url": cfg["base_url"],
                     "timeout": cfg["request_timeout"]},
        model_id=model_id or cfg["model"],
        params={"max_tokens": max_tokens or cfg["max_tokens"], "extra_body": extra_body(think, settings)},
    )
    _built = True
    install()
    return model
