"""A local model gives the GPU back before ComfyUI renders.

Ollama keeps a model loaded for minutes after its last answer. On a machine
where the agent's model and ComfyUI share one card that is exactly the wrong
moment to hold it: the agent has just handed a workflow over, and the render
that follows finds 24 of 32 GB taken by a model nobody is asking anything. It
crawls in low-VRAM mode or fails for memory, and the failure reads as a broken
workflow.

So at the handover - just before a graph is queued - :func:`release` asks Ollama
to unload what it holds. The model comes back by itself with the agent's next
call (a load from disk, around half a minute for a 20 GB model), which is the
price of one card doing both jobs.

It does nothing unless a role actually runs on Ollama, and it never raises: a
render must not fail because an unload could not be asked for. Small models are
left where they are - the memory embedder is a few hundred MB and is asked for
all the time. A graph that needs no card - API nodes and plumbing only - does
not send the model away either (:func:`needs_card`): the reload would be paid
for nothing.

The other way round is :func:`make_room`. ComfyUI keeps what it rendered with
on the card too, and for as long as it likes: measured after a session of Flux,
LTX and SD renders, 26 GB of a 32 GB card - part of it already pushed into
shared memory. A 25 GB model loaded beside that does not fail. It runs from
shared memory at 30 tokens a second instead of 2,000, and a turn that took a
minute takes half an hour and looks hung. So before a local model is called,
ComfyUI is asked how much of the card is free, and - when it is not enough, and
nothing is rendering or waiting to - asked to let go of its models (POST /free).
They load again with the next render.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request

logger = logging.getLogger("agentY.ollama")

# Below this a loaded model is not worth a reload (the embedder is ~0.3 GB).
MIN_BYTES = 2 * 1024 ** 3
TIMEOUT = 4.0
# Several graphs queued in one go ask once, not once each.
_QUIET_SECONDS = 3.0
_last = 0.0


def _settings() -> dict:
    from src.utils.settings import load_settings
    return load_settings()


def wanted(settings: dict | None = None) -> bool:
    """A role runs on Ollama, and the switch (``llm.ollama.unload_before_render``,
    default on) has not been turned off."""
    try:
        cfg = settings if settings is not None else _settings()
        llm = cfg.get("llm") or {}
        if not (llm.get("ollama") or {}).get("unload_before_render", True):
            return False
        specs = list((llm.get("tiers") or {}).values()) + list((llm.get("pipeline") or {}).values())
        return any(str(s or "").strip().lower().startswith("ollama,") for s in specs)
    except Exception:  # noqa: BLE001
        return False


# Nodes that run without the card: reading, writing, showing, wiring. Whatever is
# not listed here and is not an API node is taken to need it - a wrong "needs it"
# costs a reload, a wrong "does not" costs the render.
_LIGHT = frozenset({
    "LoadImage", "LoadImageMask", "LoadImageOutput", "SaveImage", "PreviewImage", "SaveAnimatedWEBP",
    "SaveAnimatedPNG", "SaveWEBM", "SaveVideo", "LoadVideo", "CreateVideo", "GetVideoComponents",
    "LoadAudio", "SaveAudio", "SaveAudioMP3", "SaveAudioOpus", "PreviewAudio", "PreviewAny", "Note",
    "MarkdownNote", "Reroute", "PrimitiveNode", "PrimitiveInt", "PrimitiveFloat", "PrimitiveString",
    "PrimitiveStringMultiline", "PrimitiveBoolean", "bEpicSendToViewer", "VHS_LoadImagePath",
    "VHS_LoadVideoPath", "VHS_LoadVideo", "VHS_VideoCombine", "ImageBatch", "ImageFromBatch",
})
_LIGHT_PREFIXES = ("AgentY", "agentY", "Primitive", "String")


def needs_card(prompt, object_info: dict | None = None) -> bool:
    """Would running *prompt* (an API-format graph) use the graphics card?

    No only when every node is an API node (it runs on somebody else's card) or
    plumbing. Anything unknown, and a graph that cannot be read, is a yes.
    """
    if not isinstance(prompt, dict) or not prompt:
        return True
    if object_info is None:
        try:
            from agenty_core.tools.comfyui import _get_object_info
            object_info = _get_object_info() or {}
        except Exception:  # noqa: BLE001
            object_info = {}
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        if not cls:
            return True
        if cls in _LIGHT or cls.startswith(_LIGHT_PREFIXES):
            continue
        info = object_info.get(cls) or {}
        if info.get("api_node") or str(info.get("python_module") or "").startswith("comfy_api_nodes"):
            continue
        return True
    return False


def _call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "{}")


def release(host: str | None = None, *, settings: dict | None = None, prompt=None) -> list[str]:
    """Ask Ollama to unload its large models now. Returns the names it was asked
    to drop (empty when there was nothing to do, or it could not be reached).

    With *prompt* (the graph about to be queued) it is asked only when that
    graph needs the card (:func:`needs_card`).

    An unload asked for while the model is answering takes effect when that
    answer is done - another conversation mid-call is not cut off.
    """
    global _last
    if not wanted(settings):
        return []
    if prompt is not None and not needs_card(prompt):
        logger.info("ollama: the graph needs no card - the model stays loaded")
        return []
    now = time.monotonic()
    if now - _last < _QUIET_SECONDS:
        return []
    _last = now
    try:
        if host is None:
            from src.utils.settings import ollama_host
            host = ollama_host()
        host = str(host).rstrip("/")
        loaded = _call(host + "/api/ps").get("models") or []
    except Exception as exc:  # noqa: BLE001
        logger.info("ollama: could not ask what is loaded (%s)", exc)
        return []
    dropped = []
    for model in loaded:
        name = str(model.get("name") or model.get("model") or "")
        size = int(model.get("size_vram") or model.get("size") or 0)
        if not name or size < MIN_BYTES:
            continue
        try:
            _call(host + "/api/generate", {"model": name, "keep_alive": 0})
            dropped.append(name)
        except Exception as exc:  # noqa: BLE001
            logger.info("ollama: could not unload %s (%s)", name, exc)
    if dropped:
        logger.info("ollama: unloaded %s before the render", ", ".join(dropped))
    return dropped


# ── the other way: ComfyUI gives the card back before a local model works ───────

# Asked at most this often: the answer does not change between two calls of one step.
_ROOM_QUIET_SECONDS = 20.0
# No estimate to be had (no card to ask, Ollama not answering): what a 27b model takes.
_NEED_FALLBACK = 20 * 1024 ** 3
_room_last = 0.0


def _comfy(path: str, body: dict | None = None) -> dict:
    """ComfyUI, asked directly and briefly: this runs in front of a model call,
    and a ComfyUI that does not answer must not hold that call up."""
    from agenty_core.utils.comfyui_client import get_client
    url = str(get_client().base_url).rstrip("/") + path
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=TIMEOUT) as r:
        raw = r.read().decode("utf-8", "replace").strip()
    try:
        got = json.loads(raw) if raw else {}
    except ValueError:
        got = {}
    return got if isinstance(got, dict) else {}


def make_room(host: str | None = None, *, settings: dict | None = None) -> bool:
    """Before a local model is called: if ComfyUI holds so much of the card that
    the model would not fit beside it, and nothing is rendering or queued, ask
    ComfyUI to unload its models. True when it was asked. Never raises."""
    global _room_last
    try:
        if not wanted(settings):
            return False
        now = time.monotonic()
        if now - _room_last < _ROOM_QUIET_SECONDS:
            return False
        _room_last = now
        devices = _comfy("/system_stats").get("devices") or []
        free = max((int(d.get("vram_free") or 0) for d in devices if str(d.get("type")) == "cuda"), default=None)
        if free is None:
            return False                      # no card of its own to give back
        # ComfyUI's "free" is the card less what ComfyUI ITSELF holds (measured:
        # 27.6 GB "free" with a 25 GB model of Ollama's loaded beside it) - so it
        # is the room there is for the model whether that is loaded yet or not.
        try:
            from src.utils.ollama_window import need_bytes
            need = need_bytes() or _NEED_FALLBACK
        except Exception:  # noqa: BLE001
            need = _NEED_FALLBACK
        if free >= need:
            return False
        queue = _comfy("/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            return False                      # it is using what it holds
        _comfy("/free", {"unload_models": True, "free_memory": True})
        logger.info("comfyui: asked to unload its models - %.1f GB free, the local model needs %.1f",
                    free / 1024 ** 3, need / 1024 ** 3)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info("comfyui: could not make room for the local model (%s)", exc)
        return False


class MakeRoomHook:
    """Every agent, before every model call: when its model is a local one,
    :func:`make_room`. (The model under an agent can change - a lead's turn -
    so it is asked at the call, not when the agent is built.)"""

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks import BeforeModelCallEvent
        registry.add_callback(BeforeModelCallEvent, self._before)

    def _before(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            if "ollama" in type(getattr(event.agent, "model", None)).__name__.lower():
                make_room()
        except Exception:  # noqa: BLE001 - never cost a call
            pass
