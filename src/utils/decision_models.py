"""Decision models: small local judges for the choices a turn makes.

Ollama's "decision" models do not write. Given a situation and typed questions
they answer each with a probability, in a fraction of a second. Two of the
things agentY has to decide are exactly that shape, and were measured on
2026-10-13 (40 benchmark tasks; renders with known prompts):

* **Is the request a change to the graph that is open, or a new workflow?**
  ``nimble`` (9B, text) answered 40 of 40 right. Two of three local orchestrators
  got this wrong the day before and built a second workflow beside the first.
* **Does the picture show what was asked for?** ``clef-flash`` (9B, reads images)
  caught three renders that plainly did not match their prompt - renders the
  benchmark had passed, because it checks the workflow and not the picture.

**For the local path only.** A decision model runs in Ollama, and agentY cannot
count on Ollama next to a hosted provider. So each use is ``"auto"`` by default:
it is active only when the orchestrator itself runs on Ollama AND the model is
already installed there. Nothing is downloaded, nothing is asked of a hosted
setup, and every failure - no Ollama, no model, a timeout - means "no opinion",
which leaves agentY doing exactly what it does without one.

A probability from these models is how sure the model is, not how often it is
right. Both uses therefore only ever *advise*: the router can save or cause one
extra call, the prompt check adds a line to the report. Neither rebuilds or
re-renders anything on its own.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.request

logger = logging.getLogger("agentY.decision")

ENDPOINT = "/v1/systemone"
DEFAULTS = {
    "router": "auto",                 # "auto" | "off" | a model name
    "router_model": "nimble",
    "router_build_below": 0.3,        # at most this: let the new build through at once
    "prompt_check": "auto",
    "prompt_check_model": "clef-flash",
    "prompt_check_below": 0.25,       # a picture this unlikely to show its prompt is reported
    "prompt_check_max": 8,
}
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_installed: dict = {"at": 0.0, "host": "", "names": ()}


def settings() -> dict:
    out = dict(DEFAULTS)
    try:
        from src.utils.settings import load_settings
        for key, value in (load_settings().get("decision") or {}).items():
            if key in out:
                out[key] = type(DEFAULTS[key])(value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("decision: settings unreadable (%s)", exc)
    return out


def _host() -> str:
    from src.utils.settings import ollama_host
    return str(ollama_host()).rstrip("/")


def _call(url: str, body: dict | None = None, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "{}")


def installed(model: str, host: str | None = None) -> bool:
    """Whether Ollama has *model*. Asked at most once a minute; no Ollama is no."""
    host = host or _host()
    now = time.monotonic()
    if _installed["host"] != host or now - _installed["at"] > 60:
        try:
            names = tuple(str(m.get("name") or "") for m in _call(host + "/api/tags", timeout=4).get("models") or [])
        except Exception:  # noqa: BLE001
            names = ()
        _installed.update(at=now, host=host, names=names)
    return any(n == model or n.split(":")[0] == model for n in _installed["names"])


def orchestrator_is_local() -> bool:
    try:
        from src.agent import role_model
        return str(role_model("orchestrator") or "").strip().lower().startswith("ollama,")
    except Exception:  # noqa: BLE001
        return False


def model_for(use: str, cfg: dict | None = None) -> str:
    """The model to ask for *use* ("router" / "prompt_check"), or ``""`` for none.

    ``auto``: the default model, when the orchestrator runs on Ollama and the
    model is installed. A name: that model, when installed (whatever the
    orchestrator runs on - the user asked for it). ``off``: none.
    """
    cfg = cfg or settings()
    choice = str(cfg.get(use) or "auto").strip()
    if choice.lower() in ("off", "false", "no", "0", ""):
        return ""
    try:
        if choice.lower() == "auto":
            model = str(cfg.get(f"{use}_model") or "")
            return model if model and orchestrator_is_local() and installed(model) else ""
        return choice if installed(choice) else ""
    except Exception:  # noqa: BLE001
        return ""


def ask(model: str, state, questions: dict, images=None, timeout: float = 20.0) -> dict:
    """The model's answers, keyed like *questions*. Raises on any failure."""
    body = {"model": model, "state": state, "questions": questions, "keep_alive": "5m"}
    if images:
        body["images"] = list(images)
    return _call(_host() + ENDPOINT, body, timeout=timeout).get("answers") or {}


# ── is this a change to the open graph? ──────────────────────────────────────

def graph_summary(graph) -> str:
    """What is open, in a line: the node types and how many of each."""
    counts: dict = {}
    for node in (graph or {}).values():
        if isinstance(node, dict):
            name = str(node.get("class_type") or "?")
            counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{n}" + (f" x{c}" if c > 1 else "") for n, c in sorted(counts.items()))[:1500]


def change_probability(request: str, graph, cfg: dict | None = None) -> float | None:
    """How likely *request* asks to change the open graph rather than to build a
    new workflow - or ``None`` when there is no decision model to ask."""
    cfg = cfg or settings()
    model = model_for("router", cfg)
    if not model or not str(request or "").strip():
        return None
    try:
        answers = ask(model, {"request": str(request)[:6000],
                              "canvas": "a workflow graph is open on the canvas: " + graph_summary(graph)},
                      {"change": {"type": "noul", "instructions": (
                          "Does the request ask to CHANGE the workflow graph that is already open "
                          "(add, remove, rewire or re-set something in it) rather than to build a new "
                          "workflow?")}}, timeout=15)
        value = (answers.get("change") or {}).get("noul")
        return float(value) if isinstance(value, (int, float)) else None
    except Exception as exc:  # noqa: BLE001
        logger.info("decision: router %s gave no answer (%s)", model, exc)
        return None


# ── does the picture show what was asked for? ────────────────────────────────

_TEXT_KEYS = ("text", "prompt", "positive_prompt", "t5xxl", "clip_l", "string", "value")


def _text_of(node) -> str:
    inputs = (node or {}).get("inputs") or {}
    texts = [v for k, v in inputs.items() if k in _TEXT_KEYS and isinstance(v, str) and len(v.strip()) > 3]
    return max(texts, key=len).strip() if texts else ""


def positive_prompt(graph) -> str:
    """The prompt a ComfyUI graph was sampled with: what its sampler's
    ``positive`` input leads back to. Not the longest text in the graph - that is
    as often the negative prompt."""
    if not isinstance(graph, dict):
        return ""
    for node in graph.values():
        link = ((node or {}).get("inputs") or {}).get("positive") if isinstance(node, dict) else None
        seen = set()
        while isinstance(link, list) and link and str(link[0]) not in seen:
            seen.add(str(link[0]))
            src = graph.get(str(link[0]))
            if not isinstance(src, dict):
                break
            text = _text_of(src)
            if text:
                return " ".join(text.split())
            inputs = src.get("inputs") or {}
            nxt = None
            for key in ("positive", "conditioning", "conditioning_1", "conditioning_to", "text", "prompt"):
                if isinstance(inputs.get(key), list):
                    nxt = inputs[key]
                    break
            if nxt is None:
                nxt = next((v for v in inputs.values() if isinstance(v, list) and len(v) == 2), None)
            link = nxt
    return ""


def prompt_of_file(path: str) -> str:
    """The positive prompt stored in a ComfyUI render, ``""`` when it has none."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            meta = im.info.get("prompt")
        return positive_prompt(json.loads(meta)) if meta else ""
    except Exception:  # noqa: BLE001
        return ""


def _encoded(path: str, side: int = 1024) -> str:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prompt_check(paths, cfg: dict | None = None) -> list[dict]:
    """The pictures among *paths* that probably do not show their own prompt:
    ``[{"path", "prompt", "probability"}]``. Empty when all do, when none carries
    a prompt, or when there is no decision model to ask."""
    cfg = cfg or settings()
    model = model_for("prompt_check", cfg)
    if not model:
        return []
    out = []
    todo = [str(p) for p in paths or [] if str(p).lower().endswith(_IMAGE_EXTS)][:int(cfg["prompt_check_max"])]
    for path in todo:
        prompt = prompt_of_file(path)
        if not prompt:
            continue
        try:
            answers = ask(model, "Look at the attached picture.",
                          {"shows": {"type": "noul", "instructions": f"Does the picture show this: {prompt[:1500]}"}},
                          images=[_encoded(path)], timeout=120)
            value = (answers.get("shows") or {}).get("noul")
        except Exception as exc:  # noqa: BLE001
            logger.info("decision: prompt check with %s gave no answer (%s)", model, exc)
            return out
        if isinstance(value, (int, float)) and float(value) < float(cfg["prompt_check_below"]):
            out.append({"path": path, "prompt": prompt, "probability": round(float(value), 3), "model": model})
    return out


def prompt_check_text(found: list[dict]) -> str:
    """What the chat panel is told about pictures that missed their prompt."""
    if not found:
        return ""
    import os
    lines = [f"\U0001f50e Prompt check ({found[0]['model']}, a local decision model): "
             + ("this picture" if len(found) == 1 else f"{len(found)} pictures")
             + " may not show what was asked for. Nothing was re-rendered - have a look:"]
    for f in found:
        prompt = f["prompt"] if len(f["prompt"]) <= 110 else f["prompt"][:107] + "..."
        lines.append(f"- `{os.path.basename(f['path'])}` - {f['probability']:.2f} that it shows: “{prompt}”")
    return "\n".join(lines)
