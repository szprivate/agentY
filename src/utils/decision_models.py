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

**Never without Ollama.** A decision model runs in Ollama, and agentY cannot
count on Ollama next to a hosted provider. Each use is ``"auto"`` by default:

* the *router* is used when the orchestrator itself runs on Ollama - it answers
  a question only a local orchestrator gets wrong;
* the *prompt check* is used whenever Ollama answers on this machine, whatever
  the orchestrator runs on - a hosted model renders pictures that miss their
  prompt just as a local one does.

**The weights come on first use.** They are no part of any install. The first
time a use is wanted and its model is not there, the download starts through
Ollama in the background, with a bar in the run card, and that use has "no
opinion" until the model has arrived - a turn never waits for a download. Every
other failure - no Ollama, a timeout - is "no opinion" as well, which leaves
agentY doing exactly what it does without one.

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
    "download": True,                 # fetch a missing model through Ollama on first use
}
# About how large each is, for the line that says a download has begun.
SIZE_GB = {"nimble": 9.4, "clef-flash": 10.0, "clef": 18.0, "tev1": 4.5}
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_installed: dict = {"at": 0.0, "host": "", "names": (), "up": False}
_pulling: dict = {}                    # model -> the thread fetching it
_pull_lock = __import__("threading").Lock()


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
            up = True
        except Exception:  # noqa: BLE001
            names, up = (), False
        _installed.update(at=now, host=host, names=names, up=up)
    return any(n == model or n.split(":")[0] == model for n in _installed["names"])


def ollama_up(host: str | None = None) -> bool:
    """Whether Ollama answers on this machine (asked with :func:`installed`)."""
    installed("", host)
    return bool(_installed["up"])


# ── the weights, on first use ────────────────────────────────────────────────

def bar(done: int, total: int, width: int = 16) -> str:
    """A text bar the panel's run card rewrites in place (it knows the blocks)."""
    share = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
    full = int(round(share * width))
    return "\u2588" * full + "\u2591" * (width - full)


def progress_line(model: str, done: int, total: int, why: str) -> str:
    pct = 0 if total <= 0 else int(100 * done / total)
    return (f"\u2b07\ufe0f {model} {bar(done, total)} {pct} % \u00b7 {done / 1e9:.1f} / {total / 1e9:.1f} GB"
            f" \u2014 {why}, first use")


def _pull(model: str, why: str, host: str, say, notify) -> bool:
    """Download *model* through Ollama, reporting as it goes. True when it is there."""
    req = urllib.request.Request(host + "/api/pull", data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    layers: dict = {}
    last = (-1, 0.0)
    try:
        with urllib.request.urlopen(req, timeout=24 * 3600) as r:
            for raw in r:
                try:
                    msg = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if msg.get("error"):
                    raise RuntimeError(str(msg["error"]))
                if msg.get("digest") and msg.get("total"):
                    layers[msg["digest"]] = (int(msg.get("completed") or 0), int(msg["total"]))
                    done, total = sum(v[0] for v in layers.values()), sum(v[1] for v in layers.values())
                    pct, now = int(100 * done / max(1, total)), time.monotonic()
                    if pct != last[0] and (pct >= last[0] + 2 or now - last[1] > 5):
                        last = (pct, now)
                        say(progress_line(model, done, total, why))
    except Exception as exc:  # noqa: BLE001
        notify(f"\u2b07\ufe0f {model} could not be downloaded through Ollama ({str(exc)[:160]}). "
               f"The {why} stays off until it is there: `ollama pull {model}`.")
        return False
    _installed["at"] = 0.0                               # ask Ollama afresh
    ok = installed(model, host)
    notify(f"\u2705 {model} is ready - the {why} is on from the next time it is needed." if ok else
           f"\u2b07\ufe0f Ollama finished the download of {model} but does not list it; the {why} stays off.")
    return ok


def fetch_in_background(model: str, why: str, host: str | None = None) -> bool:
    """Start the download of *model* unless one is running. Returns whether this
    call started it. The turn that asked goes on without the model."""
    host = host or _host()
    with _pull_lock:
        running = _pulling.get(model)
        if running is not None and running.is_alive():
            return False
        import contextvars
        import threading
        try:
            from agenty_core.utils.progress_signal import push as say
        except Exception:  # noqa: BLE001
            say = lambda _line: None  # noqa: E731
        try:
            from src.utils.status_bus import notify
        except Exception:  # noqa: BLE001
            notify = lambda _text, **_kw: None  # noqa: E731
        size = SIZE_GB.get(model.split(":")[0])
        notify(f"\u2b07\ufe0f Downloading {model}" + (f" (about {size:g} GB)" if size else "")
               + f" through Ollama for the {why} - first use. It is used as soon as it is there; "
                 f"nothing waits for it.")
        ctx = contextvars.copy_context()                  # so the bar reaches THIS turn's run card
        thread = threading.Thread(target=lambda: ctx.run(_pull, model, why, host, say, notify),
                                  name=f"agentY-pull-{model}", daemon=True)
        _pulling[model] = thread
        thread.start()
        return True


def orchestrator_is_local() -> bool:
    try:
        from src.agent import role_model
        return str(role_model("orchestrator") or "").strip().lower().startswith("ollama,")
    except Exception:  # noqa: BLE001
        return False


WHY = {"router": "routing of requests about the open graph", "prompt_check": "prompt check"}


def model_for(use: str, cfg: dict | None = None) -> str:
    """The model to ask for *use* ("router" / "prompt_check"), or ``""`` for none.

    ``auto``: the default model - for the router when the orchestrator runs on
    Ollama, for the prompt check whenever Ollama answers here. A name: that
    model, whatever the orchestrator runs on. ``off``: none. Never without
    Ollama. A model that is wanted but not there yet is fetched in the
    background (``download``), and the answer is ``""`` until it has arrived.
    """
    cfg = cfg or settings()
    choice = str(cfg.get(use) or "auto").strip()
    if choice.lower() in ("off", "false", "no", "0", ""):
        return ""
    try:
        if not ollama_up():
            return ""
        if choice.lower() == "auto":
            model = str(cfg.get(f"{use}_model") or "")
            if not model or (use == "router" and not orchestrator_is_local()):
                return ""
        else:
            model = choice
        if installed(model):
            return model
        if cfg.get("download", True):
            fetch_in_background(model, WHY.get(use, use))
        return ""
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


def change_probability(request: str, graph=None, cfg: dict | None = None) -> float | None:
    """How likely *request* asks to change a workflow that exists rather than to
    build one - or ``None`` when there is no decision model to ask.

    The question is put about the WORDING alone, as a choice between the two.
    The first version also told the model that a graph was open, and asked
    yes/no: with a graph open - the only time the router is asked - it then said
    "change" to 30 of 32 plain build requests (10 of 40 right; the 40 of 40
    measured the night before had told it "the canvas is empty" for every build,
    which gave the answer away). Asked this way it is 40 of 40 with a graph
    open, no edit under 0.98 and no build over 0.04. *graph* is accepted and
    not used, for that reason.
    """
    cfg = cfg or settings()
    model = model_for("router", cfg)
    if not model or not str(request or "").strip():
        return None
    try:
        answers = ask(model, {"request": str(request)[:6000]},
                      {"what": {"type": "choice", "instructions": "What does the request ask for?", "criteria": {
                          "change": "to modify a workflow that already exists ('this workflow', 'add ... to it', "
                                    "'change', 'instead of')",
                          "new": "to build a workflow, described from scratch: its model, its prompt, its settings",
                      }}}, timeout=15)
        value = ((answers.get("what") or {}).get("probabilities") or {}).get("change")
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
    ``positive`` input - or its guider's ``conditioning`` - leads back to. Not
    the longest text in the graph: that is as often the negative prompt."""
    if not isinstance(graph, dict):
        return ""
    starts = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        if isinstance(inputs.get("positive"), list):
            starts.append(inputs["positive"])
        elif "guider" in str(node.get("class_type") or "").lower() and isinstance(inputs.get("conditioning"), list):
            starts.append(inputs["conditioning"])            # BasicGuider: one conditioning, no negative
    for link in starts:
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
