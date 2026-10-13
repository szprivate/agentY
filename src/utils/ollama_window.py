"""The window a local model is loaded with, worked out from the graphics card.

``llm.ollama.num_ctx`` is the model's whole window - what goes in AND what comes
out. Too small and Ollama drops the start of a call without a word; too large and
the model no longer fits the card, spills onto the CPU and a turn takes minutes.
Neither is something to ask a person to tune by hand, so the setting's default is
-1 - "work it out" (0 and nothing mean the same): the largest window with which the largest local model the
agents use still fits the card.

What a model needs is estimated as

    its file  +  what a loaded model costs besides its weights  +  window x per token

The file's size is Ollama's own (/api/tags); the per-token cost comes from the
model's shape (/api/show: layers, width, key/value heads). The two constants
are measured, on qwen3.8:27b on a 32 GB card: 23.0 GB at 65,536, 25.2 GB at
98,304, 27.5 GB at 131,072 - a third of what the plain formula for a 16-bit
cache gives, and 2.3 GB on top of the file. A card is filled to 85 %: the rest
is what the desktop, the embedder and a decision model take.

Why not simply the model's own window: Ollama sets the cache aside for the whole
window when it loads the model, used or not. qwen3.8:27b at its own 262,144 is
about 35 GB - on a 32 GB card it spills onto the CPU. So the model's window is
the ceiling, and the card decides how much of it can be had.

A number in the setting is used as it is. Nothing here raises: with no card to
ask, or no Ollama, the answer is a window the orchestrator's prompt fits.
"""
from __future__ import annotations

import json
import logging
import subprocess
import urllib.request

logger = logging.getLogger("agentY.ollama_window")

# Up to a model's own window (262,144 for a current Qwen) where the card has the room for it.
LADDER = (32_768, 49_152, 65_536, 81_920, 98_304, 131_072, 196_608, 262_144)
FALLBACK = 65_536            # no card or no model to measure: the orchestrator's prompt fits this
CARD_SHARE = 0.85
OVERHEAD = 2.3 * 2 ** 30     # a loaded model, besides its weights and its cache
CACHE_FACTOR = 0.33          # measured cache per token / (2 x layers x key-value width x 2 bytes)

_cache: dict = {}


def _get(url: str, body: dict | None = None, timeout: float = 4.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # Never through a proxy: this is the machine's own Ollama.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def card_bytes() -> int:
    """The memory of the largest NVIDIA card, 0 when there is none to ask."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=6).stdout
        sizes = [int(float(line.strip())) for line in out.splitlines() if line.strip()]
        return max(sizes) * 2 ** 20 if sizes else 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("ollama window: no card to ask (%s)", exc)
        return 0


def model_facts(model: str, host: str) -> dict | None:
    """``{"size", "per_token", "limit"}`` of *model* from Ollama, or None."""
    try:
        tags = _get(host + "/api/tags").get("models") or []
        name = model if ":" in model else model + ":latest"
        size = next((int(m.get("size") or 0) for m in tags if m.get("name") in (model, name)), 0)
        info = _get(host + "/api/show", {"model": model}).get("model_info") or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("ollama window: %s could not be asked about (%s)", model, exc)
        return None

    def of(suffix):
        value = next((v for k, v in info.items() if k.endswith(suffix)), None)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    layers, width, heads, kv = of(".block_count"), of(".embedding_length"), of(".attention.head_count"), \
        of(".attention.head_count_kv")
    if not (size and layers and width and heads):
        return None
    per_token = 2 * layers * width * (kv or heads) / heads * 2 * CACHE_FACTOR
    return {"size": size, "per_token": per_token, "limit": int(of(".context_length") or 0)}


def fit(card: int, facts: dict) -> int:
    """The largest window of the ladder with which a model of *facts* fits *card*."""
    room = card * CARD_SHARE - facts["size"] - OVERHEAD
    limit = facts.get("limit") or LADDER[-1]
    best = LADDER[0]
    for window in LADDER:
        if window <= limit and window * facts["per_token"] <= room:
            best = window
    return best


def local_models() -> list:
    """The Ollama models the agents are set to use, by name."""
    try:
        from src.utils.settings import load_settings
        llm = load_settings().get("llm") or {}
    except Exception:  # noqa: BLE001
        return []
    names = []
    for block in ("tiers", "pipeline"):
        for value in (llm.get(block) or {}).values():
            spec = str(value or "")
            if spec.lower().startswith("ollama,"):
                name = spec.split(",", 1)[1].strip()
                if name and name not in names:
                    names.append(name)
    return names


def auto(host: str | None = None) -> dict:
    """The window worked out for this machine: ``{"window", "why"}``. Measured once per start."""
    if "auto" in _cache:
        return _cache["auto"]
    if host is None:
        try:
            from src.utils.settings import ollama_host
            host = str(ollama_host()).rstrip("/")
        except Exception:  # noqa: BLE001
            host = "http://localhost:11434"
    card = card_bytes()
    models = local_models()
    answer = {"window": FALLBACK, "why": ""}
    if not card:
        answer["why"] = "no NVIDIA card could be asked for its memory"
    elif not models:
        answer["why"] = "no agent is set to a local model"
    else:
        known = [(m, model_facts(m, host)) for m in models]
        known = [(m, f) for m, f in known if f]
        if not known:
            answer["why"] = "Ollama could not be asked about " + ", ".join(models)
        else:
            # Every local model is loaded with the one window, so the largest decides.
            model, facts = max(known, key=lambda mf: mf[1]["size"])
            window = fit(card, facts)
            need = (facts["size"] + OVERHEAD + window * facts["per_token"]) / 2 ** 30
            answer = {"window": window, "model": model, "facts": facts,
                      "why": f"{model} needs about {need:.0f} GB with it, on a {card / 2 ** 30:.0f} GB card"}
    _cache["auto"] = answer
    return answer


def resolve(setting=None) -> int:
    """The window to load a local model with: *setting* when it is a number above
    zero (``llm.ollama.num_ctx``, read when not given), else :func:`auto`."""
    if setting is None:
        try:
            from src.utils.settings import load_settings
            setting = ((load_settings().get("llm") or {}).get("ollama") or {}).get("num_ctx")
        except Exception:  # noqa: BLE001
            setting = 0
    try:
        value = int(setting or 0)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else auto()["window"]


def need_bytes() -> int:
    """What the largest local model takes of the card at the window it is loaded
    with (the setting's, or the one worked out) - 0 when that cannot be said."""
    facts = auto().get("facts")
    if not facts:
        return 0
    return int(facts["size"] + OVERHEAD + resolve() * facts["per_token"])


# The most an answer is given, and the share of a small window it may take.
OUTPUT_MOST = 16_384
OUTPUT_LEAST = 4_096
OUTPUT_SHARE = 4            # at most a quarter of the window


def output_allowance(setting=None, window: int | None = None) -> int:
    """How many tokens a local model may write in one answer (``llm.ollama.max_tokens``).

    A number above zero is used as it is. Otherwise (-1, the default) it is
    worked out from the window: 16,384 - more than an answer needs, a tool call
    with a whole workflow in it included - but never more than a quarter of the
    window, because Ollama takes the allowance OUT of the window before it
    measures what was sent. A large allowance does not buy longer answers; it
    buys a shorter memory.
    """
    if setting is None:
        try:
            from src.utils.settings import load_settings
            setting = ((load_settings().get("llm") or {}).get("ollama") or {}).get("max_tokens")
        except Exception:  # noqa: BLE001
            setting = -1
    try:
        value = int(setting or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    window = int(window or resolve())
    return max(OUTPUT_LEAST, min(OUTPUT_MOST, window // OUTPUT_SHARE))


def forget() -> None:
    """Measure again next time (the models were changed in Settings)."""
    _cache.clear()
