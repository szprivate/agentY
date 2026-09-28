"""Telling "this model cannot see" apart from "the call failed".

They need opposite handling and look identical in a traceback.

A transient failure deserves a retry, and — for QA — a pass, because a judge that
cannot be reached must never condemn the user's work. A model that is not
multimodal deserves neither: retrying it fails the same way every time, and
passing on its silence means every output is waved through by a judge with its
eyes shut. It is a setting someone has to change, and nothing else will fix it.

The case that produced this: `qa_judge` and the vision tier were both pointed at
`dashscope,qwen3.7-max`. Handed an image, DashScope answers

    invalid_parameter_error — The provided messages input is invalid.
    The error info is [Unexpected item type in content.]

which says nothing about vision. Downstream, `analyze_image` reported "the vision
agent call failed … retry", so the agent retried three times, then tried a
different path, a temp copy, and a different question — and finally told the user
the analysis was failing without ever saying why. Meanwhile QA, on the same
model, silently passed a silver hatchback against "must show a RED SPORTS CAR on
a racetrack".
"""
from __future__ import annotations

# Substrings that identify a provider refusing image content, per provider. Kept
# as fragments rather than whole messages: the wording around them varies by
# endpoint version, and the fragments are what stays put.
_BLIND_MARKERS = (
    # DashScope / Alibaba Model Studio (OpenAI-compatible endpoint)
    "unexpected item type in content",
    # OpenAI and compatible gateways
    "does not support image",
    "invalid content type",
    "image_url is not supported",
    "unsupported content type",
    # Anthropic
    "does not support image input",
    # Ollama, when the model has no vision projector
    "does not support images",
    "unable to process image",
)


def looks_blind(error) -> bool:
    """True when *error* says the model was handed an image it cannot accept.

    False for anything ambiguous — a timeout, a rate limit, a network drop. The
    cost of a wrong True is telling someone to change a setting that was fine, so
    this only fires on wording that a working model never produces.
    """
    text = str(error or "").lower()
    return any(marker in text for marker in _BLIND_MARKERS)


def model_name(agent) -> str:
    """The model id behind *agent*, or "" when it cannot be read.

    Only a real string is accepted. Anything else — a mock, a lazily-built
    config, a provider that names it differently — would otherwise be formatted
    into a message telling someone to go and change it.
    """
    try:
        name = (getattr(agent, "model", None) or object()).config.get("model_id")
    except Exception:  # noqa: BLE001
        return ""
    return name.strip() if isinstance(name, str) else ""


def supports_vision(spec: str) -> bool:
    """Whether a ``'provider,model'`` spec can accept image content blocks.

    Both mistakes hurt, and not symmetrically: a multimodal model wrongly taken
    for text-only never gets shown an image, while a text-only one wrongly taken
    for multimodal makes the provider reject EVERY turn of the conversation from
    then on (DashScope's "Unexpected item type in content."), because the rejected
    image stays in the history. So this errs toward False unless the model is
    confidently multimodal.

    The operator's own lists win, in both directions — model families move faster
    than any list kept in the code, and whoever runs the model knows:
    ``llm.text_only_models`` is checked first, then ``llm.vision_models``, both
    plain substrings matched against the model id.
    """
    provider, _, model = str(spec or "").lower().partition(",")
    provider = provider.strip()
    model = (model or provider).strip()
    if not provider:
        return False
    try:
        from src.utils.settings import load_settings
        llm = load_settings().get("llm") or {}
        for pattern in (llm.get("text_only_models") or []):
            text = str(pattern).lower().strip()
            if text and text in model:
                return False
        for pattern in (llm.get("vision_models") or []):
            text = str(pattern).lower().strip()
            if text and text in model:
                return True
    except Exception:  # noqa: BLE001 — settings must never break the check
        pass
    # Providers whose current models are multimodal across the board.
    if provider in ("claude", "anthropic", "bedrock", "google", "gemini"):
        return True
    # Otherwise require an explicit vision marker in the model id.
    markers = ("-vl", "vl-", "vl:", "vision", "omni", "4o", "gpt-4.1", "o4-",
               "llava", "minicpm-v", "gemma3", "gemma-3", "pixtral", "internvl", "moondream")
    return any(marker in model for marker in markers)


def blind_model_message(role: str, model: str = "", detail: str = "") -> str:
    """What to say when the model on *role* cannot see.

    Names the setting, because "the vision agent call failed" sends an agent
    round the retry loop that produced this, and sends a person looking in the
    wrong place.
    """
    which = f" (`{model}`)" if model else ""
    lines = [
        f"The model configured for **{role}**{which} is not multimodal — it "
        "cannot accept images at all, so this will fail identically every time.",
        "",
        "This is a configuration problem, not a transient error. **Do not retry**, "
        "and do not work around it by guessing what the image contains.",
        "",
        f"Tell the user to point the `{role}` tier at a vision-capable model "
        "(Settings > Models, or `llm.tiers` in settings.local.json) — for "
        "DashScope that means a `-vl-` model such as `qwen3-vl-flash`.",
    ]
    if detail:
        lines += ["", f"The provider said: {detail}"]
    return "\n".join(lines)
