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
