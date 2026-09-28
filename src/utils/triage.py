"""How much model does this message deserve?

agentY used to have a triage stage that read each message, classified it into a
``MessageIntent`` and handed it to a fixed handler. That went away with the staged
pipeline: the orchestrator routes natively now, and nothing here brings routing
back. What went away with it was the other thing triage was doing — noticing that
"thanks, that's perfect" and "build a five-shot video from these four references,
one room each" are not the same kind of work, and that one model answering both
is either too expensive for the first or too weak for the second.

So this module decides the **seat**, never the route: read the message, say
``simple`` or ``complex``, and point the live orchestrator at the model configured
for that weight (``llm.triage.simple`` / ``llm.triage.complex``). The turn then
runs exactly as it always did.

Three things it must never do, in order of how badly they would hurt:

* **Block a turn.** Every failure — the classifier unreachable, a timeout, a model
  answering in prose, a seat whose provider has no key — leaves the seat exactly as
  it was and carries on. The whole module is optional by construction.
* **Send an image to a model that cannot see one.** The panel decides whether to
  embed image bytes *before* the turn starts, from the configured seat. Handing
  that message to a text-only model does not fail once, it fails every turn from
  then on ("Unexpected item type in content."), because the rejected image stays in
  the history. Hence the vision guard in :func:`_pick_seat`.
* **Cost more than it saves.** The classifier is one short JSON call on the
  ``fast_utility`` tier, and the shortcuts below answer the common cases —
  an acknowledgement, a message with hooks or attachments — without any call.

A word on follow-ups: "make it brighter" is three words and it means re-running a
generation. There is no hysteresis logic here for that; the classifier is TOLD
(in ``config/system_prompts/system_prompt.triage.md``) that a follow-up inherits
the weight of what it follows, and is given the previous decision to follow.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

SIMPLE = "simple"
COMPLEX = "complex"

# Providers LLMFunctions can actually reach. A classifier pinned to anything else
# would be silently answered by Ollama (its fallback), so we decline instead.
_SPEAKABLE = {"ollama", "claude", "anthropic",
              "dashscope", "modelstudio", "qwen", "alibaba"}

# Which environment variable stands for "this provider is set up". Mirrors the
# gate _available_models applies to the model picker: a seat nobody can call is
# not a seat. Ollama is local and needs no key.
_PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "ollama": (),
    "claude": ("ANTHROPIC_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    "modelstudio": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    "qwen": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    "alibaba": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "gpt": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

# Messages that carry no work. Matched only when the WHOLE message is one of
# them (punctuation and emoji stripped), so "ok, now render all four" is not
# mistaken for "ok". Kept deliberately short: a word that is sometimes an answer
# to a question the agent asked belongs in the classifier's hands, not here.
# "yes", "no" and "go ahead" are deliberately NOT here: alone they are an answer to
# something the agent asked, and what they set running is whatever it asked about.
# The plan/review gates catch the ones agentY knows it is waiting on; the rest go to
# the classifier, which is shown the message before.
_TRIVIAL = frozenset("""
thanks thank thankyou you thx ty cheers ok okay k kk cool nice perfect great
lovely awesome amazing wow beautiful super sweet fine good done sure hi hello
hey yo morning welcome np yw
""".split())


@dataclass(frozen=True)
class Decision:
    """What triage concluded for one turn."""

    complexity: str      # SIMPLE | COMPLEX
    seat: str            # the 'provider,model' that should take the turn
    why: str             # a few words, for the status line
    confidence: float    # 0.0–1.0 (shortcuts are 1.0)
    source: str          # "shortcut" | "classifier"

    def line(self) -> str:
        why = f" — {self.why}" if self.why else ""
        return f"🧭 Triage: {self.complexity} → {self.seat}{why}"


# The last decision per conversation, so a follow-up can be judged against what
# it follows. Bounded: a long-running host must not accumulate thread ids.
_last: dict[str, Decision] = {}
# And the message that produced it, so "make it brighter" can be judged against
# what it follows. Kept here rather than read back off the orchestrator's history:
# the messages there carry the whole per-turn pin (hooks, gallery, canvas), and the
# first 400 characters of one of those is never the thing the user typed.
_last_text: dict[str, str] = {}
# The newest decision of any conversation, for `/triage` (which has none in hand).
_newest: list[Decision] = []
_MAX_REMEMBERED = 64
# Reasons already said once (a keyless seat, an unreachable classifier). Saying
# them every turn would bury the decision they are a footnote to.
_said: set[str] = set()
_prompt_cache: str = ""


def reset() -> None:
    """Forget every remembered decision (tests, and a fresh host)."""
    _last.clear()
    _last_text.clear()
    _newest.clear()
    _said.clear()
    global _prompt_cache
    _prompt_cache = ""


def last(conversation: str = "") -> Decision | None:
    """The last decision for *conversation*, or the newest one taken anywhere.

    The second form is what ``/triage`` reports: the command has no conversation
    in hand, and "what did it just decide" is the question being asked.
    """
    if conversation:
        return _last.get(conversation)
    return _newest[0] if _newest else None


# ── settings ──────────────────────────────────────────────────────────────────

def _cfg() -> dict:
    try:
        from src.utils.settings import load_settings
        return dict(((load_settings().get("llm") or {}).get("triage") or {}))
    except Exception:  # noqa: BLE001
        return {}


def enabled() -> bool:
    """Whether triage runs at all. ``AGENTY_TRIAGE=0`` turns it off for a session."""
    env = os.environ.get("AGENTY_TRIAGE")
    if env is not None:
        return str(env).strip().lower() in ("1", "true", "yes", "on")
    return bool(_cfg().get("enabled", False))


def announce() -> bool:
    return bool(_cfg().get("announce", True))


def min_confidence() -> float:
    try:
        return max(0.0, min(1.0, float(_cfg().get("min_confidence", 0.6))))
    except (TypeError, ValueError):
        return 0.6


def _timeout() -> float:
    try:
        return max(1.0, float(_cfg().get("timeout", 10.0)))
    except (TypeError, ValueError):
        return 10.0


def configured_seat() -> str:
    """The seat the orchestrator would run on with triage out of the picture."""
    try:
        from src.agent import role_model
        return str(role_model("orchestrator", default="claude,claude-haiku-4-5",
                              env_var="ORCHESTRATOR_LLM") or "")
    except Exception:  # noqa: BLE001
        return ""


def _has_key(spec: str) -> bool:
    """Whether the provider in *spec* is set up on this machine."""
    provider = str(spec or "").partition(",")[0].strip().lower()
    keys = _PROVIDER_KEYS.get(provider)
    if keys is None:      # a provider we do not know — not ours to veto
        return True
    return not keys or any(os.environ.get(k) for k in keys)


def _say(text: str, *, level: str = "info") -> None:
    """Put *text* in the panel and on the console, and never fail for either.

    Both halves can throw: the status bus needs a running host, and ``print`` on a
    Windows console that is not UTF-8 raises UnicodeEncodeError on the first emoji.
    Nothing here is worth a turn.
    """
    try:
        from src.utils import status_bus
        status_bus.notify(text, level=level, echo=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        print(f"[agentY:triage] {text}")
    except Exception:  # noqa: BLE001
        # A console that is not UTF-8 refuses the emoji, not the sentence. Saying it
        # in ASCII beats saying nothing: this is how a swallowed reason gets seen.
        try:
            print(f"[agentY:triage] {text.encode('ascii', 'replace').decode('ascii')}")
        except Exception:  # noqa: BLE001
            pass


def _note_once(key: str, text: str) -> None:
    if key in _said:
        return
    _said.add(key)
    _say(text, level="warning")


def seat_for(complexity: str) -> str:
    """The configured seat for *complexity*.

    A blank seat inherits ``llm.tiers.orchestrator``, which is what keeps the
    composer's model picker meaningful: leave ``complex`` empty and the picker
    still chooses the model that answers the hard turns. A seat whose provider has
    no key falls back the same way — the committed defaults name Anthropic models,
    and an install that only has a DashScope key must not be re-pointed at a model
    it cannot call.
    """
    spec = str(_cfg().get(complexity) or "").strip()
    if not spec or "," not in spec:
        return configured_seat()
    if not _has_key(spec):
        _note_once(f"key:{spec}",
                   f"⚠️ Triage: the {complexity} seat (`{spec}`) has no API key "
                   f"configured — using the orchestrator tier instead.")
        return configured_seat()
    return spec


def classifier_spec() -> str:
    """The model that reads the message, or "" when none can be reached."""
    try:
        from src.agent import role_model
        spec = str(role_model("triage", default="") or "").strip()
    except Exception:  # noqa: BLE001
        return ""
    spec = str(_cfg().get("classifier") or "").strip() or spec
    provider = spec.partition(",")[0].strip().lower()
    if provider not in _SPEAKABLE:
        _note_once(f"classifier:{spec}",
                   f"⚠️ Triage: `{spec or 'nothing'}` cannot be used to classify "
                   "messages (only Anthropic, DashScope and Ollama can) — set "
                   "llm.triage.classifier, or triage will leave the model alone.")
        return ""
    if not _has_key(spec):
        _note_once(f"classifier_key:{spec}",
                   f"⚠️ Triage: the classifier (`{spec}`) has no API key configured "
                   "— leaving the model alone.")
        return ""
    return spec


# ── the decision ──────────────────────────────────────────────────────────────

def shortcut(text: str, *, has_media: bool = False, has_hooks: bool = False,
             dry_run: bool = False, pending_approval: bool = False) -> str | None:
    """A weight that needs no model to see, or None to go and ask one.

    ``pending_approval`` is checked FIRST and deliberately: the shortest messages
    in agentY — "yes", "go on", "continue" — are the ones that set a whole plan or
    a halted hook chain running, and a word list would file them as small talk.
    """
    if pending_approval or has_hooks or has_media or dry_run:
        return COMPLEX
    words = re.findall(r"[a-z0-9']+", str(text or "").lower())
    if not words:
        # An empty message, or one that is nothing but emoji/punctuation.
        return SIMPLE
    if len(words) <= 3 and all(w in _TRIVIAL for w in words):
        return SIMPLE
    return None


def _prompt() -> str:
    global _prompt_cache
    if not _prompt_cache:
        from src.agent import _load_system_prompt
        _prompt_cache = _load_system_prompt("triage")
    return _prompt_cache


def _context(text: str, conversation: str, previous_user: str = "") -> str:
    """What the classifier is shown: this message, and what it follows."""
    parts = []
    before = _last.get(conversation or "")
    if previous_user:
        parts.append(f"The message before this one: {previous_user.strip()[:400]}")
    if before is not None:
        parts.append(f"That turn was judged: {before.complexity}"
                     + (f" ({before.why})" if before.why else ""))
    parts.append(f"THE MESSAGE TO JUDGE:\n{str(text or '').strip()[:2000]}")
    return "\n\n".join(parts)


def _parse(raw: str) -> tuple[str, float, str]:
    """``(complexity, confidence, why)`` from a model's answer, or ("", 0, "")."""
    from src.pipeline import _extract_json  # local: pipeline imports this module
    block = _extract_json(str(raw or ""))
    if not block:
        return "", 0.0, ""
    try:
        data = json.loads(block)
    except Exception:  # noqa: BLE001
        return "", 0.0, ""
    if not isinstance(data, dict):
        return "", 0.0, ""
    complexity = str(data.get("complexity") or "").strip().lower()
    if complexity not in (SIMPLE, COMPLEX):
        return "", 0.0, ""
    try:
        confidence = float(data.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    why = " ".join(str(data.get("why") or "").split())[:80]
    return complexity, max(0.0, min(1.0, confidence)), why


async def _classify(text: str, conversation: str, previous_user: str) -> tuple[str, float, str]:
    spec = classifier_spec()
    if not spec:
        return "", 0.0, ""
    from src.utils.llm_functions import LLMFunctions
    llm = LLMFunctions.for_spec(spec, max_tokens=256)
    messages = [{"role": "system", "content": _prompt()},
                {"role": "user", "content": _context(text, conversation, previous_user)}]
    raw = await asyncio.wait_for(llm.chat(messages, json_format=True), timeout=_timeout())
    return _parse(raw)


def _pick_seat(complexity: str, *, needs_vision: bool) -> tuple[str, str]:
    """``(seat, caveat)`` — the seat for this weight, and why it is not that one."""
    seat = seat_for(complexity)
    if not needs_vision or not seat:
        return seat, ""
    from src.utils.vision_capability import supports_vision
    if supports_vision(seat):
        return seat, ""
    other = seat_for(COMPLEX if complexity == SIMPLE else SIMPLE)
    if other and supports_vision(other):
        return other, "that seat cannot read images"
    base = configured_seat()
    return base, "neither seat can read images"


async def decide(text: str, *, conversation: str = "", previous_user: str = "",
                 has_media: bool = False, has_hooks: bool = False,
                 dry_run: bool = False, pending_approval: bool = False,
                 needs_vision: bool | None = None) -> Decision | None:
    """The seat this message deserves, or None to leave the model alone.

    ``has_media`` says this turn came with images or video, which is work by
    itself; ``needs_vision`` says the images are IN the message as content blocks
    and the seat must therefore be able to read them (defaults to ``has_media``).
    A path mentioned in the text is the first without being the second.

    Never raises: a classifier that fails, times out or answers in prose is the
    same as no opinion, and no opinion means the turn runs on whatever is already
    in the seat.
    """
    if not enabled():
        return None
    simple_seat, complex_seat = seat_for(SIMPLE), seat_for(COMPLEX)
    if not simple_seat or not complex_seat or simple_seat == complex_seat:
        return None  # nothing to choose between — do not pay for a classifier call

    weight = shortcut(text, has_media=has_media, has_hooks=has_hooks,
                      dry_run=dry_run, pending_approval=pending_approval)
    confidence, why, source = 1.0, "", "shortcut"
    if weight is None:
        try:
            weight, confidence, why = await _classify(
                text, conversation, previous_user or _last_text.get(conversation or "", ""))
        except asyncio.TimeoutError:
            _note_once("timeout", "⚠️ Triage: the classifier did not answer in time "
                                  "— leaving the model alone.")
            return None
        except Exception as exc:  # noqa: BLE001 — never cost a turn
            _note_once(f"fail:{type(exc).__name__}",
                       f"⚠️ Triage: could not classify this message ({exc}) "
                       "— leaving the model alone.")
            return None
        if not weight or confidence < min_confidence():
            return None
        source = "classifier"
    elif weight == COMPLEX:
        why = ("attachments" if has_media else
               "canvas hooks" if has_hooks else
               "a dry run" if dry_run else
               "answering the agent's question" if pending_approval else "")
    else:
        why = "nothing to do"

    seat, caveat = _pick_seat(
        weight, needs_vision=has_media if needs_vision is None else needs_vision)
    if not seat:
        return None
    decision = Decision(complexity=weight, seat=seat,
                        why=(f"{why}; {caveat}" if why and caveat else why or caveat),
                        confidence=confidence, source=source)
    if len(_last) > _MAX_REMEMBERED:
        _last.clear()
        _last_text.clear()
    key = conversation or ""
    _last[key] = decision
    _last_text[key] = str(text or "").strip()[:400]
    _newest[:] = [decision]
    return decision


def apply_to(agent, decision: Decision | None, *, role: str = "orchestrator") -> str:
    """Put *decision*'s seat in front of *agent*. Returns what changed, or "".

    Called between turns only — swapping a model inside a running request would
    leave half a tool-call round-trip with one model and half with another.
    """
    if agent is None or decision is None or not decision.seat:
        return ""
    from src.agent import agent_spec, retarget_agent
    running = agent_spec(agent)
    if running and running == decision.seat:
        # Already in the right seat: say it to the console (which model answered is
        # worth knowing while debugging) and nothing to the panel.
        try:
            print(f"[agentY:triage] {decision.complexity} - staying on {running}"
                  + (f" ({decision.why})" if decision.why else ""))
        except Exception:  # noqa: BLE001
            pass
        return ""
    try:
        landed = retarget_agent(agent, decision.seat, role=role)
    except Exception as exc:  # noqa: BLE001 — the old model is still working
        _note_once(f"retarget:{decision.seat}",
                   f"⚠️ Triage: could not switch to `{decision.seat}` ({exc}) "
                   f"— staying on `{running or 'the current model'}`.")
        return ""
    line = decision.line()
    if announce():
        _say(line)
    else:
        try:
            print(f"[agentY:triage] {line}")
        except Exception:  # noqa: BLE001
            pass
    return landed
