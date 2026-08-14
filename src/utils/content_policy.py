"""A generation the provider refused on content grounds — and what to do about it.

This is the failure the repair specialist cannot touch. Nothing is wrong with the
graph: the request reached the model and the provider's own filter said no. Sent
to the fixer it burns a repair budget rewriting a workflow that was already
correct, three times, and then reports a defect that does not exist.

What usually *does* work is running it again. Every one of these filters is
probabilistic — it scores a prompt or scores the pixels that came out — and none
of these APIs is deterministic, so the second attempt is a genuinely different
generation. Where the node has a real seed (Seedream, Gemini) rerolling it makes
that explicit; where it doesn't (OpenAI's seed says "not implemented yet in
backend"; Kling Omni discards it outright) the retry is still a fresh roll,
because the provider never promised the same answer twice.

The two stages are not the same bet, though:

* **output** — it generated something and the filter rejected the result
  (``OutputImageSensitiveContentDetected``, BFL's ``Content Moderated``,
  Ideogram's safety filter, Google's RAI). A different roll frequently passes.
* **input** — the prompt or a reference image was refused before anything was
  generated (``InputTextSensitiveContentDetected``, OpenAI's
  ``content_policy_violation``, Gemini's ``blockReason``). Rolling again mostly
  buys the same refusal; the wording is what has to change.

So both retry, output more times than input, and when the retries are spent the
agent is told plainly that this needs rewording or the user's decision — never
that the workflow is broken.

Every signature below was read out of ComfyUI's own API-node source rather than
guessed, and each is matched against the error text the executor already records.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# (pattern, provider, stage). Ordered: the specific provider forms first, the
# generic phrasings last, so a match names who refused wherever possible.
_SIGNATURES: tuple = (
    # ByteDance / Seedream / Seedance. Two shapes: their own error codes, and the
    # prose their gateway returns as a plain 400, which arrives through the shared
    # client as "API Error: <message> (Type: BadRequest)". The prose names the
    # stage itself ("the OUTPUT image may be related to…"), which is worth reading:
    # it is the difference between rewording the prompt and rolling again.
    (re.compile(r"request failed because the output \w+ may be related to copyright",
                re.I), "ByteDance", "output"),
    (re.compile(r"request failed because the input \w+ may (?:be related to copyright|"
                r"contain sensitive)", re.I), "ByteDance", "input"),
    (re.compile(r"\bInput\w*SensitiveContentDetected\b", re.I), "ByteDance", "input"),
    (re.compile(r"\bOutput\w*SensitiveContentDetected\b", re.I), "ByteDance", "output"),
    (re.compile(r"\bSensitiveContentDetected\b", re.I), "ByteDance", "unknown"),
    (re.compile(r"\bPolicyViolation\b", re.I), "ByteDance", "output"),
    # Google Gemini (Nano Banana) and Veo
    (re.compile(r"Gemini API blocked the request", re.I), "Gemini", "input"),
    (re.compile(r"\bIMAGE_PROHIBITED_CONTENT\b", re.I), "Gemini", "output"),
    (re.compile(r"\bPROHIBITED_CONTENT\b|\bIMAGE_SAFETY\b|\bBLOCKLIST\b", re.I),
     "Gemini", "unknown"),
    (re.compile(r"Responsible AI practices", re.I), "Google Veo", "output"),
    # OpenAI (GPT Image / DALL·E) — surfaced as "API Error: … (Type: …)"
    (re.compile(r"\bcontent_policy_violation\b|\bmoderation_blocked\b|"
                r"\bimage_content_policy_violation\b", re.I), "OpenAI", "input"),
    (re.compile(r"rejected as a result of our safety system", re.I), "OpenAI", "input"),
    # Black Forest Labs / Flux — polling statuses
    (re.compile(r"\bRequest Moderated\b", re.I), "Black Forest Labs", "input"),
    (re.compile(r"\bContent Moderated\b", re.I), "Black Forest Labs", "output"),
    # The rest of the API-node roster
    (re.compile(r"blocked by Ideogram's content safety filter", re.I), "Ideogram", "output"),
    (re.compile(r"flagged for content policy violation", re.I), "Reve", "output"),
    (re.compile(r"\bCONTENT_FILTERED\b"), "Stability", "output"),
    (re.compile(r"\bcontents?_moderation\b", re.I), "PixVerse", "unknown"),
    (re.compile(r"\b1301\b.{0,40}content security policy|content security policy",
                re.I), "Kling", "unknown"),
    # Generic phrasings, for a provider that words it its own way. Unnamed, but
    # still worth reading for the STAGE — "the output …" is a different bet from
    # "your prompt …" whoever said it.
    (re.compile(r"\boutput (?:image|video|content)\b[^.]{0,60}\b(?:copyright|sensitive|"
                r"blocked|filtered|rejected)\b", re.I), "", "output"),
    (re.compile(r"content (?:policy|filter|moderation|safety)", re.I), "", "unknown"),
    (re.compile(r"\bsafety (?:system|filter|policy)\b", re.I), "", "unknown"),
    (re.compile(r"\bprohibited content\b|\bmoderation\b|\bNSFW\b", re.I), "", "unknown"),
    (re.compile(r"\bcopyright(?:ed)?\b", re.I), "", "unknown"),
)

# How many times to run it again before giving up, per stage. Output-side is the
# bet worth taking twice; an input the provider read and refused rarely reads
# differently the second time, so it gets one cheap attempt and then the truth.
_DEFAULT_RETRIES = {"output": 2, "input": 1, "unknown": 2}


@dataclass(frozen=True)
class Rejection:
    """A refusal on content grounds: who refused, at which stage, in what words."""

    provider: str
    stage: str          # "input" | "output" | "unknown"
    quote: str

    def retries(self) -> int:
        """How many re-runs this is worth (``AGENTY_POLICY_RETRIES`` overrides)."""
        override = os.environ.get("AGENTY_POLICY_RETRIES", "").strip()
        if override:
            try:
                return max(0, int(override))
            except ValueError:
                pass
        return _DEFAULT_RETRIES.get(self.stage, 2)

    def who(self) -> str:
        return self.provider or "the provider"

    def describe(self) -> str:
        where = {"input": "refused the prompt or a reference image before generating",
                 "output": "generated it and then rejected the result",
                 "unknown": "refused this generation"}[self.stage]
        return f"{self.who()} {where}: {self.quote}"


def _quote(text: str, limit: int = 240) -> str:
    return " ".join(str(text or "").split())[:limit]


def classify(error_text) -> Rejection | None:
    """Read an execution failure as a content refusal, or None if it isn't one.

    Accepts the executor's error dict or a plain string.
    """
    if isinstance(error_text, dict):
        parts = [str(error_text.get("error") or "")]
        details = error_text.get("details")
        if isinstance(details, dict):
            parts.extend(str(v) for v in details.values() if isinstance(v, (str, int)))
        elif details:
            parts.append(str(details))
        body = "\n".join(p for p in parts if p)
    else:
        body = str(error_text or "")
    if not body.strip():
        return None
    for pattern, provider, stage in _SIGNATURES:
        if pattern.search(body):
            return Rejection(provider=provider, stage=stage, quote=_quote(body))
    return None


def retry_note(rej: Rejection, attempt: int, total: int, rerolled: int) -> str:
    """The progress line for one automatic re-run."""
    how = (f"new seed on {rerolled} node(s)" if rerolled
           else "same graph — these APIs don't repeat themselves anyway")
    return (f"🎲 {rej.who()} refused that generation on content grounds — "
            f"running it again ({attempt}/{total}, {how}).")


def exhausted(rej: Rejection, attempts: int) -> dict:
    """What to report once re-running has stopped being worth it.

    Deliberately not shaped like a repair failure. The graph is fine, and an agent
    told "could not be healed" will keep trying to heal it.
    """
    if rej.stage == "input":
        fix = ("Reword the prompt — name the subject and the action plainly and drop "
               "whatever reads as a real person, a brand, a franchise or a lifted "
               "style. If a reference image is what was refused, use a different one.")
    elif rej.stage == "output":
        fix = ("The prompt was accepted; what came out was not. Steer the result away "
               "from whatever the filter caught (framing, wardrobe, likeness, logo) "
               "rather than rewriting the whole prompt.")
    else:
        fix = ("Reword the prompt away from anything that reads as a real person, a "
               "brand or a copyrighted work, or try a different model.")
    return {
        "status": "rejected",
        "kind": "content_policy",
        "provider": rej.who(),
        "stage": rej.stage,
        "error": (f"{rej.who()} refused this generation on content grounds and it did "
                  f"not pass on {attempts} further attempt(s). This is the provider's "
                  f"content filter, NOT a workflow defect — there is nothing in the "
                  f"graph to repair."),
        "what_it_said": rej.quote,
        "what_to_do": fix,
        "do_not": ("Do not send this to the repair specialist and do not rebuild the "
                   "workflow — the graph is correct. Say plainly what was refused, "
                   "what you changed if you try again, and let the user decide."),
    }
