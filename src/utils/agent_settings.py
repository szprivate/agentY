"""The settings the agent may change when asked, and nothing else.

"Turn QA off", "stop auto-loading workflows onto my canvas", "let me see the
whole graph" are all one-line requests that used to mean opening the settings
dialog, finding the right group, and knowing what the key is called. There is no
reason the agent cannot do that.

There is every reason for it to be a **short, explicit list**. Settings hold the
address of the ComfyUI server, the directories things are written to, which
environment variable the API key comes from, and which model each role runs on —
and a tool that could write any of those would eventually write one of them
because a sentence was misread. Nothing here is guessed from the shape of a key:
a setting is changeable because it is named below, or it is not changeable.

What earns a place: **behavioural switches with an obvious meaning, cheap to get
wrong, and reversible in one sentence.** What does not, however harmless it looks:

* paths and directories — a wrong one loses work rather than misbehaving;
* URLs and hosts — the same, plus it can point at a machine that is not yours;
* ``api_key_env`` and anything else naming a credential;
* model choices (``llm.tiers.*``, ``llm.pipeline.*``, ``*.model``) — these change
  what every turn costs, and they already have the model picker and
  ``/switch_model``, which show what is available instead of taking a guess;
* embedder settings — changing the model or the dimensions invalidates the vector
  index on disk, which is a migration, not a toggle;
* token limits and context sizes — easy to set to a number that breaks a provider
  in a way that surfaces three turns later as an unrelated error.

Everything here is written to ``settings.local.json`` (never to the committed
defaults) and read back on the next turn: every consumer below calls
``load_settings()`` at the point of use, and writing invalidates the cache. The
one exception is ``auto_update``, which ``run_agent.ps1`` reads at startup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    """One changeable setting: what it means, what it accepts, when it bites."""

    key: str                  # dotted path into the settings tree
    kind: str                 # "bool" | "int"
    what: str                 # one line, in the user's terms
    low: int = 0
    high: int = 0
    effect: str = "on your next message"

    def describe(self, current) -> str:
        rng = f" ({self.low}–{self.high})" if self.kind == "int" else ""
        return f"{self.key} = {current!r}{rng} — {self.what}"


_ALLOWED = (
    Setting("canvas_full_graph", "bool",
            "let the agent see and edit EVERY node on the open workflow, not just "
            "the selected ones. Costs tokens on every canvas turn."),
    Setting("autoload_workflows_into_canvas", "bool",
            "drop each workflow the agent runs onto the ComfyUI canvas automatically"),
    Setting("hook_scoped_graph", "bool",
            "run only the part of the canvas the hooks actually reach, instead of "
            "the whole graph"),
    Setting("hook_tap_tensors", "bool",
            "let the agent look at a mid-graph wire (renders that branch first)"),
    Setting("comfyui_console_lines", "bool",
            "relay ComfyUI's own terminal output into the run stream"),
    Setting("auto_update", "bool",
            "fast-forward the agentY checkouts on start",
            effect="the next time the agent server is started"),
    Setting("memory.enabled", "bool", "long-term memory across conversations"),
    Setting("qa.enabled", "bool",
            "judge finished outputs against a QA briefing (does nothing without one)"),
    Setting("qa.max_retries", "int",
            "how many times a failing output is re-generated; 0 reports the verdict "
            "and stops", low=0, high=5),
    Setting("qa.max_outputs", "int", "how many outputs one QA pass judges",
            low=1, high=20),
    Setting("qa.max_references", "int", "reference images sent with a QA check",
            low=0, high=10),
    Setting("qa.video_frames", "int", "frames sampled from a video for QA",
            low=1, high=10),
    Setting("llm.history_window", "int",
            "how many past messages each turn carries", low=2, high=50),
)

_BY_KEY = {s.key: s for s in _ALLOWED}


def allowed() -> tuple:
    """Every setting the agent may change, in the order they are listed."""
    return _ALLOWED


def get(key: str) -> Setting | None:
    return _BY_KEY.get(str(key or "").strip())


def current(key: str):
    """The value in force right now, or None when the key is not readable."""
    setting = get(key)
    if setting is None:
        return None
    try:
        from src.utils.settings import load_settings
        node = load_settings()
    except Exception:  # noqa: BLE001
        return None
    for part in setting.key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def coerce(setting: Setting, value):
    """*value* as the setting's own type, or ``(None, why)`` when it cannot be.

    Deliberately tolerant about how a bool arrives — a model will send ``"true"``,
    ``"on"``, ``1`` and ``True`` for the same intent — and deliberately strict
    about what a bool MEANS: anything that is not recognisably yes or no is
    refused rather than guessed, because guessing here silently sets the opposite.
    """
    if setting.kind == "bool":
        if isinstance(value, bool):
            return value, ""
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on", "enable", "enabled"):
            return True, ""
        if text in ("false", "0", "no", "off", "disable", "disabled"):
            return False, ""
        return None, f"{value!r} is not a yes or a no"
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None, f"{value!r} is not a whole number"
    if not (setting.low <= number <= setting.high):
        return None, (f"{number} is outside {setting.low}–{setting.high}, which is "
                      f"what this setting accepts")
    return number, ""


def nest(key: str, value) -> dict:
    """``{"qa": {"max_retries": 2}}`` for a dotted key — the shape set_local takes."""
    parts = str(key).split(".")
    out: dict = {}
    node = out
    for part in parts[:-1]:
        node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return out


def apply(key: str, value) -> dict:
    """Change one setting. Returns what happened, in the shape a tool reports.

    Writes to ``settings.local.json`` only — the committed defaults are never
    touched, so a bad change is undone by removing one line from that file, and
    reinstalling never fights with it.
    """
    setting = get(key)
    if setting is None:
        return {
            "error": f"'{key}' is not a setting the agent can change.",
            "changeable": [s.key for s in _ALLOWED],
            "what_to_do": ("Everything else — model choices, paths, URLs, API key "
                           "variables — is changed by the user in Settings, on "
                           "purpose. Say which setting you mean and let them do it."),
        }
    coerced, why = coerce(setting, value)
    if coerced is None:
        return {"error": f"cannot set {setting.key}: {why}.",
                "expects": ("true or false" if setting.kind == "bool"
                            else f"a whole number {setting.low}–{setting.high}")}
    was = current(setting.key)
    if was == coerced:
        return {"status": "unchanged", "key": setting.key, "value": coerced,
                "message": f"{setting.key} was already {coerced!r} — nothing written."}
    try:
        from src.utils.settings import set_local
        set_local(nest(setting.key, coerced))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not write the setting: {exc}"}
    return {
        "status": "changed",
        "key": setting.key,
        "from": was,
        "to": coerced,
        "takes_effect": setting.effect,
        "message": (f"{setting.key}: {was!r} → {coerced!r}. Takes effect "
                    f"{setting.effect}. Saved in config/settings.local.json, so it "
                    f"survives restarts and updates."),
    }
