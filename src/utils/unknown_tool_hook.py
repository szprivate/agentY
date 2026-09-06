"""Answer a call for a tool that isn't there with the one that is.

Strands replies to a call for an unregistered tool with four words —
``Unknown tool: get_workflow_template`` — and nothing else. That is true and
useless: it does not say what the agent *does* have, so the agent improvises,
and the improvisation is expensive. Measured on one run, an orchestrator that
reached for ``get_workflow_template`` (a real tool, just not in its own set)
answered the bounce by trying ``run_script``, waiting out the full 120-second
permission timeout, listing 42 files, and re-reading two workflow JSONs: six
calls and ~297K input tokens to arrive where the first bounce could have sent
it.

So the bounce names the nearest tools the agent actually holds. Matching is
deliberately dumb — shared name tokens, then substrings — because the failure
mode it fixes is an agent reaching for a *plausible* name, and plausible names
share tokens with the real ones. When nothing scores, it lists a few names
anyway: a wrong suggestion costs one call, and silence costs six.

Registered on every agent (:func:`src.agent._make_agent`), because any of them
can hallucinate a name and none of them benefit from the bare message.
"""

from __future__ import annotations

import re

from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

# How many alternatives to name. Enough to cover a near-miss, few enough that the
# agent picks rather than browses.
_MAX_SUGGESTIONS = 5

# When nothing is close, the whole toolset is listed instead. Bounded so a very
# large registry cannot turn one bounce into a wall of text.
_MAX_LISTED = 40

# Tokens nearly every tool name carries, so sharing one says almost nothing.
# Without this, `get_node_schema` ranks `get_comfyui_dirs` over anything that
# actually inspects a graph, purely on the word "get".
_WEAK_TOKENS = frozenset({"get", "set", "list", "run", "read", "write", "new",
                          "all", "the", "a", "to", "for", "of", "and", "by"})


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def rank_alternatives(wanted: str, available) -> list[str]:
    """Available tool names most like *wanted*, best first.

    Scored on shared name tokens, with the near-universal ones (*get*, *list*,
    *run*) worth a fraction of a distinctive one — so ``get_node_schema`` is
    drawn to whatever else mentions nodes or schemas rather than to every tool
    whose name starts with "get". A pure substring match counts as a weak
    signal, which is what lets a bare ``schema`` find ``get_node_schema``.
    """
    want = _tokens(wanted)
    if not want:
        return []
    scored: list[tuple[int, int, str]] = []
    for name in available:
        have = _tokens(name)
        shared = want & have
        # Distinctive tokens are worth 3, generic ones 1 — so one real word beats
        # any number of shared "get"s.
        score = sum(1 if t in _WEAK_TOKENS else 3 for t in shared)
        if score == 0:
            low_w, low_n = str(wanted).lower(), str(name).lower()
            if not (low_w in low_n or low_n in low_w):
                continue
            score = 1
        # Ties go to the closer-sized name: `get_workflow_catalog` should beat
        # `get_workflow_recipe_for_everything` rather than lose for being short.
        scored.append((-score, abs(len(have) - len(want)), name))
    scored.sort()
    # Drop anything that only matched on a generic word while something matched
    # on a real one — five weak suggestions read as noise, and noise is what
    # sends the agent improvising again.
    if scored and -scored[0][0] >= 3:
        scored = [s for s in scored if -s[0] >= 3]
    return [name for _s, _d, name in scored[:_MAX_SUGGESTIONS]]


class UnknownToolHookProvider(HookProvider):
    """Rewrite the ``Unknown tool: X`` result into one that says what to call."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_after(self, event: AfterToolCallEvent, **kwargs) -> None:  # noqa: ARG002
        try:
            result = event.result
            if not isinstance(result, dict) or result.get("status") != "error":
                return
            content = result.get("content") or []
            if not content or not isinstance(content[0], dict):
                return
            text = str(content[0].get("text", ""))
            if not text.startswith("Unknown tool:"):
                return

            wanted = text.split(":", 1)[1].strip()
            try:
                available = list(event.agent.tool_registry.registry.keys())
            except Exception:  # noqa: BLE001
                return

            near = rank_alternatives(wanted, available)
            if near:
                advice = ("Closest tools you DO have: "
                          + ", ".join(f"`{n}`" for n in near) + ".")
            else:
                # Nothing scored, so the name was invented rather than misremembered.
                # An arbitrary handful would be worse than nothing — it reads as a
                # recommendation. Give the actual list instead; it is a few hundred
                # tokens, it happens rarely, and it ends the guessing.
                listed = sorted(available)[:_MAX_LISTED]
                more = "" if len(available) <= _MAX_LISTED else \
                    f" (+{len(available) - _MAX_LISTED} more)"
                advice = ("You have no tool by a similar name. Your tools are: "
                          + ", ".join(f"`{n}`" for n in listed) + more + ".")

            content[0]["text"] = (
                f"{text}. {advice} Use one of those, or do the job with a "
                f"specialist you can delegate to. Do NOT try to reach the missing "
                f"capability another way — not by shelling out with `run_script`, "
                f"not by reading files off disk to reconstruct it. If none of your "
                f"tools can do it, say so and stop."
            )
        except Exception:  # noqa: BLE001 — never break a turn over a nicer message
            return
