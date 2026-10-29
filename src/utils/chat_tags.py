"""``#tag`` in a chat message, resolved the way a hook directive's is.

The sidebar composer offers the same ``#`` menu as the hook prompts, so a user can
write "relight #hero_face" in chat. Left alone, the agent would see only the
literal text. This turns each name that actually resolves (canvas first, then the
project's memory - :func:`src.utils.tag_memory.resolve`) into one line of the turn's
pinned context. Names that resolve to nothing are left out: "#fyp" or "#3" in a
message is not a reference, and a note saying so would only be noise.
"""

from __future__ import annotations


def describe_chat_tags(text, canvas: dict | None) -> str:
    """A ``[TAGS]`` block for the references *text* names, or ``""``."""
    try:
        from src.utils.canvas_hooks import mentioned_tags
        from src.utils.tag_memory import resolve
        names = mentioned_tags(text)
    except Exception:  # noqa: BLE001 - never cost the user a turn
        return ""
    lines: list[str] = []
    for name in names:
        try:
            hit = resolve(name, canvas)
        except Exception:  # noqa: BLE001
            hit = None
        if not hit:
            continue
        role = str(hit.get("role") or "").strip()
        if hit.get("source") == "canvas":
            where = f"node {hit.get('node_id')} on the open canvas"
        else:
            where = "file " + (str(hit.get("path") or "").strip() or "(no path)") + " (project memory)"
        lines.append(f"- #{name} -> {where}" + (f" - {role}" if role else ""))
    if not lines:
        return ""
    return ("[TAGS] The message names these references with #tag; each points at exactly "
            "this input or file:\n" + "\n".join(lines))
