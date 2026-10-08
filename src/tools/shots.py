"""The lead conversation's tools: start shot conversations, steer them, hear back.

See src/utils/shots.py. Each tool acts for the conversation whose turn is calling
it (agenty_core.utils.turn_scope), so two leads running at once each see only
their own shots.
"""

from __future__ import annotations

import json

from strands import tool


def _lead() -> str:
    try:
        from agenty_core.utils import turn_scope
        return str(turn_scope.current().thread_id or "")
    except Exception:  # noqa: BLE001
        return ""


def _out(result) -> str:
    return json.dumps(result, ensure_ascii=False)


@tool
def start_shot(name: str, briefing: str, hook_ids: list | None = None) -> str:
    """Start a new conversation that works on one shot of a sequence, in parallel.

    Use this when the user asks for a sequence (several shots) to be worked on by
    several agents, or explicitly asks to hand shots to their own conversations.
    The shot gets its own agent and history and appears in the user's conversation
    list; it is told it is a shot of yours and gets your sequence notes
    (set_sequence_notes) with its briefing. When it finishes, its report comes back
    to you as a message and you get a turn to review it. Shots run at the same time
    up to the user's limit; the rest wait for a free agent.

    Write the plan and set_sequence_notes FIRST (characters, look, models,
    resolution, naming), then start the shots.

    A BRANCH of a hook pipeline is started the same way, with `hook_ids`: the
    conversation is handed those stages of the user's canvas to run (the ids are
    listed per branch in the PARALLEL BRANCHES block). It reports back when it
    finishes or reaches a review; you then answer it with message_shot, into the
    same conversation.

    Args:
        name: Short shot or branch name, unique in this sequence (e.g. "sh010",
            "characters").
        briefing: Everything it needs and nothing it doesn't: what to make, from
            which inputs (absolute paths), which model/workflow, duration and
            resolution, and what to report back. For a branch: in full, every
            value from earlier stages that its stages read.
        hook_ids: For a branch only - the hook ids of its stages, exactly as the
            PARALLEL BRANCHES block lists them.
    """
    from src.utils import shots
    return _out(shots.start_shot(_lead(), name, briefing, hook_ids=hook_ids))


@tool
def message_shot(name: str, text: str) -> str:
    """Send one of your shots a correction or more work.

    Reaches its running turn if it is working, or starts a new turn of it if it is
    idle. Its next report comes back to you as usual.

    Args:
        name: The shot's name (as given to start_shot).
        text: What it should do.
    """
    from src.utils import shots
    return _out(shots.message_shot(_lead(), name, text))


@tool
def shot_status(name: str = "") -> str:
    """Where your shots stand. Without a name: every shot with its state (queued,
    running, done, failed, stopped) and the start of its last report. With a name:
    that shot's whole last report and the files it made.

    Args:
        name: Optional shot name for the full report of one shot.
    """
    from src.utils import shots
    lead = _lead()
    if name:
        return _out(shots.read_shot(lead, name))
    return _out({"ok": True, "shots": shots.status(lead)})


@tool
def stop_shot(name: str) -> str:
    """Stop a shot's running turn (its conversation stays; message_shot resumes it).

    Args:
        name: The shot's name.
    """
    from src.utils import shots
    return _out(shots.stop_shot(_lead(), name))


@tool
def set_sequence_notes(notes: str) -> str:
    """Write the notes every shot of this sequence is given with its briefing:
    characters and their reference images, the look, models and settings,
    resolution and frame rate, output naming. Replaces the previous notes; shots
    already started keep what they were given (message_shot to update them).

    Args:
        notes: The sequence notes, as plain text or markdown.
    """
    from src.utils import shots
    return _out(shots.set_notes(_lead(), notes))
