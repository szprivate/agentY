"""agentY orchestrator meta-tools — live skill authoring + ad-hoc subagents.

These tools are what make the **Orchestrator** agent "free": instead of being
boxed into a fixed triage → route decision tree, the orchestrator can extend
*itself* at runtime.

- ``create_skill`` / ``list_skills`` / ``remove_skill`` let it capture a reusable
  multi-step procedure as a real Agent Skill (SKILL.md) and register it live, so
  the skill is available on the next turn (its instructions are also returned
  inline so it is usable immediately in-session). Authored skills live under
  ``skills/_scratch/`` so they are easy to identify and clean up, kept separate
  from the curated project skills in ``skills/``.
- ``spawn_subagent`` builds a fresh Strands agent with a curated toolset and runs
  it to completion on a focused sub-task, returning its text. Subagents are
  depth-1 (they do not themselves get ``spawn_subagent``) so cost stays bounded.

The tools are bound to the live orchestrator via :func:`set_orchestrator_context`
(mirrors ``image_handling.set_vision_agent``): the orchestrator's ``AgentSkills``
plugin instance is stashed module-side so ``create_skill`` can re-scan it.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from strands import tool


# ---------------------------------------------------------------------------
# Skill directories
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    # src/tools/orchestration.py → parent(tools) → parent(src) → repo root
    return Path(__file__).resolve().parent.parent.parent


def _skills_dir() -> Path:
    return _project_root() / "skills"


def _scratch_dir() -> Path:
    return _skills_dir() / "_scratch"


# ---------------------------------------------------------------------------
# Live-orchestrator context (set by the pipeline at wiring time)
# ---------------------------------------------------------------------------

_ORCH_SKILLS_PLUGIN: Any = None   # the orchestrator's AgentSkills plugin instance
_ORCH_AGENT: Any = None           # the live orchestrator Agent (unused for now)


def set_orchestrator_context(agent: Any = None, skills_plugin: Any = None) -> None:
    """Wire the live orchestrator agent + its AgentSkills plugin into this module.

    Called by the pipeline after (re)building the orchestrator so ``create_skill``
    can re-scan the skills directories on the *current* plugin instance.
    """
    global _ORCH_AGENT, _ORCH_SKILLS_PLUGIN
    if agent is not None:
        _ORCH_AGENT = agent
    if skills_plugin is not None:
        _ORCH_SKILLS_PLUGIN = skills_plugin


def _rescan_skills() -> int:
    """Re-scan both skill source dirs on the orchestrator's plugin.

    Returns the number of skills now registered, or -1 when no plugin is wired.
    """
    plugin = _ORCH_SKILLS_PLUGIN
    if plugin is None:
        return -1
    try:
        plugin.set_available_skills([str(_skills_dir()), str(_scratch_dir())])
        return len(plugin.get_available_skills())
    except Exception:  # noqa: BLE001
        return -1


# ---------------------------------------------------------------------------
# Skill authoring
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Coerce *name* into a valid skill slug (lowercase alnum + single hyphens)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:64] or "scratch-skill"


@tool
def create_skill(
    name: str,
    description: str,
    instructions: str,
    allowed_tools: Optional[str] = None,
) -> str:
    """Author a reusable skill and register it live in the agent system.

    Use this to **capture a repeatable multi-step procedure** you just worked out
    (or one you know you'll reuse) as a persistent Agent Skill. Once created, the
    skill's name + description appear in your ``<available_skills>`` list from the
    next turn on, and you can load its full instructions on demand via the
    ``skills`` tool — so you never have to re-derive the procedure. The full
    instructions are also returned here so you can follow them immediately.

    Write ``instructions`` as a clear, self-contained markdown playbook (numbered
    steps, which tools to call, what inputs/outputs to expect). Authored skills are
    stored under ``skills/_scratch/`` and can be removed with ``remove_skill``.

    Args:
        name: Short skill name; will be slugified to lowercase-with-hyphens.
        description: One line describing when to use the skill (shown in the skill list).
        instructions: The full markdown body / playbook for the skill.
        allowed_tools: Optional space-separated list of tool names the skill uses.

    Returns:
        A JSON string with the created skill's slug, path, and total skill count.
    """
    slug = _slugify(name)
    if not description or not description.strip():
        return json.dumps({"error": "description is required"})
    if not instructions or not instructions.strip():
        return json.dumps({"error": "instructions are required"})

    skill_dir = _scratch_dir() / slug
    skill_dir.mkdir(parents=True, exist_ok=True)

    # JSON-encode scalar values → always-valid YAML (handles colons/quotes safely).
    fm_lines = [
        "---",
        f"name: {slug}",
        f"description: {json.dumps(description.strip())}",
    ]
    if allowed_tools and allowed_tools.strip():
        fm_lines.append(f"allowed-tools: {allowed_tools.strip()}")
    fm_lines += [
        "metadata:",
        "  type: scratch",
        f"  created: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "---",
        "",
        instructions.strip(),
        "",
    ]
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("\n".join(fm_lines), encoding="utf-8")

    count = _rescan_skills()
    return json.dumps({
        "status": "created",
        "name": slug,
        "path": str(skill_md),
        "registered": count >= 0,
        "total_skills": count if count >= 0 else None,
        "message": (
            f"Skill '{slug}' saved and registered. It will appear in your "
            "available_skills list; activate it any time with the 'skills' tool. "
            "Its full instructions follow so you can use them now:\n\n"
            + instructions.strip()
        ),
    })


@tool
def list_skills() -> str:
    """List the skills you have authored at runtime (under skills/_scratch/).

    Returns:
        A JSON string with the authored skill names + descriptions.
    """
    scratch = _scratch_dir()
    out: list[dict[str, str]] = []
    if scratch.is_dir():
        for child in sorted(scratch.iterdir()):
            md = child / "SKILL.md"
            if not md.is_file():
                continue
            desc = ""
            try:
                for ln in md.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("description:"):
                        desc = ln.split(":", 1)[1].strip().strip('"')
                        break
            except Exception:  # noqa: BLE001
                pass
            out.append({"name": child.name, "description": desc})
    return json.dumps({"authored_skills": out, "count": len(out)})


@tool
def remove_skill(name: str) -> str:
    """Delete a skill you authored at runtime and de-register it.

    Args:
        name: The skill slug to remove (as shown by ``list_skills``).

    Returns:
        A JSON status string.
    """
    slug = _slugify(name)
    skill_dir = _scratch_dir() / slug
    if not skill_dir.exists():
        return json.dumps({"error": f"No authored skill named '{slug}'."})
    try:
        shutil.rmtree(skill_dir)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to remove '{slug}': {exc}"})
    count = _rescan_skills()
    return json.dumps({
        "status": "removed",
        "name": slug,
        "total_skills": count if count >= 0 else None,
    })


# ---------------------------------------------------------------------------
# Ad-hoc subagents
# ---------------------------------------------------------------------------

_SUBAGENT_TOOLSETS = ("research", "assembly", "info", "story", "web", "vision", "full")


@tool
async def spawn_subagent(task: str, toolset: str = "full", model: Optional[str] = None) -> str:
    """Spin up a fresh subagent with a focused toolset to handle a sub-task.

    Use this to **isolate a heavy or self-contained sub-task** in its own clean
    context (its own conversation history + a curated toolset), then fold the
    result back into your plan. The subagent runs to completion and returns its
    final text. Subagents cannot themselves spawn further subagents (depth-1),
    so keep the task well-scoped.

    Choose ``toolset`` by the job:
      - ``research``  — resolve a ComfyUI template/models/prompt into a brainbriefing.
      - ``assembly``  — assemble + validate a workflow (then signal_workflow_ready).
      - ``info``      — answer questions about installed models/workflows/capabilities.
      - ``story``     — write a synopsis / scene descriptions.
      - ``web``       — search the web + stage reference images.
      - ``vision``    — describe/analyse an image.
      - ``full``      — a general agent with the full non-meta toolset.

    Args:
        task: The complete instruction for the subagent (self-contained).
        toolset: One of research|assembly|info|story|web|vision|full.
        model: Optional 'provider,model' override (e.g. 'claude,claude-sonnet-4-5').

    Returns:
        The subagent's final text output (or a JSON error string).
    """
    ts = (toolset or "full").strip().lower()
    if ts not in _SUBAGENT_TOOLSETS:
        return json.dumps({
            "error": f"Unknown toolset '{toolset}'. Choose one of: {', '.join(_SUBAGENT_TOOLSETS)}."
        })
    try:
        from src.agent import build_subagent  # lazy import — avoids circular import
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Could not load subagent builder: {exc}"})
    try:
        sub = build_subagent(toolset=ts, model=model)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to build subagent ({ts}): {exc}"})
    try:
        result = await sub.invoke_async(task)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Subagent ({ts}) failed: {exc}"})
    finally:
        try:
            sub.messages.clear()
        except Exception:  # noqa: BLE001
            pass
