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
- ``create_custom_node`` / ``list_generated_nodes`` run the **coder** agent with the
  ``custom-node-from-github`` skill: given a model's GitHub repo, the tool clones the
  repo, then the agent reads the docs + inference code and authors a self-contained
  ComfyUI custom-node pack under ``output/custom_nodes/`` that the user can publish as
  its own repo.

The tools are bound to the live orchestrator via :func:`set_orchestrator_context`
(mirrors ``image_handling.set_vision_agent``): the orchestrator's ``AgentSkills``
plugin instance is stashed module-side so ``create_skill`` can re-scan it.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shutil
import tempfile
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


def _orch_skills_dir() -> Path:
    return _skills_dir() / "orchestrator-skills"


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
    """Re-scan all skill source dirs on the orchestrator's plugin.

    Returns the number of skills now registered, or -1 when no plugin is wired.
    """
    plugin = _ORCH_SKILLS_PLUGIN
    if plugin is None:
        return -1
    try:
        plugin.set_available_skills([str(_skills_dir()), str(_orch_skills_dir()), str(_scratch_dir())])
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

_SUBAGENT_TOOLSETS = ("research", "assembly", "info", "web", "vision", "full")

# spawn_subagent is gated: it may run ONLY when the user's current message
# explicitly asked for a subagent. The pipeline sets this per turn via
# set_subagent_allowed() from a scan of the user's message; left False, the tool
# refuses. This stops the orchestrator (a small model) from spinning up
# subagents for routine work the direct path already handles.
_SUBAGENT_ALLOWED: bool = False


def set_subagent_allowed(allowed: bool) -> None:
    """Arm/disarm ``spawn_subagent`` for the current turn (pipeline-controlled)."""
    global _SUBAGENT_ALLOWED
    _SUBAGENT_ALLOWED = bool(allowed)


@tool
async def spawn_subagent(task: str, toolset: str = "full", model: Optional[str] = None,
                         tools: Optional[list] = None, skill: Optional[str] = None) -> str:
    """Spin up a fresh subagent with a focused toolset to handle a sub-task.

    Use this to **isolate a heavy or self-contained sub-task** in its own clean
    context (its own conversation history + a curated toolset), then fold the
    result back into your plan. The subagent runs to completion and returns its
    final text. Subagents cannot themselves spawn further subagents (depth-1),
    so keep the task well-scoped.

    **Prefer a MINIMAL explicit ``tools`` list over a preset** — a subagent with
    only the ~6 tools its task needs carries far fewer tool definitions per call
    (less context, faster, and small models pick the right tool far more reliably
    from 6 than from 60). Pair it with a ``skill`` to give the subagent a fixed
    procedure. Fall back to a preset ``toolset`` only when you want a broad agent.

    Choose ``toolset`` by the job:
      - ``research``  — resolve a ComfyUI template/models/prompt into a brainbriefing.
      - ``assembly``  — assemble + validate a workflow (then signal_workflow_ready).
      - ``info``      — answer questions about installed models/workflows/capabilities.
      - ``web``       — search the web + stage reference images.
      - ``vision``    — describe/analyse an image.
      - ``full``      — a general agent with the full non-meta toolset.

    Args:
        task: The complete instruction for the subagent (self-contained).
        toolset: One of research|assembly|info|story|web|vision|full. Ignored when
            ``tools`` is given.
        model: Optional 'provider,model' override (e.g. 'claude,claude-sonnet-4-5').
        tools: Optional explicit list of tool NAMES for a lean, single-purpose
            agent (e.g. ["upload_image","get_workflow_template","apply_brainbriefing",
            "validate_workflow","duplicate_workflow","signal_workflow_ready"]).
            Takes priority over ``toolset``.
        skill: Optional skill name whose steps are baked into the subagent's prompt
            as its procedure (e.g. "batch-handoff").

    Returns:
        The subagent's final text output (or a JSON error string).
    """
    if not _SUBAGENT_ALLOWED:
        return json.dumps({
            "error": "spawn_subagent is disabled for this turn. It runs ONLY when the "
                     "user explicitly asks to use or spawn a subagent. Do this yourself "
                     "with your own tools instead (e.g. prepare_workflow, "
                     "duplicate_workflow / update_workflow, the batch-handoff skill)."
        })
    ts = (toolset or "full").strip().lower()
    if not tools and ts not in _SUBAGENT_TOOLSETS:
        return json.dumps({
            "error": f"Unknown toolset '{toolset}'. Choose one of: {', '.join(_SUBAGENT_TOOLSETS)}, "
                     f"or pass an explicit `tools` list of tool names."
        })
    try:
        from src.agent import build_subagent  # lazy import — avoids circular import
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Could not load subagent builder: {exc}"})
    try:
        sub = build_subagent(toolset=ts, model=model, tools=tools, skill=skill)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Failed to build subagent: {exc}"})
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


# ---------------------------------------------------------------------------
# create_custom_node — build a ComfyUI custom node from a model's GitHub repo
# (coder agent + custom-node-from-github skill)
# ---------------------------------------------------------------------------

def _generated_nodes_dir() -> Path:
    """Where authored node packs land: ``<repo>/output/custom_nodes/`` (git-ignored).

    Each pack is a self-contained folder the user can turn into its own GitHub repo.
    """
    return _project_root() / "output" / "custom_nodes"


_GITHUB_SHORTHAND = re.compile(r"^[\w.-]+/[\w.-]+$")


def _normalize_github_url(url: str) -> tuple[str, Optional[str]]:
    """Return ``(clone_url, branch|None)`` from a user-supplied GitHub reference.

    Accepts a full ``https://github.com/owner/repo`` URL, ``owner/repo`` shorthand,
    and ``.../tree/<branch>`` or ``.../blob/<branch>/...`` deep links (branch
    extracted). Unrecognised strings are returned unchanged for git to validate.
    """
    u = (url or "").strip()
    if _GITHUB_SHORTHAND.match(u):
        return f"https://github.com/{u}.git", None
    u = u.split("#", 1)[0].split("?", 1)[0].strip()
    m = re.match(
        r"^(https?://github\.com/[\w.-]+/[\w.-]+?)(?:\.git)?(?:/(?:tree|blob)/([^/]+).*)?/?$",
        u,
    )
    if m:
        return m.group(1) + ".git", m.group(2)
    return u, None


async def _git_clone_shallow(
    clone_url: str, branch: Optional[str], dest: Path, timeout: int = 240
) -> tuple[bool, str]:
    """Shallow-clone *clone_url* into *dest*, skipping LFS blobs (weights).

    We want the repo's docs + source, not gigabytes of model weights, so
    ``GIT_LFS_SKIP_SMUDGE`` fetches LFS pointer files instead of the real blobs.
    Returns ``(ok, error_text)``.
    """
    args = ["git", "clone", "--depth", "1", "--no-tags", "--single-branch"]
    if branch:
        args += ["--branch", branch]
    args += [clone_url, str(dest)]
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except FileNotFoundError:
        return False, "git executable not found on PATH"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return False, f"git clone timed out after {timeout}s"
    if proc.returncode != 0:
        return False, (out or b"").decode("utf-8", "replace")[-1500:]
    return True, ""


@tool
async def create_custom_node(
    github_url: str,
    node_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Turn a model's GitHub repo into a self-contained ComfyUI custom-node pack.

    Use this when the user points you at a **model repository that has no existing
    ComfyUI node** and wants one built. It runs the **coder** agent with the
    ``custom-node-from-github`` skill: the repo is shallow-cloned locally (LFS
    weights skipped), the agent reads its
    README/docs/inference code, and it authors a complete, importable node pack —
    ``__init__.py`` (the ``NODE_CLASS_MAPPINGS``), ``nodes.py`` (the node classes +
    implementation), ``requirements.txt``, ``README.md``, and ``pyproject.toml`` —
    into ``output/custom_nodes/<node_name>/``.

    That folder is self-contained and git-ignored by agentY, so the user can ``cd``
    into it, ``git init``, and push it as its own GitHub repo, or copy it into
    ``ComfyUI/custom_nodes/`` to test. The agent implements the documented behaviour
    faithfully and marks anything it could not determine from the docs with a
    ``TODO`` stub, listed under "Unresolved / TODO" in the pack's README — so read
    the returned ``agent_summary`` and relay those gaps to the user.

    Args:
        github_url: The model repo — a full ``https://github.com/owner/repo`` URL,
            ``owner/repo`` shorthand, or a ``/tree/<branch>`` deep link.
        node_name: Optional name for the pack/output folder (slugified). Defaults to
            the repo name.
        notes: Optional extra guidance for the agent (which capability to expose,
            preferred inputs/outputs, constraints).

    Returns:
        A JSON string: ``status`` (ok|incomplete), ``node_name``, ``pack_dir``,
        ``files_written``, ``has_init_py``, the agent's ``agent_summary`` (with its
        Unresolved/TODO notes), and ``next_steps``.
    """
    url = (github_url or "").strip()
    if not url:
        return json.dumps({"error": "github_url is required (a GitHub repo URL or owner/repo)."})
    clone_url, branch = _normalize_github_url(url)

    # Output pack dir under the git-ignored output/ tree; never clobber an existing pack.
    raw_name = node_name or clone_url.rstrip("/").split("/")[-1]
    slug = _slugify(raw_name.replace(".git", "")) or "custom-node"
    packs_root = _generated_nodes_dir()
    pack_dir = packs_root / slug
    _n = 2
    while pack_dir.exists() and any(pack_dir.iterdir()):
        pack_dir = packs_root / f"{slug}-{_n}"
        _n += 1
    try:
        pack_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Could not create output folder {pack_dir}: {exc}"})

    # Shallow-clone the repo so the agent can read the actual docs + inference code.
    tmp_root = Path(tempfile.mkdtemp(prefix="cnc_clone_"))
    repo_dir = tmp_root / "repo"
    ok, err = await _git_clone_shallow(clone_url, branch, repo_dir)
    if not ok:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return json.dumps({
            "error": f"Could not clone {clone_url}: {err}",
            "hint": "Ensure it is a public GitHub repo reachable from this machine; "
                    "private repos need git credentials configured on PATH.",
        })

    # Build and run the coder agent with the custom-node-from-github skill (lazy
    # import avoids a cycle). The cloning + output-dir setup above is the fat-tool
    # scaffold; the skill carries the ComfyUI-node domain knowledge.
    try:
        from src.agent import create_coder_agent
        agent = create_coder_agent(skill="custom-node-from-github")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_root, ignore_errors=True)
        return json.dumps({"error": f"Could not build coder agent: {exc}"})

    task = (
        "Build a ComfyUI custom-node pack for the model in this repository.\n\n"
        f"repo_url: {clone_url}\n"
        f"repo_dir (local clone to READ): {repo_dir}\n"
        f"pack_dir (empty output folder to FILL): {pack_dir}\n"
        f"node_name: {slug}\n"
        f"notes: {notes or '(none)'}\n\n"
        "Read the repo, understand how the model loads and runs, then write every "
        "pack file (__init__.py, nodes.py, requirements.txt, README.md, "
        "pyproject.toml) into pack_dir using write_text_file. Follow your system "
        "instructions exactly — keep __init__.py import cheap, do heavy imports "
        "inside the node FUNCTION, and mark anything undetermined with a TODO stub. "
        "When done, return your summary including every Unresolved/TODO item."
    )
    agent_summary = ""
    try:
        result = await agent.invoke_async(task)
        agent_summary = str(result).strip()
    except Exception as exc:  # noqa: BLE001
        agent_summary = f"[coder agent error: {exc}]"
    finally:
        try:
            agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(tmp_root, ignore_errors=True)

    files = sorted(str(p.relative_to(pack_dir)) for p in pack_dir.rglob("*") if p.is_file())
    has_init = (pack_dir / "__init__.py").is_file()
    status = "ok" if (has_init and files) else "incomplete"
    return json.dumps({
        "status": status,
        "node_name": slug,
        "pack_dir": str(pack_dir),
        "repo_url": clone_url,
        "files_written": files,
        "has_init_py": has_init,
        "agent_summary": agent_summary[:4000],
        "next_steps": (
            f"Review the pack at {pack_dir}. To publish it as its own GitHub repo: "
            "cd into that folder, `git init`, commit, and push. To test it, copy or "
            "symlink the folder into ComfyUI/custom_nodes/ and restart ComfyUI."
        ),
    }, indent=2)


@tool
def list_generated_nodes() -> str:
    """List the ComfyUI custom-node packs the coder agent has written.

    Returns:
        A JSON string with each pack's name, absolute path, file count, and whether
        it has an ``__init__.py`` (i.e. looks importable by ComfyUI).
    """
    root = _generated_nodes_dir()
    out: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            n_files = sum(1 for p in child.rglob("*") if p.is_file())
            out.append({
                "name": child.name,
                "path": str(child),
                "files": n_files,
                "has_init_py": (child / "__init__.py").is_file(),
            })
    return json.dumps({"generated_nodes": out, "count": len(out)})
