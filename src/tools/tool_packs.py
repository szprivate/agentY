"""Rarely used tools join a conversation when it needs them.

Every tool definition goes out with every model call of the orchestrator. Some
are called almost never — in three weeks and 4,900 orchestrator tool calls:
annotate_image 0, bake_hooks_to_canvas 0, create_custom_node 0, start_batch_job
0, upload_file_to_url 0, create_skill 0, spawn_subagent 1, calculator 11 — and
together they are about a fifth of the tool definitions. With the setting
``tool_packs_on_demand`` (the default) they are grouped into packs by purpose:
the prompt lists each pack with its tools, and ``load_tools(pack)`` puts one in
the tool list.

Two packs are of another kind: their tools are not rare, they belong to a KIND
OF TURN - the stages of a hook pipeline (``hooks``), changing and running the
graph that is open (``graph``). Whether a turn is of that kind is known before
it starts (there are hook nodes on the canvas; a graph came with the message),
so the pipeline pins the pack (``_tool_packs_pinned``) and the model never has
to ask: on such a turn the tools are simply there, and on every other turn
their ~4,600 tokens of definitions are not sent. They are listed in the prompt
like the others, so a turn that needs one unexpectedly can still load it.

The same rules as MCP servers (src/tools/mcp_on_demand.py), for the same
reasons: which packs a conversation has is read off its own history at the start
of every turn (pooled agents, restarts), and a direct call to a tool whose pack
is not loaded loads the pack and runs — the names are in the prompt, so a model
may well call one straight away.
"""
from __future__ import annotations

import json
import logging
import re

from strands import tool
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.types.tools import ToolContext

logger = logging.getLogger("agentY.tools")

TOOL_NAME = "load_tools"

# pack -> (what it is for, its tools). Only tools the orchestrator rarely calls
# and that nothing in an ordinary turn depends on.
PACKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "batch": ("run one workflow over many inputs unattended (a folder of images, a shot list)",
              ("start_batch_job", "get_batch_status", "stop_batch_job", "list_batch_jobs")),
    "extend": ("save a working procedure as a skill, write a custom ComfyUI node, or start a "
               "sub-agent (only when the user asks for one)",
               ("create_skill", "list_skills", "remove_skill", "create_custom_node",
                "list_generated_nodes", "spawn_subagent")),
    "annotate": ("circle, box or arrow things in an image (an overlay, not a re-render)",
                 ("annotate_image",)),
    "bake": ("bake a hook chain's workflows into subgraphs on the canvas (a hook's `bake` switch)",
             ("bake_hooks_to_canvas",)),
    "upload": ("PUT a local file to a presigned URL — the upload step of an MCP flow (Magnific)",
               ("upload_file_to_url",)),
    "calc": ("exact arithmetic and symbolic maths", ("calculator",)),
    # Added 2026-10-12, when the tool definitions had grown to ~31k of the ~41k
    # tokens every orchestrator step sent (106 tools). Each group below is a job
    # most conversations never do; together about 12k tokens.
    "edit": ("cut shots into a film on an `agentY edit` node: the timeline, its texts and their "
             "presets, reading rhythm and beats, rendering the cut",
             ("edit_timeline", "render_edit", "probe_media", "read_rhythm", "read_beats",
              "list_text_presets", "find_text_demos", "try_text_preset", "keep_text_preset")),
    "models": ("look for a model that is NOT installed, on Hugging Face - check_model and "
               "find_local_models (what IS installed) and download_hf_model are always in your list",
               ("find_hf_file", "hf_search", "hf_file", "hf_repo")),
    "shots": ("running the shots of a sequence you lead (start_shot, which begins one, is always "
              "in your list)",
              ("close_shot", "message_shot", "shot_status", "stop_shot", "set_sequence_notes")),
    "loops": ("keep refining unattended until a goal is met: a canvas value, a prompt, an iteration plan; "
              "write a revised prompt into its canvas node",
              ("refine_canvas_until", "prompt_autoloop", "iterate", "revise_prompt")),
    "video": ("take a video apart: understand what is in it, its shots, single frames, its technical facts",
              ("analyze_video", "split_video_into_shots", "video_frames", "media_info")),
    "canvas_more": ("less common canvas work: a screenshot of the canvas, bypassing or muting nodes, "
                    "adding a second workflow as a tab",
                    ("screenshot_canvas", "set_canvas_node_mode", "add_canvas_workflow")),
    "admin": ("the agent's own settings and logs, Slack, freeing GPU memory, clearing the history, the "
              "ComfyUI queue's history and a prompt's status, submitting a workflow file straight to ComfyUI",
              ("set_agent_setting", "list_agent_settings", "get_logs", "send_to_slack",
               "free_memory", "get_system_stats", "clear_history", "get_history",
               "get_prompt_status_by_id", "submit_prompt")),
    "layout": ("make the open canvas readable: arrange nodes without overlaps, split into named coloured "
               "groups, rename groups, save the workflow (read the `canvas-layout` skill first)",
               ("arrange_canvas", "group_canvas_nodes", "auto_group_canvas", "save_canvas_workflow")),
    "worlds": ("build a walkable 3D world from reference images, and fly a camera through a 3D scene "
               "to render a camera move (read the `camera-move-control` skill first); make a 3D scene "
               "with World Labs Atlas from a picture or a prompt, render new views of it from any camera; capture a place all round and have World Labs build a Gaussian splat from it (or train one locally), or make a ready splat world with World Labs Marble and export SPZ / PLY (read the `worldlabs-atlas` skill first)",
               ("run_world_builder", "render_camera_move", "atlas_create_world", "atlas_expand_world",
                "atlas_rebuild_world_mesh", "atlas_splat_to_mesh", "atlas_fetch", "atlas_list",
                "atlas_render_view", "atlas_open_in_viewer", "atlas_capture_around", "train_splat",
                "splat_training_status", "atlas_views_to_splat", "marble_create_world", "marble_export")),
    "dcc": ("model a building / object / room in Blender, Maya or Houdini from a reference photo: run the one "
            "build script (live in the open app, or headless in batch) and check the silhouette by numbers; measure the photo with MoGe "
            "(read the `model-from-reference` skill first); destroy a model in Houdini with a rigid-body sim, dust, smoke and cloth "
            "(read the `dcc-destruction-houdini` skill first)",
            ("dcc_build", "dcc_silhouette_check", "measure_reference")),
    "motion": ("editable motion graphics (titles, lower thirds, logo reveals, animated text / shapes / images): "
               "write a *.mgscene.json scene as DATA, check it, look at stills, render it with Remotion; the user edits "
               "the same file in the viewer's Motion panel (read the `motion-graphics-scene` skill first)",
               ("mg_scene_validate", "mg_scene_preview_frame", "mg_render")),
    "lora": ("train a LoRA (a person, pet, product, style or motion) for an image or video model: build and caption the "
             "dataset, have the USER approve it in the viewer, train locally or hosted, follow the run, stop it "
             "(read the `lora-training` skill first)",
             ("lora_dataset_prepare", "lora_dataset_set_captions", "lora_dataset_review", "train_lora",
              "lora_training_status", "lora_training_stop")),
    "routing": ("see which models are available and what each role runs, and move the orchestrator (or a "
                "tier) to another one - always after the user agrees (see 'Switching models')",
                ("list_models", "switch_model")),
    "ayon": ("the studio's Ayon pipeline, when ComfyUI was launched through Ayon: which project / folder / task "
             "and frame range this session is in, load a published product into the canvas, mark an output "
             "for publish, open the Ayon Publisher (the user presses Publish) - see docs/dev/ayon.md",
             ("ayon_context", "ayon_load", "ayon_mark_publish", "ayon_publish", "ayon_open_tool")),
    # Added 2026-10-22 (speed-ups): the least-called of what stayed always-on - in 22,000 orchestrator
    # tool calls each of these was called 0-2 % of the time, and together they were ~7,000 tokens of
    # definitions on every call. A direct call to a held tool still loads its pack and runs.
    "dashscope": ("make an image or video on the user's own DashScope key (not Comfy credits) and "
                  "collect one that timed out",
                  ("dashscope_generate", "dashscope_fetch")),
    "viewer": ("show files in the bEpic viewer, or in the chat panel; read again what the user has "
               "picked in a history panel (pinned whenever something is); what the viewer shows (tab, "
               "item, frame, in/out range, compare, camera) and the marks the user drew on it - also "
               "as an inpaint mask (pinned when the item on screen has some)",
               ("show_in_viewer", "show_in_panel", "get_history_selection", "get_viewer_state",
                "get_viewer_annotations")),
    "web_more": ("find reference IMAGES on the web, find other photos of the SAME thing from one picture "
                 "(reverse image search), or hand a web search to the scout "
                 "(web_search, read_web_page, download_image and run_info are always in your list)",
                 ("web_search_images", "reverse_image_search", "run_web_search")),
    "inspect": ("read arbitrary files, list a workflow file's loaded models, search folders for workflow "
                "files (read_text_file, list_files and get_node_source are always in your list)",
                ("file_read", "inspect_workflow_file", "find_workflows")),
    "mcp": ("see which MCP servers are set up and whether each is loaded, install one the user asks for (the "
            "preset catalog first: install_mcp_preset sets the whole thing up, app add-on included; a server "
            "that is not in it: draft_mcp_recipe, then the user's yes), re-check a preset's setup, install a "
            ".mcpb bundle, or see which creative apps (Blender, Houdini, Maya, Nuke, After Effects, Premiere, "
            "Resolve, Cavalry, Unity) are installed and which version",
            ("list_mcp_servers", "add_mcp_server", "list_mcp_presets", "install_mcp_preset", "setup_mcp_preset",
             "draft_mcp_recipe", "install_mcp_bundle", "find_app_installs")),
    "nodes": ("install a ComfyUI custom node pack the user asks for (its https GitHub link) into "
              "custom_nodes, its requirements into ComfyUI's Python - asks the user first, never clone "
              "with run_script; ComfyUI needs a restart afterwards, say so - or find which pack provides "
              "a node class (pinned on a turn that asks for an install)",
              ("install_custom_node", "find_custom_node_for")),
    "dirs": ("where ComfyUI keeps its folders (models, input, output) and where agentY saves what it makes",
             ("get_comfyui_dirs", "get_agent_output_dirs")),
    # Pinned by the pipeline on the turns they belong to (CONTEXT_PACKS).
    "farm": ("render on the studio's Deadline render farm instead of the local ComfyUI: send a workflow as a "
             "job, follow it, see the pools and workers (read the `deadline-farm` skill first)",
             ("submit_to_deadline", "deadline_job_status", "deadline_farm_info")),
    "hooks": ("the stages of a hook pipeline on the canvas - already in your list whenever the canvas "
              "has `agentY hook` / review / loop nodes",
              ("apply_canvas_hooks", "stop_hook_run", "halt_for_review", "release_review", "loop_check",
               "forward_outputs", "place_canvas_text")),
    "graph": ("changing the wiring of the graph the user has open, deleting its nodes, running it as it "
              "is - already in your list whenever a graph is open or you work in the open graph "
              "(`set_canvas_node_params` and `get_canvas_node` always are)",
              ("edit_canvas_graph", "replace_canvas_node", "delete_canvas_nodes", "run_canvas")),
}
# Experimental, off by default ([atlas] camera_move_enabled): the paid Atlas camera move joins the worlds pack.
try:
    from src.utils.settings import atlas_camera_move_enabled as _atlas_move_on
    if _atlas_move_on():
        _w_why, _w_names = PACKS["worlds"]
        PACKS["worlds"] = (_w_why.replace(
            "render new views of it from any camera;",
            "render new views of it from any camera, and render a camera move through it with Atlas-painted "
            "frames (paid; price it with dry_run first);"), _w_names + ("atlas_render_camera_move",))
except Exception:  # noqa: BLE001
    pass
# Packs the pipeline loads by what the turn is, rather than the model by asking.
CONTEXT_PACKS = ("hooks", "graph")
_PACK_OF = {name: pack for pack, (_why, names) in PACKS.items() for name in names}


# Packs a user's message asks for in so many words: "install this node pack",
# "set up https://github.com/x/ComfyUI-Foo". Pinned for that turn, like the
# context packs, so the tool is in the list without the model having to know the
# pack exists - in the turn that showed the gap, it did not, and told the user it
# had no way to install a node.
_INSTALL_VERB = re.compile(r"\b(install|re-?install|set\s*up|setup|add|clone|get|update|upgrade)\b", re.I)
_NODE_PACK_WORDS = re.compile(
    r"node[\s_-]*packs?|custom[\s_-]*nodes?|comfyui[\s_-]*manager|"
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|codeberg\.org|bitbucket\.org)/[^\s/]+/[^\s/]*comfy",
    re.I)


def packs_asked_for(text: str) -> set:
    """The packs a user's message plainly asks for (see _NODE_PACK_WORDS)."""
    text = str(text or "")
    if _INSTALL_VERB.search(text) and _NODE_PACK_WORDS.search(text):
        return {"nodes"}
    return set()


def enabled() -> bool:
    """Setting ``tool_packs_on_demand`` (default on)."""
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("tool_packs_on_demand", True))
    except Exception:  # noqa: BLE001
        return True


def name_of(tool_obj) -> str:
    """A tool's name, for a decorated function and for a module-style tool."""
    name = getattr(tool_obj, "tool_name", None)
    if name:
        return str(name)
    spec = getattr(tool_obj, "TOOL_SPEC", None)
    if isinstance(spec, dict) and spec.get("name"):
        return str(spec["name"])
    return str(getattr(tool_obj, "__name__", "")).rsplit(".", 1)[-1]


def split(tools: list) -> tuple[list, dict]:
    """``(always, held)``: the tools every conversation has, and
    ``{pack: [tool, …]}`` for the ones that wait to be loaded."""
    always, held = [], {}
    for t in tools:
        pack = _PACK_OF.get(name_of(t))
        if pack:
            held.setdefault(pack, []).append(t)
        else:
            always.append(t)
    return always, held


def catalogue(held: dict) -> str:
    """The prompt section naming each pack and its tools."""
    lines = [f"- **{pack}** — {PACKS[pack][0]}: " + ", ".join(f"`{name_of(t)}`" for t in tools)
             for pack, tools in held.items()]
    if not lines:
        return ""
    return ("\n\n## More tools — load on demand\n"
            f"These tools are yours but NOT in your list until you call `{TOOL_NAME}(pack)` — "
            "once per conversation; they then stay for the rest of it. Load a pack when a "
            "request needs it (and only then); its tools then appear with their full "
            "parameters.\n" + "\n".join(lines) + "\n")


def packs_in(messages) -> set:
    """The packs a conversation has loaded or called a tool of."""
    found = set()
    for msg in messages or ():
        content = msg.get("content") if isinstance(msg, dict) else None
        for block in content if isinstance(content, list) else ():
            use = block.get("toolUse") if isinstance(block, dict) else None
            if not isinstance(use, dict):
                continue
            name = str(use.get("name") or "")
            if name == TOOL_NAME:
                pack = str((use.get("input") or {}).get("pack") or "").strip().lower()
                if pack in PACKS:
                    found.add(pack)
            elif name in _PACK_OF:
                found.add(_PACK_OF[name])
    return found


def _held(agent) -> dict:
    return getattr(agent, "_tool_packs", None) or {}


def loaded(agent) -> set:
    got = getattr(agent, "_tool_packs_loaded", None)
    if got is None:
        got = set()
        try:
            agent._tool_packs_loaded = got
        except Exception:  # noqa: BLE001
            pass
    return got


def load(agent, pack: str) -> list:
    """Put *pack*'s tools in *agent*'s list. Returns their names."""
    tools = _held(agent).get(pack)
    if not tools:
        raise LookupError(f"no tool pack named {pack!r} (there are: {', '.join(_held(agent)) or 'none'})")
    reg = agent.tool_registry
    missing = [t for t in tools if name_of(t) not in reg.registry]
    if missing:
        reg.process_tools(missing)
    loaded(agent).add(pack)
    return [name_of(t) for t in tools]


def unload(agent, pack: str) -> None:
    reg = agent.tool_registry
    for t in _held(agent).get(pack) or ():
        reg.registry.pop(name_of(t), None)
        reg.dynamic_tools.pop(name_of(t), None)
    loaded(agent).discard(pack)


def pinned(agent) -> set:
    """Packs the pipeline says this turn needs (an edit node on the canvas, a
    lead conversation): loaded without the model having to ask."""
    return set(getattr(agent, "_tool_packs_pinned", None) or ())


def sync(agent) -> set:
    """Give *agent* the packs its current conversation uses, and only those."""
    wanted = (packs_in(getattr(agent, "messages", None)) | pinned(agent)) & set(_held(agent))
    for pack in loaded(agent) - wanted:
        unload(agent, pack)
    for pack in wanted - loaded(agent):
        try:
            load(agent, pack)
        except Exception as exc:  # noqa: BLE001
            logger.info("tool pack %s not loaded: %s", pack, exc)
    return loaded(agent)


@tool(context=True)
def load_tools(pack: str, tool_context: ToolContext) -> str:
    """Load one of your extra tool packs for this conversation.

    The packs and the tools in each are listed in your prompt under "More tools —
    load on demand". Call this once before the first use of a pack's tools; they
    are then in your list for the rest of the conversation.

    Args:
        pack: The pack's name, e.g. "batch".
    """
    pack = str(pack or "").strip().lower()
    try:
        names = load(tool_context.agent, pack)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})
    return json.dumps({"ok": True, "pack": pack, "tools": names,
                       "message": "In your list now: " + ", ".join(names) + ". Call them directly."})


class ToolPacksHook(HookProvider):
    """Each turn: the conversation's packs. Each call to a held tool that isn't
    loaded: load its pack and run the call."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        registry.add_callback(BeforeInvocationEvent, self._on_turn)
        registry.add_callback(BeforeToolCallEvent, self._on_tool)

    def _on_turn(self, event: BeforeInvocationEvent, **kwargs) -> None:  # noqa: ARG002
        try:
            sync(event.agent)
        except Exception as exc:  # noqa: BLE001 — never break a turn over this
            logger.warning("tool packs: sync failed: %s", exc)

    def _on_tool(self, event: BeforeToolCallEvent, **kwargs) -> None:  # noqa: ARG002
        if event.selected_tool is not None:
            return
        try:
            name = str((event.tool_use or {}).get("name") or "")
            pack = _PACK_OF.get(name)
            if not pack or pack not in _held(event.agent):
                return
            load(event.agent, pack)
            found = event.agent.tool_registry.registry.get(name)
            if found is not None:
                event.selected_tool = found
        except Exception as exc:  # noqa: BLE001 — the unknown-tool message follows
            logger.info("tool packs: could not load for %s: %s", event.tool_use, exc)
