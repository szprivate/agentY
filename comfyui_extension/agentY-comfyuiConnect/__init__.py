"""agentY-comfyuiConnect — canvas ↔ agent bridge.

Two responsibilities:

1. **Push** — the agentY pipeline POSTs a graph-format workflow to
   ``/agent/load_workflow``; this broadcasts it over the websocket and
   ``web/agent_canvas.js`` loads it onto the canvas, so the user sees exactly
   what the agent just ran without clicking.

2. **Hook node** — the ``AgentYHook`` node lets the user annotate any node's
   output with a natural-language directive ("sweep the seed", "iterate this
   folder"). It is a pure identity passthrough, so on a normal Queue Prompt it is
   never executed (ComfyUI only runs nodes on the path to an output node, and an
   unreferenced hook is skipped by validation/execution). When the user asks the
   agentY agent to run the on-canvas graph, ``web/agent_chat.js`` ships the
   captured API prompt + the hook directives and the pipeline expands them.
"""
from aiohttp import web

try:
    from server import PromptServer
    _routes = PromptServer.instance.routes

    @_routes.post("/agent/load_workflow")
    async def _agent_load_workflow(request):  # noqa: ANN001
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        graph = data.get("workflow", data) if isinstance(data, dict) else None
        if not isinstance(graph, dict) or "nodes" not in graph:
            return web.json_response({"ok": False, "error": "not a graph workflow"}, status=400)
        # Broadcast to every connected ComfyUI frontend (sid=None => all).
        PromptServer.instance.send_sync("agent.load_workflow", graph)
        return web.json_response({"ok": True, "nodes": len(graph.get("nodes", []))})
except Exception as _e:  # noqa: BLE001
    # Never break ComfyUI startup if the server API shape changes.
    print(f"[agentY-comfyuiConnect] could not register /agent/load_workflow: {_e}")


class _AnyType(str):
    """A type that equals any other ComfyUI slot type (the standard ``*`` trick).

    Lets ``AgentYHook`` attach to — and pass through — an output of any type.
    """

    def __eq__(self, _other):  # noqa: ANN001
        return True

    def __ne__(self, _other):  # noqa: ANN001
        return False


_ANY = _AnyType("*")


class AgentYHook:
    """An agent directive attached to an upstream node's output.

    Wire this node's ``anchor`` input from any node's output and type a directive
    (e.g. "create prompt variations", "sweep the seed 6×", "iterate the files in
    this folder"). On a normal ComfyUI run the node is inert — it's an identity
    passthrough that nothing downstream needs, so it is never executed. When the
    agentY agent runs the on-canvas graph, it reads this directive and applies it
    to the anchored node.

    Recommended usage: wire only the ``anchor`` input and leave the output
    unwired (the node is then pruned entirely on a normal run). Splicing it inline
    also works — the agent removes it from the graph before running.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "directive": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "e.g. sweep the seed, 6 variations",
                }),
                "mode": (
                    ["auto", "prompt-variations", "seed-sweep", "file-iterate", "freeform"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "anchor": (_ANY, {}),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "hook"
    CATEGORY = "agentY"
    DESCRIPTION = (
        "Attach an agent directive to a node's output. Inert on a normal run; "
        "applied by the agentY agent when it runs the on-canvas graph."
    )

    def hook(self, directive="", mode="auto", anchor=None):  # noqa: ANN001, ARG002
        # Pure identity passthrough — only ever runs if spliced inline, in which
        # case it must not alter the data flowing through it.
        return (anchor,)


WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {"AgentYHook": AgentYHook}
NODE_DISPLAY_NAME_MAPPINGS = {"AgentYHook": "agentY hook"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
