""""Install this node pack" reaches install_custom_node.

The assembly refactor took install_custom_node off the orchestrator (only the
repair specialist kept it). A user who asked "can you set up this node pack for
me: https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git" got "my script tool
won't clone into custom_nodes, and I have no node-install tool of my own" - true
on both counts: run_script is refused custom_nodes on purpose, and nothing else
could install. The tool is back, held in the `nodes` pack, pinned on a turn whose
message asks for an install, and still asks the user before it runs.
"""

import pathlib
import unittest
from types import SimpleNamespace

from pipeline_stub import pipeline_stub
from src.tools import ORCHESTRATOR_TOOLS, FIX_WORKFLOW_ASSEMBLY_TOOLS
from src.tools import tool_packs as tp
from src.utils import tool_permissions as perms

PROMPT = (pathlib.Path(__file__).resolve().parent.parent
          / "config" / "system_prompts" / "system_prompt.orchestrator.md")
THE_MESSAGE = "mmh can you set up this node pack for me: https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git"


class TheOrchestratorCanInstall(unittest.TestCase):

    def test_it_carries_the_install_tools(self):
        names = {tp.name_of(t) for t in ORCHESTRATOR_TOOLS}
        self.assertIn("install_custom_node", names)
        self.assertIn("find_custom_node_for", names)
        # The repair specialist keeps them too.
        self.assertIn("install_custom_node", {tp.name_of(t) for t in FIX_WORKFLOW_ASSEMBLY_TOOLS})

    def test_they_are_held_in_the_nodes_pack_and_listed_in_the_prompt(self):
        always, held = tp.split(ORCHESTRATOR_TOOLS)
        self.assertNotIn("install_custom_node", {tp.name_of(t) for t in always})
        self.assertEqual({tp.name_of(t) for t in held["nodes"]}, {"install_custom_node", "find_custom_node_for"})
        self.assertIn("`install_custom_node`", tp.catalogue(held))

    def test_the_prompt_says_how_and_that_comfyui_needs_a_restart(self):
        text = PROMPT.read_text(encoding="utf-8")
        self.assertIn("install_custom_node(source)", text)
        self.assertIn("ComfyUI must be restarted", text)


class AskingForAnInstallPinsThePack(unittest.TestCase):

    def test_messages_that_ask_for_an_install(self):
        for text in (THE_MESSAGE, "install the ComfyUI-KJNodes custom nodes",
                     "please add https://github.com/kijai/ComfyUI-WanVideoWrapper",
                     "can you get me the impact pack node pack?", "Install ComfyUI Manager"):
            self.assertEqual(tp.packs_asked_for(text), {"nodes"}, text)

    def test_messages_that_do_not(self):
        for text in ("", "make an image of a cat", "add a KSampler node to my graph",
                     "what does this custom node do?", "set up a wan 2.2 workflow",
                     "update the prompt on node 12"):
            self.assertEqual(tp.packs_asked_for(text), set(), text)

    def test_the_pipeline_pins_it_for_that_turn(self):
        agent = SimpleNamespace()
        p = pipeline_stub(_orchestrator_agent=agent, _canvas_hooks=[], _review_halt=None, _loop_states={},
                          _canvas_graph={}, _canvas_selection=[])
        p._pin_tool_packs(False, THE_MESSAGE)
        self.assertIn("nodes", agent._tool_packs_pinned)
        p._pin_tool_packs(False, "make an image of a cat")
        self.assertNotIn("nodes", agent._tool_packs_pinned)


class ItStillAsks(unittest.TestCase):

    def test_ask_means_ask_not_hide(self):
        # The user's settings.local.json: ask_before_tools = ["iterate", "install_custom_node"].
        self.assertTrue(perms.must_ask("install_custom_node", {"source": "https://github.com/a/b"},
                                       ["iterate", "install_custom_node"]))
        self.assertIn("install_custom_node", perms.DEFAULT_ASK_TOOLS)

    def test_the_prompt_names_the_folder_and_a_restart(self):
        out = perms.describe("install_custom_node", {"source": "https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git",
                                                     "restart": True})
        self.assertIn("custom_nodes/ComfyUI-DLSS5-Enhancer", out)
        self.assertIn("RESTART ComfyUI", out)


if __name__ == "__main__":
    unittest.main()
