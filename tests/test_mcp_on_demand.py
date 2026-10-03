"""MCP servers' tools join a conversation only when it needs them.

Their definitions are most of what each orchestrator step sends (Magnific and
Blender: ~86k tokens) and are used in about one step in a hundred, so a
conversation loads a server with use_mcp_server and keeps it for as long as its
history shows it, and a direct call to an unloaded server's tool still runs.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from strands import tool
from strands.tools.registry import ToolRegistry

from src.tools import mcp_on_demand as od
from src.tools import mcp_tools


@tool(name="magnific__video_generate")
def _mag_video(prompt: str) -> str:
    """Make a video."""
    return "video:" + prompt


@tool(name="magnific__images_generate")
def _mag_image(prompt: str) -> str:
    """Make an image."""
    return "image:" + prompt


@tool(name="blender__execute_blender_code")
def _blender(code: str) -> str:
    """Run code in Blender."""
    return "ran"


@tool
def plain_tool() -> str:
    """Always there."""
    return "x"


CLIENTS = {"magnific": object(), "blender": object()}
LISTS = {"magnific": [_mag_video, _mag_image], "blender": [_blender]}
CONFIG = {"servers": {"magnific": {"enabled": True}, "blender": {"enabled": True},
                      "off": {"enabled": False}}}


def _agent(messages=()):
    reg = ToolRegistry()
    reg.process_tools([plain_tool])
    return SimpleNamespace(tool_registry=reg, messages=list(messages))


def _use(name, **inp):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": "u", "name": name, "input": inp}}]}


class OnDemand(unittest.TestCase):

    def setUp(self):
        patches = [
            mock.patch.object(mcp_tools, "_CLIENTS", dict(CLIENTS)),
            mock.patch.object(mcp_tools, "_LISTS", {k: (id(CLIENTS[k]), v) for k, v in LISTS.items()}),
            mock.patch.object(mcp_tools, "load_mcp_config", return_value=CONFIG),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        od._LISTS = mcp_tools._LISTS
        self.addCleanup(setattr, od, "_LISTS", mcp_tools._LISTS)

    def names(self, agent):
        return sorted(agent.tool_registry.registry)

    def test_a_new_conversation_has_none(self):
        a = _agent([{"role": "user", "content": [{"text": "make a picture"}]}])
        od.sync(a)
        self.assertEqual(self.names(a), ["plain_tool"])

    def test_loading_a_server_adds_its_tools(self):
        a = _agent()
        out = json.loads(od.use_mcp_server._tool_func("magnific", SimpleNamespace(agent=a)))
        self.assertTrue(out["ok"])
        self.assertEqual(out["tools"], 2)
        self.assertEqual(self.names(a), ["magnific__images_generate", "magnific__video_generate", "plain_tool"])
        specs = {s["name"] for s in a.tool_registry.get_all_tool_specs()}
        self.assertIn("magnific__video_generate", specs)

    def test_unknown_or_switched_off_servers_say_why(self):
        a = _agent()
        out = json.loads(od.use_mcp_server._tool_func("nope", SimpleNamespace(agent=a)))
        self.assertFalse(out["ok"])
        out = json.loads(od.use_mcp_server._tool_func("off", SimpleNamespace(agent=a)))
        self.assertIn("switched off", out["error"])

    def test_the_conversation_decides_each_turn(self):
        # A pooled agent answered a Magnific conversation; the next one is about Blender.
        a = _agent([_use("use_mcp_server", server="magnific")])
        od.sync(a)
        self.assertIn("magnific__video_generate", self.names(a))
        a.messages = [_use("blender__execute_blender_code", code="")]
        od.sync(a)
        self.assertEqual(self.names(a), ["blender__execute_blender_code", "plain_tool"])
        a.messages = []
        od.sync(a)
        self.assertEqual(self.names(a), ["plain_tool"])

    def test_names_that_only_look_like_a_server_are_ignored(self):
        self.assertEqual(od.servers_in([_use("my__tool"), _use("off__x"),
                                        _use("use_mcp_server", server="ghost")]), set())

    def test_a_call_to_an_unloaded_tool_loads_its_server(self):
        a = _agent()
        ev = SimpleNamespace(selected_tool=None, agent=a,
                             tool_use={"name": "magnific__video_generate", "input": {}})
        od.MCPOnDemandHook()._on_tool(ev)
        self.assertIs(ev.selected_tool, _mag_video)
        # Anything else is left to the unknown-tool message.
        ev = SimpleNamespace(selected_tool=None, agent=a, tool_use={"name": "made_up", "input": {}})
        od.MCPOnDemandHook()._on_tool(ev)
        self.assertIsNone(ev.selected_tool)

    def test_the_prompt_lists_every_tool_name(self):
        text = od.catalogue()
        self.assertIn("use_mcp_server", text)
        self.assertIn("**magnific** (2 tools): video_generate, images_generate", text)
        self.assertIn("**blender** (1 tools): execute_blender_code", text)

    def test_the_setting_switches_it_off(self):
        with mock.patch("src.utils.settings.load_settings", return_value={"mcp_tools_on_demand": False}):
            self.assertFalse(od.enabled())
        with mock.patch("src.utils.settings.load_settings", return_value={}):
            self.assertTrue(od.enabled())


if __name__ == "__main__":
    unittest.main()
