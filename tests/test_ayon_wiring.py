"""The ayon tools reach the orchestrator: wrapped as shared core tools, held in the `ayon` pack, and given the
host's canvas and edit path (src/tools/ayon_host.py)."""

import unittest
from unittest import mock

import src.tools as T
from agenty_core.tools import ayon
from agenty_core.utils import turn_scope
from src.tools import ayon_host, tool_packs as tp

NAMES = ("ayon_context", "ayon_load", "ayon_mark_publish", "ayon_publish", "ayon_open_tool")


class Wiring(unittest.TestCase):
    def test_shared_core_and_wrapped(self):
        for name in NAMES:
            self.assertIn(name, T._SHARED_CORE_TOOLS)
            self.assertEqual(getattr(T, name).tool_name, name)

    def test_in_the_ayon_pack_and_the_orchestrator(self):
        for name in NAMES:
            self.assertEqual(tp._PACK_OF[name], "ayon")
            self.assertTrue(any(getattr(t, "tool_name", None) == name for t in T.ORCHESTRATOR_TOOLS), name)

    def test_the_host_adapters_are_installed(self):
        self.assertIs(ayon._canvas_provider, ayon_host.canvas_provider)
        self.assertIs(ayon._canvas_placer, ayon_host.canvas_placer)


class HostAdapters(unittest.TestCase):
    def setUp(self):
        self.scope = turn_scope.Scope("req", "thread")
        token = turn_scope._current.set(self.scope)
        self.addCleanup(turn_scope._current.reset, token)

    def test_provider_returns_the_turns_graph(self):
        graph = {"7": {"class_type": "AYON Context", "inputs": {}}}
        self.scope.set("canvas_prompt", graph)
        self.assertIs(ayon_host.canvas_provider(), graph)

    def test_placer_pushes_one_edit_and_keeps_the_turns_graph_in_step(self):
        graph = {"12": {"class_type": "VAEDecode", "inputs": {}}}
        self.scope.set("canvas_prompt", graph)
        ops = [{"op": "add", "class_type": "AYON Image Save", "node_id": "13", "params": {"ayon_info": "{}"}},
               {"op": "connect", "from": "12", "output": 0, "to": "13", "input": "images_in"}]
        with mock.patch("src.utils.canvas_patch.push") as push:
            self.assertTrue(ayon_host.canvas_placer(ops, "why"))
        push.assert_called_once_with({"op": "edit_graph", "ops": ops, "reason": "why"})
        self.assertEqual(graph["13"]["class_type"], "AYON Image Save")
        self.assertEqual(graph["13"]["inputs"]["images_in"], ["12", 0])


if __name__ == "__main__":
    unittest.main()
