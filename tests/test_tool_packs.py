"""Rarely used tools wait in packs until a conversation needs them.

Their definitions are about a fifth of what each orchestrator step sent, for
tools called a handful of times in thousands of calls. A pack is loaded with
load_tools, kept for as long as the conversation's history shows it, and a
direct call to a held tool still runs.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from strands import tool
from strands.tools.registry import ToolRegistry

from src.tools import tool_packs as tp


@tool
def start_batch_job(folder: str) -> str:
    """Run a workflow over a folder."""
    return "started"


@tool
def get_batch_status() -> str:
    """How the batch is doing."""
    return "running"


@tool
def annotate_image(path: str) -> str:
    """Draw on an image."""
    return path


@tool
def check_model(name: str) -> str:
    """Always there."""
    return name


ALL = [check_model, start_batch_job, get_batch_status, annotate_image]


def _agent(messages=()):
    always, held = tp.split(ALL)
    reg = ToolRegistry()
    reg.process_tools(always)
    return SimpleNamespace(tool_registry=reg, messages=list(messages), _tool_packs=held)


def _use(name, **inp):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": "u", "name": name, "input": inp}}]}


class Packs(unittest.TestCase):

    def names(self, agent):
        return sorted(agent.tool_registry.registry)

    def test_held_tools_are_out_of_the_list_and_the_rest_stay(self):
        always, held = tp.split(ALL)
        self.assertEqual([tp.name_of(t) for t in always], ["check_model"])
        self.assertEqual({k: [tp.name_of(t) for t in v] for k, v in held.items()},
                         {"batch": ["start_batch_job", "get_batch_status"], "annotate": ["annotate_image"]})
        self.assertEqual(self.names(_agent()), ["check_model"])

    def test_every_pack_names_real_orchestrator_tools(self):
        """A pack naming a tool that no longer exists would silently hold nothing."""
        from src.tools import ORCHESTRATOR_TOOLS
        have = {tp.name_of(t) for t in ORCHESTRATOR_TOOLS}
        for pack, (_why, names) in tp.PACKS.items():
            for name in names:
                with self.subTest(pack=pack, tool=name):
                    self.assertIn(name, have)

    def test_what_every_turn_needs_is_never_held(self):
        for name in ("prepare_workflow", "signal_workflow_ready", "withdraw_workflow", "check_model",
                     "find_local_models", "update_workflow", "open_workflow_in_canvas", "memory_read",
                     "project_memory_read", "start_shot", "stop", "queue", "download_hf_model"):
            self.assertNotIn(name, tp._PACK_OF)

    def test_loading_a_pack_adds_its_tools(self):
        a = _agent()
        out = json.loads(tp.load_tools._tool_func("batch", SimpleNamespace(agent=a)))
        self.assertTrue(out["ok"])
        self.assertEqual(out["tools"], ["start_batch_job", "get_batch_status"])
        self.assertEqual(self.names(a), ["check_model", "get_batch_status", "start_batch_job"])
        self.assertIn("start_batch_job", {s["name"] for s in a.tool_registry.get_all_tool_specs()})

    def test_an_unknown_pack_says_which_there_are(self):
        out = json.loads(tp.load_tools._tool_func("nope", SimpleNamespace(agent=_agent())))
        self.assertFalse(out["ok"])
        self.assertIn("batch", out["error"])

    def test_the_conversation_decides_each_turn(self):
        a = _agent([_use("load_tools", pack="batch")])
        tp.sync(a)
        self.assertIn("start_batch_job", self.names(a))
        a.messages = [_use("annotate_image", path="x.png")]      # another conversation, same agent
        tp.sync(a)
        self.assertEqual(self.names(a), ["annotate_image", "check_model"])
        a.messages = []
        tp.sync(a)
        self.assertEqual(self.names(a), ["check_model"])

    def test_a_call_to_a_held_tool_loads_its_pack(self):
        a = _agent()
        ev = SimpleNamespace(selected_tool=None, agent=a, tool_use={"name": "get_batch_status", "input": {}})
        tp.ToolPacksHook()._on_tool(ev)
        self.assertIs(ev.selected_tool, get_batch_status)
        self.assertIn("start_batch_job", self.names(a), "the whole pack, not one tool")
        ev = SimpleNamespace(selected_tool=None, agent=a, tool_use={"name": "made_up", "input": {}})
        tp.ToolPacksHook()._on_tool(ev)
        self.assertIsNone(ev.selected_tool)

    def test_the_prompt_lists_every_held_tool(self):
        _always, held = tp.split(ALL)
        text = tp.catalogue(held)
        self.assertIn("load_tools", text)
        for name in ("start_batch_job", "get_batch_status", "annotate_image"):
            self.assertIn(f"`{name}`", text)
        self.assertEqual(tp.catalogue({}), "")

    def test_the_setting_switches_it_off(self):
        with mock.patch("src.utils.settings.load_settings", return_value={"tool_packs_on_demand": False}):
            self.assertFalse(tp.enabled())
        with mock.patch("src.utils.settings.load_settings", return_value={}):
            self.assertTrue(tp.enabled())


if __name__ == "__main__":
    unittest.main()
