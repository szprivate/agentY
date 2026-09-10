"""Naming the tool that can answer, not the one that sounds like it.

The bounce for an unregistered tool already refuses the escape route — *"not by
shelling out with `run_script`, not by reading files off disk"* — and then names
the closest tools by spelling. For a whole class of questions that is nobody.

Measured against the orchestrator's own registry, asking for `get_node_schema`
returned `get_canvas_node`, `create_custom_node`, `set_canvas_node_params`. Not
one of them can say what values an input accepts: the first reads a node the user
has on screen, the second builds a node, the third writes to one. The tool that
CAN is `get_node_schema` itself, held by the info specialist — and `run_info`
shares no word with it, so ranking by name could never find it.

So the run did what the message forbids. `run_script` is on the orchestrator's
toolset and in DEFAULT_ASK_TOOLS, so every attempt was a permission prompt with a
120-second timeout, and the loop ran until the user stopped it.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.unknown_tool_hook import (names_the_same_tool, rank_alternatives,
                                         route_to_specialist)

# A registry shaped like the orchestrator's: it holds the specialist, and holds
# several tools whose names mention nodes without inspecting any.
REGISTRY = ["run_info", "get_canvas_node", "set_canvas_node_params", "run_script",
            "delete_canvas_nodes", "create_custom_node", "prepare_workflow",
            "web_search", "signal_workflow_ready"]


def _route(wanted, registry=REGISTRY):
    return route_to_specialist(wanted, registry,
                               strong_match=names_the_same_tool(wanted, registry))


class TheQuestionThatHadNoAnswer(unittest.TestCase):

    def test_a_schema_question_is_sent_to_the_specialist(self):
        self.assertEqual(_route("get_node_schema")[0], "run_info")

    def test_the_reason_says_what_it_can_answer(self):
        why = _route("get_node_schema")[1]
        self.assertIn("values it accepts", why)

    def test_name_ranking_alone_still_finds_nobody_useful(self):
        """The reason routing had to exist — pinned so it is not mistaken for a
        ranking bug that someone later 'fixes' by loosening the scores."""
        near = rank_alternatives("get_node_schema", REGISTRY)
        self.assertNotIn("run_info", near)

    def test_the_other_spellings_route_too(self):
        for wanted in ("get_object_info", "list_node_options", "search_nodes",
                       "get_node_widgets", "node_parameters"):
            with self.subTest(wanted=wanted):
                self.assertEqual(_route(wanted)[0], "run_info")


class WhatMustNotBeRerouted(unittest.TestCase):

    def test_a_tool_that_exists_under_another_spelling_wins(self):
        # `delete_canvas_node` means `delete_canvas_nodes`, not "ask the info
        # agent about nodes".
        self.assertEqual(_route("delete_canvas_node"), ())
        self.assertEqual(_route("get_canvas_nodes"), ())

    def test_sharing_one_word_is_not_the_same_tool(self):
        # `search_nodes` and `delete_canvas_nodes` share "nodes" and nothing else.
        self.assertFalse(names_the_same_tool("search_nodes", REGISTRY))

    def test_an_unrelated_name_is_left_to_the_ranking(self):
        self.assertEqual(_route("get_workflow_template"), ())

    def test_nothing_is_offered_when_the_specialist_is_not_held(self):
        # An agent that cannot delegate must not be told to.
        without = [n for n in REGISTRY if n != "run_info"]
        self.assertEqual(route_to_specialist("get_node_schema", without), ())

    def test_an_empty_name_routes_nowhere(self):
        self.assertEqual(_route(""), ())


class TheBounceItself(unittest.TestCase):
    """What the agent actually reads."""

    def _bounce(self, wanted, registry=REGISTRY):
        from unittest import mock

        from src.utils.unknown_tool_hook import UnknownToolHookProvider
        result = {"status": "error",
                  "content": [{"text": f"Unknown tool: {wanted}"}]}
        agent = mock.Mock()
        agent.tool_registry.registry = {n: object() for n in registry}
        event = mock.Mock(result=result, agent=agent)
        UnknownToolHookProvider()._on_after(event)
        return result["content"][0]["text"]

    def test_the_specialist_is_named_first(self):
        text = self._bounce("get_node_schema")
        self.assertIn("Call `run_info`", text)
        self.assertLess(text.index("run_info"), text.index("get_canvas_node"))

    def test_the_near_misses_are_still_offered_as_the_alternative(self):
        text = self._bounce("get_node_schema")
        self.assertIn("if you meant something else", text)

    def test_the_door_to_the_shell_stays_shut(self):
        self.assertIn("not by shelling out", self._bounce("get_node_schema"))

    def test_an_ordinary_bounce_is_unchanged(self):
        text = self._bounce("delete_canvas_node")
        self.assertIn("Closest tools you DO have", text)
        self.assertNotIn("Call `run_info`", text)


if __name__ == "__main__":
    unittest.main()
