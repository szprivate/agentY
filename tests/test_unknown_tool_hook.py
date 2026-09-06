"""A bounce that names the neighbouring tool, instead of one that names nothing.

``Unknown tool: get_workflow_template`` is the whole message Strands sends back
when an agent calls a tool it does not hold. It is true and it is a dead end: it
never says what the agent *does* have, so the agent improvises. In the run this
was written for, that improvisation was `run_script` (120 seconds of permission
timeout), a directory listing of 42 files, and two workflow JSONs re-read — six
calls and ~297K input tokens to reach a conclusion the first bounce could have
handed over.

What matters here is the ranking (a plausible wrong name shares tokens with the
right one) and that nothing in it can break a turn.
"""

import unittest

from src.utils.unknown_tool_hook import UnknownToolHookProvider, rank_alternatives

ORCHESTRATOR_ISH = [
    "prepare_workflow", "signal_workflow_ready", "update_workflow",
    "duplicate_workflow", "open_workflow_in_canvas", "get_workflow_catalog",
    "get_comfyui_dirs", "run_script", "run_info", "memory_read", "file_read",
    "read_text_file", "analyze_image", "get_node_schema",
]


class Ranking(unittest.TestCase):
    def test_the_real_miss_points_at_the_real_neighbour(self):
        # The call that actually happened.
        self.assertEqual(rank_alternatives("get_workflow_template", ORCHESTRATOR_ISH)[0],
                         "get_workflow_catalog")

    def test_an_exact_name_that_exists_ranks_itself_first(self):
        self.assertEqual(rank_alternatives("get_node_schema", ORCHESTRATOR_ISH)[0],
                         "get_node_schema")

    def test_a_generic_verb_alone_is_not_a_match(self):
        # "get" is in half the tool names; matching on it would rank
        # get_comfyui_dirs alongside anything else beginning with "get".
        near = rank_alternatives("get_node_schema", ORCHESTRATOR_ISH)
        self.assertNotIn("get_comfyui_dirs", near)

    def test_a_substring_still_finds_it(self):
        self.assertEqual(rank_alternatives("schema", ORCHESTRATOR_ISH), ["get_node_schema"])

    def test_a_name_with_nothing_in_common_suggests_nothing(self):
        self.assertEqual(rank_alternatives("totally_made_up_xyz", ORCHESTRATOR_ISH), [])

    def test_it_never_suggests_more_than_a_handful(self):
        self.assertLessEqual(len(rank_alternatives("workflow", ORCHESTRATOR_ISH)), 5)

    def test_an_empty_name_is_not_a_crash(self):
        self.assertEqual(rank_alternatives("", ORCHESTRATOR_ISH), [])


class _Registry:
    def __init__(self, names):
        self.registry = {n: object() for n in names}


class _Agent:
    def __init__(self, names):
        self.tool_registry = _Registry(names)


class _Event:
    def __init__(self, result, names=ORCHESTRATOR_ISH):
        self.result = result
        self.agent = _Agent(names)


def _bounce(name="get_workflow_template"):
    return {"toolUseId": "t1", "status": "error",
            "content": [{"text": f"Unknown tool: {name}"}]}


class Rewriting(unittest.TestCase):
    def setUp(self):
        self.hook = UnknownToolHookProvider()

    def _fire(self, event):
        self.hook._on_after(event)  # noqa: SLF001
        return event.result["content"][0]["text"]

    def test_the_bounce_gains_the_alternatives(self):
        text = self._fire(_Event(_bounce()))
        self.assertIn("Unknown tool: get_workflow_template", text)
        self.assertIn("get_workflow_catalog", text)

    def test_it_forbids_the_workaround_that_cost_the_time(self):
        # Shelling out to reconstruct a missing capability is exactly what
        # happened; the message has to close that door explicitly.
        text = self._fire(_Event(_bounce()))
        self.assertIn("run_script", text)
        self.assertIn("Do NOT", text)

    def test_an_invented_name_gets_the_actual_toolset(self):
        # No arbitrary sample: a handful chosen alphabetically reads as a
        # recommendation and sends the agent somewhere irrelevant.
        text = self._fire(_Event(_bounce("totally_made_up_xyz")))
        self.assertIn("no tool by a similar name", text)
        for name in ORCHESTRATOR_ISH:
            self.assertIn(name, text)

    def test_a_huge_registry_does_not_become_a_wall_of_text(self):
        names = [f"tool_{i:03d}" for i in range(200)]
        text = self._fire(_Event(_bounce("totally_made_up_xyz"), names))
        self.assertIn("+160 more", text)
        self.assertLess(len(text), 1500)

    def test_an_ordinary_error_is_left_alone(self):
        result = {"toolUseId": "t1", "status": "error",
                  "content": [{"text": "ComfyUI returned 500"}]}
        event = _Event(result)
        self.assertEqual(self._fire(event), "ComfyUI returned 500")

    def test_a_successful_result_is_left_alone(self):
        result = {"toolUseId": "t1", "status": "success",
                  "content": [{"text": "Unknown tool: not really"}]}
        event = _Event(result)
        self.assertEqual(self._fire(event), "Unknown tool: not really")

    def test_a_broken_event_does_not_raise(self):
        class Exploding:
            result = {"status": "error", "content": [{"text": "Unknown tool: x"}]}

            @property
            def agent(self):
                raise RuntimeError("no agent")

        self.hook._on_after(Exploding())  # noqa: SLF001 — must not raise

    def test_a_result_with_no_content_does_not_raise(self):
        event = _Event({"toolUseId": "t1", "status": "error", "content": []})
        self.hook._on_after(event)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
