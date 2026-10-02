"""On start, the example workflows of the installed node packs join the custom templates.

A pack's ``example_workflows`` folder is the only place templates for its nodes
exist, and the agent had none of them. Pinned here: it can be switched off, it
waits for a ComfyUI that is still starting, a failure changes nothing, a real
change reaches the running pipeline, and it runs after the official sync rather
than beside it (both rebuild the same recipe database).

    python -m unittest discover -s tests
"""
import unittest
from unittest import mock

from agenty_core.templates_sync import ComfyUIUnreachable

from src import agenty_ui_server as U


def _synced(**over):
    result = {"status": "synced", "packs": 2, "added": ["A__x.json"], "changed": [],
              "removed": [], "skipped": [], "recipes": {"recipe_count": 5}}
    result.update(over)
    return result


class StartupNodePackExamples(unittest.TestCase):

    def test_switched_off_it_asks_nothing(self):
        calls = []
        out = U._sync_node_pack_examples(lambda *a, **k: calls.append(a),
                                         settings={"sync_node_pack_examples": False})
        self.assertIsNone(out)
        self.assertEqual(calls, [])

    def test_it_asks_the_configured_comfyui(self):
        seen = []
        U._sync_node_pack_examples(lambda base, log: seen.append(base) or {"status": "current"},
                                   settings={"comfyui_url": "http://box:8188/"})
        self.assertEqual(seen, ["http://box:8188"])

    def test_it_waits_for_a_comfyui_that_is_still_starting(self):
        answers = [ComfyUIUnreachable("refused"), {"status": "current"}]

        def refresh(base, log):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        sleeps = []
        out = U._sync_node_pack_examples(refresh, settings={}, sleep=sleeps.append, clock=lambda: 0.0)
        self.assertEqual(out, {"status": "current"})
        self.assertEqual(len(sleeps), 1)

    def test_a_failure_is_not_fatal_and_not_retried(self):
        sleeps = []

        def refresh(base, log):
            raise RuntimeError("did not return a map")

        self.assertIsNone(U._sync_node_pack_examples(refresh, settings={}, sleep=sleeps.append))
        self.assertEqual(sleeps, [])

    def test_a_change_reaches_the_running_pipeline_and_nothing_new_does_not(self):
        from src.utils import agentY_server
        pipeline = type("Pipeline", (), {})()
        pipeline._recipe_tasks_cache = [{"task": "stale"}]
        with mock.patch.object(agentY_server, "_agent_ref", pipeline):
            U._sync_node_pack_examples(lambda base, log: {"status": "current"}, settings={})
            self.assertEqual(pipeline._recipe_tasks_cache, [{"task": "stale"}])
            U._sync_node_pack_examples(lambda base, log: _synced(), settings={})
        self.assertIsNone(pipeline._recipe_tasks_cache)

    def test_it_runs_after_the_official_sync_not_beside_it(self):
        order = []
        with mock.patch.object(U, "_sync_official_templates", lambda: order.append("official")), \
                mock.patch.object(U, "_sync_node_pack_examples", lambda: order.append("packs")):
            U._sync_templates_at_start()
        self.assertEqual(order, ["official", "packs"])


if __name__ == "__main__":
    unittest.main()
