"""On start, the official templates follow the ComfyUI install.

The mirror lagged behind the local ComfyUI, so a model ComfyUI had a template for
(MiniMax H3's local reference-to-video) was one the agent had to build by hand —
and it guessed the sampler. The host now asks ComfyUI on start and mirrors what it
ships. Pinned here: it can be switched off, it waits for a ComfyUI that is still
starting, it gives up quietly, and a real change reaches the running pipeline.

    python -m unittest discover -s tests
"""
import unittest
from unittest import mock

from agenty_core.templates_sync import ComfyUIUnreachable

from src import agenty_ui_server as U


def _synced(**over):
    result = {"status": "synced", "version": "0.11.48", "added": ["a.json"], "changed": [],
              "removed": [], "recipes": {"recipe_count": 5}}
    result.update(over)
    return result


class StartupTemplateSync(unittest.TestCase):

    def test_switched_off_it_asks_nothing(self):
        calls = []
        out = U._sync_official_templates(lambda *a, **k: calls.append(a),
                                         settings={"sync_templates_from_comfyui": False})
        self.assertIsNone(out)
        self.assertEqual(calls, [])

    def test_it_asks_the_comfyui_in_settings(self):
        seen = []
        U._sync_official_templates(lambda base, log: seen.append(base) or {"status": "current"},
                                   settings={"comfyui_url": "http://comfy.local:8188/"})
        self.assertEqual(seen, ["http://comfy.local:8188"])

    def test_it_waits_for_a_comfyui_that_is_still_starting(self):
        answers = [ComfyUIUnreachable("refused"), ComfyUIUnreachable("refused"),
                   {"status": "current"}]

        def refresh(base, log):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        sleeps = []
        out = U._sync_official_templates(refresh, settings={}, sleep=sleeps.append,
                                         clock=lambda: 0.0)
        self.assertEqual(out, {"status": "current"})
        self.assertEqual(len(sleeps), 2)

    def test_it_gives_up_once_the_wait_is_over(self):
        ticks = iter([0.0, U._TEMPLATE_SYNC_WAIT_S + 1])

        def refresh(base, log):
            raise ComfyUIUnreachable("refused")

        sleeps = []
        self.assertIsNone(U._sync_official_templates(refresh, settings={}, sleep=sleeps.append,
                                                     clock=lambda: next(ticks)))
        self.assertEqual(sleeps, [])

    def test_any_other_failure_keeps_the_templates_and_does_not_retry(self):
        sleeps = []

        def refresh(base, log):
            raise RuntimeError("lists only 3 templates")

        self.assertIsNone(U._sync_official_templates(refresh, settings={}, sleep=sleeps.append))
        self.assertEqual(sleeps, [])

    def test_a_change_reaches_the_running_pipeline(self):
        from src.utils import agentY_server
        pipeline = type("Pipeline", (), {})()
        pipeline._recipe_tasks_cache = [{"task": "stale"}]
        with mock.patch.object(agentY_server, "_agent_ref", pipeline):
            U._sync_official_templates(lambda base, log: _synced(), settings={})
        self.assertIsNone(pipeline._recipe_tasks_cache)

    def test_nothing_new_leaves_the_pipeline_alone(self):
        from src.utils import agentY_server
        pipeline = type("Pipeline", (), {})()
        pipeline._recipe_tasks_cache = [{"task": "kept"}]
        with mock.patch.object(agentY_server, "_agent_ref", pipeline):
            U._sync_official_templates(lambda base, log: _synced(added=[]), settings={})
        self.assertEqual(pipeline._recipe_tasks_cache, [{"task": "kept"}])


if __name__ == "__main__":
    unittest.main()
