"""A workflow no template covers is looked up on the web, not guessed.

The from-scratch builder works from the recipe database, which only knows the
workflows already in the library. The agent has web_search / read_web_page and
used them for an unknown workflow only when it thought of it. Now it is a rule
in the prompt, and the build result itself says so when no template fitted.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

from src import pipeline

PROMPT = (Path(__file__).resolve().parents[1] / "config" / "system_prompts"
          / "system_prompt.orchestrator.md").read_text(encoding="utf-8")


def _briefing(name):
    return SimpleNamespace(template=SimpleNamespace(name=name))


class TheBuildResultSaysSo(unittest.TestCase):

    def test_a_from_scratch_build_is_flagged(self):
        for name in ("build_new", "BUILD_NEW", "none", "", None):
            with self.subTest(template=name):
                result = {"status": "ready", "workflow_path": "x.json"}
                pipeline._mark_from_scratch(result, _briefing(name))
                self.assertIs(result.get("from_scratch"), True)
                for word in ("web_search", "read_web_page", "update_workflow", "BEFORE you signal"):
                    self.assertIn(word, result["verify"])

    def test_a_template_build_is_not(self):
        result = {"status": "ready", "workflow_path": "x.json"}
        pipeline._mark_from_scratch(result, _briefing("flux_dev_t2i"))
        self.assertEqual(result, {"status": "ready", "workflow_path": "x.json"})

    def test_a_build_that_failed_is_not(self):
        for status in ("failed", "needs_fix", "error", "blocked"):
            result = {"status": status}
            pipeline._mark_from_scratch(result, _briefing("build_new"))
            self.assertEqual(result, {"status": status})

    def test_prepare_workflow_marks_before_it_answers(self):
        import inspect
        src = inspect.getsource(pipeline)
        body = src.split("async def prepare_workflow(", 1)[1].split("@_tool", 1)[0]
        self.assertIn("_mark_from_scratch(result, briefing)", body)


class TheToolsAndTheRule(unittest.TestCase):

    def test_the_orchestrator_has_the_tools_the_rule_names(self):
        from src.tools import ORCHESTRATOR_TOOLS, tool_packs
        have = {tool_packs.name_of(t) for t in ORCHESTRATOR_TOOLS}
        for name in ("web_search", "read_web_page", "update_workflow"):
            self.assertIn(name, have)
            self.assertNotIn(name, tool_packs._PACK_OF, "must not wait in an on-demand pack")

    def test_the_prompt_states_the_rule_before_the_set_up_step(self):
        self.assertIn("A workflow you do not know is looked up, not guessed.", PROMPT)
        rule = PROMPT.index("A workflow you do not know is looked up")
        self.assertLess(rule, PROMPT.index("1. **Set up (always start here):**"))
        section = PROMPT[rule:PROMPT.index("1. **Set up (always start here):**")]
        for word in ("web_search", "read_web_page", "from_scratch: true", "verify"):
            self.assertIn(word, section)


if __name__ == "__main__":
    unittest.main()
