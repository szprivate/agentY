"""One switch for "keep what this produced", named by purpose.

The hook node carried three booleans — `bake_to_canvas`, `freeze`, `memorize` —
then two, and they were always one question asked three ways: should what this
hook produced outlive the run? They are now a single switch.

What ON *does* follows the purpose, because the purposes produce different things
and there is only one sensible way to keep each:

* `make_workflow` produces a workflow, so keeping it means nesting it into a
  subgraph. That is the one purpose where the switch is still called **bake**.
* every other producing purpose produces a result — text, a prompt, a script,
  images, videos — so keeping it means **memorize**: write it to the store beside
  the outputs and put it straight back next time.

The switch deliberately no longer rewires anything. `freeze` used to bake a text
hook's value into its target input and take over the hook's downstream link,
which destroyed the thing the user drew: the hook chain IS the graph's readable
statement of what happens, and a switch about keeping a result has no business
rewriting it.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import (_is_general, _is_iterate, _is_qa, _is_standin,
                                    _is_text, _wants_bake)
from src.utils.hook_cache import remembering


def hook(purpose, remember=False, **kw):
    """A hook as the current panel sends it: one switch, one field."""
    h = {"hook_node_id": "5", "purpose": purpose, "directive": "do the thing",
         "anchors": [], "targets": [], "remember": remember}
    h.update(kw)
    return h


class OneSwitchTest(unittest.TestCase):

    def test_on_a_make_workflow_hook_it_asks_for_a_subgraph(self):
        self.assertTrue(_wants_bake(hook("make_workflow", remember=True)))
        self.assertFalse(_wants_bake(hook("make_workflow", remember=False)))

    def test_the_same_switch_memorizes_everywhere_else(self):
        for purpose in ("text", "inline_parameter", "general_request"):
            with self.subTest(purpose=purpose):
                self.assertTrue(remembering(hook(purpose, remember=True)))
                self.assertFalse(remembering(hook(purpose, remember=False)))

    def test_baking_is_offered_for_exactly_one_purpose(self):
        """Every other purpose reads the same bit as "memorize", never as "bake"."""
        for purpose in ("text", "inline_parameter", "general_request", "qa", "iterate"):
            with self.subTest(purpose=purpose):
                self.assertFalse(_wants_bake(hook(purpose, remember=True)),
                                 "there is no workflow here to nest")

    def test_a_string_from_a_widget_still_counts(self):
        """Widget booleans arrive as "true" from some frontends."""
        for truthy in (True, "true", "True", "1", "on", "yes"):
            with self.subTest(v=truthy):
                self.assertTrue(_wants_bake(hook("make_workflow", remember=truthy)))
        for falsy in (False, "false", "0", "", None):
            with self.subTest(v=falsy):
                self.assertFalse(_wants_bake(hook("make_workflow", remember=falsy)))

    def test_the_purposes_that_read_the_switch_are_the_ones_the_node_shows_it_for(self):
        """qa and iterate produce nothing to keep, which is why it is hidden there."""
        for purpose in ("inline_parameter", "text", "general_request"):
            with self.subTest(purpose=purpose):
                h = hook(purpose, remember=True)
                self.assertFalse(_is_standin(h), "would be baked as a subgraph instead")
                self.assertFalse(_is_iterate(h))
                self.assertFalse(_is_qa(h))
        self.assertTrue(_is_qa(hook("qa")))
        self.assertTrue(_is_iterate(hook("iterate")))
        self.assertTrue(_is_standin(hook("make_workflow")))
        self.assertTrue(_is_text(hook("text")))
        self.assertTrue(_is_general(hook("general_request")))

    def test_a_make_workflow_hook_that_bakes_also_remembers_what_it_produced(self):
        """The subgraph is the recipe; the record is the results it already made.

        Keeping only the recipe would mean the most expensive hook on the canvas
        is the one that re-renders every time you open the graph.
        """
        h = hook("make_workflow", remember=True)
        self.assertTrue(_wants_bake(h))
        self.assertTrue(remembering(h), "bake implies keeping the outputs too")


class LegacyCanvasTest(unittest.TestCase):
    """A canvas saved before the merge sent two fields. Read them as they meant."""

    def test_bake_was_the_one_make_workflow_looked_at(self):
        self.assertTrue(remembering({"purpose": "make_workflow", "bake": True,
                                     "memorize": False}))
        self.assertTrue(_wants_bake({"purpose": "make_workflow", "bake": True}))

    def test_memorize_was_the_one_every_other_purpose_looked_at(self):
        self.assertTrue(remembering({"purpose": "text", "memorize": True,
                                     "bake": False}))
        self.assertFalse(remembering({"purpose": "text", "memorize": False,
                                      "bake": True}),
                         "`freeze: bake` rode along on every hook — it must reach nothing")

    def test_a_hook_that_says_nothing_at_all_keeps_nothing(self):
        self.assertFalse(remembering({}))
        self.assertFalse(_wants_bake({}))


class VocabularyTest(unittest.TestCase):
    """What the agent is told must match what the user sees on the node."""

    def test_the_prompt_no_longer_says_freeze_or_bake_to_canvas(self):
        from src.pipeline import _ORCH_PARTIALS_DIR
        text = (_ORCH_PARTIALS_DIR / "canvas_hooks.md").read_text(encoding="utf-8")
        self.assertNotIn("frozen", text)
        self.assertNotIn("`freeze`", text)
        self.assertNotIn("bake_to_canvas", text)

    def test_the_hook_block_describes_what_a_remembered_hook_already_did(self):
        from src.utils.canvas_hooks import describe_hooks
        h = hook("text")
        h["_cached"] = {"value": "warm sodium light", "targets": ["20"],
                        "outputs": [], "when": ""}
        block = describe_hooks([h], {})
        self.assertIn("ALREADY DONE", block)
        self.assertNotIn("if frozen", block)


if __name__ == "__main__":
    unittest.main()
