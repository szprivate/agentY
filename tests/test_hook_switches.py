"""One switch for "make it permanent", read differently by purpose.

The hook node carried three booleans — `bake_to_canvas`, `freeze`, `memorize` —
and they read as three versions of one idea. Two of them were: `bake_to_canvas`
and `freeze` ask the same question ("do I want a graph I can re-run without the
agent?") of two different products, and are never both applicable. They are now
one `bake` switch, resolved by purpose on this side.

`memorize` is a different axis and stays its own switch: it is about paying for
an answer twice, not about permanence, and a hook can reasonably be both.

The wire keeps two fields because the two consumers are in different places, and
this pins down that a single switch still reaches both of them — and, more
importantly, that neither one fires for a purpose that does not read it.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import (_is_general, _is_iterate, _is_qa, _is_standin,
                                    _is_text, _wants_bake)


def hook(purpose, **kw):
    """A hook as the panel sends it: one switch, mirrored onto both fields."""
    bake = kw.pop("bake", False)
    h = {"hook_node_id": "5", "purpose": purpose, "directive": "do the thing",
         "anchors": [], "targets": [], "bake": bake, "freeze": bake}
    h.update(kw)
    return h


class OneSwitchTest(unittest.TestCase):

    def test_bake_on_a_make_workflow_hook_asks_for_a_subgraph(self):
        self.assertTrue(_wants_bake(hook("make_workflow", bake=True)))
        self.assertFalse(_wants_bake(hook("make_workflow", bake=False)))

    def test_the_same_switch_freezes_a_text_hook_s_value(self):
        """place_canvas_text reads `freeze`; the panel sets it from the one switch."""
        self.assertTrue(hook("text", bake=True)["freeze"])
        self.assertFalse(hook("text", bake=False)["freeze"])

    def test_a_string_from_a_widget_still_counts(self):
        """Widget booleans arrive as "true" from some frontends."""
        for truthy in (True, "true", "True", "1", "on", "yes"):
            with self.subTest(v=truthy):
                self.assertTrue(_wants_bake({"bake": truthy}))
        for falsy in (False, "false", "0", "", None):
            with self.subTest(v=falsy):
                self.assertFalse(_wants_bake({"bake": falsy}))

    def test_baking_is_only_ever_offered_for_make_workflow_hooks(self):
        """`freeze` riding along on a make_workflow hook must reach nothing.

        The panel sets both fields from one switch, which is only safe because
        each consumer is gated by purpose on this side.
        """
        h = hook("make_workflow", bake=True)
        self.assertTrue(_is_standin(h))          # baked as a subgraph
        self.assertFalse(_is_general(h))         # never place_canvas_text'd
        self.assertFalse(_is_qa(h))

    def test_the_purposes_that_read_a_switch_are_the_ones_the_node_shows_it_for(self):
        """The node hides a switch its purpose does not read — same list, both sides.

        `bake` is shown for make_workflow (subgraph) and the place_canvas_text
        purposes (freeze the value); `memorize` only for the latter. qa and
        iterate read neither, which is why both are hidden there.
        """
        for purpose in ("inline_parameter", "text", "general_request"):
            with self.subTest(purpose=purpose):
                h = hook(purpose, bake=True)
                self.assertFalse(_is_standin(h), "would be baked as a subgraph instead")
                self.assertFalse(_is_iterate(h))
                self.assertFalse(_is_qa(h))

        # Neither consumer exists for these two: _wants_bake is only ever asked of
        # standin hooks, and `freeze` is only read on the place_canvas_text path.
        for purpose in ("qa", "iterate"):
            with self.subTest(purpose=purpose):
                h = hook(purpose, bake=True)
                self.assertFalse(_is_standin(h))     # never reaches _wants_bake
                self.assertFalse(_is_text(h))        # never reaches place_canvas_text
                self.assertFalse(_is_general(h))
        self.assertTrue(_is_qa(hook("qa")))
        self.assertTrue(_is_iterate(hook("iterate")))

    def test_a_hook_can_be_both_memorized_and_baked(self):
        """The reason memorize stayed a separate switch."""
        h = hook("text", bake=True, memorize=True)
        self.assertTrue(h["freeze"])
        self.assertTrue(h["memorize"])


class VocabularyTest(unittest.TestCase):
    """What the agent is told must match what the user sees on the node."""

    def test_the_prompt_no_longer_says_freeze_or_bake_to_canvas(self):
        from pathlib import Path
        from src.pipeline import _ORCH_PARTIALS_DIR
        text = (_ORCH_PARTIALS_DIR / "canvas_hooks.md").read_text(encoding="utf-8")
        self.assertNotIn("frozen", text)
        self.assertNotIn("`freeze`", text)
        self.assertNotIn("bake_to_canvas", text)
        self.assertIn("`bake`", text)

    def test_the_hook_block_says_bake_too(self):
        from src.utils.canvas_hooks import describe_hooks
        block = describe_hooks([hook("text", bake=False)], {})
        self.assertIn("'bake' switch", block)
        self.assertNotIn("if frozen", block)


if __name__ == "__main__":
    unittest.main()
