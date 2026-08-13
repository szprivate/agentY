"""The UNRESOLVED block: what the turn is about to assume, said before it acts.

The RUN PLAN already says what will happen. This says what is open about it —
computed from the wiring, the directives and the project's own memory, so it costs
nothing at run time and cannot disagree with the graph. It is deliberately silent
on an ordinary turn: a block that appears every time stops being read by the third.

    python -m unittest tests.test_hook_sitrep
"""

import unittest
from unittest import mock

from src.utils.canvas_hooks import describe_hooks, sitrep_lines


def _no_project_memory():
    """describe_hooks reads the live project store; pin it so these tests assert
    the code and not whatever ComfyUI happens to be serving on this machine."""
    return mock.patch("src.utils.canvas_hooks._project_memory_names", return_value=[])


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    return {"43": _node("LoadImage", image="ref.png", upload="image"),
            "242": _node("OpenAIGPTImageNodeV2", prompt="old"),
            "250": _node("SaveImage", images=["242", 0])}


def _hook(hid, directive, anchors=(), targets=(), purpose="inline_parameter"):
    return {"hook_node_id": str(hid), "purpose": purpose, "directive": directive,
            "anchors": [{"node_id": str(a), "type": "LoadImage", "widgets": {}} for a in anchors],
            "targets": [{"node_id": str(t), "to_input": "prompt", "to_input_type": "STRING",
                         "type": "OpenAIGPTImageNodeV2"} for t in targets]}


def _text(hid, directive, **kw):
    return _hook(hid, directive, purpose="text", **kw)


class SilenceTests(unittest.TestCase):
    def test_an_ordinary_hook_says_nothing(self):
        hooks = [_hook(5, "write a prompt for a portrait", anchors=[43], targets=[242])]
        self.assertEqual(sitrep_lines(hooks, _canvas(), known=[]), [])

    def test_no_hooks_no_block(self):
        self.assertEqual(sitrep_lines([], _canvas(), known=[]), [])

    def test_it_stays_out_of_the_hooks_block_when_it_has_nothing_to_add(self):
        hooks = [_hook(5, "write a prompt for a portrait", anchors=[43], targets=[242])]
        with _no_project_memory():
            self.assertNotIn("UNRESOLVED", describe_hooks(hooks, _canvas()))


class ChainOnlyTests(unittest.TestCase):
    """The 18:18 failure, stated up front instead of discovered by running it."""

    def test_a_producer_that_reaches_no_renderer_is_called_out(self):
        hooks = [_hook(44, "analyse the image and write a STYLEGUIDE", anchors=[43], targets=[27]),
                 _hook(27, "break the story into shots", targets=[242])]
        hooks[0]["targets"][0]["type"] = "AgentYHook"
        text = "\n".join(sitrep_lines(hooks, _canvas(), known=[]))
        self.assertIn("hook 44 feeds only hook 27", text)
        self.assertIn("nothing it produces reaches a node that renders", text)
        self.assertIn("written value, not a generation", text)

    def test_a_text_hook_is_not_called_out_for_it(self):
        # A text hook is *supposed* to produce writing; saying so adds nothing.
        hooks = [_text(4, "extract the characters", anchors=[43], targets=[27]),
                 _hook(27, "shots", targets=[242])]
        hooks[0]["targets"][0]["type"] = "AgentYHook"
        self.assertEqual(sitrep_lines(hooks, _canvas(), known=[]), [])


class DanglingHookTests(unittest.TestCase):
    def test_a_hook_wired_to_nothing_at_all(self):
        text = "\n".join(sitrep_lines([_hook(9, "make it cinematic")], _canvas(), known=[]))
        self.assertIn("hook 9 has nothing wired in and nothing wired out", text)
        self.assertIn("wire an anchor", text)

    def test_a_standin_needs_neither(self):
        # make_workflow with no anchor is text-to-media — the documented use.
        hooks = [_hook(9, "a slow dolly through fog", purpose="make_workflow")]
        self.assertEqual(sitrep_lines(hooks, _canvas(), known=[]), [])


class ConditionalTests(unittest.TestCase):
    def test_a_condition_with_nothing_to_wait_on(self):
        hooks = [_hook(30, "Wait for the references. If ANY failed - STOP.",
                       anchors=[43], targets=[242])]
        text = "\n".join(sitrep_lines(hooks, _canvas(), known=[]))
        self.assertIn("hook 30 waits on how something turns out", text)
        self.assertIn("nothing to wait for", text)

    def test_a_condition_with_a_producer_is_fine(self):
        hooks = [_hook(5, "write three prompts", anchors=[43], targets=[30]),
                 _hook(30, "Wait for all of them. If ANY failed - STOP.", targets=[242])]
        hooks[0]["targets"][0]["type"] = "AgentYHook"
        text = "\n".join(sitrep_lines(hooks, _canvas(), known=[]))
        self.assertNotIn("nothing to wait for", text)


class ProjectMemoryTests(unittest.TestCase):
    """The continuity failure: writing your own hero when the project has one."""

    def test_a_directive_naming_a_stored_entry_is_flagged(self):
        hooks = [_hook(5, "put the hero on the dock at night", anchors=[43], targets=[242])]
        text = "\n".join(sitrep_lines(hooks, _canvas(), known=["hero", "grade"]))
        self.assertIn('hook 5 mentions "hero"', text)
        self.assertIn('project_memory_read("hero")', text)
        self.assertNotIn("grade", text, "only what the directive actually names")

    def test_a_multi_word_entry_matches_the_phrase_as_written(self):
        # The entry is "alley-night"; nobody types that. They type the words.
        hooks = [_hook(5, "shoot it in the alley at night", targets=[242], anchors=[43])]
        text = "\n".join(sitrep_lines(hooks, _canvas(), known=["alley-night"]))
        self.assertIn('mentions "alley-night"', text)

    def test_a_multi_word_entry_needs_all_of_its_words(self):
        hooks = [_hook(5, "shoot it in the alley", targets=[242], anchors=[43])]
        self.assertEqual(sitrep_lines(hooks, _canvas(), known=["alley-night"]), [])

    def test_it_does_not_fire_inside_another_word(self):
        # "grade" inside "upgraded" is exactly the wrong that makes a block ignorable.
        hooks = [_hook(5, "the upgraded pipeline handles it", anchors=[43], targets=[242])]
        self.assertEqual(sitrep_lines(hooks, _canvas(), known=["grade"]), [])

    def test_no_project_memory_means_no_lines(self):
        hooks = [_hook(5, "put the hero on the dock", anchors=[43], targets=[242])]
        self.assertEqual(sitrep_lines(hooks, _canvas(), known=[]), [])


class BlockTests(unittest.TestCase):
    def setUp(self):
        p = _no_project_memory()
        p.start()
        self.addCleanup(p.stop)

    def test_it_reaches_the_block_the_agent_reads(self):
        hooks = [_hook(9, "make it cinematic")]
        block = describe_hooks(hooks, _canvas())
        self.assertIn("UNRESOLVED", block)
        self.assertIn("hook 9 has nothing wired in", block)

    def test_the_run_plan_still_comes_first(self):
        hooks = [_hook(5, "write three prompts", anchors=[43], targets=[30]),
                 _hook(30, "Wait for all of them. If ANY failed - STOP.")]
        hooks[0]["targets"][0]["type"] = "AgentYHook"
        block = describe_hooks(hooks, _canvas())
        self.assertLess(block.index("RUN PLAN"), block.index("UNRESOLVED"))


if __name__ == "__main__":
    unittest.main()
