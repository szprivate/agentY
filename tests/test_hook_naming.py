"""Name a hook by something the user can actually see, and shut up until asked.

Two complaints, one cause: the hook block is written for the agent and leaks
straight through into what the user reads.

**Ids are invisible.** A ComfyUI canvas does not show node ids without going
looking for them, so an answer that says "hook 30" names something the user
cannot point at. What they CAN see is the node's title bar and the directive in
its box.

**The block is not a request.** The canvas is attached to every turn, so the
hooks arrive whether or not anyone asked for a run — and an agent that opens a
question with "Hook scope: 17 node(s)… I've reviewed all the canvas hooks" has
announced a run that nobody started.

    python -m unittest discover -s tests
"""

import unittest

from pipeline_stub import pipeline_stub, tools
from src.pipeline import Pipeline
from src.utils.canvas_hooks import (describe_hooks, hook_label, hook_name,
                                    hook_title)


def _GRAPH():
    return {"20": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}


def _hook(hid="30", purpose="text", title="", directive="write a caption for this"):
    return {"hook_node_id": str(hid), "purpose": purpose, "title": title,
            "directive": directive, "anchors": [], "targets": []}


class TitleTest(unittest.TestCase):

    def test_the_users_own_title_is_the_name(self):
        self.assertEqual(hook_title(_hook(title="Reference frames")), "Reference frames")

    def test_the_default_title_names_nothing(self):
        """Every hook on the canvas is called "agentY hook" — it identifies none."""
        for default in ("agentY hook", "AgentYHook", "agenty hook", "", "   "):
            with self.subTest(title=default):
                self.assertEqual(hook_title(_hook(title=default)), "")

    def test_a_missing_title_field_is_not_a_crash(self):
        self.assertEqual(hook_title({}), "")
        self.assertEqual(hook_title(None), "")

    def test_whitespace_is_normalised(self):
        self.assertEqual(hook_title(_hook(title="  Reference   frames \n")),
                         "Reference frames")

    def test_a_very_long_title_is_cut(self):
        self.assertLessEqual(len(hook_title(_hook(title="x" * 200))), 60)


class NameTest(unittest.TestCase):

    def test_an_untitled_hook_falls_back_to_what_it_visibly_says(self):
        """The directive is the text the user can read on the node itself."""
        self.assertEqual(hook_name(_hook(directive="one reference per character")),
                         "one reference per character")

    def test_a_title_wins_over_the_directive(self):
        got = hook_name(_hook(title="Reference frames",
                              directive="one reference per character"))
        self.assertEqual(got, "Reference frames")

    def test_a_long_directive_is_trimmed_to_something_glanceable(self):
        self.assertLessEqual(len(hook_name(_hook(directive="w " * 200))), 56)

    def test_the_label_carries_the_id_AND_the_name(self):
        """The id is what the tools take; the name is what the user can point at."""
        self.assertEqual(hook_label(_hook(hid="30", title="Reference frames")),
                         'hook 30 "Reference frames"')

    def test_a_hook_with_nothing_to_say_is_just_its_id(self):
        self.assertEqual(hook_label({"hook_node_id": "30"}), "hook 30")


class BlockTest(unittest.TestCase):

    def test_a_titled_hook_is_named_in_the_block(self):
        block = describe_hooks([_hook(title="Reference frames")], {})
        self.assertIn('"Reference frames"', block)
        self.assertIn("hook 30", block, "the id has to stay — it is what tools take")

    def test_an_untitled_hook_is_not_padded_with_its_own_directive(self):
        """Every line already quotes the directive; printing it twice is noise."""
        block = describe_hooks([_hook(directive="write a caption for this")], {})
        self.assertEqual(block.count("write a caption for this"), 1)

    def test_every_purpose_carries_the_title(self):
        for purpose in ("text", "inline_parameter", "make_workflow",
                        "general_request", "qa", "review"):
            with self.subTest(purpose=purpose):
                block = describe_hooks(
                    [_hook(purpose=purpose, title=f"My {purpose}")], {})
                self.assertIn(f'"My {purpose}"', block)

    def test_the_block_tells_the_agent_to_use_both(self):
        block = describe_hooks([_hook()], {})
        self.assertIn("a node id is not visible", block.lower())
        self.assertIn("give its name and its id", block)


class NotARunTest(unittest.TestCase):
    """The block arrives every turn; that is not a request to act on it."""

    def test_the_block_leads_with_which_kind_of_turn_this_is(self):
        block = describe_hooks([_hook()], {})
        head = block.split("]")[0]
        self.assertIn("REFERENCE, NOT AN INSTRUCTION", head)
        self.assertIn("FIRST decide which kind of turn this is", head)

    def test_it_says_plainly_not_to_open_with_an_inventory(self):
        block = describe_hooks([_hook()], {})
        self.assertIn("Do not open with an inventory", block)
        self.assertIn("do not 'review the workflow structure' first", block)

    def test_the_reason_is_given_not_just_the_rule(self):
        self.assertIn("reads as a run starting", describe_hooks([_hook()], {}))


class ScopeNoiseTest(unittest.TestCase):
    """"🎯 Hook scope: 17 node(s)…" in front of an ANSWER is a lie about state."""

    def test_the_note_is_held_rather_than_pushed_at_setup(self):
        pipe = pipeline_stub(_hook_scope_note="🎯 Hook scope: 17 node(s)…")
        self.assertEqual(pipe._hook_scope_note, "🎯 Hook scope: 17 node(s)…",
                         "still unpushed — nothing has run")

    def test_a_run_is_what_publishes_it(self):
        import asyncio
        pipe = pipeline_stub(_hook_scope_note="🎯 Hook scope: 17 node(s)…",
                             _canvas_base_prompt=_GRAPH())
        asyncio.run(tools(pipe)["apply_canvas_hooks"](resolutions=[]))
        self.assertEqual(pipe._hook_scope_note, "",
                         "consumed — a run started, so saying so is now true")

    def test_it_is_said_once_not_per_call(self):
        import asyncio
        pipe = pipeline_stub(_hook_scope_note="🎯 Hook scope: 17 node(s)…",
                             _canvas_base_prompt=_GRAPH())
        t = tools(pipe)["apply_canvas_hooks"]
        asyncio.run(t(resolutions=[]))
        asyncio.run(t(resolutions=[]))
        self.assertEqual(pipe._hook_scope_note, "")

    def test_a_canvas_that_was_not_scoped_has_nothing_to_say(self):
        pipe = pipeline_stub()
        self.assertEqual(pipe._hook_scope_note, "")


if __name__ == "__main__":
    unittest.main()
