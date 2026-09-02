"""Handing what a hook MADE to the hook wired after it.

Two runs from one session, both of which failed, and for different reasons. The
canvas in each: a screenplay (hook 3), a storyboard prompt (hook 25), reference
images (hook 6), and a video that wants the references and the storyboard as
image inputs (hook 37).

**Wired hook→hook.** Ordering was right and the run still failed, because a hook's
wire carries the value it AUTHORS — hook 6's three prompts — and never the three
images those prompts produced. Hook 37 was told its inputs were three strings and
that its IMAGE slot had "nothing here to choose from"; it supplied the id of the
node that generates references, which wires the generator in and runs it again.

**Wired through an agentY collector.** The dependency was invisible: hook 6 writes
into node 41 and hook 37 reads node 41, two facts nothing joined up. Hook 37 was
ordered FIRST and read an empty collector.

So: a hook's files are its product, the wiring says who is waiting for them, and
producers whose product is needed run this turn rather than being queued for after
it — which used to depend on the user writing the word "wait" in a directive.
"""

import json
import unittest
from unittest import mock

from src.utils import canvas_hooks as ch


def hook(hid, purpose="inline_parameter", directive="", anchors=(), targets=(),
         prev=()):
    """One hook as the frontend records it.

    The split matters and is easy to get wrong: hook→hook links go in
    ``prev_hook_ids``, real-node links in ``anchors`` with the type their output
    carries. So a hook fed only by other hooks has an EMPTY anchor list — which is
    exactly why its IMAGE target reported "nothing here to choose from" even though
    three wires ran into it.
    """
    return {
        "hook_node_id": str(hid), "purpose": purpose, "directive": directive,
        "prev_hook_ids": [str(x) for x in prev],
        "anchors": [{"node_id": str(a[0]), "from_output_type": a[1]} for a in anchors],
        "targets": [{"node_id": str(t[0]), "type": t[1], "to_input": t[2],
                     "to_input_type": t[3]} for t in targets],
    }


def screenplay():
    return hook(3, "text", "Write a screenplay, 5 seconds.",
                anchors=[(2, "IMAGE")],
                targets=[(6, "AgentYHook", "anchors.anchor0", "*"),
                         (25, "AgentYHook", "anchors.anchor0", "*"),
                         (37, "AgentYHook", "anchors.anchor2", "*")])


def storyboard():
    return hook(25, directive="Turn the screenplay into a storyboard prompt.",
                prev=[3],
                targets=[(24, "OpenAIGPTImageNodeV2", "prompt", "STRING"),
                         (37, "AgentYHook", "anchors.anchor1", "*")])


def refs(collector=False):
    targets = [(28, "OpenAIGPTImageNodeV2", "prompt", "STRING")]
    if collector:
        targets.append((41, "AgentYImageCollector", "files", "STRING"))
    else:
        targets.append((37, "AgentYHook", "anchors.anchor0", "*"))
    return hook(6, directive="Make a prompt for each character; one image per prompt.",
                prev=[3], targets=targets)


def video(anchors=(), prev=(6, 25, 3),
          directive="Collect the outputs from the previous passes and make a video."):
    return hook(37, directive=directive, anchors=anchors, prev=prev, targets=[
        (40, "ByteDance2ReferenceNodeV2", "model.reference_images.image_1", "IMAGE"),
        (40, "ByteDance2ReferenceNodeV2", "model.prompt", "STRING"),
    ])


def wired_directly():
    """The first attempt: hooks 3, 6, 25 wired straight into hook 37.

    Hook 37 has no anchors at all — every wire into it comes from a hook — so
    nothing in reach of its IMAGE input carries an image.
    """
    return [screenplay(), storyboard(), refs(), video()]


def through_collector():
    """The second attempt: hook 6 fills an agentY collector, hook 37 reads it."""
    return [screenplay(), storyboard(), refs(collector=True),
            video(anchors=[(41, "IMAGE")], prev=(25, 3))]


def order(hooks):
    return [str(h["hook_node_id"]) for h in ch._order_by_dependency(hooks)]


class TheCollectorNoLongerHidesTheWire(unittest.TestCase):
    """A dependency that runs through a plain node is still a dependency."""

    def test_the_producer_comes_before_the_hook_that_reads_its_collector(self):
        got = order(through_collector())
        self.assertLess(got.index("6"), got.index("37"),
                        f"hook 6 fills the collector hook 37 reads; got {got}")

    def test_the_logged_order_was_this_one(self):
        """Pinned as the regression it is: 3 → 25 → 37 → 6 is what shipped, and
        hook 37 read an empty collector because of it."""
        self.assertNotEqual(order(through_collector()), ["3", "25", "37", "6"])

    def test_the_producer_is_named_as_a_producer(self):
        self.assertIn("6", ch._producers_of(through_collector())["37"])

    def test_a_node_nobody_writes_to_creates_no_dependency(self):
        """The direction that matters: two hooks anchored on the same LoadImage
        are not thereby ordered. Inventing an edge is worse than missing one — it
        can serialise a graph, or make a cycle out of two independent branches."""
        a = hook(1, anchors=[(50, "IMAGE")], targets=[(60, "KSampler", "seed", "INT")])
        b = hook(2, anchors=[(50, "IMAGE")], targets=[(61, "KSampler", "seed", "INT")])
        self.assertEqual(ch._producers_of([a, b]), {"1": set(), "2": set()})

    def test_a_hook_reading_a_node_it_also_writes_is_not_its_own_producer(self):
        h = hook(1, anchors=[(41, "AgentYImageCollector")],
                 targets=[(41, "AgentYImageCollector", "files", "STRING")])
        self.assertEqual(ch._producers_of([h])["1"], set())


class TheMediaGap(unittest.TestCase):
    """A hook that must connect an image, with no image wired into it."""

    def test_it_is_found_when_every_anchor_carries_a_string(self):
        gaps = ch.media_gap(video(), {"3", "6", "25", "37"})
        self.assertEqual(len(gaps), 1)
        self.assertIn("40.model.reference_images.image_1", gaps[0])
        self.assertIn("IMAGE", gaps[0])

    def test_a_wired_image_is_not_a_gap(self):
        self.assertEqual(
            ch.media_gap(video(anchors=[(48, "IMAGE")]), {"3", "6", "25", "37"}), [])

    def test_a_string_target_is_never_a_gap(self):
        """`prompt` takes words. Only inputs that must carry a WIRE can be short
        of one."""
        ids = {"3", "6", "25", "37"}
        self.assertEqual(ch.media_targets(storyboard(), ids), [])
        self.assertEqual(ch.media_gap(storyboard(), ids), [])

    def test_feeding_another_hooks_anchor_is_not_a_media_input(self):
        """A hook's `out` carries any type, so hook 25 → hook 37's anchor looks
        exactly like an unfilled IMAGE slot. Counted as one, every chained hook
        claimed it was short of a file and the plan filled up with steps telling
        the agent to generate images for a wire that only carries a prompt."""
        self.assertIn("37.anchors.anchor1 (*)",
                      [f"{t['node_id']}.{t['to_input']} ({t['to_input_type']})"
                       for t in storyboard()["targets"]])
        self.assertEqual(ch.media_targets(storyboard(), {"3", "6", "25", "37"}), [])

    def test_an_unknown_anchor_type_is_given_the_benefit_of_the_doubt(self):
        """A type the frontend did not report is not evidence of a mismatch, and
        a false gap would force an inline run nobody needed."""
        self.assertEqual(ch.media_gap(video(anchors=[(48, "")]), {"3", "6", "25", "37"}), [])


class ProducersRunWhenTheirFilesAreNeeded(unittest.TestCase):
    """The rule that used to be spelled 'wait' in a directive."""

    def test_the_reference_maker_must_run_this_turn(self):
        self.assertIn("6", ch.gating_hook_ids(wired_directly()))

    def test_it_no_longer_depends_on_the_word_wait(self):
        """The logged run only worked as far as it did because the directive said
        "Wait for the other stages to finish". Delete that sentence and hook 6 was
        queued for after the turn, so its files never existed while hook 37 was
        being worked — silently."""
        plain = wired_directly()
        plain[-1]["directive"] = "Make a video from the references and the storyboard."
        self.assertFalse(ch.is_conditional(plain[-1]), "no conditional wording left")
        self.assertIn("6", ch.gating_hook_ids(plain))

    def test_it_holds_through_a_collector_too(self):
        self.assertIn("6", ch.gating_hook_ids(through_collector()))

    def test_a_text_hook_is_never_told_to_run(self):
        """place_canvas_text writes a string; there is no execution to run early,
        and sending it to apply_canvas_hooks can only answer "no batch"."""
        self.assertNotIn("3", ch.gating_hook_ids(wired_directly()))

    def test_a_graph_with_no_gap_and_no_condition_is_left_alone(self):
        """Every inline run serialises the turn. Only pay for it when the wiring
        says a later hook cannot proceed without the files."""
        self.assertEqual(ch.gating_hook_ids([screenplay(), storyboard()]), set())


class ThePlanSaysIt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = "\n".join(ch.plan_lines(wired_directly()))

    def test_it_names_the_input_that_cannot_be_filled(self):
        self.assertIn("40.model.reference_images.image_1", self.plan)

    def test_it_names_the_hooks_whose_files_would_fill_it(self):
        self.assertIn("hook 6, 25", self.plan)

    def test_the_screenplay_is_not_named_as_a_source_of_files(self):
        """Hook 3 is upstream of hook 37 and is a real dependency — for its WORDS.
        It writes a string with place_canvas_text and produces no file, so listing
        it here sends the agent to run a hook that has nothing to run."""
        self.assertNotIn("hook 3, 6, 25", self.plan)
        self.assertNotIn("3", ch.file_dependencies(wired_directly(), video()))

    def test_it_warns_against_the_mistake_that_was_made(self):
        """The run passed "28" — the id of the node that GENERATES references —
        which wires the generator into the video graph and runs it again."""
        self.assertIn("runs it again", self.plan)

    def test_a_plan_exists_even_with_no_conditional_directive(self):
        """It used to return [] unless some directive said "wait" or "stop"."""
        plain = wired_directly()
        plain[-1]["directive"] = "Make a video from the references."
        self.assertTrue(ch.plan_lines(plain))

    def test_the_steps_are_numbered_once_each(self):
        """They were hardcoded, so two conditional hooks printed two step 4s."""
        import re
        two = wired_directly()
        two[1]["directive"] = "Only continue once the storyboard exists."
        nums = [int(m) for m in re.findall(r"^  (\d+)\. ", "\n".join(ch.plan_lines(two)), re.M)]
        self.assertEqual(nums, list(range(1, len(nums) + 1)))


class TheConnectionHint(unittest.TestCase):
    """What the block says at the input that has nothing to choose from."""

    @classmethod
    def setUpClass(cls):
        cls.line = ch._target_context(video(), {"3", "6", "25", "37"})

    def test_it_asks_for_a_file(self):
        self.assertIn("FILE PATH", self.line)

    def test_it_does_not_lead_with_supply_a_node_id(self):
        """The old wording opened with "supply a node id, not a value" and buried
        "or a file" at the end. The agent did exactly what it was told first."""
        self.assertNotIn("supply a node id, not a value", self.line)

    def test_it_warns_that_a_generator_id_re_runs_the_generator(self):
        self.assertIn("run again", self.line)

    def test_a_hook_with_a_fitting_anchor_still_gets_the_node_id_offer(self):
        """Unchanged where it was right: an image IS wired in, and choosing
        between the wired ones is exactly what a node id is for."""
        line = ch._target_context(video(anchors=[(48, "IMAGE"), (47, "IMAGE")]),
                                  {"3", "6", "25", "37"})
        self.assertIn("connect one of 48, 47", line)


class TheProductLedger(unittest.TestCase):
    """What the pipeline records when a hook's stage finishes."""

    def setUp(self):
        from src.pipeline import Pipeline
        self.p = Pipeline.__new__(Pipeline)
        self.p._hook_products = {}
        self.p._canvas_hooks = wired_directly()

    def test_it_records_the_files_against_the_hook(self):
        rec = self.p._record_hook_products("6", ["/out/a.png", "/out/b.png"])
        self.assertEqual(self.p._hook_products["6"], ["/out/a.png", "/out/b.png"])
        self.assertEqual(rec["files"], ["/out/a.png", "/out/b.png"])

    def test_it_names_the_hook_waiting_for_them(self):
        self.assertEqual(self.p._record_hook_products("6", ["/out/a.png"])["waiting"], ["37"])

    def test_a_second_batch_adds_to_the_first(self):
        """A hook can run more than once in a turn — a QA retry, a repaired
        member — and the later files are as much its product as the earlier."""
        self.p._record_hook_products("6", ["/a.png"])
        self.p._record_hook_products("6", ["/b.png"])
        self.assertEqual(self.p._hook_products["6"], ["/a.png", "/b.png"])

    def test_a_run_with_no_hook_records_nothing(self):
        """An ordinary workflow run is not a hook's product, and filing it under
        "" would hand it to whichever hook asked next."""
        self.assertEqual(self.p._record_hook_products("", ["/a.png"]), {})
        self.assertEqual(self.p._hook_products, {})

    def test_a_hook_that_produced_nothing_records_nothing(self):
        self.assertEqual(self.p._record_hook_products("6", []), {})

    def test_a_broken_hook_list_does_not_break_the_run(self):
        """The note is a courtesy on the end of a successful generation. It must
        never be the thing that turns one into an error."""
        self.p._canvas_hooks = "not a list at all"
        with mock.patch("src.utils.canvas_hooks.product_consumers",
                        side_effect=RuntimeError("boom")):
            rec = self.p._record_hook_products("6", ["/a.png"])
        self.assertEqual(rec["waiting"], [])


class TheNoteInTheToolResult(unittest.TestCase):
    """Where the hand-off is actually delivered.

    Not the [CANVAS HOOKS] block: that is built once, before anything runs, so by
    the time hook 37 is reached its description of these inputs predates the files.
    """

    def setUp(self):
        from src.pipeline import Pipeline
        self.p = Pipeline.__new__(Pipeline)
        self.p._hook_products = {}
        self.p._canvas_hooks = wired_directly()

    def note(self, hook_id="6", files=("/out/a.png",)):
        return self.p._products_note(self.p._record_hook_products(hook_id, list(files)))

    def test_it_names_the_waiting_hook(self):
        self.assertIn("hook 37", self.note())

    def test_it_says_to_pass_the_paths(self):
        self.assertIn("PATHS", self.note())

    def test_it_says_not_to_pass_the_generator_id(self):
        self.assertIn("generated them", self.note())

    def test_nothing_is_said_when_nobody_is_waiting(self):
        """Hook 37 is last. A note telling the agent to hand its files onward
        would be inventing a stage that does not exist."""
        self.assertEqual(self.note("37"), "")

    def test_nothing_is_said_for_a_run_with_no_hook(self):
        self.assertEqual(self.p._products_note({}), "")


if __name__ == "__main__":
    unittest.main()
