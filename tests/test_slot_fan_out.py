"""One wire, one slot, several images — and how many runs that means.

Reported from a real session. A hook's output was wired into ONE numbered slot,
``model.reference_images.image_1`` on a Seedance node, and its directive said to
use ALL the generated references in a SINGLE video. The agent handed over three
paths, the batch builder read the only shape it had — a list of values for one
input is a product axis — and queued three paid video generations with one
reference each. `notes` was empty: from the builder's side nothing unusual
happened.

The apparatus for exactly this already existed (`expand_image_batches` inserts an
`agentY expand image batch` and fans a batch across the slots) but every part of
it is gated on the source being a COLLECTOR NODE on the graph. Handed bare paths,
none of it fires.

So there are two readings of "several values at one numbered slot", both
legitimate — N runs with one image each, or one run using all N — and they differ
by N paid generations. `fan_out` expresses the second, `sweep: true` the first,
and being given neither is refused rather than guessed.
"""

import unittest
from unittest import mock

from src.utils import canvas_hooks as ch

# A model node with reference slots, written as the graph holds it. The class name
# is deliberately one no ComfyUI has: `slot_family` prefers the live schema, so a
# real class would make these tests read whatever the machine running them happens
# to have installed — the first draft asked for a 3-slot node and got the 30-slot
# Seedance schema off the developer's own ComfyUI. The graph's own sibling slots
# are the fallback, and testing the fallback is what keeps this hermetic; the
# schema path gets its own test, with the schema mocked.
NODE = "40"
CLASS = "TestReferenceNodeV2"


def graph(slots=3, prompt="a prompt"):
    inputs = {f"model.reference_images.image_{i}": None for i in range(1, slots + 1)}
    inputs["model.prompt"] = prompt
    return {NODE: {"class_type": CLASS, "inputs": inputs}}


def res(param="model.reference_images.image_1", values=("/a.png", "/b.png", "/c.png"),
        **over):
    return {"target_node_id": NODE, "param": param, "values": list(values), **over}


# The hook whose output lands on image_1 — where the declared IMAGE type comes
# from. The frontend records it, and it is what tells the guard this input takes a
# wire without having to ask ComfyUI.
HOOK = {"hook_node_id": "37", "targets": [
    {"node_id": NODE, "type": CLASS, "to_input": "model.reference_images.image_1",
     "to_input_type": "IMAGE"},
    {"node_id": NODE, "type": CLASS, "to_input": "model.reference_images.image_3",
     "to_input_type": "IMAGE"},
    {"node_id": NODE, "type": CLASS, "to_input": "model.prompt",
     "to_input_type": "STRING"}]}


def build(base, resolutions):
    labels: list = []
    prompts, notes = ch.build_batch(
        base, resolutions, cap=32, labels=labels,
        connection_inputs={f"{NODE}.model.reference_images.image_1"})
    return prompts, labels, notes


class TheSlotFamily(unittest.TestCase):
    """Knowing image_2 exists is the whole licence for wiring one."""

    def test_it_reads_the_siblings_the_graph_shows(self):
        self.assertEqual(
            ch.slot_family(graph(3), NODE, "model.reference_images.image_1"),
            ["model.reference_images.image_1", "model.reference_images.image_2",
             "model.reference_images.image_3"])

    def test_the_schema_is_preferred_when_it_can_be_asked(self):
        """ComfyUI knows the whole declared family, including slots the graph has
        not opened yet — thirty of them on the node in the report, where the graph
        showed one."""
        with mock.patch.object(ch, "autogrow_slots", return_value={
                "model.reference_images": [f"image_{i}" for i in range(1, 9)]}):
            fam = ch.slot_family(graph(1), NODE, "model.reference_images.image_1")
        self.assertEqual(len(fam), 8)

    def test_an_unnumbered_input_has_no_family(self):
        self.assertEqual(ch.slot_family(graph(), NODE, "model.prompt"), [])

    def test_an_unknown_node_has_no_family(self):
        self.assertEqual(ch.slot_family(graph(), "999", "image_1"), [])

    def test_a_schema_that_cannot_be_asked_is_not_an_error(self):
        """ComfyUI down is the normal case for half the tests in this repo, and
        the honest answer is the graph's own slots — never an invented name."""
        with mock.patch.object(ch, "autogrow_slots", side_effect=OSError("down")):
            self.assertEqual(len(ch.slot_family(graph(2), NODE,
                                                "model.reference_images.image_1")), 2)

    def test_slots_are_ordered_by_number_not_by_text(self):
        """image_10 sorts before image_2 as a string, and a fan that filled the
        slots in that order would put the second reference in the tenth slot."""
        fam = ch.slot_family(graph(12), NODE, "model.reference_images.image_1")
        self.assertEqual(fam[1].rpartition(".")[2], "image_2")
        self.assertEqual(fam[-1].rpartition(".")[2], "image_12")


class FanningAcrossTheSlots(unittest.TestCase):
    def test_three_images_become_one_run(self):
        """The whole point. Three references, one video — not three videos."""
        prompts, labels, _notes = build(graph(3), [res(mode="fan_out")])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(labels[0], {
            f"{NODE}.model.reference_images.image_1": "/a.png",
            f"{NODE}.model.reference_images.image_2": "/b.png",
            f"{NODE}.model.reference_images.image_3": "/c.png"})

    def test_without_it_the_same_call_is_three_runs(self):
        """The behaviour that produced the report, kept as the contrast: a plain
        value_list on one input is still an ordinary sweep."""
        prompts, _labels, _notes = build(graph(3), [res(sweep=True)])
        self.assertEqual(len(prompts), 3)

    def test_it_says_what_it_did(self):
        _p, _l, notes = build(graph(3), [res(mode="fan_out")])
        said = " ".join(notes)
        self.assertIn("ONE run", said)
        self.assertIn("not 3 runs", said)

    def test_more_values_than_slots_are_reported_not_dropped_quietly(self):
        prompts, labels, notes = build(graph(2), [res(mode="fan_out")])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(len(labels[0]), 2)
        self.assertIn("used the first 2", " ".join(notes))

    def test_it_starts_from_the_slot_the_wire_landed_on(self):
        """A hook wired into image_2 fills image_2 onward, not image_1 — the user
        put the wire where they wanted the first one."""
        _p, labels, _n = build(graph(4), [res(param="model.reference_images.image_2",
                                              values=("/a.png", "/b.png"),
                                              mode="fan_out")])
        self.assertEqual(labels[0], {
            f"{NODE}.model.reference_images.image_2": "/a.png",
            f"{NODE}.model.reference_images.image_3": "/b.png"})

    def test_it_still_crosses_with_other_sweeps(self):
        """A fan is one run's worth of references, not a promise that the batch
        has one member: two prompts over the same three references is two runs."""
        prompts, _l, _n = build(graph(3), [
            res(mode="fan_out"),
            {"target_node_id": NODE, "param": "model.prompt", "values": ["x", "y"]}])
        self.assertEqual(len(prompts), 2)

    def test_an_unreadable_family_does_not_silently_sweep(self):
        """The failure mode to avoid is answering an ambiguous request with the
        expensive reading. With no slots to fan across, one run is produced and
        the shortfall is stated — never N runs nobody asked for."""
        with mock.patch.object(ch, "autogrow_slots", return_value={}):
            prompts, labels, notes = build(graph(1), [res(mode="fan_out")])
        self.assertEqual(len(prompts), 1, "one run, never three")
        self.assertEqual(len(labels[0]), 1)
        self.assertIn("2 could not be placed", " ".join(notes))

    def test_a_single_placed_value_is_not_described_as_a_spread(self):
        """"spread 1 value … not 1 runs using one each" reads as a bug in the
        receipt, and the shortfall note above it already says what happened."""
        with mock.patch.object(ch, "autogrow_slots", return_value={}):
            _p, _l, notes = build(graph(1), [res(mode="fan_out")])
        self.assertNotIn("not 1 runs", " ".join(notes))

    def test_fan_out_with_no_values_is_skipped_not_crashed(self):
        prompts, _l, notes = build(graph(3), [res(values=(), mode="fan_out")])
        self.assertIn("nothing to spread", " ".join(notes))
        self.assertEqual(prompts, [])


class TheAmbiguityIsAskedNotGuessed(unittest.TestCase):
    def flag(self, base, resolutions, hooks=None):
        return ch.ambiguous_slot_sweeps(base, hooks if hooks is not None else [HOOK],
                                        resolutions)

    def test_it_stays_quiet_when_nothing_declares_the_input_type(self):
        """No hook target and no schema means no evidence this input takes a wire,
        and a refusal on a guess is worse than the sweep it would prevent — a
        prompt sweep bounced for no reason is a run the user cannot get started."""
        self.assertEqual(self.flag(graph(3), [res()], hooks=[]), [])

    def test_several_images_on_one_numbered_slot_is_flagged(self):
        problems = self.flag(graph(3), [res()])
        self.assertEqual(len(problems), 1)
        self.assertIn("3 separate runs", problems[0])
        self.assertIn("fan_out", problems[0])

    def test_fan_out_has_already_answered(self):
        self.assertEqual(self.flag(graph(3), [res(mode="fan_out")]), [])

    def test_sweep_true_has_already_answered(self):
        """The escape hatch has to exist: one video per character is a real thing
        to want, and a guard with no way past it is a guard people route around."""
        self.assertEqual(self.flag(graph(3), [res(sweep=True)]), [])

    def test_one_value_is_never_ambiguous(self):
        self.assertEqual(self.flag(graph(3), [res(values=("/a.png",))]), [])

    def test_a_text_input_is_never_ambiguous(self):
        """Two prompts on `prompt` is a sweep and always was; there is no second
        reading to ask about."""
        self.assertEqual(
            self.flag(graph(3), [res(param="model.prompt", values=("x", "y"))]), [])

    def test_the_last_slot_has_nowhere_to_fan_to(self):
        """Nothing free beside it, so the only reading left IS a sweep — asking
        would be a question with one answer."""
        self.assertEqual(
            self.flag(graph(3), [res(param="model.reference_images.image_3")]), [])

    def test_a_seed_sweep_is_left_alone(self):
        self.assertEqual(
            self.flag(graph(3), [{"target_node_id": NODE, "param": "model.prompt",
                                  "mode": "sweep_seed", "count": 4}]), [])


class TheBlockNamesTheFamily(unittest.TestCase):
    """Nothing used to say the slot had siblings.

    From the agent's side there was one slot, so several values for it could only
    mean several runs. It was not ignoring the directive; it had no verb for what
    the directive asked.
    """

    def line(self, slots=3):
        hook = {"hook_node_id": "37", "targets": [
            {"node_id": NODE, "type": CLASS,
             "to_input": "model.reference_images.image_1", "to_input_type": "IMAGE"}]}
        return ch._target_context(hook, {"37"}, graph(slots))

    def test_it_says_how_many_slots_there_are(self):
        self.assertIn("one of 3 numbered slots", self.line())

    def test_it_names_the_free_ones_as_a_range(self):
        """As a range, not a list: the node in the report declares thirty, and
        spelling them out put a paragraph into a block re-sent every turn."""
        line = self.line(12)
        self.assertIn("image_2 … image_12", line)
        self.assertNotIn("image_7,", line)

    def test_it_says_what_several_values_would_mean(self):
        self.assertIn("several RUNS", self.line())

    def test_it_offers_the_way_out(self):
        self.assertIn("fan_out", self.line())

    def test_a_lone_slot_says_nothing(self):
        """No siblings, no choice, no line — the block is re-sent every turn and
        pays for every word."""
        self.assertNotIn("numbered slots", self.line(1))


class NothingElseChanged(unittest.TestCase):
    """The regression surface, written down.

    Two things here can bounce or reshape a batch that used to run: the fan-out
    rewrite sits in front of every build_batch call, and the ambiguity guard
    refuses before queueing. Both are new gates on a path every canvas run goes
    through, so what they must NOT touch is worth pinning harder than what they do.
    """

    def test_a_batch_with_no_fan_is_untouched(self):
        """The rewrite runs on every call. With no fan mode it must be the
        identity — same variants, same labels, and no notes invented."""
        rs = [{"target_node_id": NODE, "param": "model.prompt",
               "mode": "value_list", "values": ["a", "b", "c"]}]
        prompts, labels, notes = build(graph(3), rs)
        self.assertEqual(len(prompts), 3)
        self.assertEqual([l[f"{NODE}.model.prompt"] for l in labels], ["a", "b", "c"])
        self.assertEqual(notes, [])

    def test_a_seed_sweep_still_sweeps(self):
        prompts, _l, _n = build(graph(3), [{"target_node_id": NODE,
                                            "param": "model.prompt",
                                            "mode": "sweep_seed", "count": 5}])
        self.assertEqual(len(prompts), 5)

    def test_a_zip_still_zips(self):
        """The pairing case, and the one the guard has to keep its hands off: a
        zip already says these advance together, one value each per run. Bouncing
        it would refuse a batch that had ALREADY said which reading it meant."""
        prompts, labels, _n = build(graph(3), [
            {"target_node_id": NODE, "param": "model.reference_images.image_1",
             "values": ["/a.png", "/b.png"], "zip_group": "run"},
            {"target_node_id": NODE, "param": "model.prompt",
             "values": ["x", "y"], "zip_group": "run"}])
        self.assertEqual(len(prompts), 2)
        self.assertEqual(labels[0][f"{NODE}.model.prompt"], "x")
        self.assertEqual(ch.ambiguous_slot_sweeps(graph(3), [HOOK], [
            {"target_node_id": NODE, "param": "model.reference_images.image_1",
             "values": ["/a.png", "/b.png"], "zip_group": "run"}]), [])

    def test_a_cartesian_product_still_crosses(self):
        prompts, _l, _n = build(graph(3), [
            {"target_node_id": NODE, "param": "model.prompt", "values": ["x", "y"]},
            {"target_node_id": NODE, "param": "model.reference_images.image_1",
             "values": ["/a.png", "/b.png"], "sweep": True}])
        self.assertEqual(len(prompts), 4)

    def test_one_value_on_a_numbered_slot_is_never_touched(self):
        """The overwhelmingly common case — one reference, one run — must not
        acquire a note, a bounce, or an extra variant."""
        prompts, _l, notes = build(graph(3), [res(values=("/a.png",))])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(notes, [])
        self.assertEqual(ch.ambiguous_slot_sweeps(graph(3), [HOOK],
                                                  [res(values=("/a.png",))]), [])

    def test_the_guard_asks_for_no_schema_it_does_not_need(self):
        """`slot_family` is called per target while rendering the block, which is
        built every turn. A param with no number must return before asking ComfyUI
        at all — otherwise a graph of ordinary widgets pays an HTTP round trip per
        input to be told there is no family."""
        with mock.patch.object(ch, "autogrow_slots") as asked:
            ch.slot_family(graph(3), NODE, "model.prompt")
        asked.assert_not_called()

    def test_a_non_wire_numbered_widget_gets_no_fan_advice(self):
        """A LoRA stacker's `lora_1`, `lora_2` is a numbered family too, and
        offering fan_out on it would advertise a behaviour nothing here was
        designed against. The hint belongs to inputs that take a WIRE."""
        base = {NODE: {"class_type": CLASS,
                       "inputs": {"lora_1": "", "lora_2": "", "lora_3": ""}}}
        hook = {"hook_node_id": "9", "targets": [
            {"node_id": NODE, "type": CLASS, "to_input": "lora_1",
             "to_input_type": "STRING"}]}
        line = ch._target_context(hook, {"9"}, base)
        self.assertNotIn("fan_out", line)
        self.assertNotIn("numbered slots", line)

    def test_running_the_canvas_as_it_stands_is_not_gated(self):
        """`resolutions=[]` means "run the graph exactly as it is" — every hook
        answered from memory, nothing to sweep. There are no values to be
        ambiguous about, and a guard that fired here would block the one call that
        cannot be wrong."""
        self.assertEqual(ch.ambiguous_slot_sweeps(graph(3), [HOOK], []), [])
        self.assertEqual(ch.ambiguous_slot_sweeps(graph(3), [HOOK], None), [])

    def test_junk_resolutions_do_not_raise(self):
        """Both new gates run before the build's own validation, so they see
        whatever the model sent — including things the builder would have rejected."""
        for junk in ([None], ["nope"], [{}], [{"param": "x"}],
                     [{"target_node_id": NODE}], [{"target_node_id": NODE,
                                                   "param": "image_1"}]):
            with self.subTest(junk=junk):
                self.assertIsInstance(
                    ch.ambiguous_slot_sweeps(graph(3), [HOOK], junk), list)
                self.assertIsInstance(ch._expand_fan_outs(graph(3), junk), tuple)


if __name__ == "__main__":
    unittest.main()
