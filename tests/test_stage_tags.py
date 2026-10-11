"""A tag on a stage's output names that stage's result.

A tag (`agentY add tag`) named the node up its wire - a loader with a file in
it. Put on the output of a review, to name "the approved apartment reference",
it named nothing: the review is spliced out of the graph for a run, the tag was
left on no wire, and `#appartmentref` in a later directive was reported as "not
wired to anything". It now names what the host has on record for that stage.
"""
import unittest

from src.utils import canvas_hooks as ch
from src.utils import run_ledger as rl


def graph():
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "face.png"}},
        "2": {"class_type": "AgentYRefNote", "inputs": {"input": ["1", 0], "tag": "hero face"}},
        "118": {"class_type": "AgentYHook", "inputs": {"directive": "main location"}},
        "26": {"class_type": "Gen", "inputs": {"prompt": ["118", 0]}},
        "31": {"class_type": "SaveImage", "inputs": {"images": ["26", 0]}},
        "136": {"class_type": "AgentYReview", "inputs": {"anchors.anchor0": ["31", 0]}},
        "50": {"class_type": "AgentYRefNote", "inputs": {"input": ["136", 0], "tag": "#appartmentref",
                                                           "role": "the room, not the light"}},
        "51": {"class_type": "AgentYRefNote", "inputs": {"input": ["50", 0], "tag": "flat"}},
        "52": {"class_type": "AgentYRefNote", "inputs": {"input": ["118", 0], "tag": "location_prompt"}},
        "125": {"class_type": "AgentYHook",
                "inputs": {"anchors.anchor0": ["51", 0], "directive": "backgrounds in #appartmentref"}},
    }


class WhichTagsNameAStage(unittest.TestCase):
    def setUp(self):
        self.tags = ch.stage_tags(graph())

    def test_a_tag_on_a_reviews_output(self):
        self.assertEqual(self.tags["appartmentref"],
                         {"stage_id": "136", "note_id": "50", "role": "the room, not the light"})

    def test_a_tag_on_a_hooks_output(self):
        self.assertEqual(self.tags["location_prompt"]["stage_id"], "118")

    def test_a_tag_stacked_on_a_tag_names_the_same_stage(self):
        self.assertEqual(self.tags["flat"]["stage_id"], "136")

    def test_a_tag_on_a_loader_is_not_one_of_them(self):
        self.assertNotIn("hero_face", self.tags)

    def test_no_tags_no_answer(self):
        self.assertEqual(ch.stage_tags({"1": {"class_type": "LoadImage", "inputs": {}}}), {})
        self.assertEqual(ch.stage_tags(None), {})


class WhatTheResultIs(unittest.TestCase):
    def test_approved_files_are_listed_exactly(self):
        said = rl.result_words({"state": rl.APPROVED, "approved": ["a.png", "b.png"]})
        self.assertIn("APPROVED by the user - exactly these 2 file(s), in order: a.png; b.png", said)
        self.assertIn("They are files, not nodes", said)

    def test_an_approved_text_is_the_file_it_was_saved_to(self):
        said = rl.result_words({"state": rl.APPROVED, "approved": []}, ["s_v002.md"])
        self.assertEqual(said, "APPROVED by the user - the text saved at s_v002.md. Read the file.")

    def test_a_result_still_under_review_is_not_to_be_used(self):
        said = rl.result_words({"state": rl.WAITING, "outputs": ["a.png", "b.png"]})
        self.assertIn("NOT SETTLED", said)
        self.assertIn("Do not use it until they answer", said)

    def test_a_stage_that_has_not_run(self):
        self.assertIn("NOTHING YET", rl.result_words({}))

    def test_made_but_never_reviewed(self):
        self.assertIn("made, nobody has approved it: a.png", rl.result_words(
            {"state": rl.DONE, "outputs": ["a.png"]}))

    def test_inputs_changed_since_is_said(self):
        said = rl.result_words({"state": rl.APPROVED, "approved": ["a.png"], "changed": True})
        self.assertIn("its inputs have changed since - ask before relying on it", said)


class WhatTheAgentReads(unittest.TestCase):
    def block(self, results):
        g = graph()
        hooks = [{"hook_node_id": "118", "purpose": "set_parameter", "directive": "main location",
                  "anchors": [], "targets": [{"node_id": "26", "to_input": "prompt"}]},
                 {"hook_node_id": "125", "purpose": "set_parameter",
                  "directive": "backgrounds in #appartmentref", "anchors": [], "targets": []}]
        clean, _ = ch.splice_hook_nodes(g, hooks)
        return ch.describe_hooks(hooks, clean, stage_results=results)

    def test_without_the_record_it_named_everything_the_stage_made(self):
        # Wired through the review to the stage's save node: every picture of
        # the stage, where the user meant the one they approved.
        text = self.block(None)
        self.assertIn("- #appartmentref → node 31 (SaveImage)", text)
        self.assertIn("node 31 (SaveImage) [#appartmentref]", text)

    def test_with_it_the_tag_is_the_stages_result(self):
        text = self.block({"appartmentref": {
            "stage_id": "136", "title": "pick the location",
            "result": rl.result_words({"state": rl.APPROVED, "approved": ["//srv/loc_v002.png"]})}})
        self.assertIn('- #appartmentref → THE RESULT OF stage 136 "pick the location": '
                      "APPROVED by the user - exactly these 1 file(s), in order: //srv/loc_v002.png",
                      text)
        self.assertIn('"the room, not the light" (take only that from it)', text)
        self.assertNotIn("- #appartmentref → node 31", text)
        # ...and the stage that names it is told the same thing, not the save node.
        self.assertIn("#appartmentref = the result of stage 136, as given in the TAGS list above", text)
        self.assertNotIn("node 31 (SaveImage) [#appartmentref]", text)

    def test_a_loaders_tag_is_unchanged(self):
        text = self.block({})
        self.assertIn("- #hero_face → node 1 (LoadImage)", text)

    def test_the_pipeline_hands_the_record_over(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("stage_results=self._stage_tag_results(),", src)
        body = src.split("    def _stage_tag_results(self)", 1)[1].split("\n    def ", 1)[0]
        self.assertIn('stage_tags(getattr(self, "_canvas_graph", None))', body)
        self.assertIn("self._ledger_stages_of(sid)", body)


if __name__ == "__main__":
    unittest.main()
