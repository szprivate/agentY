"""An `agentY add tag` says what a reference input is FOR.

Wired onto the wire that carries a reference — LoadImage → add tag → wherever —
it carries a line like "the face, not the styling". The agent reads it with the
input, so a reference stops being just an image and becomes an image with a job.
(The node was called "agentY ref note" until it grew a tag field; its class id,
and everything below, is unchanged. The naming half lives in test_canvas_tags.)

Living on the wire is what makes it trustworthy: there is no node id recorded
anywhere to drift out of date, because whatever is plugged into the note is what
the note is about. Everything below follows from that.

    python -m unittest tests.test_ref_notes
"""

import unittest

from src.utils.canvas_hooks import build_batch, describe_hooks, ref_notes


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    """LoadImage 43 → ref note 51 → the GPT node; hook 5 anchors on the note."""
    return {
        "43": _node("LoadImage", image="hero_face.png", upload="image"),
        "51": _node("AgentYRefNote", input=["43", 0],
                    role="the face only — not the hair, not the wardrobe"),
        "242": _node("OpenAIGPTImageNodeV2", prompt="old", images=["51", 0]),
        "250": _node("SaveImage", images=["242", 0]),
    }


def _hook(anchor_id, directive="make a portrait in this style"):
    return {"hook_node_id": "5", "purpose": "inline_parameter", "directive": directive,
            "anchors": [{"node_id": str(anchor_id), "type": "AgentYRefNote", "widgets": {}}],
            "targets": [{"node_id": "242", "to_input": "prompt",
                         "to_input_type": "STRING", "type": "OpenAIGPTImageNodeV2"}]}


class AnchorOnTheNoteTests(unittest.TestCase):
    """The common wiring: the hook is anchored on the note itself."""

    def setUp(self):
        self.block = describe_hooks([_hook("51")], _canvas())

    def test_the_anchor_reads_as_the_loader_not_the_note(self):
        # "node 51 (AgentYRefNote)" tells the agent nothing about what it's looking
        # at, and hides the filename it needs.
        self.assertIn("node 43 (LoadImage)", self.block)
        self.assertNotIn("AgentYRefNote", self.block)
        self.assertIn("hero_face.png", self.block)

    def test_the_role_is_stated_as_an_instruction(self):
        self.assertIn("USE THIS FOR", self.block)
        self.assertIn("the face only — not the hair, not the wardrobe", self.block)
        self.assertIn("take only that from it", self.block)

    def test_the_role_sits_on_the_input_it_belongs_to(self):
        line = next(l for l in self.block.splitlines() if "node 43 (LoadImage)" in l)
        self.assertIn("USE THIS FOR", line)


class NoteElsewhereOnTheLoaderTests(unittest.TestCase):
    """The note annotates the loader; the hook happens to anchor the loader direct."""

    def test_the_role_still_reaches_the_anchor(self):
        hook = _hook("43")
        hook["anchors"][0]["type"] = "LoadImage"
        block = describe_hooks([hook], _canvas())
        self.assertIn("node 43 (LoadImage)", block)
        self.assertIn("the face only", block)


class FrontendResolvedTests(unittest.TestCase):
    """The production path: the panel resolves the note before it sends anything.

    It has to, because the raw anchor id is read by more than the hook block — the
    iterate hook's feedback node and the QA reference list both look for a real
    node there, and would find an annotation instead.
    """

    def test_a_role_sent_with_the_anchor_is_used(self):
        hook = _hook("43")
        hook["anchors"][0] = {"node_id": "43", "type": "LoadImage", "widgets": {},
                              "role": "the light, not the architecture"}
        block = describe_hooks([hook], {"43": _node("LoadImage", image="alley.png"),
                                        "242": _node("OpenAIGPTImageNodeV2", prompt="x")})
        self.assertIn("node 43 (LoadImage)", block)
        self.assertIn("the light, not the architecture", block)

    def test_the_sent_role_wins_over_a_stale_graph_read(self):
        hook = _hook("43")
        hook["anchors"][0] = {"node_id": "43", "type": "LoadImage", "widgets": {},
                              "role": "what the user typed just now"}
        block = describe_hooks([hook], _canvas())
        self.assertIn("what the user typed just now", block)
        self.assertNotIn("the face only", block)

    def test_an_older_frontend_sending_no_role_still_works(self):
        hook = _hook("51")            # no "role" key at all
        self.assertIn("the face only", describe_hooks([hook], _canvas()))


class ResolutionTests(unittest.TestCase):
    def test_both_directions_are_reported(self):
        roles, wrapped = ref_notes(_canvas())
        self.assertEqual(wrapped["51"], "43")
        self.assertEqual(roles["43"], roles["51"])
        self.assertIn("the face only", roles["43"])

    def test_a_note_stacked_on_a_note_still_names_the_loader(self):
        prompt = _canvas()
        prompt["52"] = _node("AgentYRefNote", input=["51", 0], role="and the jawline")
        roles, wrapped = ref_notes(prompt)
        self.assertEqual(wrapped["52"], "43", "walk back past the note below it")
        self.assertEqual(roles["43"], "the face only — not the hair, not the wardrobe",
                         "the innermost note keeps the loader's role")

    def test_a_cycle_does_not_hang(self):
        prompt = {"1": _node("AgentYRefNote", input=["2", 0], role="a"),
                  "2": _node("AgentYRefNote", input=["1", 0], role="b")}
        roles, wrapped = ref_notes(prompt)   # must simply return
        self.assertEqual(set(roles) >= {"1", "2"}, True)

    def test_an_empty_note_adds_nothing(self):
        prompt = _canvas()
        prompt["51"]["inputs"]["role"] = "   "
        block = describe_hooks([_hook("51")], prompt)
        self.assertIn("node 43 (LoadImage)", block, "still resolved to the loader")
        self.assertNotIn("USE THIS FOR", block)

    def test_a_graph_without_notes_is_untouched(self):
        prompt = {"43": _node("LoadImage", image="hero.png", upload="image")}
        hook = _hook("43")
        hook["anchors"][0]["type"] = "LoadImage"
        block = describe_hooks([hook], prompt)
        self.assertIn("node 43 (LoadImage)", block)
        self.assertNotIn("USE THIS FOR", block)
        self.assertEqual(ref_notes(None), ({}, {}))


class RunnableGraphTests(unittest.TestCase):
    """A ref note is a real node, unlike a hook: it stays in the graph that runs."""

    def test_the_note_is_not_spliced_out_and_a_sweep_still_works(self):
        prompts, notes = build_batch(_canvas(), [
            {"target_node_id": "242", "param": "prompt", "mode": "value_list",
             "values": ["a", "b"]},
        ])
        self.assertEqual([p["242"]["inputs"]["prompt"] for p in prompts], ["a", "b"])
        self.assertEqual(notes, [])
        self.assertIn("51", prompts[0], "the note must survive into the run")
        self.assertEqual(prompts[0]["242"]["inputs"]["images"], ["51", 0],
                         "and stay on the wire it annotates")


if __name__ == "__main__":
    unittest.main()
