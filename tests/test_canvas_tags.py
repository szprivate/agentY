"""An `agentY add tag` node NAMES the reference on the wire it sits on.

The node has said what a reference is FOR since it was called "agentY ref note"
("the face, not the styling"); the tag is the other half — a short handle for the
same wire, so a hook directive can write `#hero_face` and mean one exact node
instead of describing an input and hoping the right one is picked. The canvas
offers the list when you type `#` in a hook's prompt box (web/agent_tags.js);
this is the half that makes the word resolve to a node once it arrives.

    python -m unittest tests.test_canvas_tags
"""

import unittest
from unittest import mock

from src.utils import preflight
from src.utils.hook_cache import fingerprint
from src.utils.canvas_hooks import (canvas_tags, describe_hooks, hook_scope_ids,
                                    mentioned_tags, normalise_tag, prune_to_hooks,
                                    tagged_inputs)


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    """Two tagged references and one untagged, all feeding the same GPT node."""
    return {
        "43": _node("LoadImage", image="hero_face.png", upload="image"),
        "51": _node("AgentYRefNote", input=["43", 0], tag="hero_face",
                    role="the face only — not the hair, not the wardrobe"),
        "60": _node("LoadImage", image="alley.png", upload="image"),
        "61": _node("AgentYRefNote", input=["60", 0], tag="alley_light", role=""),
        "70": _node("LoadImage", image="plain.png", upload="image"),
        "242": _node("OpenAIGPTImageNodeV2", prompt="old", images=["51", 0]),
    }


def _canvas_untagged():
    """The same shape, but no `agentY add tag` node carries a name."""
    return {
        "43": _node("LoadImage", image="hero.png"),
        "51": _node("AgentYRefNote", input=["43", 0], role="the face"),
        "242": _node("OpenAIGPTImageNodeV2", prompt="x"),
    }


def _hook(anchor_id, directive="put #hero_face under #alley_light", **anchor):
    a = {"node_id": str(anchor_id), "type": "AgentYRefNote", "widgets": {}}
    a.update(anchor)
    return {"hook_node_id": "5", "purpose": "inline_parameter", "directive": directive,
            "anchors": [a],
            "targets": [{"node_id": "242", "to_input": "prompt",
                         "to_input_type": "STRING", "type": "OpenAIGPTImageNodeV2"}]}


class NormaliseTests(unittest.TestCase):
    """The field is free text, so two spellings of one name must land on one tag."""

    def test_a_typed_hash_and_spaces_are_forgiven(self):
        self.assertEqual(normalise_tag("#hero face"), "hero_face")
        self.assertEqual(normalise_tag("  Hero-Face  "), "Hero-Face")
        self.assertEqual(normalise_tag("hero_face"), "hero_face")

    def test_a_name_with_nothing_in_it_is_no_tag(self):
        self.assertEqual(normalise_tag("   "), "")
        self.assertEqual(normalise_tag("#"), "")
        self.assertEqual(normalise_tag(None), "")


class ResolutionTests(unittest.TestCase):
    def test_a_tag_names_the_node_up_the_wire_not_the_annotation(self):
        tags = canvas_tags(_canvas())
        self.assertEqual(tags["hero_face"]["node_id"], "43")
        self.assertEqual(tags["hero_face"]["note_id"], "51")
        self.assertIn("the face only", tags["hero_face"]["role"])

    def test_a_tag_needs_no_role(self):
        self.assertEqual(canvas_tags(_canvas())["alley_light"]["node_id"], "60")

    def test_a_stacked_tag_still_names_the_loader(self):
        prompt = _canvas()
        prompt["52"] = _node("AgentYRefNote", input=["51", 0], tag="jawline")
        self.assertEqual(canvas_tags(prompt)["jawline"]["node_id"], "43")

    def test_the_same_tag_twice_is_one_entry(self):
        prompt = _canvas()
        prompt["62"] = _node("AgentYRefNote", input=["70", 0], tag="hero_face")
        tags = canvas_tags(prompt)
        self.assertEqual(tags["hero_face"]["node_id"], "43", "lowest node id wins")
        self.assertEqual(len([t for t in tags if t == "hero_face"]), 1)

    def test_a_graph_with_no_tags_has_none(self):
        prompt = {"43": _node("LoadImage", image="hero.png"),
                  "51": _node("AgentYRefNote", input=["43", 0], role="the face")}
        self.assertEqual(canvas_tags(prompt), {})
        self.assertEqual(canvas_tags(None), {})

    def test_a_cycle_does_not_hang(self):
        prompt = {"1": _node("AgentYRefNote", input=["2", 0], tag="a"),
                  "2": _node("AgentYRefNote", input=["1", 0], tag="b")}
        self.assertEqual(set(canvas_tags(prompt)), {"a", "b"})


class GlossaryTests(unittest.TestCase):
    """What the agent is handed: `#name` → the node it names."""

    def setUp(self):
        self.block = describe_hooks([_hook("51")], _canvas())

    def test_every_tag_is_mapped_to_its_node(self):
        self.assertIn("TAGS —", self.block)
        self.assertIn("#hero_face → node 43 (LoadImage)", self.block)
        self.assertIn("#alley_light → node 60 (LoadImage)", self.block)

    def test_the_mapping_carries_the_file_so_it_can_be_looked_at(self):
        line = next(l for l in self.block.splitlines() if l.startswith("- #hero_face"))
        self.assertIn("hero_face.png", line)
        self.assertIn("the face only", line)

    def test_an_unwired_tag_says_it_names_nothing(self):
        prompt = _canvas()
        prompt["80"] = _node("AgentYRefNote", tag="orphan", role="")
        block = describe_hooks([_hook("51")], prompt)
        self.assertIn("#orphan", block)
        self.assertIn("NOT WIRED", block)

    def test_a_canvas_without_tags_gets_no_glossary(self):
        prompt = {"43": _node("LoadImage", image="hero.png"),
                  "51": _node("AgentYRefNote", input=["43", 0], role="the face")}
        block = describe_hooks([_hook("51")], prompt)
        self.assertNotIn("TAGS —", block)


class AnchorLineTests(unittest.TestCase):
    """The tag is repeated where the input is used, so the two can be matched up."""

    def test_the_anchor_line_carries_the_tag(self):
        line = next(l for l in describe_hooks([_hook("51")], _canvas()).splitlines()
                    if "node 43 (LoadImage)" in l and "#1" in l)
        self.assertIn("[#hero_face]", line)
        self.assertIn("USE THIS FOR", line, "the role still reads as the instruction")

    def test_a_tag_sent_with_the_anchor_is_used(self):
        # The panel resolves the annotation before it sends anything, so the tag
        # arrives on the anchor rather than being read back off the graph.
        hook = _hook("43", type="LoadImage", tag="hero_face", role="the face only")
        block = describe_hooks([hook], {"43": _node("LoadImage", image="hero_face.png"),
                                        "242": _node("OpenAIGPTImageNodeV2", prompt="x")})
        self.assertIn("[#hero_face]", block)

    def test_a_loose_spelling_sent_with_the_anchor_still_resolves(self):
        hook = _hook("43", type="LoadImage", tag="#hero face")
        block = describe_hooks([hook], {"43": _node("LoadImage", image="hero_face.png"),
                                        "242": _node("OpenAIGPTImageNodeV2", prompt="x")})
        self.assertIn("[#hero_face]", block)

    def test_an_untagged_anchor_is_unchanged(self):
        # Its own segment of the context line — the rest of that line now also
        # carries the references the directive named, which do have tags.
        hook = _hook("70", type="LoadImage", directive="describe this")
        block = describe_hooks([hook], _canvas())
        seg = next(s for l in block.splitlines() for s in l.split("; ")
                   if "node 70 (LoadImage)" in s)
        self.assertNotIn("[#", seg)


class NamedAsAnInputTests(unittest.TestCase):
    """A tag in the prompt hands the hook a reference, with no wire drawn."""

    def _hook_named(self, purpose="inline_parameter", anchors=None):
        h = _hook("51", directive="put #hero_face under #alley_light")
        h["purpose"] = purpose
        h["anchors"] = anchors if anchors is not None else []
        return h

    def test_a_named_reference_is_reported_as_an_input(self):
        h = self._hook_named()
        got = [(nid, tag) for nid, _cls, _in, _role, tag in tagged_inputs(h, _canvas())]
        self.assertEqual(got, [("43", "hero_face"), ("60", "alley_light")])

    def test_a_reference_already_wired_is_not_listed_twice(self):
        # Anchored on the tag node for #hero_face, which resolves to node 43.
        h = self._hook_named(anchors=[{"node_id": "51", "type": "AgentYRefNote",
                                       "widgets": {}}])
        got = [nid for nid, *_ in tagged_inputs(h, _canvas())]
        self.assertEqual(got, ["60"], "the wired one is described as the anchor it is")

    def test_a_tag_naming_nothing_is_not_an_input(self):
        prompt = _canvas()
        prompt["80"] = _node("AgentYRefNote", tag="orphan", role="")
        h = self._hook_named()
        h["directive"] = "use #orphan"
        self.assertEqual(tagged_inputs(h, prompt), [])

    def test_the_hook_block_says_it_is_an_input(self):
        block = describe_hooks([self._hook_named()], _canvas())
        self.assertIn("NAMED IN THE DIRECTIVE", block)
        self.assertIn("hero_face.png", block)
        self.assertNotIn("no input wired", block)

    def test_a_make_workflow_hook_is_no_longer_text_to_media(self):
        block = describe_hooks([self._hook_named("make_workflow")], _canvas())
        self.assertNotIn("no input wired — treat the prompt as text-to-media", block)
        self.assertIn("still an input to what you generate", block)
        self.assertIn("hero_face.png", block)
        self.assertIn("alley.png", block)

    def test_a_hook_naming_nothing_is_still_text_to_media(self):
        h = self._hook_named("make_workflow")
        h["directive"] = "a wide shot of an empty street"
        block = describe_hooks([h], _canvas())
        self.assertIn("no input wired — treat the prompt as text-to-media", block)


class ScopeTests(unittest.TestCase):
    """A named reference must survive the trim, or the name resolves to nothing.

    The hook block is rendered from the SCOPED graph (pipeline.py), so a reference
    trimmed here is one the agent is never told about — the `#name` in the
    directive would then point at a node id that is not in the prompt at all.
    """

    def _graph(self, directive):
        return {
            "43": _node("LoadImage", image="hero_face.png"),
            "51": _node("AgentYRefNote", input=["43", 0], tag="hero_face",
                        role="the face only"),
            "9": _node("AgentYHook", directive=directive, purpose="make_workflow"),
            "20": _node("CLIPTextEncode", text="x"),
            "21": _node("SaveImage", images=["20", 0]),
        }

    def test_a_named_reference_is_kept(self):
        keep = hook_scope_ids(self._graph("build it from #hero_face"), ["9"])
        self.assertIn("43", keep, "the loader the tag names")
        self.assertIn("51", keep, "and the annotation carrying the name")

    def test_an_unnamed_reference_is_still_trimmed(self):
        keep = hook_scope_ids(self._graph("build something"), ["9"])
        self.assertNotIn("43", keep)

    def test_the_pruned_graph_keeps_it(self):
        graph = self._graph("build it from #hero_face")
        scoped, dropped = prune_to_hooks(graph, ["9"])
        self.assertIn("43", scoped)
        self.assertIn("51", scoped)
        self.assertNotIn("43", dropped)

    def test_a_tag_does_not_drag_in_what_consumes_it(self):
        # #hero_face also feeds an unrelated render. Naming the reference must not
        # queue that render as well — the user named an input, not a second job.
        graph = self._graph("build it from #hero_face")
        graph["30"] = _node("ImageScale", image=["43", 0])
        graph["31"] = _node("SaveImage", images=["30", 0])
        keep = hook_scope_ids(graph, ["9"])
        self.assertIn("43", keep)
        self.assertNotIn("31", keep, "the other chain's saver stays out")


class MemorizeTests(unittest.TestCase):
    """A named reference is hashed into the keep switch's key, like a wired one.

    The promise the keep switch makes is that memory is released the moment
    anything feeding the hook changes. It held for a reference you wired and
    quietly failed for the identical reference you NAMED — the fingerprint walked
    up from the anchors, and a named reference is not an anchor. You swapped the
    image and got the old answer back, which is the one failure a cache must not
    have.
    """

    HOOK = {"hook_node_id": "9", "purpose": "text", "anchors": [], "targets": [],
            "directive": "describe #hero_face", "remember": True}

    @staticmethod
    def _graph(image="hero_a.png", tag="hero_face", role="the face", src="43"):
        return {"43": _node("LoadImage", image=image),
                "60": _node("LoadImage", image="other.png"),
                "51": _node("AgentYRefNote", input=[src, 0], tag=tag, role=role)}

    def _moved(self, **changed):
        return fingerprint(self.HOOK, self._graph()) !=             fingerprint(self.HOOK, self._graph(**changed))

    def test_an_unchanged_graph_keeps_its_key(self):
        self.assertFalse(self._moved(), "or nothing could ever be remembered")

    def test_a_different_image_releases_it(self):
        self.assertTrue(self._moved(image="hero_b.png"))

    def test_rewiring_the_tag_to_another_image_releases_it(self):
        self.assertTrue(self._moved(src="60"))

    def test_editing_the_stated_role_releases_it(self):
        # The note is DOWNSTREAM of the loader, so seeding the walk at the loader
        # would miss it — and the agent is being asked a different question.
        self.assertTrue(self._moved(role="the jawline"))

    def test_renaming_the_tag_releases_it(self):
        # #hero_face now names nothing; the answer was built from something the
        # directive can no longer point at.
        self.assertTrue(self._moved(tag="hero_shot"))

    def test_a_hook_naming_nothing_is_unaffected(self):
        plain = dict(self.HOOK, directive="write a caption")
        self.assertEqual(fingerprint(plain, self._graph()),
                         fingerprint(plain, self._graph(image="hero_b.png")))


class MentionTests(unittest.TestCase):
    """What counts as a directive naming a tag."""

    def test_a_hash_at_the_start_of_a_word_is_a_name(self):
        self.assertEqual(mentioned_tags("put #hero_face under #alley_light"),
                         ["hero_face", "alley_light"])
        self.assertEqual(mentioned_tags("(#hero_face)"), ["hero_face"])

    def test_a_hash_mid_word_is_not(self):
        self.assertEqual(mentioned_tags("shot#3 and colour #a1b2c3"), ["a1b2c3"])
        self.assertEqual(mentioned_tags("issue123#hero"), [])

    def test_a_number_is_not_a_name(self):
        # "#2 of the references" is a rank, not a reference to something called 2.
        self.assertEqual(mentioned_tags("use #2 of the references"), [])

    def test_the_same_name_twice_is_listed_once(self):
        self.assertEqual(mentioned_tags("#hero then #hero again"), ["hero"])


class PreflightTests(unittest.TestCase):
    """A typo'd tag is a reference to nothing, and reads like an instruction."""

    def setUp(self):
        self.enterContext(mock.patch.object(preflight, "_schema", return_value={}))
        # Whether `#word` is a broken reference or ordinary prose depends on
        # whether the PROJECT has a tag vocabulary, and that store is a real
        # directory on the machine running the tests. Left alone, these tests pass
        # on an empty install and fail on a working one — which is what happened.
        # The store is stated per test instead; `_remembered_project` covers the
        # branch where it is not empty.
        self._remembered = set()
        self.enterContext(mock.patch.object(
            preflight, "_any_remembered", lambda: bool(self._remembered)))
        self.enterContext(mock.patch.object(
            preflight, "_remembered", lambda t: t in self._remembered))

    def _notes(self, directive, prompt):
        hook = _hook("51", directive=directive)
        return [f.text for f in preflight.check([hook], prompt) if f.level == "note"]

    def test_a_name_no_node_carries_is_called_out(self):
        notes = self._notes("put #her0_face in the frame", _canvas())
        self.assertTrue(any("#her0_face" in n and "names nothing" in n for n in notes))

    def test_a_name_that_exists_is_not(self):
        notes = self._notes("put #hero_face in the frame", _canvas())
        self.assertFalse(any("names nothing" in n for n in notes))

    def test_prose_on_an_untagged_canvas_is_left_alone(self):
        # No tags anywhere means "#" is punctuation, not a broken reference.
        notes = self._notes("shortlist #1 and #2, ship it #soon", _canvas_untagged())
        self.assertFalse(any("names nothing" in n for n in notes))

    def test_a_remembered_name_resolves_without_a_node(self):
        # The project remembers it from another graph, so it points at a file
        # rather than at nothing.
        self._remembered = {"hero_face"}
        notes = self._notes("put #hero_face in the frame", _canvas_untagged())
        self.assertFalse(any("names nothing" in n for n in notes))

    def test_a_project_vocabulary_makes_prose_checkable_again(self):
        # Once the project HAS tags, a `#name` outside them is worth calling out
        # even on a canvas that carries none of its own.
        self._remembered = {"hero_face"}
        notes = self._notes("ship it #soon", _canvas_untagged())
        self.assertTrue(any("#soon" in n and "names nothing" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
