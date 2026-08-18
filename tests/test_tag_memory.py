"""A tag that outlives its graph, and the editor that forgets it.

A canvas tag names a wire: `#hero_face` means node 43, and node 43 exists only in
the graph that is open. The picture does not — it is still there tomorrow, in
another workflow, in a Claude Desktop session on the same ComfyUI. So the tag node
carries a `remember` switch that writes the reference into the project's memory,
and from then on the name resolves in two places.

The rule that needs pinning hardest is the one about NOT forgetting: the sync runs
against whatever graph happens to be open, and almost every graph contains none of
the remembered tags. "Forget what is not here" would empty the store the first time
someone opened an unrelated workflow, so off means stop refreshing, never delete.

    python -m unittest tests.test_tag_memory
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import project_memory as PM
from src.utils import agentY_server as srv
from src.utils import preflight
from src.utils.canvas_hooks import describe_hooks
from src.utils.tag_memory import remembered_reference, resolve, sync


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas(remember=True, role="the face only — not the hair"):
    return {
        "43": _node("LoadImage", image="hero_face.png"),
        "51": _node("AgentYRefNote", input=["43", 0], tag="hero_face",
                    role=role, remember=remember),
        "60": _node("LoadImage", image="alley.png"),
        "61": _node("AgentYRefNote", input=["60", 0], tag="alley_light",
                    role="", remember=False),
    }


class _Store(unittest.TestCase):
    """Each test gets its own project store, so none can see another's facts."""

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        self.enterContext(mock.patch.object(PM, "user_dir", lambda: self.root))


class SyncTests(_Store):
    def test_a_remembered_tag_becomes_a_project_reference(self):
        self.assertEqual(sync(_canvas()), ["hero_face"])
        ref = remembered_reference("hero_face")
        self.assertEqual(ref["path"], "hero_face.png")
        self.assertIn("the face only", ref["role"])
        self.assertIn("agentY add tag", ref["text"], "the file says where it came from")

    def test_a_tag_with_the_switch_off_is_not_written(self):
        sync(_canvas())
        self.assertIsNone(remembered_reference("alley_light"))

    def test_nothing_is_written_when_no_tag_is_remembered(self):
        self.assertEqual(sync(_canvas(remember=False)), [])
        self.assertEqual(PM.list_entries(), [])

    def test_a_tag_naming_no_file_is_skipped(self):
        # A mid-graph tensor has no file of its own, so there is nothing to
        # remember but the name — and an entry that says only "there was a tag
        # here" is not a fact worth carrying into every future turn.
        graph = {"70": _node("ImageScale", image=["9", 0], upscale_method="lanczos",
                             crop="disabled"),
                 "71": _node("AgentYRefNote", input=["70", 0], tag="decoded",
                             role="the look", remember=True)}
        self.assertEqual(sync(graph), [])
        self.assertIsNone(remembered_reference("decoded"))

    def test_syncing_twice_updates_rather_than_duplicates(self):
        sync(_canvas())
        sync(_canvas(role="the jawline, nothing else"))
        # The store slugs a name to its filename form, so "hero_face" is filed as
        # "hero-face"; lookup slugs both sides, which is why the round-trip works.
        entries = [e for e in PM.list_entries() if e.name == PM.slug("hero_face")]
        self.assertEqual(len(entries), 1)
        self.assertIn("jawline", remembered_reference("hero_face")["role"])

    def test_turning_the_switch_off_does_not_forget(self):
        # THE rule. The sync runs against whatever graph is open; a graph that
        # does not contain this tag must not delete it, or opening any unrelated
        # workflow would empty the store.
        sync(_canvas())
        sync(_canvas(remember=False))
        self.assertIsNotNone(remembered_reference("hero_face"))

    def test_an_unrelated_graph_forgets_nothing(self):
        # The same rule from the direction that actually happens: you remember a
        # reference, then open a completely different workflow. Checked with a
        # graph that DOES remember something of its own, so a sync that clears the
        # store before writing cannot hide behind an early return.
        sync(_canvas())
        other = {"1": _node("LoadImage", image="other.png"),
                 "2": _node("AgentYRefNote", input=["1", 0], tag="other_ref",
                            role="", remember=True)}
        self.assertEqual(sync(other), ["other_ref"])
        self.assertIsNotNone(remembered_reference("hero_face"),
                             "another graph's tags must not evict this one")
        self.assertIsNotNone(remembered_reference("other_ref"))


class ResolutionTests(_Store):
    def test_the_canvas_wins_over_memory(self):
        sync(_canvas())
        # A tag on the graph in front of you is the more specific statement.
        self.assertEqual(resolve("hero_face", _canvas())["source"], "canvas")

    def test_memory_answers_when_the_canvas_does_not(self):
        sync(_canvas())
        hit = resolve("hero_face", {})
        self.assertEqual(hit["source"], "memory")
        self.assertEqual(hit["path"], "hero_face.png")

    def test_a_name_nobody_ever_remembered_resolves_to_nothing(self):
        self.assertIsNone(resolve("never_seen", {}))

    def test_only_reference_entries_answer(self):
        # A character sheet named "hero" is a fact about the project, not a file
        # to wire in; reading it as a reference would hand back a path-less entry.
        PM.write_entry("hero", "Anna, 30s, red coat.", type="character")
        self.assertIsNone(remembered_reference("hero"))


class HookBlockTests(_Store):
    """What the agent is told in a graph that has never seen the tag node."""

    def setUp(self):
        super().setUp()
        sync(_canvas())
        self.hook = {"hook_node_id": "9", "purpose": "make_workflow", "anchors": [],
                     "directive": "restyle #hero_face as a night shot"}
        self.graph = {"9": _node("AgentYHook", directive="restyle #hero_face")}

    def test_a_remembered_name_is_listed_with_its_file(self):
        block = describe_hooks([self.hook], self.graph)
        self.assertIn("#hero_face", block)
        self.assertIn("REMEMBERED", block)
        self.assertIn("hero_face.png", block)
        self.assertIn("the face only", block)

    def test_it_says_a_file_is_not_a_node(self):
        # The difference matters: a canvas tag can be anchored, a remembered one
        # has to be uploaded and wired before it is an input to anything.
        block = describe_hooks([self.hook], self.graph)
        self.assertIn("It is a file, not a node", block)

    def test_a_name_that_is_neither_is_still_called_out(self):
        hook = dict(self.hook, directive="restyle #her0_face as a night shot")
        notes = [f.text for f in preflight.check([hook], self.graph)]
        self.assertTrue(any("names nothing" in t for t in notes))

    def test_a_remembered_name_is_not_called_a_mistake(self):
        notes = [f.text for f in preflight.check([self.hook], self.graph)]
        self.assertFalse(any("names nothing" in t for t in notes))


class EditorRouteTests(_Store):
    """Inspect and delete over HTTP — the routes the editor page drives."""

    def setUp(self):
        super().setUp()
        PM.write_entry("hero", "Anna, 30s, red coat.", type="character")
        PM.write_entry("grade", "Cool shadows, warm skin.", type="style")
        app = srv._build_app()
        app.testing = True
        self.client = app.test_client()

    def _list(self):
        return json.loads(self.client.get("/agentY/project_memory").data)

    def test_listing_returns_every_fact_with_its_body(self):
        data = self._list()
        self.assertTrue(data["ok"])
        names = {e["name"]: e for e in data["entries"]}
        self.assertEqual(set(names), {"hero", "grade"})
        self.assertIn("red coat", names["hero"]["body"])
        self.assertEqual(names["grade"]["type"], "style")
        self.assertTrue(data["store"], "the page shows where the files live")

    def test_forgetting_one_removes_it_and_leaves_the_rest(self):
        r = json.loads(self.client.post("/agentY/project_memory/delete",
                                        json={"names": ["hero"]}).data)
        self.assertEqual(r["deleted"], ["hero"])
        self.assertEqual({e["name"] for e in self._list()["entries"]}, {"grade"})

    def test_forgetting_something_absent_is_reported_not_an_error(self):
        r = json.loads(self.client.post("/agentY/project_memory/delete",
                                        json={"names": ["nope"]}).data)
        self.assertTrue(r["ok"])
        self.assertEqual(r["not_found"], ["nope"])

    def test_a_delete_with_no_names_is_refused(self):
        resp = self.client.post("/agentY/project_memory/delete", json={"names": []})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(len(self._list()["entries"]), 2, "nothing was touched")

    def test_the_editor_page_is_served(self):
        resp = self.client.get("/agentY/project_memory_viewer")
        self.assertEqual(resp.status_code, 200)
        body = resp.data.decode("utf-8", "replace")
        self.assertIn("/agentY/project_memory/delete", body)
        self.assertNotIn("project_memory/write", body, "the editor never authors")

    def test_the_command_is_offered_in_the_panel(self):
        names = {c["name"] for c in json.loads(
            self.client.get("/agentY/commands").data)}
        self.assertIn("/project_memory", names)


if __name__ == "__main__":
    unittest.main()
