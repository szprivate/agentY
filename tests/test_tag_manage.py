"""Managing remembered tags: list, rename, delete, untag one picture, merge.

The viewer's "Remove tag #name" and the panel's Manage tags page both land here
(src/utils/tag_memory.py, behind /agentY/tags*). A remembered tag is one reference
entry in the project store - one name, one file - so these are operations on
entries, kept in the format the tag node's `remember` switch and the viewer's
"Tag for agentY" write. Every test gets its own store; no real project is read.

    python -m unittest tests.test_tag_manage
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import project_memory as PM
from src.utils import tag_memory as TM


class _Store(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        self.enterContext(mock.patch.object(PM, "user_dir", lambda: self.root))
        self.a = self.root / "anna.png"
        self.b = self.root / "ben.png"
        for f in (self.a, self.b):
            f.write_bytes(b"png")

    def names(self):
        return [t["tag"] for t in TM.list_tags()]


class Listing(_Store):
    def test_every_remembered_tag_with_its_file_and_who_made_it(self):
        TM.remember_file("Hero_Face", str(self.a), "the face only")
        TM.sync({"1": {"class_type": "LoadImage", "inputs": {"image": str(self.b)}},
                 "2": {"class_type": "AgentYRefNote",
                       "inputs": {"input": ["1", 0], "tag": "alley_light", "role": "", "remember": True}}})
        PM.write_entry("grade", "Teal shadows.", type="style")          # a fact, not a tag
        PM.write_entry("set", "The set.\npath: " + str(self.b), type="reference")   # the agent's own
        tags = {t["tag"]: t for t in TM.list_tags()}
        self.assertEqual(list(tags), ["alley_light", "Hero_Face", "set"])   # by name, any case
        hero = tags["Hero_Face"]
        self.assertEqual((hero["key"], hero["origin"], hero["name"], hero["exists"]),
                         ("hero-face", "viewer", "anna.png", True))
        self.assertEqual(hero["role"], "the face only")
        self.assertEqual(tags["alley_light"]["origin"], "canvas")
        self.assertEqual(tags["set"]["origin"], "agent")

    def test_a_missing_file_is_listed_as_missing(self):
        TM.remember_file("hero", str(self.a))
        self.a.unlink()
        [t] = TM.list_tags()
        self.assertFalse(t["exists"])

    def test_no_comfyui_is_a_503_not_an_empty_list(self):
        with mock.patch.object(PM, "user_dir", lambda: None):
            with self.assertRaises(TM.TagError) as cm:
                TM.list_tags()
        self.assertEqual(cm.exception.status, 503)


class Rename(_Store):
    def test_the_entry_moves_and_still_resolves_under_the_new_name(self):
        TM.remember_file("hero", str(self.a), "Anna")
        out = TM.rename_tag("hero", "#Lead Face")
        self.assertEqual((out["tag"], out["from"]), ("Lead_Face", "hero"))
        self.assertIn("#hero", out["warning"])
        self.assertEqual(self.names(), ["Lead_Face"])
        self.assertIsNone(TM.remembered_reference("hero"))
        ref = TM.resolve("lead_face", {})
        self.assertTrue(ref["path"].endswith("anna.png"))
        self.assertEqual(ref["role"], "Anna")
        self.assertIn("`#Lead_Face`", ref["text"])
        self.assertNotIn("#hero", ref["text"])

    def test_a_change_of_case_only_rewrites_in_place(self):
        TM.remember_file("hero", str(self.a))
        TM.rename_tag("hero", "Hero")
        self.assertEqual(self.names(), ["Hero"])
        self.assertEqual(len(list(self.root.glob("agentY/project/*/*.md"))), 1)

    def test_an_agent_entry_gets_its_name_written_down(self):
        PM.write_entry("set", "The set.\npath: " + str(self.b), type="reference")
        TM.rename_tag("set", "stage_set")
        self.assertEqual(self.names(), ["stage_set"])
        self.assertIn("`#stage_set`", PM.read_entry("stage_set").body)

    def test_onto_an_existing_tag_is_a_conflict(self):
        TM.remember_file("hero", str(self.a))
        TM.remember_file("villain", str(self.b))
        with self.assertRaises(TM.TagError) as cm:
            TM.rename_tag("hero", "villain")
        self.assertEqual(cm.exception.status, 409)
        self.assertEqual(self.names(), ["hero", "villain"])

    def test_invalid_names_are_refused_like_at_creation(self):
        TM.remember_file("hero", str(self.a))
        for bad in ("", "   ", "#", "123", "__", "x" * 65):
            with self.assertRaises(ValueError, msg=bad):
                TM.rename_tag("hero", bad)
            with self.assertRaises(ValueError, msg=bad):
                TM.remember_file(bad, str(self.b))
        # markup is no name: it collapses to letters, or is refused
        out = TM.rename_tag("hero", "<img src=x onerror=alert(1)>")
        self.assertRegex(out["tag"], r"^[A-Za-z0-9_\-]+$")
        self.assertEqual(self.names(), [out["tag"]])

    def test_an_unknown_tag_is_a_404(self):
        with self.assertRaises(TM.TagError) as cm:
            TM.rename_tag("nobody", "someone")
        self.assertEqual(cm.exception.status, 404)

    def test_a_fact_is_not_a_tag(self):
        PM.write_entry("grade", "Teal shadows.", type="style")
        for call in (lambda: TM.rename_tag("grade", "x_grade"), lambda: TM.delete_tag("grade")):
            with self.assertRaises(TM.TagError) as cm:
                call()
            self.assertEqual(cm.exception.status, 404)
        self.assertIsNotNone(PM.read_entry("grade"))


class DeleteAndUntag(_Store):
    def test_delete_forgets_the_tag_and_leaves_the_picture(self):
        TM.remember_file("hero", str(self.a))
        out = TM.delete_tag("#hero")
        self.assertTrue(out["path"].endswith("anna.png"))
        self.assertEqual(self.names(), [])
        self.assertTrue(self.a.is_file())

    def test_untag_takes_it_off_that_picture_only(self):
        TM.remember_file("hero", str(self.a))
        with self.assertRaises(TM.TagError) as cm:
            TM.untag_file("hero", str(self.b))
        self.assertEqual(cm.exception.status, 404)
        self.assertEqual(self.names(), ["hero"])
        # the viewer's spelling of the same file still matches
        TM.untag_file("hero", str(self.a).replace("/", "\\"))
        self.assertEqual(self.names(), [])

    def test_untag_wants_a_path(self):
        TM.remember_file("hero", str(self.a))
        for bad in ("", None, "x" * 5000, "a\x00b"):
            with self.assertRaises(TM.TagError, msg=repr(bad)):
                TM.untag_file("hero", bad)
        self.assertEqual(self.names(), ["hero"])


class Merge(_Store):
    def test_into_a_new_name_is_a_rename(self):
        TM.remember_file("hero", str(self.a))
        out = TM.merge_tags("hero", "lead")
        self.assertEqual((out["tag"], out["dropped"]), ("lead", ""))
        self.assertEqual(self.names(), ["lead"])

    def test_both_remembered_keep_the_target_picture_by_default(self):
        TM.remember_file("hero", str(self.a))
        TM.remember_file("lead", str(self.b))
        out = TM.merge_tags("hero", "lead")
        self.assertTrue(out["path"].endswith("ben.png"))
        self.assertTrue(out["dropped"].endswith("anna.png"))
        self.assertEqual(self.names(), ["lead"])
        self.assertTrue(TM.remembered_reference("lead")["path"].endswith("ben.png"))

    def test_or_keep_the_source_picture(self):
        TM.remember_file("hero", str(self.a), "Anna")
        TM.remember_file("lead", str(self.b))
        out = TM.merge_tags("hero", "lead", keep="from")
        self.assertTrue(out["path"].endswith("anna.png"))
        self.assertEqual(self.names(), ["lead"])
        ref = TM.remembered_reference("lead")
        self.assertTrue(ref["path"].endswith("anna.png"))
        self.assertIn("`#lead`", ref["text"])

    def test_refusals(self):
        TM.remember_file("hero", str(self.a))
        PM.write_entry("grade", "Teal.", type="style")
        for args, status in ((("hero", "hero"), 400), (("hero", "grade"), 409), (("ghost", "lead"), 404)):
            with self.assertRaises(TM.TagError, msg=args) as cm:
                TM.merge_tags(*args)
            self.assertEqual(cm.exception.status, status, args)
        with self.assertRaises(ValueError):
            TM.merge_tags("hero", "lead", keep="both")


class Routes(_Store):
    def setUp(self):
        super().setUp()
        from src.utils import agentY_server as srv
        from tests.route_client import authorised_client
        app = srv._build_app()
        app.testing = True
        self.app = app
        self.client = authorised_client(app)

    def call(self, path, body=None):
        r = (self.client.get(path) if body is None else self.client.post(path, json=body))
        return r.status_code, json.loads(r.data)

    def test_list_rename_untag_merge_delete(self):
        TM.remember_file("hero", str(self.a))
        TM.remember_file("lead", str(self.b))
        code, d = self.call("/agentY/tags")
        self.assertEqual(code, 200)
        self.assertEqual([t["tag"] for t in d["tags"]], ["hero", "lead"])
        code, d = self.call("/agentY/tags/rename", {"tag": "hero", "to": "anna"})
        self.assertEqual((code, d["tag"]), (200, "anna"))
        self.assertTrue(d["warning"])
        code, d = self.call("/agentY/tags/rename", {"tag": "anna", "to": "lead"})
        self.assertEqual(code, 409)
        code, d = self.call("/agentY/tags/rename", {"tag": "anna", "to": "1"})
        self.assertEqual(code, 400)
        code, d = self.call("/agentY/tags/untag", {"tag": "anna", "path": str(self.b)})
        self.assertEqual(code, 404)
        code, d = self.call("/agentY/tags/merge", {"from": "anna", "into": "lead", "keep": "from"})
        self.assertEqual(code, 200)
        self.assertTrue(d["path"].endswith("anna.png"))
        code, d = self.call("/agentY/tags/untag", {"tag": "lead", "path": str(self.a)})
        self.assertEqual(code, 200)
        TM.remember_file("hero", str(self.a))
        code, d = self.call("/agentY/tags/delete", {"tag": "hero"})
        self.assertEqual(code, 200)
        self.assertEqual(self.call("/agentY/tags")[1]["tags"], [])
        code, d = self.call("/agentY/tags/delete", {"tag": "hero"})
        self.assertEqual(code, 404)
        code, d = self.call("/agentY/tags/delete", {})
        self.assertEqual(code, 400)

    def test_every_route_is_behind_the_guard(self):
        TM.remember_file("hero", str(self.a))
        from flask.testing import FlaskClient
        bare = FlaskClient(self.app)           # no session token (test_client() now adds one)
        for path, body in (("/agentY/tags", None), ("/agentY/tags/rename", {"tag": "hero", "to": "x_y"}),
                           ("/agentY/tags/delete", {"tag": "hero"}),
                           ("/agentY/tags/untag", {"tag": "hero", "path": str(self.a)}),
                           ("/agentY/tags/merge", {"from": "hero", "into": "lead"})):
            r = bare.get(path) if body is None else bare.post(path, json=body)
            self.assertEqual(r.status_code, 403, path)
        # a page of another site, even with the token, is refused too
        r = self.client.post("/agentY/tags/delete", json={"tag": "hero"},
                             headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.names(), ["hero"])

    def test_no_store_says_so(self):
        with mock.patch.object(PM, "user_dir", lambda: None):
            code, d = self.call("/agentY/tags")
        self.assertEqual(code, 503)
        self.assertIn("ComfyUI", d["error"])


if __name__ == "__main__":
    unittest.main()
