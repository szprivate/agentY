"""Conversation management endpoints: detail list, rename, search, bulk delete."""
import os
import tempfile
import unittest
from unittest import mock

from src.utils import conversation_store as cs


class ThreadManagement(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(tmp.name, "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        from src.utils import agentY_server as srv
        self.srv = srv
        p = mock.patch.object(srv._api_guard, "check", return_value=None, create=True)
        p.start()
        self.addCleanup(p.stop)
        client = srv._build_app().test_client()
        try:
            from src.utils import api_guard
            tok = api_guard.session_token(None)
        except Exception:  # noqa: BLE001
            tok = ""
        h = {"X-AgentY-Token": tok} if tok else {}
        self.get = lambda u: client.get(u, headers=h)
        self.post = lambda u, j: client.post(u, json=j, headers=h)
        self.delete = lambda u: client.delete(u, headers=h)
        self.running: set = set()
        p2 = mock.patch.object(srv, "running_threads", lambda: list(self.running))
        p2.start()
        self.addCleanup(p2.stop)
        self.a = cs.create_thread("Alpha")
        self.b = cs.create_thread("Beta")
        cs.add_message(self.a, "user", "relight the hero_face please")
        cs.add_message(self.a, "assistant", "done, 100% lit")
        cs.add_message(self.b, "user", "make a beach")

    def test_detail_list_has_preview_count_and_size(self):
        rows = {r["id"]: r for r in self.get("/agentY/threads?detail=1").get_json()}
        self.assertEqual(rows[self.a]["message_count"], 2)
        self.assertEqual(rows[self.a]["preview"], "relight the hero_face please")
        self.assertGreater(rows[self.a]["size_chars"], 20)
        self.assertIn("running", rows[self.a])

    def test_rename_persists_and_validates(self):
        r = self.post(f"/agentY/threads/{self.a}/rename", {"title": "  Hero   relight "})
        self.assertEqual(r.get_json()["title"], "Hero relight")
        self.assertEqual(cs.get_thread(self.a)["title"], "Hero relight")
        self.assertEqual(self.post(f"/agentY/threads/{self.a}/rename", {"title": " "}).status_code, 400)
        self.assertEqual(self.post("/agentY/threads/nope/rename", {"title": "x"}).status_code, 404)

    def test_search_title_and_content_with_literal_wildcards(self):
        ids = lambda q: [r["id"] for r in self.get("/agentY/threads/search?q=" + q).get_json()]
        self.assertEqual(ids("beta"), [self.b])                 # title
        self.assertEqual(ids("HERO_face"), [self.a])            # content, case-insensitive
        self.assertEqual(ids("100%25"), [self.a])               # % is literal
        self.assertEqual(ids("hero%25face"), [])
        self.assertEqual(ids(""), [])

    def test_bulk_delete_is_one_call_and_skips_running(self):
        c = cs.create_thread("Gamma")
        self.running.add(self.b)
        body = self.post("/agentY/threads/delete", {"ids": [self.a, self.b, c]}).get_json()
        self.assertEqual(body["deleted"], 2)
        self.assertEqual(body["skipped"], [self.b])
        left = [t["id"] for t in cs.list_threads()]
        self.assertEqual(left, [self.b])

    def test_single_delete_refuses_a_running_conversation(self):
        self.running.add(self.a)
        self.assertEqual(self.delete(f"/agentY/threads/{self.a}").status_code, 409)
        self.assertIsNotNone(cs.get_thread(self.a))
        self.running.clear()
        self.assertEqual(self.delete(f"/agentY/threads/{self.a}").status_code, 200)
        self.assertIsNone(cs.get_thread(self.a))


if __name__ == "__main__":
    unittest.main()
