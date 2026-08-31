"""The agent can change a few settings when asked, and only those few.

"Turn QA off", "stop putting workflows on my canvas", "let yourself see the whole
graph" are one-line requests that used to mean opening the dialog and knowing
what the key is called.

The list is short on purpose, and the reason is the rest of the file: settings
hold the ComfyUI address, the directories things are written to, which
environment variable the API key comes from, and which model each role runs on. A
tool that could write any of those would eventually write one of them because a
sentence was misread — so nothing is inferred from the shape of a key. A setting
is changeable because it is named, or it is not changeable.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import agent_settings as st


class AllowlistTest(unittest.TestCase):

    def test_the_dangerous_kinds_are_absent(self):
        """Not by a rule that could be relaxed — they are simply not on the list."""
        keys = [s.key for s in st.allowed()]
        for banned in ("comfyui_url", "agent_server_url", "agent_server_url_macos",
                       "output_dir", "comfyui_dir",
                       "conversation_db", "memory.store_dir", "qa.briefing_dir",
                       "memory.embedder.model", "memory.embedder.embedding_dims",
                       "memory.embedder.api_key_env", "llm.tiers.orchestrator",
                       "llm.pipeline.coder", "llm.anthropic.max_tokens",
                       "system_prompts.orchestrator"):
            with self.subTest(key=banned):
                self.assertNotIn(banned, keys)
                self.assertIsNone(st.get(banned))

    def test_nothing_is_changeable_by_resembling_something_changeable(self):
        for near in ("qa.enabled.extra", "QA.enabled", "qa_enabled",
                     "", "qa", "memory", "canvas_full_graph.value"):
            with self.subTest(key=near):
                self.assertIsNone(st.get(near))

    def test_a_stray_space_is_tolerated(self):
        """A model sending " qa.enabled" means qa.enabled; refusing that is theatre."""
        self.assertIsNotNone(st.get(" canvas_full_graph "))

    def test_every_entry_says_what_it_is_for(self):
        for s in st.allowed():
            with self.subTest(key=s.key):
                self.assertTrue(s.what.strip())
                self.assertIn(s.kind, ("bool", "int"))
                if s.kind == "int":
                    self.assertLess(s.low, s.high)

    def test_every_entry_is_a_real_setting(self):
        """A key that no longer exists is a switch that silently does nothing."""
        from src.utils.settings import load_settings
        live = load_settings()
        for s in st.allowed():
            with self.subTest(key=s.key):
                node = live
                for part in s.key.split("."):
                    self.assertIsInstance(node, dict, s.key)
                    self.assertIn(part, node, f"{s.key} is not in the settings tree")
                    node = node[part]


class CoerceTest(unittest.TestCase):

    def setUp(self):
        self.flag = st.get("qa.enabled")
        self.num = st.get("qa.max_retries")

    def test_a_model_says_yes_in_several_ways(self):
        for yes in (True, "true", "True", "yes", "on", 1, "1", "enabled"):
            with self.subTest(v=yes):
                self.assertEqual(st.coerce(self.flag, yes), (True, ""))
        for no in (False, "false", "no", "off", 0, "0", "disabled"):
            with self.subTest(v=no):
                self.assertEqual(st.coerce(self.flag, no), (False, ""))

    def test_an_unrecognisable_answer_is_refused_not_guessed(self):
        """Guessing here silently sets the opposite of what was asked."""
        for junk in ("maybe", "sure why not", "", None, "2"):
            with self.subTest(v=junk):
                value, why = st.coerce(self.flag, junk)
                self.assertIsNone(value)
                self.assertTrue(why)

    def test_a_number_out_of_range_is_refused_with_the_range(self):
        value, why = st.coerce(self.num, 99)
        self.assertIsNone(value)
        self.assertIn("0–5", why)

    def test_the_ends_of_the_range_are_allowed(self):
        self.assertEqual(st.coerce(self.num, 0), (0, ""))
        self.assertEqual(st.coerce(self.num, 5), (5, ""))

    def test_a_number_written_as_text_still_counts(self):
        self.assertEqual(st.coerce(self.num, " 3 "), (3, ""))

    def test_a_word_is_not_a_number(self):
        self.assertIsNone(st.coerce(self.num, "three")[0])

    def test_the_dotted_key_becomes_the_shape_set_local_takes(self):
        self.assertEqual(st.nest("qa.max_retries", 2), {"qa": {"max_retries": 2}})
        self.assertEqual(st.nest("canvas_full_graph", True), {"canvas_full_graph": True})


class ApplyTest(unittest.TestCase):

    def setUp(self):
        self.written = []
        self.enterContext(mock.patch("src.utils.settings.set_local",
                                     side_effect=self.written.append))

    def _live(self, tree):
        return mock.patch("src.utils.settings.load_settings", return_value=tree)

    def test_a_change_is_written_and_reported_both_ways(self):
        with self._live({"qa": {"enabled": True}}):
            out = st.apply("qa.enabled", False)
        self.assertEqual(out["status"], "changed")
        self.assertEqual((out["from"], out["to"]), (True, False))
        self.assertEqual(self.written, [{"qa": {"enabled": False}}])

    def test_it_says_when_the_change_bites(self):
        with self._live({"qa": {"enabled": True}}):
            self.assertIn("next message", st.apply("qa.enabled", False)["message"])
        with self._live({"auto_update": True}):
            self.assertIn("started", st.apply("auto_update", False)["message"])

    def test_setting_it_to_what_it_already_is_writes_nothing(self):
        with self._live({"qa": {"enabled": True}}):
            out = st.apply("qa.enabled", True)
        self.assertEqual(out["status"], "unchanged")
        self.assertEqual(self.written, [])

    def test_a_key_that_is_not_on_the_list_is_refused_and_says_who_owns_it(self):
        out = st.apply("comfyui_url", "http://somewhere-else:8188")
        self.assertIn("not a setting the agent can change", out["error"])
        self.assertIn("changeable", out)
        self.assertIn("let them do it", out["what_to_do"])
        self.assertEqual(self.written, [])

    def test_a_bad_value_never_reaches_the_file(self):
        with self._live({"qa": {"max_retries": 1}}):
            out = st.apply("qa.max_retries", 99)
        self.assertIn("outside 0–5", out["error"])
        self.assertEqual(self.written, [])

    def test_an_unwritable_settings_file_is_reported_not_swallowed(self):
        with mock.patch("src.utils.settings.set_local", side_effect=OSError("read-only")):
            with self._live({"qa": {"enabled": True}}):
                out = st.apply("qa.enabled", False)
        self.assertIn("could not write", out["error"])


class ThroughTheToolsTest(unittest.TestCase):

    def _call(self, name, **kw):
        import asyncio
        return json.loads(asyncio.run(tools(pipeline_stub())[name](**kw)))

    def test_listing_gives_the_keys_with_their_current_values(self):
        out = self._call("list_agent_settings")
        keys = [s["key"] for s in out["settings"]]
        self.assertIn("qa.enabled", keys)
        self.assertIn("canvas_full_graph", keys)
        for s in out["settings"]:
            self.assertIn("what", s)
            self.assertIn("takes_effect", s)

    def test_the_listing_never_carries_a_secret_or_a_path(self):
        blob = json.dumps(self._call("list_agent_settings")).lower()
        for leak in ("api_key", "http://", "c:/", "\\\\", ".sqlite"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, blob)

    def test_setting_something_off_the_list_is_refused_through_the_tool(self):
        out = self._call("set_agent_setting", key="output_dir", value="D:/elsewhere")
        self.assertIn("not a setting the agent can change", out["error"])

    def test_a_real_change_goes_through(self):
        with mock.patch("src.utils.agent_settings.apply",
                        return_value={"status": "changed", "key": "qa.enabled",
                                      "from": True, "to": False,
                                      "takes_effect": "on your next message",
                                      "message": "ok"}) as applied:
            out = self._call("set_agent_setting", key="qa.enabled", value=False)
        applied.assert_called_once_with("qa.enabled", False)
        self.assertEqual(out["status"], "changed")


if __name__ == "__main__":
    unittest.main()
