"""The Canvas settings: the new one, and whether the panel can explain any of them.

Dropping every finished render onto the canvas is right until you are generating
in bulk, at which point it buries the graph you are working in. So it is a switch
now — and because a result that is neither drawn nor dropped would exist only as a
file nobody mentioned, turning it off has to leave the chat saying where the file
went.

The rest of this file is about the settings *form*. It is generated from the TOML,
so a key listed in a section that no longer exists renders nothing at all, and a
switch whose name does not carry its trade-off is a switch people leave alone.
Neither failure is visible from either file on its own.
"""

import re
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "settings.default.toml"

# The panel lives in its own repo, normally beside ComfyUI.
_PANEL = None
for _base in (ROOT.parent / "ComfyUI" / "custom_nodes", ROOT.parent):
    _candidate = _base / "agentY-comfyuiConnect" / "web" / "agent_settings.js"
    if _candidate.exists():
        _PANEL = _candidate
        break


def defaults():
    with SETTINGS.open("rb") as fh:
        return tomllib.load(fh)


class TheSetting(unittest.TestCase):
    def test_it_is_on_by_default(self):
        """A result you can wire into the next step beats a filename in a log, so
        the default is the helpful one and the switch is for opting out."""
        self.assertIs(defaults()["drop_outputs_into_canvas"], True)

    def _decide(self, settings=None, env=None):
        from src.utils import agentY_server as srv
        with mock.patch.dict("os.environ", env or {}, clear=False):
            if env is None:
                import os
                os.environ.pop("AGENTY_CANVAS_DROP", None)
            with mock.patch("src.agent._load_settings", return_value=settings or {}):
                return srv._drop_outputs_into_canvas()

    def test_an_unset_setting_means_on(self):
        """An older settings.local.json has no such key, and an install that
        upgrades must not quietly stop putting results on the canvas."""
        self.assertTrue(self._decide({}))

    def test_it_can_be_switched_off(self):
        self.assertFalse(self._decide({"drop_outputs_into_canvas": False}))

    def test_the_env_var_wins_both_ways(self):
        self.assertFalse(self._decide({"drop_outputs_into_canvas": True},
                                      {"AGENTY_CANVAS_DROP": "0"}))
        self.assertTrue(self._decide({"drop_outputs_into_canvas": False},
                                     {"AGENTY_CANVAS_DROP": "1"}))

    def test_a_meaningless_env_value_defers_to_the_setting(self):
        self.assertFalse(self._decide({"drop_outputs_into_canvas": False},
                                      {"AGENTY_CANVAS_DROP": "maybe"}))

    def test_unreadable_settings_mean_on(self):
        """Failing closed here would silently drop the feature, not protect
        anything."""
        from src.utils import agentY_server as srv
        with mock.patch("src.agent._load_settings", side_effect=OSError("boom")):
            self.assertTrue(srv._drop_outputs_into_canvas())


class EveryRouteAResultArrivesBy(unittest.TestCase):
    """One answer, decided by the host.

    A result reaches the canvas from a turn's own stream, from a Magnific
    completion minutes later, and from a Slack-driven run watched in the sidebar.
    A browser-side toggle would have to be repeated in each, and would disagree
    with itself the moment one was missed.
    """

    def test_the_host_stamps_the_decision_on_what_it_sends(self):
        source = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        emitters = source.count('"type": "output"')
        self.assertGreater(emitters, 0)
        self.assertEqual(source.count('"drop": _drop_outputs_into_canvas()'),
                         emitters + 1,
                         "every output event, plus the background Magnific drop")


@unittest.skipIf(_PANEL is None, "the agentY-comfyuiConnect extension is not beside this checkout")
class TheSettingsForm(unittest.TestCase):
    """The Canvas section, as the panel will render it."""

    @classmethod
    def setUpClass(cls):
        cls.src = _PANEL.read_text(encoding="utf-8")
        block = re.search(r'\{ title: "Canvas", keys: \[(.*?)\] \}', cls.src, re.S)
        assert block, "the Canvas section is gone or was reshaped"
        cls.canvas_keys = re.findall(r'"([a-z_]+)"', block.group(1))
        notes = re.search(r"const KEY_NOTES = \{(.*?)\n\};", cls.src, re.S)
        cls.noted = set(re.findall(r"^  ([a-z_]+):", notes.group(1), re.M)) if notes else set()

    def test_the_new_switch_is_offered(self):
        self.assertIn("drop_outputs_into_canvas", self.canvas_keys)

    def test_every_canvas_setting_explains_itself(self):
        """The TOML comments never reach the browser, so this is the only place a
        switch can say what turning it on costs."""
        for key in self.canvas_keys:
            with self.subTest(key=key):
                self.assertIn(key, self.noted)

    def test_every_canvas_setting_actually_exists(self):
        """The form is generated from the TOML: a section listing a key that is
        gone renders nothing, and the section quietly shrinks."""
        top_level = defaults()
        for key in self.canvas_keys:
            with self.subTest(key=key):
                self.assertIn(key, top_level)

    def test_no_note_is_a_restatement_of_the_name(self):
        """A note has to add something. "Canvas full graph: shows the full graph"
        is worse than none — it costs a line and teaches people not to read them.
        """
        for key in self.canvas_keys:
            with self.subTest(key=key):
                note = re.search(rf"^  {key}:\n?(.*?)(?=^  [a-z_]+:|\Z)",
                                 self.src, re.S | re.M)
                self.assertIsNotNone(note)
                words = len(re.findall(r"[A-Za-z']+", note.group(1)))
                self.assertGreater(words, 25, f"{key}'s note is too short to explain it")


@unittest.skipIf(_PANEL is None, "the agentY-comfyuiConnect extension is not beside this checkout")
class ThePanelHonoursIt(unittest.TestCase):
    def setUp(self):
        self.chat = (_PANEL.parent / "agent_chat.js").read_text(encoding="utf-8")

    def test_it_checks_before_it_drops(self):
        inject = self.chat[self.chat.index("\n  injectNode(ev) {"):]
        inject = inject[:inject.index("\n  _attachRefNote(")]
        self.assertIn("ev.drop === false", inject)
        self.assertLess(inject.index("ev.drop === false"), inject.index("createNode"),
                        "the check must come before the node is made")

    def test_only_an_explicit_false_turns_it_off(self):
        """A host older than this setting sends no `drop` field at all, and the
        old behaviour is the on one — `!ev.drop` would have made every upgrade
        stop putting results on the canvas."""
        self.assertIn("ev.drop === false", self.chat)
        self.assertNotIn("!ev.drop", self.chat)

    def test_it_still_says_where_the_file_went(self):
        """The panel renders no media inline. With no node and no line, a result
        would exist only as a file nobody was told about."""
        inject = self.chat[self.chat.index("\n  injectNode(ev) {"):]
        inject = inject[:inject.index("\n  _attachRefNote(")]
        skipped = inject[:inject.index("createNode")]
        self.assertIn("ev.path", skipped)


if __name__ == "__main__":
    unittest.main()
