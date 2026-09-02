"""The Canvas settings, and whether the panel can explain any of them.

Two switches about the same instinct: agentY puts what it makes onto the canvas,
and past a certain volume that buries the graph you are working in.

Dropping every finished render onto the canvas is right until you are generating
in bulk. And because a result that is neither drawn nor dropped would exist only
as a file nobody mentioned, turning it off has to leave the chat saying where the
file went.

Placing a text hook's answer as an "agentY text" node is the same trade at a
smaller scale, but with a difference worth pinning: that node is a readable copy
and nothing else. The hook stays wired and the answer is injected into the graph
at run time either way, so the switch must not touch the injection — and the tool
result must stop claiming a node was placed, or the agent tells the user to go
look at something that is not there.

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



def decide(fn, settings=None, env=None):
    """Call one of the server's canvas switches with a known settings/env pair."""
    from src.utils import agentY_server as srv
    clean = {"AGENTY_CANVAS_DROP": "", "AGENTY_CANVAS_TEXT": ""}
    with mock.patch.dict("os.environ", {**clean, **(env or {})}, clear=False):
        for name, value in clean.items():
            if not (env or {}).get(name):
                import os
                os.environ.pop(name, None)
        with mock.patch("src.agent._load_settings", return_value=settings or {}):
            return getattr(srv, fn)()


class ThePlaceTextSetting(unittest.TestCase):
    def test_it_is_on_by_default(self):
        self.assertIs(defaults()["place_text_nodes_on_canvas"], True)

    def test_an_unset_setting_means_on(self):
        """Same upgrade path as the drop switch: an older settings.local.json has
        no such key, and installing an update must not change what the canvas
        does."""
        self.assertTrue(decide("_place_text_nodes_on_canvas", {}))

    def test_it_can_be_switched_off(self):
        self.assertFalse(decide("_place_text_nodes_on_canvas",
                                {"place_text_nodes_on_canvas": False}))

    def test_the_env_var_wins_both_ways(self):
        self.assertFalse(decide("_place_text_nodes_on_canvas",
                                {"place_text_nodes_on_canvas": True},
                                {"AGENTY_CANVAS_TEXT": "0"}))
        self.assertTrue(decide("_place_text_nodes_on_canvas",
                               {"place_text_nodes_on_canvas": False},
                               {"AGENTY_CANVAS_TEXT": "1"}))

    def test_unreadable_settings_mean_on(self):
        from src.utils import agentY_server as srv
        with mock.patch("src.agent._load_settings", side_effect=OSError("boom")):
            self.assertTrue(srv._place_text_nodes_on_canvas())

    def test_the_two_switches_do_not_move_together(self):
        """They share one helper, so a copied env var name or settings key would
        tie them silently: turning off bulk media drops would also stop text
        answers being placed, which is not what either switch says it does."""
        both_off = {"drop_outputs_into_canvas": False,
                    "place_text_nodes_on_canvas": False}
        self.assertTrue(decide("_place_text_nodes_on_canvas", both_off,
                               {"AGENTY_CANVAS_TEXT": "1"}))
        self.assertFalse(decide("_drop_outputs_into_canvas", both_off,
                                {"AGENTY_CANVAS_TEXT": "1"}))
        self.assertTrue(decide("_drop_outputs_into_canvas", both_off,
                               {"AGENTY_CANVAS_DROP": "1"}))
        self.assertFalse(decide("_place_text_nodes_on_canvas", both_off,
                                {"AGENTY_CANVAS_DROP": "1"}))


class TheTextHookPath(unittest.TestCase):
    """What `place_canvas_text` does with the answer once it has one."""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        start = cls.src.index("async def place_canvas_text(")
        cls.tool = cls.src[start:cls.src.index("async def iterate_step(", start)]

    def test_the_host_decides_and_stamps_it_on_the_event(self):
        """The panel must not read this setting itself — it is the same argument
        as the drop switch, one answer for every route."""
        self.assertIn('"place": place', self.tool)
        self.assertIn("_place_text_nodes_on_canvas()", self.tool)

    def test_the_answer_still_reaches_the_graph_when_the_node_does_not(self):
        """The switch governs the visible copy only. Reading it before the
        injection would let a `False` skip the injection too, and a text hook
        would then deliver nothing to the nodes wired downstream of it."""
        self.assertLess(self.tool.index("_inject(self._canvas_base_prompt"),
                        self.tool.index("_place_text_nodes_on_canvas()"),
                        "the value must be injected before the switch is read")

    def test_the_tool_does_not_report_a_node_it_did_not_place(self):
        """The agent repeats this message to the user almost verbatim. "Placed an
        agentY text node on the canvas" with the switch off sends someone hunting
        the graph for something that was never added."""
        self.assertIn('"status": "placed" if place else "injected"', self.tool)
        self.assertIn("Placed NO node on the canvas", self.tool)


class SlackSaysTheSameThing(unittest.TestCase):
    """The sidebar is not the only place a turn is narrated."""

    def render(self, event):
        from src.utils.slack_render import TurnRender
        renderer = TurnRender.__new__(TurnRender)
        seen = []
        renderer._log = lambda text: seen.append(text) or []
        renderer._on_canvas_patch(event)
        return " ".join(seen)

    def test_a_placed_node_is_still_announced(self):
        self.assertIn("canvas", self.render({"op": "place_text", "place": True}))

    def test_a_host_without_the_setting_still_announces(self):
        self.assertIn("canvas", self.render({"op": "place_text"}))

    def test_nothing_placed_is_not_reported_as_placed(self):
        said = self.render({"op": "place_text", "place": False})
        self.assertNotIn("Placed", said)
        self.assertIn("graph", said, "the injection is the part nobody can see")


@unittest.skipIf(_PANEL is None, "the agentY-comfyuiConnect extension is not beside this checkout")
class ThePanelHonoursTheTextSetting(unittest.TestCase):
    def setUp(self):
        self.chat = (_PANEL.parent / "agent_chat.js").read_text(encoding="utf-8")
        body = self.chat[self.chat.index("\n  _placeCanvasText(ev) {"):]
        self.body = body[:body.index("\n  // \u2500\u2500 canvas hooks")]

    def test_it_checks_before_it_creates_the_node(self):
        self.assertIn("ev.place === false", self.body)
        self.assertLess(self.body.index("ev.place === false"),
                        self.body.index("createNode"),
                        "the check must come before the node is made")

    def test_only_an_explicit_false_turns_it_off(self):
        """A host older than this setting sends no `place` field, and its
        behaviour was to place."""
        self.assertNotIn("!ev.place", self.chat)

    def test_it_still_says_the_value_reached_the_graph(self):
        """With no node and no line, the one invisible half of a text hook — the
        run-time injection — would have nothing anywhere saying it happened."""
        skipped = self.body[:self.body.index("createNode")]
        self.assertIn("injected", skipped)

    def test_it_does_not_claim_something_landed_offscreen(self):
        """_noteOffscreenDrop() tells the user to go and look at another tab. With
        placement off there is nothing there to look at."""
        case = self.chat[self.chat.index('case "canvas_patch":'):]
        case = case[:case.index('case "system":')]
        guard = case[:case.index("_noteOffscreenDrop()")]
        self.assertIn('ev.op === "place_text" && ev.place === false', guard)


if __name__ == "__main__":
    unittest.main()
