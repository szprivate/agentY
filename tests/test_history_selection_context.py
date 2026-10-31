"""What the user picked in a history panel reaches the orchestrator.

The panel sends ``history_selection`` ({viewer, assets}) with each message; the
server puts it on the turn's scope; the pipeline lists it at the top of the
turn from ``config/system_prompts/orchestrator/history_selection.md``, only when
something is picked, and pins the viewer pack whose ``get_history_selection``
reads it again mid-turn.
"""

import inspect
import unittest
from unittest import mock

from agenty_core.utils import turn_scope
from src import pipeline as pl
from tests.pipeline_stub import pipeline_stub

SEL = {"viewer": {"tab": "named:agentY", "tab_label": "agentY", "mode": "multi", "count": 2,
                  "items": [{"index": 0, "path": "D:/out/a.png", "kind": "image", "label": "Hero"},
                            {"index": 2, "path": "D:/out/b.mp4", "kind": "video"}]}}


class Block(unittest.TestCase):

    def setUp(self):
        tok = turn_scope.enter(turn_scope.Scope("r", "t"))
        self.addCleanup(turn_scope.leave, tok)

    def test_the_header_lives_in_a_partial(self):
        self.assertTrue((pl._ORCH_PARTIALS_DIR / "history_selection.md").is_file())
        text = pl._orch_partial("history_selection")
        self.assertIn("{list}", text)
        self.assertIn("get_history_selection", text)

    def test_nothing_picked_is_no_block(self):
        p = pipeline_stub()
        self.assertEqual(p._describe_history_selection(), "")

    def test_a_pick_is_listed_with_where_it_was_made(self):
        turn_scope.current().set("history_selection", SEL)
        p = pipeline_stub()
        block = p._describe_history_selection()
        self.assertTrue(block.startswith("[HISTORY SELECTION"), block[:60])
        self.assertIn('bEpic viewer history, tab "agentY" (selected)', block)
        self.assertIn("1. image: D:/out/a.png", block)
        self.assertIn("2. video: D:/out/b.mp4", block)
        self.assertNotIn("{list}", block)

    def test_it_is_resolved_once_per_turn(self):
        turn_scope.current().set("history_selection", SEL)
        p = pipeline_stub()
        with mock.patch("agenty_core.tools.history_selection.resolve",
                        return_value={"items": [], "count": 0}) as res:
            p._history_selection()
            p._history_selection()
        self.assertEqual(res.call_count, 1)

    def test_the_turn_input_carries_it_and_the_input_guidance(self):
        turn_scope.current().set("history_selection", SEL)
        p = pipeline_stub(_canvas_base_prompt=None)
        p._extract_hard_constraints = lambda text: []
        p._describe_canvas_selection = lambda: ""
        p._get_memory_context = lambda text: ""
        p._get_project_memory_context = lambda: ""
        p._detect_plan_approval = lambda text, ctx: None
        p._prepend_gallery = lambda text: text
        p._annotate_attachments = lambda user_input, text: text
        seen = {}
        p._set_turn_guides = lambda names: seen.setdefault("g", list(names)) is None
        with mock.patch("src.executor._autoload_workflows_into_canvas", return_value=True):
            got = p._build_orchestrator_input("use the selected ones", "use the selected ones")
        self.assertIn("[HISTORY SELECTION", got)
        self.assertIn("input_images", seen.get("g", []))


class Plumbing(unittest.TestCase):

    def test_the_server_hands_it_to_the_turn(self):
        from src.utils import agentY_server as srv
        self.assertIn("history_selection", inspect.signature(srv._run_pipeline_stream).parameters)
        src = inspect.getsource(srv)
        self.assertIn('turn_scope.current().set("history_selection", history_selection)', src)
        self.assertIn('"history_selection": history_selection,', src)

    def test_remember_canvas_keeps_it_for_a_slack_turn(self):
        from src.utils import agentY_server as srv
        srv.remember_canvas(None, [], [], client="c-hist", history_selection=SEL)
        self.assertEqual(srv._canvas_by_client["c-hist"]["history_selection"], SEL)

    def test_tool_is_shared_core_and_in_the_viewer_pack(self):
        import src.tools as tools
        from src.tools import tool_packs
        self.assertIn("get_history_selection", tools._SHARED_CORE_TOOLS)
        names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools.ORCHESTRATOR_TOOLS}
        self.assertIn("get_history_selection", names)
        self.assertIn("get_history_selection", tool_packs.PACKS["viewer"][1])


if __name__ == "__main__":
    unittest.main()
