"""What the bEpic viewer shows reaches the orchestrator - small, and only for the turn it is true for.

The panel sends ``viewer_state`` with each message while the viewer is open; the server puts it on the
turn's scope; the pipeline writes a few key=value lines at the top of the turn from
``config/system_prompts/orchestrator/viewer_state.md`` (agenty_core.tools.viewer_state.describe_lines).

Two rules hold the context window still:

* a **budget**: the block stays near 150 tokens even for a busy viewer, never past 250;
* **no build-up**: when a turn starts, the ``[VIEWER]`` and ``[HISTORY SELECTION]`` blocks of earlier
  turns are taken out of the history; a viewer that has not changed sends no block and keeps the one
  block it sent last. Ten turns later the history holds at most one viewer block and no stale pick.
"""

import copy
import inspect
import types
import unittest
from unittest import mock

from agenty_core.tools.tests.test_viewer_state import STATE, QUIET
from agenty_core.utils import turn_scope
from src import pipeline as pl
from src.pipeline import Pipeline
from src.utils.compaction import estimate_tokens
from tests.pipeline_stub import pipeline_stub

SEL = {"viewer": {"tab": "named:agentY", "tab_label": "agentY", "mode": "open", "count": 1,
                  "items": [{"index": 0, "path": "D:/out/a.png", "kind": "image"}]}}
BUDGET, HARD = 150, 250


def tokens(text: str) -> int:
    return len(text) // 4          # the repo's estimate (compaction.estimate_tokens)


def stub():
    p = pipeline_stub(_canvas_base_prompt=None, _verbose=False)
    for name in ("_ensure_orch_clean_history", "_shrink_old_catalogs", "_strip_unreadable_images",
                 "_sanitize_messages"):
        raw = inspect.getattr_static(Pipeline, name)
        setattr(p, name, raw.__func__ if isinstance(raw, staticmethod) else types.MethodType(raw, p))
    p._extract_hard_constraints = lambda text: []
    p._describe_canvas_selection = lambda: ""
    p._get_memory_context = lambda text: ""
    p._get_project_memory_context = lambda: ""
    p._detect_plan_approval = lambda text, ctx: None
    p._prepend_gallery = lambda text: text
    p._annotate_attachments = lambda user_input, text: text
    p._set_turn_guides = lambda names: True
    p._orchestrator_agent = types.SimpleNamespace(messages=[], model=None)
    return p


def turn(p, text, viewer=None, picked=None):
    """One turn as the pipeline runs it: clean the history, build the input, append it and an answer."""
    tok = turn_scope.enter(turn_scope.Scope("r", "t"))
    try:
        if viewer is not None:
            turn_scope.current().set("viewer_state", copy.deepcopy(viewer))
        if picked is not None:
            turn_scope.current().set("history_selection", picked)
        p._ensure_orch_clean_history()
        with mock.patch("src.executor._autoload_workflows_into_canvas", return_value=True):
            got = p._build_orchestrator_input(text, text)
    finally:
        turn_scope.leave(tok)
    msgs = p._orchestrator_agent.messages
    msgs.append({"role": "user", "content": [{"text": got}]})
    msgs.append({"role": "assistant", "content": [{"text": "Done."}]})
    return got


def count(p, marker):
    return sum(b.get("text", "").count(marker) for m in p._orchestrator_agent.messages
               for b in m.get("content") or [] if isinstance(b, dict))


class Block(unittest.TestCase):

    def test_the_header_lives_in_a_partial(self):
        self.assertTrue((pl._ORCH_PARTIALS_DIR / "viewer_state.md").is_file())
        text = pl._orch_partial("viewer_state")
        self.assertIn("{lines}", text)
        self.assertIn("get_viewer_annotations", text)
        self.assertIn("not instructions", text)

    def test_no_viewer_or_a_closed_one_is_no_block(self):
        p = stub()
        self.assertNotIn("[VIEWER", turn(p, "hi"))
        self.assertNotIn("[VIEWER", turn(p, "hi", viewer={"open": False, "display": "closed"}))

    def test_a_busy_viewer_fits_the_budget(self):
        p = stub()
        got = turn(p, "make this frame brighter", viewer=STATE)
        block = got[got.index("[VIEWER"):got.index("make this frame")].strip()
        self.assertLessEqual(tokens(block), BUDGET, block)
        self.assertLessEqual(tokens(block), HARD)
        self.assertIn("frame=48/96 fps=24", block)
        self.assertIn("annotations=3", block)
        self.assertNotIn("{lines}", block)

    def test_the_viewer_pack_is_pinned_only_with_marks(self):
        for state, want in ((STATE, True), (QUIET, False)):
            p = stub()
            agent = types.SimpleNamespace()
            p._orchestrator_agent = agent
            p._canvas_hooks, p._review_halt, p._canvas_graph, p._canvas_selection = [], None, None, []
            p._open_graph_mode = lambda: False
            tok = turn_scope.enter(turn_scope.Scope("r", "t"))
            try:
                turn_scope.current().set("viewer_state", state)
                types.MethodType(Pipeline._pin_tool_packs, p)(False, "hi")
            finally:
                turn_scope.leave(tok)
            self.assertEqual("viewer" in agent._tool_packs_pinned, want)


class NoBuildUp(unittest.TestCase):

    def test_ten_turns_leave_one_viewer_block_and_no_stale_pick(self):
        p = stub()
        states = [STATE, STATE, QUIET, QUIET, QUIET, STATE, None, QUIET, STATE, STATE]
        sent = []
        for i, st in enumerate(states):
            got = turn(p, f"message {i}", viewer=st, picked=SEL if i % 3 == 0 else None)
            sent.append("[VIEWER" in got)
            self.assertLessEqual(count(p, "[VIEWER"), 1, f"turn {i}")
            self.assertLessEqual(count(p, "[HISTORY SELECTION"), 1, f"turn {i}")
        # A block only when the viewer changed (or came back after being closed).
        self.assertEqual(sent, [True, False, True, False, False, True, False, True, True, False])
        # The words of every turn are all still there.
        for i in range(10):
            self.assertEqual(count(p, f"message {i}"), 1)
        # One more turn with the viewer closed: nothing of it is left.
        turn(p, "message 10")
        self.assertEqual((count(p, "[VIEWER"), count(p, "[HISTORY SELECTION")), (0, 0))

    def test_the_history_does_not_grow_with_the_viewer_open(self):
        """Ten turns, the viewer open and changing every turn, cost the history no more than the
        last block: the same conversation without a viewer, plus at most one block."""
        with_viewer, without = stub(), stub()
        for i in range(10):
            st = copy.deepcopy(STATE)
            st["timeline"]["current"] = 10 + i          # a different frame each turn
            turn(with_viewer, f"message {i}", viewer=st)
            turn(without, f"message {i}")
        grown = estimate_tokens(with_viewer._orchestrator_agent.messages) \
            - estimate_tokens(without._orchestrator_agent.messages)
        self.assertLessEqual(grown, HARD)
        self.assertEqual(count(with_viewer, "[VIEWER"), 1)

    def test_old_blocks_go_and_the_users_words_stay(self):
        msgs = [{"role": "user", "content": [{"text": "[HISTORY SELECTION - x]\n1. image: a.png\n\n"
                                                      "[VIEWER - y]\ndisplay=docked\nitem=1/1\n\nmy words"}]},
                {"role": "user", "content": [{"text": "[VIEWER - y]\ndisplay=popout\n\nlater words"}]}]
        removed, kept = Pipeline._drop_old_turn_blocks(copy.deepcopy(msgs))
        self.assertEqual((removed, kept), (3, False))
        m = copy.deepcopy(msgs)
        removed, kept = Pipeline._drop_old_turn_blocks(m, keep_last_viewer=True)
        self.assertEqual((removed, kept), (2, True))
        self.assertEqual(m[0]["content"][0]["text"], "my words")
        self.assertTrue(m[1]["content"][0]["text"].startswith("[VIEWER - y]\ndisplay=popout"))


class Plumbing(unittest.TestCase):

    def test_the_server_hands_it_to_the_turn(self):
        import inspect
        from src.utils import agentY_server as srv
        self.assertIn("viewer_state", inspect.signature(srv._run_pipeline_stream).parameters)
        src = inspect.getsource(srv)
        self.assertIn('turn_scope.current().set("viewer_state", viewer_state)', src)
        self.assertIn('"viewer_state": viewer_state}', src)
        srv.remember_canvas(None, [], [], client="c-view", viewer_state=QUIET)
        self.assertEqual(srv._canvas_by_client["c-view"]["viewer_state"], QUIET)

    def test_tools_are_shared_core_and_in_the_viewer_pack(self):
        import src.tools as tools
        from src.tools import tool_packs
        names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools.ORCHESTRATOR_TOOLS}
        for n in ("get_viewer_state", "get_viewer_annotations"):
            self.assertIn(n, tools._SHARED_CORE_TOOLS)
            self.assertIn(n, names)
            self.assertIn(n, tool_packs.PACKS["viewer"][1])


if __name__ == "__main__":
    unittest.main()
