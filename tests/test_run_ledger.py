"""The run ledger: the host's own record of what happened to each stage.

Over a long pipeline run the agent lost track of what was settled - an approved
screenplay was rewritten, a review long answered was asked again, last round's
pictures went on as the approved set. Its only record was its memory of the
conversation. The host now keeps one, at the moments it knows something, and
tells the agent every turn.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import conversation_store as cs
from src.utils import review_gate as rg
from src.utils import run_ledger as rl

ROOT = Path(__file__).resolve().parent.parent


def hook(hid, purpose="set_parameter", title="", **more):
    return {"hook_node_id": hid, "purpose": purpose, "title": title, "directive": "", **more}


HOOKS = [hook("4", "text_only", "screenplay"), hook("131", "human_review", "read it"),
         hook("8", title="characters"), hook("135", "human_review", "pick"),
         hook("9", "loop_start"), hook("142", title="final shot")]


class Store(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._dir.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(self._dir.name, "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        cs.create_thread("lead", thread_id="lead")
        cs.create_thread("branch", thread_id="b1")
        cs.set_shot("b1", "lead", "characters")
        rl.sync("lead", HOOKS, {"8": "k1", "4": "t1"})


class WhereItIsKept(Store):
    def test_with_the_lead_conversation(self):
        self.assertEqual(rl.lead_of("lead"), "lead")

    def test_a_branch_writes_the_leads_copy(self):
        self.assertEqual(rl.lead_of("b1"), "lead")
        rl.produced(rl.lead_of("b1"), "8", outputs=["a.png"], by="characters")
        self.assertEqual(rl.stage("lead", "8")["outputs"], ["a.png"])
        self.assertIsNone(rl.stage("b1", "8"))

    def test_it_survives_a_restart(self):
        rl.produced("lead", "8", outputs=["a.png"])
        cs._INITIALISED = False
        self.assertEqual(rl.stage("lead", "8")["state"], rl.DONE)


class WhatHappensToAStage(Store):
    def test_it_starts_pending(self):
        self.assertEqual(rl.stage("lead", "8")["state"], rl.PENDING)

    def test_producing_is_a_round(self):
        rl.produced("lead", "8", outputs=["r1_a.png", "r1_b.png"], key="k1")
        st = rl.produced("lead", "8", outputs=["r2_a.png"], key="k1")
        self.assertEqual((st["round"], st["outputs"], st["state"]), (2, ["r2_a.png"], rl.DONE))

    def test_a_stop_is_waiting_for_the_user(self):
        rl.produced("lead", "8", outputs=["a.png", "b.png"])
        rl.waiting("lead", "135", ["8"], ["a.png", "b.png"])
        self.assertEqual(rl.stage("lead", "8")["state"], rl.WAITING)
        self.assertEqual(rl.stage("lead", "135")["state"], rl.WAITING)

    def test_an_approval_records_exactly_what_goes_on(self):
        rl.produced("lead", "8", outputs=["a.png", "b.png", "c.png"])
        rl.waiting("lead", "135", ["8"], ["a.png", "b.png", "c.png"])
        rl.approved("lead", "135", ["8"], ["b.png"])
        st = rl.stage("lead", "8")
        self.assertEqual((st["state"], st["approved"], st["review"]), (rl.APPROVED, ["b.png"], "135"))
        self.assertEqual(st["outputs"], ["a.png", "b.png", "c.png"])     # what it made is kept

    def test_redoing_an_approved_stage_takes_the_approval_away(self):
        rl.produced("lead", "8", outputs=["a.png"])
        rl.approved("lead", "135", ["8"], ["a.png"])
        st = rl.produced("lead", "8", outputs=["new.png"])
        self.assertEqual(st["state"], rl.DONE)
        self.assertNotIn("approved", st)

    def test_a_reopened_review_asks_again(self):
        rl.produced("lead", "8", outputs=["a.png"])
        rl.approved("lead", "135", ["8"], ["a.png"])
        rl.reopened("lead", ["135"])
        self.assertEqual(rl.stage("lead", "135")["state"], rl.PENDING)
        self.assertEqual(rl.stage("lead", "8")["state"], rl.DONE)


class ChangedSinceApproved(Store):
    def setUp(self):
        super().setUp()
        rl.produced("lead", "8", outputs=["a.png"], key="k1")
        rl.approved("lead", "135", ["8"], ["a.png"])

    def test_the_same_inputs_leave_it_settled(self):
        rl.sync("lead", HOOKS, {"8": "k1"})
        self.assertFalse(rl.stage("lead", "8")["changed"])

    def test_edited_inputs_are_flagged(self):
        rl.sync("lead", HOOKS, {"8": "k2"})
        st = rl.stage("lead", "8")
        self.assertTrue(st["changed"])
        self.assertEqual(st["state"], rl.APPROVED)            # flagged, not thrown away
        self.assertIn("BUT ITS INPUTS HAVE CHANGED SINCE", rl.index("lead"))

    def test_running_it_again_clears_the_flag(self):
        rl.sync("lead", HOOKS, {"8": "k2"})
        self.assertFalse(rl.produced("lead", "8", outputs=["b.png"], key="k2")["changed"])

    def test_a_stage_taken_off_the_canvas_leaves_the_index_but_not_the_record(self):
        rl.sync("lead", [h for h in HOOKS if h["hook_node_id"] != "8"], {})
        self.assertNotIn('"characters"', rl.index("lead"))
        self.assertEqual(rl.stage("lead", "8")["approved"], ["a.png"])

    def test_a_branch_seeing_only_its_stages_does_not_retire_the_others(self):
        rl.sync("lead", [HOOKS[5]], {}, partial=True)
        self.assertNotIn("gone", rl.stage("lead", "8"))


class WhatTheAgentIsTold(Store):
    def test_nothing_until_something_has_happened(self):
        self.assertEqual(rl.index("lead"), "")

    def test_one_entry_per_stage_in_canvas_order_without_the_loop_nodes(self):
        rl.produced("lead", "8", outputs=["a.png"])
        text = rl.index("lead")
        self.assertLess(text.index('- 4 "screenplay"'), text.index('- 8 "characters"'))
        self.assertLess(text.index('- 8 "characters"'), text.index('- 142 "final shot"'))
        self.assertNotIn("- 9", text)
        self.assertIn('- 142 "final shot" [set / sweep parameter]: not done yet', text)

    def test_an_approved_stage_lists_what_goes_on(self):
        rl.produced("lead", "8", outputs=["a.png", "b.png"], by="characters")
        rl.approved("lead", "135", ["8"], ["b.png"])
        text = rl.index("lead")
        self.assertIn("APPROVED by the user (review 135) · round 1 · by branch characters", text)
        self.assertIn("goes on (1): b.png", text)
        self.assertIn("it outranks your memory of the conversation", text)

    def test_a_long_list_is_cut_and_the_tool_named(self):
        files = [f"f{n}.png" for n in range(9)]
        rl.produced("lead", "8", outputs=files)
        self.assertIn("made (9): f0.png; f1.png; f2.png; f3.png; +5 more (ledger_stage(8))",
                      rl.index("lead"))

    def test_a_branch_is_shown_its_own_stages(self):
        rl.produced("lead", "8", outputs=["a.png"])
        rl.produced("lead", "4", text_path="s.md")
        text = rl.index("lead", only=["8", "135"])
        self.assertIn('- 8 "characters"', text)
        self.assertNotIn("screenplay", text)

    def test_the_counts_for_a_status_line(self):
        rl.produced("lead", "4", text_path="s.md")
        rl.approved("lead", "131", ["4"], [])
        rl.produced("lead", "8", outputs=["a.png"])
        rl.waiting("lead", "135", ["8"], ["a.png"])
        self.assertEqual(rl.summary("lead"), {"total": 3, "approved": 1, "waiting": 1, "done": 0})


class WrittenResults(Store):
    def test_in_the_project_layout_when_a_shot_is_named(self):
        path = rl.text_file("lead", "4", "Screenplay (draft)", 3,
                            ctx={"sequence": "Seq010", "shot": "Sh020"}, root=self._dir.name)
        self.assertEqual(Path(path), Path(self._dir.name) / "Seq010" / "Sh020" / "text"
                         / "sh020_v003_screenplay_draft.md")

    def test_with_the_agents_notes_otherwise(self):
        with mock.patch("src.utils.hook_cache.cache_dir", return_value=Path(self._dir.name) / "mem"):
            path = rl.text_file("lead", "4", "screenplay", 1, ctx={}, root=self._dir.name)
        self.assertEqual(Path(path), Path(self._dir.name) / "mem" / "texts" / "lead" / "screenplay_v001.md")

    def test_the_ledger_holds_the_path_and_the_file_holds_the_text(self):
        path = os.path.join(self._dir.name, "t", "s_v001.md")
        self.assertTrue(rl.write_text(path, "INT. KITCHEN - MORNING"))
        rl.produced("lead", "4", text_path=path)
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "INT. KITCHEN - MORNING")
        self.assertIn(f"text → {path}", rl.index("lead"))
        self.assertNotIn("KITCHEN", rl.index("lead"))


class ThePlanThePanelShows(Store):
    """GET /agentY/threads/<id>/plan - the ledger, grouped the way the run goes."""

    SHOTS = [{"shot": "characters", "thread_id": "b1", "status": "running"},
             {"shot": "places", "thread_id": "b2", "status": "done"}]

    def scope(self, tid):
        return {"b1": ["8", "135"], "b2": ["118"]}.get(tid, [])

    def setUp(self):
        super().setUp()
        rl.sync("lead", HOOKS + [hook("118", title="main location")], {})
        rl.set_groups("lead", ["4", "131"],
                      [{"after": "131", "members": ["8", "135"]},
                       {"after": "131", "members": ["9", "118"]}], ["142"])
        rl.produced("lead", "4", text_path="s.md")
        rl.approved("lead", "131", ["4"], [])
        rl.produced("lead", "8", outputs=["a.png", "b.png"])
        rl.waiting("lead", "135", ["8"], ["a.png", "b.png"])

    def test_trunk_then_each_branch_then_what_waits_for_them(self):
        groups = rl.plan("lead", self.SHOTS, self.scope)["groups"]
        self.assertEqual([(g["kind"], g["name"]) for g in groups],
                         [("trunk", "Before the branches"), ("branch", "characters"),
                          ("branch", "places"), ("joined", "After the branches")])
        self.assertEqual([r["id"] for r in groups[1]["stages"]], ["8", "135"])
        self.assertEqual([r["id"] for r in groups[2]["stages"]], ["118"])     # no loop node

    def test_a_branch_carries_the_conversation_working_on_it(self):
        branch = rl.plan("lead", self.SHOTS, self.scope)["groups"][1]
        self.assertEqual((branch["thread_id"], branch["status"]), ("b1", "running"))

    def test_a_row_says_where_the_stage_stands(self):
        row = rl.plan("lead", self.SHOTS, self.scope)["groups"][1]["stages"][0]
        self.assertEqual((row["state"], row["round"], row["files"], row["review"]),
                         (rl.WAITING, 1, 2, "135"))

    def test_the_one_line(self):
        line = rl.plan("lead", self.SHOTS, self.scope)["line"]
        self.assertEqual(line, "1/4 approved · 1 waiting for you · 1 branch running")

    def test_a_run_with_no_branches_is_one_group(self):
        rl.set_groups("lead", [], [], [])
        groups = rl.plan("lead", [], self.scope)["groups"]
        self.assertEqual([g["kind"] for g in groups], ["stages"])
        self.assertEqual([r["id"] for r in groups[0]["stages"]], ["4", "131", "8", "135", "142", "118"])

    def test_no_run_is_no_plan(self):
        cs.create_thread("x", thread_id="other")
        self.assertEqual(rl.plan("other")["groups"], [])

    def test_the_host_serves_it(self):
        src = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        self.assertIn('@app.route("/agentY/threads/<tid>/plan", methods=["GET"])', src)
        self.assertIn("plan = _rl.plan(lead, shots, _shots.scope_of)", src)

    def test_the_lead_records_the_layout_each_turn(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        start = src.split("    def _ledger_turn_start(self)", 1)[1].split("\n    def ", 1)[0]
        self.assertIn('self._ledger("set_groups", par.trunk if par else [],', start)


def pipeline(thread="lead", branch=None, reply="", halt=None, others=()):
    obj = types.SimpleNamespace(_verbose=False, _dry_run=False, _canvas_hooks=[dict(h) for h in HOOKS],
                                _session=types.SimpleNamespace(session_id=thread),
                                _review_halt=halt, _review_others=list(others),
                                _review_reply=reply, _hook_products={})
    obj._thread_id_now = lambda: thread
    obj._branch_info = lambda: branch
    obj._review_goes_on = lambda h: (list(h.produced)[:1], "the collector as it stands now")
    obj._close_finished_branches = lambda: []
    obj._remember_approved = lambda halt, files: ""      # project memory is not this test's
    for name in ("_ledger_lead", "_ledger", "_ledger_stages_of", "_ledger_turn_start",
                 "_ledger_approve", "_ledger_block", "_ledger_text", "_arm_review",
                 "_record_hook_products"):
        setattr(obj, name, getattr(Pipeline, name).__get__(obj))
    return obj


def stages_before(_hooks, rid):
    return {"131": ["4"], "135": ["8"]}.get(str(rid), [])


@mock.patch("src.utils.canvas_hooks.stage_before_review", stages_before)
@mock.patch("src.utils.canvas_hooks.product_consumers", lambda hooks, hid: [])
class ThePipelineWritesIt(Store):
    def test_a_stage_that_ran(self):
        obj = pipeline()
        obj._canvas_hooks[2]["_cache_key"] = "k1"
        obj._record_hook_products("8", ["a.png", "b.png"])
        st = rl.stage("lead", "8")
        self.assertEqual((st["state"], st["round"], st["outputs"], st["key"]),
                         (rl.DONE, 1, ["a.png", "b.png"], "k1"))

    def test_a_branch_writes_under_its_name_into_the_leads_ledger(self):
        obj = pipeline("b1", {"lead_id": "lead", "name": "characters"})
        obj._record_hook_products("8", ["a.png"])
        self.assertEqual(rl.stage("lead", "8")["by"], "characters")

    def test_a_dry_run_records_nothing(self):
        obj = pipeline()
        obj._dry_run = True
        obj._record_hook_products("8", ["stand-in.png"])
        self.assertEqual(rl.stage("lead", "8")["state"], rl.PENDING)

    def test_a_stop_raised(self):
        obj = pipeline()
        obj._review_armed, obj._review_armed_more = None, []
        obj._arm_review(rg.ReviewHalt(hook_node_id="135", collector_key="k", produced=("a.png", "b.png")))
        self.assertEqual(rl.stage("lead", "8")["state"], rl.WAITING)
        self.assertEqual(rl.stage("lead", "135")["outputs"], ["a.png", "b.png"])

    def test_a_continue_at_the_start_of_a_turn_approves_every_standing_stop(self):
        first = rg.ReviewHalt(hook_node_id="135", collector_key="k", produced=("a.png", "b.png"))
        text = rg.ReviewHalt(hook_node_id="131", text_hooks=("4",))
        pipeline(reply="continue", halt=first, others=[text])._ledger_turn_start()
        self.assertEqual(rl.stage("lead", "8")["approved"], ["a.png"])
        self.assertEqual(rl.stage("lead", "4")["state"], rl.APPROVED)
        self.assertEqual(rl.stage("lead", "4")["approved"], [])

    def test_a_message_that_is_not_an_answer_approves_nothing(self):
        halt = rg.ReviewHalt(hook_node_id="135", collector_key="k", produced=("a.png",))
        rl.waiting("lead", "135", ["8"], ["a.png"])
        pipeline(reply="", halt=halt)._ledger_turn_start()
        self.assertEqual(rl.stage("lead", "8")["state"], rl.WAITING)

    def test_a_written_stage_is_saved_as_a_file_one_version_per_round(self):
        obj = pipeline()
        with mock.patch("src.utils.hook_cache.cache_dir", return_value=Path(self._dir.name) / "mem"), \
                mock.patch("src.utils.output_context.current", return_value={}):
            first = obj._ledger_text(obj._canvas_hooks[0], "draft one")
            second = obj._ledger_text(obj._canvas_hooks[0], "draft two")
        self.assertTrue(first.endswith("screenplay_v001.md"))
        self.assertTrue(second.endswith("screenplay_v002.md"))
        self.assertEqual(Path(first).read_text(encoding="utf-8"), "draft one")
        self.assertEqual(rl.stage("lead", "4")["text"], second)
        self.assertEqual(rl.stage("lead", "4")["round"], 2)

    def test_the_block_a_branch_gets_is_its_own_stages(self):
        rl.produced("lead", "8", outputs=["a.png"])
        rl.produced("lead", "4", text_path="s.md")
        obj = pipeline("b1", {"lead_id": "lead", "name": "characters"})
        obj._canvas_hooks = [dict(HOOKS[2]), dict(HOOKS[3])]
        text = obj._ledger_block()
        self.assertIn('"characters"', text)
        self.assertNotIn("screenplay", text)


class WhereItIsWired(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def tool(self, name):
        return self.SRC.split(f"        async def {name}(", 1)[1].split("        @_tool", 1)[0]

    def test_every_turn_lines_it_up_after_the_stages_are_fingerprinted(self):
        self.assertIn("        self._apply_hook_cache()\n        self._ledger_turn_start()\n", self.SRC)

    def test_the_agent_is_given_it_with_the_stages(self):
        build = self.SRC.split("    def _build_orchestrator_input(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("_led = self._ledger_block()", build)

    def test_each_way_a_review_is_lifted_records_it(self):
        release = self.tool("release_review")
        self.assertEqual(release.count("self._ledger_approve("), 2)
        self.assertIn('self._ledger("approved", hid, _stages, _made)', release)

    def test_redoing_a_stage_reopens_its_review_on_record(self):
        body = self.SRC.split("    def _reopen_reviews_after(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn('_led("reopened", [r for r in passed if r not in keep])', body)

    def test_a_written_stage_is_saved_when_it_is_placed(self):
        place = self.tool("place_canvas_text")
        self.assertIn("_saved_as = self._ledger_text(hook, str(text))", place)
        self.assertIn('**({"saved_as": _saved_as} if _saved_as else {})', place)

    def test_the_agent_can_ask_for_one_stage_in_full(self):
        self.assertIn("list_outputs, show_in_panel, ledger_stage, ", self.SRC)
        self.assertIn('st.pop("key", None)', self.tool("ledger_stage"))


if __name__ == "__main__":
    unittest.main()
