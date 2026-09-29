"""The prompt loop's unsupervised mode: the bookkeeping, and the round tool.

The supervised loop (revise_prompt) stays exactly as it is; these cover the mode
asked for in chat, where agentY runs the graph, the QA agent judges, and the
orchestrator writes the next version until it passes, runs out, or circles:

* circling — a criterion failing twice is re-rolled with a fresh seed, a third
  failure stops the loop;
* keeping the best version rather than the last;
* the round itself: a version on the canvas, a run agentY makes (the panel is
  told NOT to queue), the QA card, and the best version put back at the end.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import prompt_autoloop as auto
from src.utils import prompt_loop as pl


def _run(v, passed=False, missed=(), score=None, reroll=False, error="", output="x.png"):
    return {"v": v, "text": f"p{v}", "passed": passed, "missed": list(missed),
            "score": score, "reroll": reroll, "error": error, "output": output}


class Circling(unittest.TestCase):

    def _session(self, *runs, budget=10):
        s = auto.new_session(budget, object())
        s["runs"] = list(runs)
        return s

    def test_a_fresh_failure_is_revised(self):
        s = self._session(_run(1, missed=["screens lit — dark"]))
        self.assertEqual(auto.decide(s)[0], "revise")

    def test_the_same_criterion_twice_is_rerolled(self):
        s = self._session(_run(1, missed=["screens lit — dark"]),
                          _run(2, missed=["Screens lit — still dim"]))
        nxt, why = auto.decide(s)
        self.assertEqual(nxt, "reroll")
        self.assertIn("screens lit", why)

    def test_a_reroll_is_not_rerolled_again(self):
        s = self._session(_run(1, missed=["a — x"]), _run(2, missed=["a — y"], reroll=True))
        self.assertEqual(auto.decide(s)[0], "revise")

    def test_three_in_a_row_is_stalled(self):
        s = self._session(_run(1, missed=["a — x"]), _run(2, missed=["a — y"]),
                          _run(3, missed=["a — z"], reroll=True))
        self.assertEqual(auto.decide(s)[0], "stalled")

    def test_a_break_in_the_streak_resets_it(self):
        s = self._session(_run(1, missed=["a — x"]), _run(2, missed=["b — y"]),
                          _run(3, missed=["a — z"]))
        self.assertEqual(auto.streaks(s["runs"]), {"a": 1})
        self.assertEqual(auto.decide(s)[0], "revise")

    def test_a_pass_ends_it(self):
        s = self._session(_run(1, missed=["a — x"]), _run(2, passed=True))
        self.assertEqual(auto.decide(s)[0], "passed")

    def test_the_budget_ends_it(self):
        s = self._session(_run(1, missed=["a — x"]), _run(2, missed=["b — y"]), budget=2)
        self.assertEqual(auto.decide(s)[0], "budget")

    def test_an_unreadable_judge_ends_it_unpassed(self):
        s = self._session(_run(1, passed=False, error="no vision"))
        self.assertEqual(auto.decide(s)[0], "unjudged")

    def test_the_user_speaking_ends_it(self):
        s = self._session(_run(1, missed=["a — x"]))
        self.assertEqual(auto.decide(s, interrupted=True)[0], "interrupted")


class KeepingTheBest(unittest.TestCase):

    def test_a_pass_wins(self):
        runs = [_run(1, missed=["a"], score=0.9), _run(2, passed=True, score=0.4)]
        self.assertEqual(auto.best(runs)["v"], 2)

    def test_fewer_misses_beat_a_later_version(self):
        runs = [_run(1, missed=["a"], score=0.5), _run(2, missed=["a", "b"], score=0.9)]
        self.assertEqual(auto.best(runs)["v"], 1)

    def test_the_score_breaks_a_tie(self):
        runs = [_run(1, missed=["a"], score=0.8), _run(2, missed=["b"], score=0.6)]
        self.assertEqual(auto.best(runs)["v"], 1)

    def test_unmeasured_ranks_below_measured(self):
        runs = [_run(1, missed=["a"], score=None), _run(2, missed=["b"], score=0.1)]
        self.assertEqual(auto.best(runs)["v"], 2)

    def test_the_summary_names_the_winner_and_the_objections(self):
        s = auto.new_session(3, object())
        s["runs"] = [_run(1, missed=["a — x", "b — y"], score=0.7),
                     _run(2, missed=["a — z"], score=0.5), _run(3, missed=["a — w", "b"])]
        out = auto.summary(s, "budget", "all 3 runs used")
        self.assertEqual(out["best"]["v"], 2)
        self.assertFalse(out["best"]["is_last"])
        self.assertEqual(out["kept_objecting_to"][0], "a (failed 3×)")


class Store(dict):
    def get_prompt_loop(self, thread_id):
        return self.get(thread_id)

    def set_prompt_loop(self, thread_id, loop):
        if loop is None:
            self.pop(thread_id, None)
        else:
            self[thread_id] = loop


class TheRound(unittest.TestCase):
    """prompt_autoloop, with ComfyUI and the judge stood in for."""

    def setUp(self):
        self.store = Store()
        for target, kw in (("src.utils.prompt_loop._store", {"return_value": self.store}),
                           ("src.utils.fitness.score_file", {"side_effect": self._score}),
                           ("src.utils.qa.check_output", {"side_effect": self._judge}),
                           ("src.executor.execute_workflow", {"side_effect": self._execute}),
                           ("src.utils.interject_bus.pending_count", {"return_value": 0})):
            p = mock.patch(target, **kw)
            p.start()
            self.addCleanup(p.stop)
        from src.utils import canvas_patch, tool_activity
        canvas_patch.clear()
        tool_activity.clear()
        self.addCleanup(canvas_patch.clear)
        self.patches, self.cards = canvas_patch, tool_activity
        pl.start("t1", "6", "text")
        self.verdicts = []          # what the judge says, round by round
        self.scores = {}
        self.ran = []

    # -- stand-ins ------------------------------------------------------------
    async def _execute(self, wf, brief, user_message="", verbose=False,
                       collected_paths=None, qa_briefing=None):
        from pathlib import Path
        graph = json.loads(Path(wf).read_text(encoding="utf-8"))
        self.ran.append(graph)
        collected_paths.append(f"W:/out/r{len(self.ran)}.png")
        yield "done"

    def _judge(self, path, briefing, request=""):
        from src.utils.qa import QaResult
        passed, missed = self.verdicts.pop(0)
        return QaResult(path=path, passed=passed, summary="",
                        checks=[{"criterion": m, "result": "fail", "note": "n"}
                                for m in missed])

    def _score(self, path):
        return {"score": self.scores.get(path, 0.5)}

    def _pipe(self):
        graph = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
                 "7": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        from src.utils.qa import QaBriefing
        return pipeline_stub(_canvas_graph=graph, _canvas_base_prompt=graph,
                             _session=mock.Mock(session_id="t1", current_output_paths=[]),
                             _qa_briefing=QaBriefing(criteria="screens lit"),
                             _autoloop=None, _loop_writing=False)

    def _call(self, pipe, **kw):
        return json.loads(asyncio.run(tools(pipe)["prompt_autoloop"](**kw)))

    # -- tests ------------------------------------------------------------------
    def test_a_round_writes_runs_and_judges(self):
        self.verdicts = [(False, ["screens lit"])]
        out = self._call(self._pipe(), text="stadium, lit screens", max_runs=4)
        self.assertEqual((out["next"], out["version"], out["of"]), ("revise", 1, 4))
        self.assertEqual(out["missed"], ["screens lit — n"])
        self.assertEqual(self.ran[0]["6"]["inputs"]["text"], "stadium, lit screens")
        strip = [e for e in self.patches.drain() if e.get("op") == "prompt_version"]
        self.assertFalse(strip[0]["queue"], "agentY runs it — the panel must not queue it")
        cards = [c for c in self.cards.drain() if c.get("agent") == "qa"]
        self.assertEqual([c["phase"] for c in cards], ["call", "result"])

    def test_it_ends_at_the_first_pass(self):
        self.verdicts = [(False, ["a"]), (True, [])]
        pipe = self._pipe()
        self._call(pipe, text="one", max_runs=5)
        out = self._call(pipe, text="two")
        self.assertEqual(out["next"], "passed")
        self.assertEqual(out["summary"]["best"]["v"], 2)
        self.assertEqual(len(self.ran), 2)

    def test_a_reroll_reruns_the_same_prompt_with_a_new_seed(self):
        self.verdicts = [(False, ["a"]), (False, ["a"]), (False, ["b"])]
        pipe = self._pipe()
        self._call(pipe, text="one", max_runs=5)
        out = self._call(pipe, text="two")
        self.assertEqual(out["next"], "reroll")
        out = self._call(pipe, reroll_seed=True)
        self.assertEqual(self.ran[2]["6"]["inputs"]["text"], "two")
        self.assertNotEqual(self.ran[2]["7"]["inputs"]["seed"], 1)
        self.assertEqual(out["version"], 3)

    def test_the_best_version_goes_back_on_the_canvas(self):
        self.verdicts = [(False, ["a"]), (False, ["a", "b"])]
        self.scores = {"W:/out/r1.png": 0.9, "W:/out/r2.png": 0.2}
        pipe = self._pipe()
        self._call(pipe, text="good", max_runs=2)
        self.patches.drain()
        out = self._call(pipe, text="worse")
        self.assertEqual(out["next"], "budget")
        self.assertEqual(out["summary"]["best"]["v"], 1)
        events = self.patches.drain()
        self.assertIn({"text": "good"}, [e.get("params") for e in events if e.get("params")])
        self.assertEqual(pl.current("t1")["v"], 1)

    def test_no_qa_node_and_no_goal_is_refused(self):
        pipe = self._pipe()
        pipe._qa_briefing = None
        out = self._call(pipe, text="one")
        self.assertIn("nothing to judge against", out["error"])

    def test_a_goal_alone_is_enough(self):
        self.verdicts = [(True, [])]
        pipe = self._pipe()
        pipe._qa_briefing = None
        out = self._call(pipe, text="one", goal="the screens are lit")
        self.assertEqual(out["next"], "passed")

    def test_the_loop_must_be_on(self):
        pl.stop("t1")
        out = self._call(self._pipe(), text="one")
        self.assertIn("prompt loop is not on", out["error"])

    def test_it_is_offered_to_the_orchestrator_beside_revise_prompt(self):
        names = set(tools(self._pipe()))
        self.assertTrue({"revise_prompt", "prompt_autoloop"} <= names)


if __name__ == "__main__":
    unittest.main()
