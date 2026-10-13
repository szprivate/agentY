"""The ``agentY condition`` node as the judge of a loop.

On the execution wire, inside a loop, the node's decision model judges each
round by the node's statements; the QA agent is asked only about what passed,
and only if the break has a condition of its own.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import condition_judge as cj
from src.utils import hook_flow as hf
from src.utils import loop_judge as lj
from src.utils.canvas_hooks import strip_exec_links


def _hook(hid, purpose="make_workflow", via=(), **kw):
    via = [str(v) for v in via]
    return {"hook_node_id": str(hid), "purpose": purpose, "title": "",
            "directive": "" if hf.is_flow({"purpose": purpose}) else f"stage {hid}",
            "prev_hook_ids": [], "prev_hook_id": None, "prev_links": [], "anchors": [], "targets": [],
            "via_hook_ids": via, "exec_wired": True, **kw}


def _pipe(condition="the woman looks frustrated\nnot: the woman looks angry", brk="", **judge):
    return [
        _hook(10, hf.LOOP_START),
        _hook(4, via=[10]),
        _hook(20, hf.CONDITION, via=[4], condition=condition, model=judge.get("model", "clef"),
              threshold=judge.get("threshold", 0.6), batch=judge.get("batch", cj.BATCH_ALL)),
        _hook(11, hf.LOOP_BREAK, via=[20], condition=brk, max_rounds=3, forward="best"),
        _hook(5, via=[11]),
    ]


class OnTheWire(unittest.TestCase):

    def test_a_condition_node_in_a_loop_is_its_judge_and_not_a_stage(self):
        flow = hf.plan(_pipe())
        loop = flow.loops[0]
        self.assertEqual(loop.members, ["4"])
        self.assertEqual(loop.judge, {"node_id": "20", "model": "clef", "threshold": 0.6, "batch": cj.BATCH_ALL,
                                      "condition": "the woman looks frustrated\nnot: the woman looks angry"})
        self.assertEqual([h["hook_node_id"] for h in flow.hooks], ["4", "5"])
        self.assertEqual(flow.hooks[1]["via_hook_ids"], ["4"])          # the wire runs through it
        self.assertEqual(flow.problems, [])

    def test_it_is_enough_to_say_when_the_loop_is_finished(self):
        """A break with no condition and no review used to be a wiring note."""
        self.assertFalse(any("nothing says when" in p for p in hf.plan(_pipe()).problems))

    def test_what_is_wired_wrong_is_said(self):
        empty = hf.plan(_pipe(condition="  "))
        self.assertEqual(empty.loops[0].judge, {})
        self.assertTrue(any("no statement" in p for p in empty.problems))
        outside = hf.plan([_hook(4), _hook(20, hf.CONDITION, via=[4], condition="x"), _hook(5, via=[20])])
        self.assertTrue(any("not inside" in p and "20" in p for p in outside.problems))
        self.assertEqual(outside.hooks[1]["via_hook_ids"], ["4"])

    def test_what_the_node_says_is_held_to_what_it_offers(self):
        got = cj.settings({"hook_node_id": 7, "condition": " a \n\n b ", "model": "gpt", "threshold": 7, "batch": "?"})
        self.assertEqual(got, {"node_id": "7", "condition": "a\nb", "model": "clef-flash", "threshold": 0.95,
                               "batch": cj.BATCH_ALL})

    def test_the_agent_is_told_who_judges(self):
        text = "\n".join(hf.loop_lines(hf.plan(_pipe(brk="the hands have five fingers"))))
        self.assertIn("JUDGED BY THE CONDITION NODE 20", text)
        self.assertIn("clef", text)
        self.assertIn('and then, by the QA agent: "the hands have five fingers"', text)

    def test_its_exec_wire_comes_off_and_the_node_stays(self):
        graph = {"20": {"class_type": "AgentYCondition", "inputs": {"exec": ["4", 1], "images": ["8", 0],
                                                                    "condition": "x"}}}
        strip_exec_links(graph)
        self.assertEqual(graph["20"]["inputs"], {"images": ["8", 0], "condition": "x"})


class TheStatements(unittest.TestCase):

    def test_they_are_read_as_the_node_reads_them(self):
        self.assertEqual(cj.parse("- the cup is red\n\n# a note\nNOT: there is text\nnever: a watermark"),
                         [("the cup is red", True), ("there is text", False), ("a watermark", False)])

    def test_a_picture_passes_when_every_statement_holds(self):
        st = cj.parse("frustrated\nnot: angry")
        ok = cj.verdict([0.8, 0.2], st, 0.6)
        self.assertTrue(ok["met"])
        self.assertEqual(ok["score"], 0.8)
        bad = cj.verdict([0.8, 0.7], st, 0.6)
        self.assertFalse(bad["met"])
        self.assertEqual(cj.missed(bad), ["must NOT be the case: angry (the judge saw it at 70 %)"])

    def test_the_frames_of_a_film_are_one_verdict(self):
        a, b = {"met": True, "score": 0.9, "rows": []}, {"met": False, "score": 0.3, "rows": []}
        self.assertFalse(cj.combine([a, b], cj.BATCH_ALL)["met"])
        self.assertEqual(cj.combine([a, b], cj.BATCH_ALL)["score"], 0.3)
        self.assertTrue(cj.combine([b, a], cj.BATCH_ANY)["met"])
        self.assertIs(cj.combine([b, a], cj.BATCH_FIRST), b)

    def test_the_model_is_asked_once_per_picture_with_every_statement(self):
        asked = []

        def ask(model, state, questions, images=None, timeout=0):
            asked.append((model, sorted(questions), images))
            return {"s0": {"noul": 0.9}, "s1": {"noul": 0.1}}
        got = cj.judge_file("a.png", {"model": "clef", "threshold": 0.5, "batch": cj.BATCH_ALL,
                                      "condition": "frustrated\nnot: angry"}, ask=ask, encode=lambda p: "B64")
        self.assertTrue(got["met"])
        self.assertEqual(asked, [("clef", ["s0", "s1"], ["B64"])])

    def test_no_answer_is_an_error_and_not_a_verdict(self):
        with self.assertRaises(cj.ConditionError):
            cj.judge_file("a.png", {"model": "clef", "condition": "x"}, ask=lambda *a, **k: {}, encode=lambda p: "")

    def test_without_ollama_it_says_what_to_do(self):
        with mock.patch("src.utils.decision_models.ollama_up", return_value=False):
            with self.assertRaises(cj.ConditionError) as said:
                cj.ready("clef")
        self.assertIn("Ollama is not running", str(said.exception))


class ARoundJudgedByIt(unittest.TestCase):

    def _loop(self, brk=""):
        return hf.plan(_pipe(brk=brk)).loops[0]

    def test_the_decision_model_judges_and_the_qa_agent_is_not_asked(self):
        checked = []
        results = lj.judge(["good.png", "bad.png"], self._loop(), check=lambda *a, **k: checked.append(a),
                           score=lambda p: {"score": 0.4},
                           decide=lambda path, judge: ({"met": True, "score": 0.9, "rows": []} if "good" in path else
                                                       {"met": False, "score": 0.2, "rows": [
                                                           {"statement": "frustrated", "wanted": True,
                                                            "probability": 0.2, "holds": 0.2, "ok": False}]}))
        self.assertEqual([r["passed"] for r in results], [True, False])
        self.assertEqual(results[1]["missed"], ["frustrated (the judge saw it at 20 %)"])
        self.assertEqual(results[1]["score"], 0.2)                 # how near it came, for `closest`
        self.assertEqual(results[0]["judged_by"], "clef")
        self.assertEqual(checked, [])

    def test_the_breaks_own_condition_is_judged_on_what_passed_only(self):
        checked = []

        def check(path, briefing, request=""):
            checked.append(path)
            return SimpleNamespace(passed=False, failed_criteria=lambda: ["six fingers"], summary="", error="")
        results = lj.judge(["good.png", "bad.png"], self._loop("the hands have five fingers"), check=check,
                           score=lambda p: {"score": 0.0},
                           decide=lambda path, judge: {"met": "good" in path, "score": 0.5, "rows": []})
        self.assertEqual(checked, ["good.png"])
        self.assertEqual(results[0]["passed"], False)
        self.assertEqual(results[0]["missed"], ["six fingers"])

    def test_a_round_it_decides_ends_the_loop(self):
        loop = self._loop()
        state = hf.new_state(loop)
        miss = lj.judge(["a.png"], loop, score=lambda p: {}, decide=lambda p, j: {"met": False, "score": 0.3, "rows": []})
        self.assertFalse(hf.record_round(state, miss, loop)["finished"])
        hit = lj.judge(["b.png"], loop, score=lambda p: {}, decide=lambda p, j: {"met": True, "score": 0.8, "rows": []})
        verdict = hf.record_round(state, hit, loop)
        self.assertTrue(verdict["finished"] and verdict["condition_met"])
        self.assertEqual(verdict["forward"], ["b.png"])

    def test_a_judge_that_cannot_be_asked_stops_the_round(self):
        with mock.patch("src.utils.condition_judge.ready", side_effect=cj.ConditionError("no Ollama")):
            with self.assertRaises(cj.ConditionError):
                lj.judge(["a.png"], self._loop())


if __name__ == "__main__":
    unittest.main()
