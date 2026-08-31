"""QA that can see the set, and a verdict that reaches whoever can act on it.

Two reports from the same afternoon.

The first: *"the qa checker seems to see only a single image (although multiple
images were created) — and always kicks the generation back for exactly that
reason."* It was right to. Outputs are judged one at a time, so a criterion about
the SET ("all the references must share one grade") has nothing in front of it
but a single picture, and the honest answer is a fail — which sends a whole batch
back to be re-generated for a reason no re-generation can address. The per-file
judge is now told to mark those `n/a`, which is only honest because
:func:`qa.check_set` answers them instead.

The second: a QA verdict whose retries were spent went to the log and nowhere
else. The agent that could have fixed the one bad shot never heard about it. Now
it comes back per variant, with what was missed — and, when the briefing says so,
with which hook to re-enter.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub
from src.pipeline import Pipeline
from src.utils.qa import QaBriefing, briefing_from_hooks, check_set, parse_retry


class JudgeInputTest(unittest.TestCase):
    """What an `agentY qa` node's `judge` input means.

    The node merges what used to be two: a `qa` hook carrying prose and a separate
    briefing node carrying the measured checks. The report that prompted it was
    not about either half — it was "it's quite unclear what to wire into the qa
    node", after wiring a save node's image tensor in and getting nothing.

    That instinct was right and the old node read it backwards. A wired image was
    a REFERENCE — something to compare against — while the thing being judged was
    named by wiring the briefing's `out` INTO a hook, an arrow pointing from the
    standard to the work. Now `judge` says what to assess and `reference` says
    what to compare it against, and the panel resolves whatever is wired into
    `judge` to the stage it belongs to.
    """

    def test_files_named_by_judge_are_judged(self):
        b = briefing_from_hooks([{
            "purpose": "qa", "directive": "warm grade",
            "judged": [{"node_id": "9", "type": "AgentYImageCollector",
                        "widgets": {"files": "/out/a.png\n/out/b.png"}}],
        }])
        self.assertEqual(b.judge_paths, ("/out/a.png", "/out/b.png"))

    def test_judged_files_do_not_become_references(self):
        """The whole point of two inputs. A file wired into `judge` is being
        marked; the same file wired into `reference` is the marking scheme."""
        b = briefing_from_hooks([{
            "purpose": "qa", "directive": "warm grade",
            "judged": [{"node_id": "9", "type": "AgentYImageCollector",
                        "widgets": {"files": "/out/a.png"}}],
        }])
        self.assertEqual(b.reference_paths, ())
        self.assertEqual(b.judge_paths, ("/out/a.png",))

    def test_references_do_not_become_judged(self):
        b = briefing_from_hooks([{
            "purpose": "qa", "directive": "warm grade",
            "anchors": [{"node_id": "3", "type": "AgentYImageCollector",
                         "widgets": {"files": "/ref/mood.png"}}],
        }])
        self.assertEqual(b.reference_paths, ("/ref/mood.png",))
        self.assertEqual(b.judge_paths, ())

    def test_judged_files_are_added_to_the_run_outputs_never_substituted(self):
        """`judge` says which outputs the briefing is ABOUT. Reading it as "only
        these" would let one mis-wire quietly excuse every other output from being
        checked — and an unchecked output is the failure QA exists to prevent."""
        b = QaBriefing(criteria="x", judge_paths=("/named/extra.png",))
        self.assertEqual(b.outputs_with(["/run/out1.png", "/run/out2.png"]),
                         ["/run/out1.png", "/run/out2.png", "/named/extra.png"])

    def test_a_file_arriving_both_ways_is_judged_once(self):
        """A collector wired into `judge` that holds this very run's output. Judged
        twice, one failure would be reported as two."""
        b = QaBriefing(criteria="x", judge_paths=("/run/out1.png", "/named/extra.png"))
        self.assertEqual(b.outputs_with(["/run/out1.png"]),
                         ["/run/out1.png", "/named/extra.png"])

    def test_no_judge_input_still_judges_the_run(self):
        """Unwired is the common case and means "everything this run produced"."""
        b = QaBriefing(criteria="x")
        self.assertEqual(b.outputs_with(["/run/a.png"]), ["/run/a.png"])

    def test_judge_paths_survive_a_merge(self):
        a = QaBriefing(criteria="a", judge_paths=("/a.png",))
        b = QaBriefing(criteria="b", judge_paths=("/b.png",))
        self.assertEqual(a.merged_with(b).judge_paths, ("/a.png", "/b.png"))

    def test_a_merge_does_not_duplicate_a_shared_file(self):
        a = QaBriefing(criteria="a", judge_paths=("/same.png",))
        b = QaBriefing(criteria="b", judge_paths=("/same.png", "/other.png"))
        self.assertEqual(a.merged_with(b).judge_paths, ("/same.png", "/other.png"))

    def test_scoping_still_travels_as_applies_to(self):
        """Whatever is wired into `judge` — a hook, an image, a collector — the
        panel resolves it to the stage it belongs to and sends that as
        `applies_to`. The agent side is unchanged by the new node, which is why
        per-stage scoping did not have to be rebuilt."""
        from src.utils.qa import briefing_for_hook
        hooks = [
            {"purpose": "qa", "directive": "for stage 5", "applies_to": ["5"]},
            {"purpose": "qa", "directive": "for everything"},
        ]
        five = briefing_for_hook(hooks, "5")
        self.assertIn("for stage 5", five.criteria)
        self.assertIn("for everything", five.criteria)
        seven = briefing_for_hook(hooks, "7")
        self.assertNotIn("for stage 5", seven.criteria)
        self.assertIn("for everything", seven.criteria)


class RetryScopeTest(unittest.TestCase):
    """What a failing verdict should cause, in the user's own words."""

    def test_a_budget(self):
        self.assertEqual(parse_retry("skin tones warm.\nretry: 3"), (3, ""))

    def test_a_hook_to_re_enter(self):
        self.assertEqual(parse_retry("retry: hook 5"), (None, "5"))
        self.assertEqual(parse_retry("re-run hook 5 x2"), (2, "5"))

    def test_prose_about_retries_changes_nothing(self):
        """'retry' in a sentence is not a setting."""
        for text in ["no retry artefacts please", "the retry looked worse",
                     "avoid re-running the same pose"]:
            with self.subTest(text=text):
                self.assertEqual(parse_retry(text), (None, ""))

    def test_a_qa_hook_carries_it_into_the_briefing(self):
        hooks = [{"hook_node_id": "9", "purpose": "qa", "anchors": [],
                  "directive": "All shots must match the style guide.\nretry: hook 5"}]
        b = briefing_from_hooks(hooks)
        self.assertEqual((b.retry_budget, b.retry_hook), (None, "5"))
        self.assertIn("re-run hook 5 on a fail", b.describe())

    def test_a_briefing_without_one_is_unchanged(self):
        b = briefing_from_hooks([{"hook_node_id": "9", "purpose": "qa",
                                  "anchors": [], "directive": "warm skin tones"}])
        self.assertEqual((b.retry_budget, b.retry_hook), (None, ""))
        self.assertNotIn("retr", b.describe())

    def test_the_executor_takes_the_budget_from_the_briefing(self):
        import inspect
        from src import executor
        src = inspect.getsource(executor.execute_workflows_batch)
        self.assertIn('getattr(qa_briefing, "retry_budget", None)', src)


class SetCheckTest(unittest.TestCase):
    """The question a single image cannot answer."""

    def _briefing(self):
        return QaBriefing(criteria="All the references must share one grade.",
                          sources=("canvas qa hook",))

    def test_one_output_is_not_a_set(self):
        res = check_set(["a.png"], self._briefing())
        self.assertTrue(res.passed)
        self.assertIn("at least two", res.error)

    def test_it_judges_every_output_together(self):
        seen = {}

        def fake_agent(blocks):
            seen["blocks"] = blocks
            return json.dumps({"verdict": "fail", "summary": "shot 3 is cooler",
                               "checks": [{"criterion": "one grade", "result": "fail",
                                           "note": "the third is bluer"}]})

        agent = mock.Mock(side_effect=fake_agent)
        agent.messages = mock.Mock()
        with mock.patch("src.utils.qa._output_blocks",
                        side_effect=lambda p, f: ([{"image": p}], f"OUTPUT {p}")):
            res = check_set(["a.png", "b.png", "c.png"], self._briefing(), agent=agent)
        self.assertFalse(res.passed)
        self.assertEqual(res.failed_criteria(), ["one grade — the third is bluer"])
        text = seen["blocks"][-1]["text"]
        self.assertIn("ALL 3 outputs of one run", text)
        self.assertIn("Judge them AS A SET", text)
        self.assertIn("was already judged elsewhere", text)

    def test_an_unreadable_judge_never_condemns_the_set(self):
        agent = mock.Mock(side_effect=RuntimeError("model down"))
        agent.messages = mock.Mock()
        with mock.patch("src.utils.qa._output_blocks",
                        side_effect=lambda p, f: ([{"image": p}], "x")):
            res = check_set(["a.png", "b.png"], self._briefing(), agent=agent)
        self.assertTrue(res.passed)
        self.assertIn("model down", res.error)

    def test_the_per_file_judge_is_told_it_sees_one_of_many(self):
        from src.utils.qa import load_qa_prompts
        system = load_qa_prompts().get("system", "")
        self.assertIn("judging ONE output", system)
        self.assertIn("not judgeable here", system)


class VerdictReachesTheAgentTest(unittest.TestCase):
    """A gate that only talks to the log is not a gate."""

    @staticmethod
    def _run(qa_verdicts, briefing=None, outputs=("C:/out/a.png", "C:/out/b.png")):
        async def fake_batch(paths, *a, **kw):
            kw["collected_paths"].extend(outputs)
            if kw.get("qa_verdicts") is not None:
                kw["qa_verdicts"].update(qa_verdicts)
            yield "ran"

        pipe = pipeline_stub(_qa_briefing=briefing)
        with mock.patch("src.pipeline._execute_workflows_batch", fake_batch), \
             mock.patch("src.pipeline._clear_exec_errors"), \
             mock.patch("src.pipeline._get_exec_errors", return_value=[]):
            raw = asyncio.run(Pipeline._run_canvas_batch(
                pipe, ["wf0.json", "wf1.json"], [], [{"6.text": "A"}, {"6.text": "B"}]))
        return json.loads(raw)

    def test_a_spent_verdict_comes_back_on_the_variant_that_earned_it(self):
        out = self._run({"wf1.json": {"tries": 1, "missed": ["skin reads orange"],
                                      "summary": "too warm", "outputs": ["C:/out/b.png"]}})
        self.assertEqual(out["qa_failed_count"], 1)
        self.assertNotIn("qa", out["variants"][0])
        self.assertEqual(out["variants"][1]["qa"]["missed"], ["skin reads orange"])
        self.assertEqual(out["variants"][1]["made_from"]["6.text"], "B")
        self.assertIn("Do not re-run the ones that passed", out["message"])

    def test_the_briefing_can_name_the_hook_to_re_enter(self):
        b = QaBriefing(criteria="x", retry_hook="5")
        out = self._run({"wf0.json": {"tries": 0, "missed": ["wrong character"],
                                      "summary": "", "outputs": []}}, briefing=b)
        self.assertIn("re-run hook 5 for those", out["message"])
        self.assertIn("addressing exactly what was missed", out["message"])

    def test_a_clean_run_says_nothing_about_qa(self):
        out = self._run({})
        self.assertNotIn("qa_failed_count", out)
        self.assertNotIn("qa_set", out)

    def test_the_set_verdict_rides_along_when_it_fails(self):
        b = QaBriefing(criteria="one grade across the set")
        with mock.patch("src.utils.qa.check_set") as cs:
            cs.return_value = mock.Mock(error="", passed=False, summary="mixed grades",
                                        failed_criteria=lambda: ["one grade — mixed"])
            out = self._run({}, briefing=b)
        self.assertFalse(out["qa_set"]["passed"])
        self.assertIn("judged as a whole and missed", out["message"])
        self.assertIn("one grade — mixed", out["message"])

    def test_no_briefing_means_no_set_verdict(self):
        with mock.patch("src.utils.qa.check_set") as cs:
            out = self._run({})
        cs.assert_not_called()
        self.assertNotIn("qa_set", out)


class SingleOutputTest(unittest.TestCase):
    """One output cannot be inconsistent with itself — but say so, don't go quiet.

    The per-file judge marks set-criteria `n/a`. If nothing else spoke, "was that
    criterion checked or skipped?" would have no answer anywhere.
    """

    def _verdict(self, paths, briefing=QaBriefing(criteria="one grade across the set")):
        pipe = pipeline_stub(_qa_briefing=briefing)
        with mock.patch("src.utils.qa.check_set") as cs, \
             mock.patch("src.pipeline._push_progress") as push:
            got = asyncio.run(Pipeline._qa_set_verdict(pipe, paths))
        return got, cs, push

    def test_a_single_output_is_reported_as_not_applicable(self):
        got, cs, push = self._verdict(["C:/out/only.png"])
        self.assertIsNone(got)
        cs.assert_not_called()
        self.assertTrue(any("not applicable" in str(c) for c in push.call_args_list),
                        push.call_args_list)

    def test_no_outputs_at_all_says_nothing(self):
        got, _cs, push = self._verdict([])
        self.assertIsNone(got)
        push.assert_not_called()

    def test_without_a_briefing_it_stays_silent(self):
        got, _cs, push = self._verdict(["C:/out/only.png"], briefing=None)
        self.assertIsNone(got)
        push.assert_not_called()

    def test_the_queued_batch_gets_the_set_verdict_too(self):
        """Not only run_now — the default path is where most sets are made."""
        import inspect
        src = inspect.getsource(Pipeline._astream_orchestrator)
        self.assertIn("self._qa_set_verdict(", src)
        self.assertIn("as a SET they miss", src)


if __name__ == "__main__":
    unittest.main()
