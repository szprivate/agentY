"""An agent that keeps making the call it has already made is stopped.

The first class replays the turn that prompted this: benchmark case CX04 with
qwen3.8:27b on 2026-10-12, where ``hf_file`` was called 136 times in one turn.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import repeat_guard as rg

URL_A = {"url": "https://huggingface.co/fal/depth_anything_v2_base_sd15_depth_controlnet/resolve/main/"
                "control_v11f1p_sd15_depth.safetensors"}
URL_B = {"url": "https://huggingface.co/hpcai-tech/cv-diffs/resolve/main/models/control_v11f1e_sd15_tile.safetensors"}


def run(turn, name, args, answer):
    """Make one call as an agent would; returns the refusal ("" when it ran)."""
    why = turn.refusal(name, args)
    if not why:
        turn.answered(name, args, {"status": "success", "content": [{"text": answer}]})
    return why


class TheRunawayThatHappened(unittest.TestCase):

    def test_two_urls_asked_over_and_over_are_stopped_after_three_answers_each(self):
        turn = rg.Turn()
        ran = refused = 0
        for i in range(125):                               # as in the trace: A, B, A, B, ...
            args = URL_A if i % 2 == 0 else URL_B
            if run(turn, "hf_file", args, '{"ok": true, "path": "x"}'):
                refused += 1
            else:
                ran += 1
        self.assertEqual(ran, 6)                           # three answers per url, then no more
        self.assertEqual(refused, 119)

    def test_the_refusal_says_what_to_do_instead(self):
        turn = rg.Turn()
        for _ in range(3):
            run(turn, "hf_file", URL_A, "same")
        why = run(turn, "hf_file", URL_A, "same")
        self.assertIn("the answer was the same every time", why)
        self.assertIn("tell the user", why)


class TheSameCall(unittest.TestCase):

    def test_a_call_whose_answer_changes_is_never_refused(self):
        turn = rg.Turn()
        for i in range(25):                               # under the per-tool notice
            self.assertEqual(run(turn, "get_canvas_node", {"node_id": "5"}, f"state {i}"), "")

    def test_other_arguments_are_another_call(self):
        turn = rg.Turn()
        for i in range(10):
            self.assertEqual(run(turn, "hf_file", {"url": f"https://x/{i}"}, "same"), "")

    def test_argument_order_does_not_make_a_new_call(self):
        turn = rg.Turn()
        for _ in range(3):
            run(turn, "check_model", {"a": 1, "b": 2}, "same")
        self.assertTrue(run(turn, "check_model", {"b": 2, "a": 1}, "same"))

    def test_polling_gets_more_patience_while_nothing_changes(self):
        turn = rg.Turn()
        for i in range(rg.POLLING_PATIENCE):
            self.assertEqual(run(turn, "queue", {}, "running"), "", i)
        self.assertTrue(run(turn, "queue", {}, "running"))

    def test_and_starts_over_when_the_answer_moves(self):
        turn = rg.Turn()
        for _ in range(rg.POLLING_PATIENCE - 1):
            run(turn, "queue", {}, "running")
        run(turn, "queue", {}, "done")
        self.assertEqual(run(turn, "queue", {}, "done"), "")

    def test_a_new_turn_forgets(self):
        turn = rg.Turn()
        for _ in range(3):
            run(turn, "hf_file", URL_A, "same")
        self.assertEqual(run(rg.Turn(), "hf_file", URL_A, "same"), "")


class OneToolOverAndOver(unittest.TestCase):

    def test_a_single_notice_at_the_count_then_it_carries_on(self):
        turn = rg.Turn(per_tool=30)
        answers = [run(turn, "analyze_image", {"file": f"ref_{i}.png"}, f"picture {i}") for i in range(45)]
        self.assertEqual(sum(1 for a in answers if a), 1)
        self.assertTrue(answers[30])
        self.assertIn("carry on", answers[30])

    def test_the_count_doubles_between_notices_and_the_third_closes_the_tool(self):
        turn = rg.Turn(per_tool=5)
        refused_at = [i for i in range(60) if run(turn, "web_search", {"q": str(i)}, str(i))]
        self.assertEqual(refused_at[:3], [5, 11, 22])        # at 5, 10 and 20 calls let through
        self.assertEqual(refused_at[3:], list(range(23, 60)))
        self.assertIn("closed for the rest of this turn", turn.refusal("web_search", {"q": "again"}))

    def test_another_tool_has_its_own_count(self):
        turn = rg.Turn(per_tool=5)
        for i in range(5):
            run(turn, "web_search", {"q": str(i)}, str(i))
        self.assertEqual(run(turn, "read_web_page", {"url": "u"}, "page"), "")


class TheHook(unittest.TestCase):

    def _provider(self, **cfg):
        base = {"enabled": True, "same_call": 3, "per_tool": 30}
        base.update(cfg)
        with mock.patch.object(rg, "settings", return_value=base):
            provider = rg.RepeatGuardHookProvider()
            provider._start(None)
        return provider

    def _call(self, provider, name="hf_file", args=None, answer="same"):
        event = SimpleNamespace(tool_use={"name": name, "input": args or URL_A}, cancel_tool=False)
        provider._before(event)
        if not event.cancel_tool:
            provider._after(SimpleNamespace(tool_use=event.tool_use,
                                            result={"status": "success", "content": [{"text": answer}]}))
        return event.cancel_tool

    def test_the_fourth_identical_call_is_cancelled_with_the_reason(self):
        provider = self._provider()
        self.assertEqual([bool(self._call(provider)) for _ in range(5)], [False, False, False, True, True])

    def test_the_setting_turns_it_off(self):
        provider = self._provider(enabled=False)
        self.assertEqual([bool(self._call(provider)) for _ in range(6)], [False] * 6)

    def test_a_call_something_else_already_cancelled_is_left_alone(self):
        provider = self._provider()
        event = SimpleNamespace(tool_use={"name": "run_script", "input": {}}, cancel_tool="Not run - declined.")
        provider._before(event)
        self.assertEqual(event.cancel_tool, "Not run - declined.")

    def test_every_agent_is_given_one(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("RepeatGuardHookProvider()", src)


class OneBudgetForTheWholeJobTest(unittest.TestCase):
    """An agent restarted over the same job must not get a fresh budget each time.

    The repair specialist is built anew for every heal of a workflow, with its
    messages cleared, so a per-invocation budget let each attempt explore from
    zero. On 2026-10-18 that spread one runaway over several attempts: 67
    ``get_node_source`` calls, each with different arguments so ``same_call``
    never fired, and no attempt on its own near the per-tool notice.
    """

    CFG = {"enabled": True, "same_call": 3, "per_tool": 5}

    def setUp(self):
        from agenty_core.utils import turn_scope
        self._turn_scope = turn_scope
        self._token = turn_scope.enter(turn_scope.Scope(request_id="test"))

    def tearDown(self):
        self._turn_scope.leave(self._token)

    def _start(self, scope=""):
        with mock.patch.object(rg, "settings", return_value=dict(self.CFG)):
            provider = rg.RepeatGuardHookProvider(scope=scope)
            provider._start(None)
        return provider

    def _attempt(self, provider, calls):
        """One agent invocation's worth of calls; returns how many were refused."""
        refused = 0
        for i in range(calls):
            # Different arguments every time, as in the incident: this is
            # exploration sprawl, so only the per-tool budget can catch it.
            event = SimpleNamespace(
                tool_use={"name": "get_node_source", "input": {"node": f"n{i}"}},
                cancel_tool=False)
            provider._before(event)
            if event.cancel_tool:
                refused += 1
            else:
                provider._after(SimpleNamespace(
                    tool_use=event.tool_use,
                    result={"status": "success", "content": [{"text": "source"}]}))
        return refused

    def test_two_attempts_at_one_workflow_share_one_budget(self):
        first, second = self._start("heal:/tmp/wf.json"), self._start("heal:/tmp/wf.json")
        self.assertEqual(self._attempt(first, 5), 0)     # fills the budget exactly
        self.assertEqual(self._attempt(second, 5), 1)    # and the next call trips it

    def test_without_a_scope_every_attempt_still_starts_fresh(self):
        first, second = self._start(), self._start()
        self.assertEqual(self._attempt(first, 5), 0)
        self.assertEqual(self._attempt(second, 5), 0)

    def test_another_workflow_is_another_budget(self):
        a, b = self._start("heal:/tmp/a.json"), self._start("heal:/tmp/b.json")
        self.assertEqual(self._attempt(a, 5), 0)
        self.assertEqual(self._attempt(b, 5), 0)         # untouched by a's spend

    def test_the_same_scope_names_the_same_turn(self):
        self.assertIs(rg.shared_turn("heal:/tmp/wf.json", dict(self.CFG)),
                      rg.shared_turn("heal:/tmp/wf.json", dict(self.CFG)))

    def test_a_new_turn_starts_clean(self):
        rg.shared_turn("heal:/tmp/wf.json", dict(self.CFG)).refusal("x", {"n": 1})
        self._turn_scope.leave(self._token)
        self._token = self._turn_scope.enter(self._turn_scope.Scope(request_id="next"))
        fresh = rg.shared_turn("heal:/tmp/wf.json", dict(self.CFG))
        self.assertEqual(fresh.refusal("x", {"n": 1}), "")


if __name__ == "__main__":
    unittest.main()


class AsStrandsRunsIt(unittest.TestCase):
    """Strands fires the after-call hook for a cancelled call too, with the refusal
    as its result. Counted as an answer, it broke the run of identical answers and
    the next copy ran again (CX10, night_20261029: refused, ran, refused, ran ...)."""

    def _provider(self):
        with mock.patch.object(rg, "settings", return_value={"enabled": True, "same_call": 3,
                                                                  "per_tool": 1000}):
            provider = rg.RepeatGuardHookProvider()
            provider._start(None)
        return provider

    def _call(self, provider, args):
        event = SimpleNamespace(tool_use={"name": "run_script", "input": args}, cancel_tool=False)
        provider._before(event)
        cancel = event.cancel_tool
        result = ({"status": "error", "content": [{"text": cancel}]} if cancel
                  else {"status": "success", "content": [{"text": "same"}]})
        provider._after(SimpleNamespace(tool_use=event.tool_use, result=result,
                                        cancel_message=cancel or None))
        return bool(cancel)

    def test_once_refused_the_call_stays_refused(self):
        provider = self._provider()
        a, b = {"code": "scan gemma"}, {"code": "scan proj_linear"}
        ran = [not self._call(provider, a if i % 2 == 0 else b) for i in range(40)]
        self.assertEqual(sum(ran), 6)                      # three answers each, then never again


def _round(i, name="run_script", args=None):
    uid = f"t{i}"
    return [{"role": "assistant", "content": [{"toolUse": {"toolUseId": uid, "name": name,
                                                           "input": args or {"code": "scan"}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": uid, "status": "success",
                                                         "content": [{"text": "same"}]}}]}]


class RepeatedRoundsLeaveTheHistory(unittest.TestCase):
    """DashScope refuses a history with the same call over many rounds, and the
    conversation stays refused for every later turn."""

    def test_the_first_and_the_last_copy_stay(self):
        msgs = [{"role": "user", "content": [{"text": "build it"}]}]
        for i in range(6):
            msgs += _round(i)
        msgs.append({"role": "assistant", "content": [{"text": "done"}]})
        self.assertEqual(rg.collapse_repeats(msgs), 4)
        ids = [m["content"][0]["toolUse"]["toolUseId"] for m in msgs if "toolUse" in m["content"][0]]
        self.assertEqual(ids, ["t0", "t5"])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant", "user", "assistant"])

    def test_two_copies_and_other_calls_are_left_alone(self):
        msgs = [{"role": "user", "content": [{"text": "x"}]}]
        msgs += _round(0) + _round(1) + _round(2, args={"code": "other"})
        before = list(msgs)
        self.assertEqual(rg.collapse_repeats(msgs), 0)
        self.assertEqual(msgs, before)

    def test_words_around_the_call_do_not_make_it_another_round(self):
        msgs = [{"role": "user", "content": [{"text": "x"}]}]
        for i in range(4):
            r = _round(i)
            r[0]["content"].insert(0, {"text": "checking again"})
            msgs += r
        self.assertEqual(rg.collapse_repeats(msgs), 2)
