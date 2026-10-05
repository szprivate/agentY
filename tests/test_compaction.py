"""History compaction: a long conversation stays small, and nothing is lost.

What these hold:

* old tool output is cut to its head — recent output, and the turn in progress,
  never are;
* an old turn's input loses its injected guidance but KEEPS what the user wrote
  (their words are the end of that block — cutting to the head threw them away);
* a summary only ever cuts at a real user message, so no tool call is separated
  from its result, and a summariser that fails costs a digest, never the turns;
* everything a pass changes is archived whole first;
* between turns it rewrites the conversation's SAVED history — which is what a
  restart resumes from — and never overwrites a history a newer turn replaced.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import compaction as c

CFG = dict(c.DEFAULTS, keep_recent_messages=2, history_budget_tokens=10 ** 9)


def user(text):
    return {"role": "user", "content": [{"text": text}]}


def call(i, name="tool"):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t{i}", "name": name,
                                                          "input": {"q": i}}}]}


def result(i, text):
    return {"role": "user", "content": [{"toolResult": {"toolUseId": f"t{i}", "status": "success",
                                                        "content": [{"text": text}]}}]}


def said(text):
    return {"role": "assistant", "content": [{"text": text}]}


def turn(n, tools=2, size=5000, ask=None):
    out = [user(ask or f"request {n}")]
    for i in range(tools):
        out += [call(f"{n}-{i}"), result(f"{n}-{i}", f"RESULT {n}-{i} " + "x" * size)]
    return out + [said(f"answer {n}")]


def pairs_intact(messages):
    seen = set()
    for m in messages:
        for b in m["content"]:
            if "toolUse" in b:
                seen.add(b["toolUse"]["toolUseId"])
            if "toolResult" in b and b["toolResult"]["toolUseId"] not in seen:
                return False
    return True


class Ageing(unittest.TestCase):

    def test_old_tool_output_is_cut_to_its_head(self):
        msgs = turn(1) + turn(2)
        res = c.compact(msgs, CFG, allow_summary=False)
        text = res.messages[2]["content"][0]["toolResult"]["content"][0]["text"]
        self.assertTrue(text.startswith(c.TRIM_MARK))
        self.assertIn("RESULT 1-0", text)
        self.assertLess(len(text), 800)
        self.assertLess(res.after, res.before // 2)

    def test_recent_output_is_left_whole(self):
        msgs = turn(1) + turn(2)
        res = c.compact(msgs, dict(CFG, keep_recent_messages=3), allow_summary=False)
        last = res.messages[-2]["content"][0]["toolResult"]["content"][0]["text"]
        self.assertFalse(last.startswith(c.TRIM_MARK))
        self.assertEqual(len(last), len(msgs[-2]["content"][0]["toolResult"]["content"][0]["text"]))

    def test_short_output_is_never_touched(self):
        msgs = turn(1, size=100) + turn(2, size=100)
        self.assertFalse(c.compact(msgs, CFG, allow_summary=False).changed)

    def test_it_is_idempotent(self):
        once = c.compact(turn(1) + turn(2), CFG, allow_summary=False)
        twice = c.compact(once.messages, CFG, allow_summary=False)
        self.assertFalse(twice.changed)
        self.assertEqual(twice.messages, once.messages)

    def test_the_original_messages_are_not_mutated(self):
        msgs = turn(1) + turn(2)
        before = json.dumps(msgs)
        c.compact(msgs, CFG, allow_summary=False)
        self.assertEqual(json.dumps(msgs), before)

    def test_json_tool_output_is_aged_too(self):
        msgs = turn(1) + turn(2)
        msgs[2]["content"][0]["toolResult"]["content"] = [{"json": {"rows": ["y" * 100] * 80}}]
        res = c.compact(msgs, CFG, allow_summary=False)
        self.assertTrue(res.messages[2]["content"][0]["toolResult"]["content"][0]["text"]
                        .startswith(c.TRIM_MARK))

    def test_tool_calls_keep_their_results(self):
        self.assertTrue(pairs_intact(c.compact(turn(1) + turn(2) + turn(3), CFG,
                                               allow_summary=False).messages))


class AnOldTurnsInput(unittest.TestCase):
    """Injected guidance first, the user's words last — in ONE block."""

    def _ask(self, words):
        return "[CANVAS HOOKS] guidance " + "g" * 20_000 + "\n\n" + words

    def test_the_users_words_survive(self):
        msgs = turn(1, ask=self._ask("make the screens brighter, keep the sky")) + turn(2)
        res = c.compact(msgs, CFG, allow_summary=False)
        text = res.messages[0]["content"][0]["text"]
        self.assertLess(len(text), 4000)
        self.assertTrue(text.endswith("make the screens brighter, keep the sky"))
        self.assertTrue(text.startswith(c.TRIM_MARK))

    def test_the_turn_in_progress_keeps_its_guidance(self):
        """Its injected blocks are still in force."""
        msgs = turn(1) + turn(2, ask=self._ask("now the roof"))
        res = c.compact(msgs, dict(CFG, keep_recent_messages=0), allow_summary=False)
        live = res.messages[c.turn_starts(res.messages)[-1]]["content"][0]["text"]
        self.assertEqual(len(live), len(self._ask("now the roof")))


class Summarising(unittest.TestCase):

    def _run(self, msgs, **over):
        seen = []

        def fake(old):
            seen.append(old)
            return "GOAL — a stadium. STATE — W:/out/a.png exists."
        cfg = dict(CFG, history_budget_tokens=0, **over)
        return c.compact(msgs, cfg, summarise=fake), seen

    def test_older_turns_become_one_summary(self):
        res, seen = self._run(turn(1) + turn(2) + turn(3) + turn(4))
        self.assertEqual(len(c.turn_starts(res.messages)), 2, "the last two turns are kept")
        first = res.messages[0]
        self.assertEqual(first["role"], "user")
        self.assertTrue(first["content"][0]["text"].startswith(c.SUMMARY_MARK))
        self.assertIn("W:/out/a.png", first["content"][0]["text"])
        self.assertEqual(first["content"][1]["text"], "request 3")
        self.assertEqual(len(seen[0]), 12, "turns 1 and 2, whole")

    def test_no_tool_call_is_separated_from_its_result(self):
        res, _ = self._run(turn(1, tools=5) + turn(2, tools=5) + turn(3, tools=5))
        self.assertTrue(pairs_intact(res.messages))

    def test_too_few_turns_are_not_summarised(self):
        res, seen = self._run(turn(1) + turn(2))
        self.assertEqual(seen, [])
        self.assertEqual(res.summarised, 0)

    def test_under_budget_nothing_is_summarised(self):
        seen = []
        res = c.compact(turn(1, size=10) + turn(2, size=10) + turn(3, size=10),
                        dict(CFG, history_budget_tokens=10 ** 6),
                        summarise=lambda old: seen.append(old) or "x" * 50)
        self.assertEqual((seen, res.summarised), ([], 0))

    def test_a_previous_summary_is_carried_forward(self):
        res, _ = self._run(turn(1) + turn(2) + turn(3))
        again = res.messages + turn(4) + turn(5)
        shown = c.render_for_summary(again[:c.turn_starts(again)[-2]])
        self.assertIn("PREVIOUS SUMMARY:", shown)
        self.assertIn("W:/out/a.png", shown)

    def test_a_failing_summariser_costs_a_digest_not_the_turns(self):
        def boom(old):
            raise RuntimeError("model unreachable")
        with self.assertLogs("agentY.compaction", level="WARNING"):
            res = c.compact(turn(1, ask="guidance " + "g" * 9000 + "\n\nbrighten it")
                            + turn(2) + turn(3),
                            dict(CFG, history_budget_tokens=0), summarise=boom)
        text = res.messages[0]["content"][0]["text"]
        self.assertIn("written without a model", text)
        self.assertIn("User: brighten it", text)
        self.assertIn("answer 1", text)

    def test_mid_turn_never_summarises(self):
        res = c.compact(turn(1) + turn(2) + turn(3), dict(CFG, history_budget_tokens=0),
                        allow_summary=False, summarise=lambda old: self.fail("called"))
        self.assertEqual(res.summarised, 0)


class TheArchive(unittest.TestCase):

    def test_what_is_changed_is_archived_as_it_was(self):
        msgs = turn(1) + turn(2) + turn(3)
        res = c.compact(msgs, dict(CFG, history_budget_tokens=0), summarise=lambda o: "s" * 60)
        whole = [m for m in res.archived if "toolResult" in m["content"][0]]
        self.assertTrue(whole)
        for m in whole:
            self.assertFalse(m["content"][0]["toolResult"]["content"][0]["text"]
                             .startswith(c.TRIM_MARK), "a stub was archived, not the original")
        # every message of the summarised turn is there, once
        self.assertEqual(sum(1 for m in res.archived if m in msgs[:6]), 6)

    def test_it_is_appended_to_the_conversations_file(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(c, "ARCHIVE_DIR", Path(tmp)):
            c.archive("thread-1", [user("one")], "between turns")
            path = c.archive("thread-1", [user("two")], "mid-turn")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["messages"][0]["content"][0]["text"] for r in rows], ["one", "two"])
        self.assertEqual(rows[1]["reason"], "mid-turn")

    def test_an_unwritable_archive_never_raises(self):
        with mock.patch.object(c, "ARCHIVE_DIR", Path("Z:/nope/nope")):
            with self.assertLogs("agentY.compaction", level="WARNING"):
                self.assertIsNone(c.archive("t", [user("x")]))


class TheMidTurnGuard(unittest.TestCase):

    def _fire(self, messages, **cfg):
        agent = mock.Mock(messages=messages)
        with mock.patch.object(c, "settings", return_value=dict(CFG, **cfg)), \
                mock.patch.object(c, "archive") as arch:
            c.CompactionHookProvider(role="info")._before_model(mock.Mock(agent=agent))
        return agent.messages, arch

    def test_below_the_hard_budget_history_is_byte_stable(self):
        msgs = turn(1) + turn(2)
        before = json.dumps(msgs)
        got, _ = self._fire(msgs, hard_budget_tokens=10 ** 9)
        self.assertEqual(json.dumps(got), before)

    def test_past_it_old_output_is_aged_in_place(self):
        msgs = turn(1, tools=6)                 # one long turn, still running
        got, arch = self._fire(msgs, hard_budget_tokens=100)
        self.assertIs(got, msgs, "the agent's own list is edited, not replaced")
        self.assertTrue(got[2]["content"][0]["toolResult"]["content"][0]["text"]
                        .startswith(c.TRIM_MARK))
        self.assertTrue(pairs_intact(got))
        arch.assert_not_called()                # a specialist's history is transient

    def test_switched_off_it_does_nothing(self):
        msgs = turn(1, tools=6)
        before = json.dumps(msgs)
        got, _ = self._fire(msgs, hard_budget_tokens=100, enabled=False)
        self.assertEqual(json.dumps(got), before)


class BetweenTurns(unittest.TestCase):
    """The server's pass over a conversation's saved history."""

    def setUp(self):
        from src.utils import agentY_server as server
        self.server = server
        self.saved = {}
        for target, kw in (
                (mock.patch.object(c, "settings", return_value=CFG), {}),
                (mock.patch.object(c, "archive"), {}),
                (mock.patch.object(server.cs, "update_brain_messages",
                                   side_effect=lambda t, m: self.saved.__setitem__(t, m)), {})):
            got = target.start()
            self.addCleanup(target.stop)
            if target.attribute == "archive":
                self.archive = got
        self.addCleanup(server._thread_brain_cache.pop, "tc", None)

    def test_the_saved_history_is_what_gets_compacted(self):
        self.server._thread_brain_cache["tc"] = turn(1) + turn(2)
        self.server._compact_thread("tc")
        cached = self.server._thread_brain_cache["tc"]
        self.assertTrue(cached[2]["content"][0]["toolResult"]["content"][0]["text"]
                        .startswith(c.TRIM_MARK))
        self.assertEqual(len(self.saved["tc"]), len(cached), "a restart resumes from this")
        self.archive.assert_called_once()

    def test_a_history_a_newer_turn_replaced_is_not_overwritten(self):
        self.server._thread_brain_cache["tc"] = turn(1) + turn(2)
        newer = turn(1) + turn(2) + turn(3)

        def slow(*a, **k):
            self.server._thread_brain_cache["tc"] = newer      # a turn finished meanwhile
        self.archive.side_effect = slow
        self.server._compact_thread("tc")
        self.assertIs(self.server._thread_brain_cache["tc"], newer)
        self.assertEqual(self.saved, {})

    def test_the_next_turn_waits_for_a_running_pass(self):
        import inspect
        src = inspect.getsource(self.server._restore_state)
        self.assertLess(src.index("_await_compaction(thread_id)"),
                        src.index("_reset_pipeline_state(pipeline)"))

    def test_it_runs_after_done_so_nobody_waits_for_it(self):
        import inspect
        src = inspect.getsource(self.server._run_pipeline_turn)
        self.assertLess(src.rindex('out_q.put({"type": "done"})'),
                        src.index("_schedule_compaction(thread_id)"))


class TheCompactCommand(unittest.TestCase):
    """`/compact` — the same pass, asked for, and it says what it did."""

    def setUp(self):
        from src.utils import agentY_server as server
        self.server = server
        self.saved, self.state = {}, {}
        self.cfg = dict(CFG, enabled=False)          # /compact works even when it is off
        patches = [
            mock.patch.object(c, "settings", side_effect=lambda: dict(self.cfg)),
            mock.patch.object(c, "archive"),
            mock.patch.object(c, "llm_summary", return_value="GOAL — a stadium. " + "s" * 40),
            mock.patch.object(server.cs, "update_brain_messages",
                              side_effect=lambda t, m: self.saved.__setitem__(t, m)),
            mock.patch.object(server.cs, "load_state", side_effect=lambda t: self.state.get(t)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(server._thread_brain_cache.pop, "tc", None)

    def _run(self, text="/compact"):
        events = self.server._handle_command("tc", text)
        return " ".join(str(e.get("data") or e.get("message") or "") for e in events)

    def test_it_trims_and_summarises_whatever_the_budget_says(self):
        self.server._thread_brain_cache["tc"] = turn(1) + turn(2) + turn(3)
        said_ = self._run()
        self.assertIn("Compacted", said_)
        self.assertIn("summarised", said_)
        self.assertIn("history_archive/tc.jsonl", said_)
        cached = self.server._thread_brain_cache["tc"]
        self.assertTrue(cached[0]["content"][0]["text"].startswith(c.SUMMARY_MARK))
        self.assertEqual(len(self.saved["tc"]), len(cached))

    def test_the_bare_word_works_too(self):
        """Slack swallows a leading slash."""
        self.server._thread_brain_cache["tc"] = turn(1) + turn(2)
        self.assertTrue(self.server._is_command("compact", "tc"))
        self.assertIn("Compacted", self._run("compact"))

    def test_after_a_restart_it_reads_the_saved_history(self):
        self.state["tc"] = {"brain_messages": turn(1) + turn(2)}
        self.assertIn("Compacted", self._run())
        self.assertIn("tc", self.saved)

    def test_nothing_to_do_is_said_plainly(self):
        self.assertIn("no history yet", self._run())
        self.server._thread_brain_cache["tc"] = turn(1, size=10)
        self.assertIn("Already compact", self._run())

    def test_it_is_offered_in_the_command_list(self):
        names = [cmd["name"] for cmd in self.server.SLASH_COMMANDS] \
            if hasattr(self.server, "SLASH_COMMANDS") else None
        if names is None:
            import inspect
            self.assertIn('"/compact"', inspect.getsource(self.server))
        else:
            self.assertIn("/compact", names)

    def test_slack_answers_it_without_starting_a_turn(self):
        import inspect
        src = inspect.getsource(self.server._slack_start_turn)
        self.assertLess(src.index("_slack_compact(thread_id)"),
                        src.index("_run_pipeline_stream"))


class Wiring(unittest.TestCase):

    def test_the_message_window_no_longer_cuts_first(self):
        import inspect
        from src import agent
        src = inspect.getsource(agent._wrap_agent)
        self.assertIn("max(window_size, 400)", src)
        self.assertIn("CompactionHookProvider(role=role)", src)

    def test_only_the_history_column_is_rewritten(self):
        import inspect
        from src.utils import conversation_store as cs
        src = inspect.getsource(cs.update_brain_messages)
        self.assertIn("UPDATE thread_state SET brain_messages=?", src)
        self.assertNotIn("agent_session", src)


if __name__ == "__main__":
    unittest.main()
