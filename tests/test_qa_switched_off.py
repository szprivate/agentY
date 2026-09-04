"""Who decides whether a QA briefing gets read.

From the report: *"I tried to add a qa node to a workflow, but the orchestrator
just said: QA hook 49 — not mine to run. After the render, the QA agent compares
the output against the reference … So the qa checker agent actually never ran."*

It never ran because `qa.enabled` was false in settings. Two things were wrong
with that, and they are separate.

**Nothing said so.** `resolve_briefing` returns None both for "nobody wrote a
briefing" and for "one is wired to the canvas but QA is off", and the
`[CANVAS HOOKS]` block promised a QA agent either way — so the orchestrator
faithfully told the user a QA agent would compare the render against the
reference, and no line anywhere admitted it would not.

**And the switch should not have won.** Wiring a QA node into the graph in front
of you and leaving it live is a decision about THIS run; `qa.enabled` is a
standing default about runs in general. A live node now overrides it. Taking it
back is the standard ComfyUI gesture — bypass (Ctrl+B) or mute (Ctrl+M) — which
the panel already honours by never collecting a disabled hook. `AGENTY_QA=0` in
the environment stays absolute, so a cost-capped or CI run still can't be made to
spend a judge's tokens by a canvas it never sees.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.canvas_hooks import describe_hooks
from src.utils.qa import QaBriefing, resolve_briefing

QA_HOOK = {
    "hook_node_id": "49",
    "purpose": "qa",
    "directive": "reads as a trans woman in her 40s, not pregnant; match the "
                 "reference for lighting and mood",
    "anchors": [],
}

# The sentence that made the promise. Its absence is half the fix.
PROMISE = "A separate QA agent applies it to every image/video the run produces"


def _cfg(*, enabled: bool, forced_off: bool = False) -> dict:
    return {"enabled": enabled, "forced_off": forced_off, "max_retries": 1,
            "max_outputs": 6, "max_references": 4, "video_frames": 3,
            "briefing_dir": "./config/qa/"}


class ALiveNodeOutranksTheSetting(unittest.TestCase):
    """`resolve_briefing` — the decision itself."""

    def _resolve(self, *, enabled, forced_off=False, hooks=(), thread=None):
        with mock.patch("src.utils.qa.qa_settings",
                        return_value=_cfg(enabled=enabled, forced_off=forced_off)), \
             mock.patch("src.utils.qa.briefing_from_thread", return_value=thread):
            return resolve_briefing(hooks=list(hooks), thread_id="t1")

    def test_the_canvas_node_is_judged_with_the_switch_off(self):
        got = self._resolve(enabled=False, hooks=[dict(QA_HOOK)])
        self.assertIsNotNone(got)
        self.assertIn("not pregnant", got.criteria)

    def test_the_thread_briefing_is_still_shelved_by_the_switch(self):
        # `/qa` is the standing-default surface; the switch is exactly the right
        # thing to govern it. Only the canvas node is a per-run decision.
        shelved = QaBriefing(criteria="sharp", sources=("thread",))
        self.assertIsNone(self._resolve(enabled=False, hooks=[], thread=shelved))

    def test_the_env_kill_switch_beats_the_canvas_too(self):
        self.assertIsNone(self._resolve(enabled=False, forced_off=True,
                                        hooks=[dict(QA_HOOK)]))

    def test_switched_on_is_unchanged(self):
        self.assertIsNotNone(self._resolve(enabled=True, hooks=[dict(QA_HOOK)]))
        self.assertIsNotNone(self._resolve(enabled=True, hooks=[],
                                           thread=QaBriefing(criteria="sharp",
                                                             sources=("thread",))))

    def test_no_briefing_anywhere_is_still_none(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.assertIsNone(self._resolve(enabled=enabled, hooks=[]))

    def test_a_disabled_node_never_reaches_the_decision(self):
        # Bypass/mute is the way to take the override back, and it is enforced in
        # the panel (`_collectCanvasHooks` skips mode 4 and mode 2) — a disabled
        # hook is simply not in the list. Assert the contract this relies on, so
        # deleting that filter fails here rather than silently forcing QA on.
        js = ("../ComfyUI/custom_nodes/agentY-comfyuiConnect/web/agent_chat.js")
        try:
            with open(js, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            self.skipTest("sidebar checkout not present")
        self.assertIn("if (hn.mode === 4 || hn.mode === 2) continue;", src)

    def test_the_env_var_is_what_sets_forced_off(self):
        from src.utils.qa import qa_settings
        with mock.patch.dict("os.environ", {"AGENTY_QA": "0"}):
            cfg = qa_settings()
        self.assertTrue(cfg["forced_off"])
        self.assertFalse(cfg["enabled"])
        with mock.patch.dict("os.environ", {"AGENTY_QA": "1"}):
            cfg = qa_settings()
        self.assertFalse(cfg["forced_off"])
        self.assertTrue(cfg["enabled"])


class TheBlockSaysWhichWorldItIsIn(unittest.TestCase):
    """`describe_hooks` is the orchestrator's only account of what QA will do."""

    def _block(self, *, enabled: bool, forced_off: bool = False) -> str:
        with mock.patch("src.utils.qa.qa_settings",
                        return_value=_cfg(enabled=enabled, forced_off=forced_off)):
            return describe_hooks([dict(QA_HOOK)])

    def test_a_switched_off_setting_no_longer_changes_the_promise(self):
        # The node overrides it, so the normal wording is now the true one.
        block = self._block(enabled=False)
        self.assertIn(PROMISE, block)
        self.assertNotIn("NOTHING WILL CHECK", block)

    def test_a_force_disabled_process_does_not_promise_a_judge(self):
        block = self._block(enabled=False, forced_off=True)
        self.assertNotIn(PROMISE, block)
        self.assertIn("NOTHING WILL CHECK", block)

    def test_it_names_the_thing_that_actually_silenced_it(self):
        # Pointing at the settings switch here would send the user to change
        # something that is no longer in charge.
        block = self._block(enabled=False, forced_off=True)
        self.assertIn("AGENTY_QA", block)
        self.assertNotIn("Settings ▸ qa ▸ enabled", block)

    def test_the_agent_is_told_to_pass_that_on(self):
        block = self._block(enabled=False, forced_off=True)
        self.assertIn("say plainly that QA did NOT run", block)
        self.assertIn("Do not describe a check that will not happen", block)

    def test_the_criteria_survive_either_way(self):
        # Force-disabled, satisfying the briefing up front is the ONLY thing left
        # standing behind it — so dropping the criteria would be the worse bug.
        for off in (False, True):
            with self.subTest(forced_off=off):
                block = self._block(enabled=False, forced_off=off)
                self.assertIn("QA hook 49", block)
                self.assertIn("not pregnant", block)

    def test_neither_branch_offers_the_node_as_work(self):
        for off in (False, True):
            with self.subTest(forced_off=off):
                block = self._block(enabled=False, forced_off=off)
                self.assertIn("these are NOT work for you", block)
                self.assertIn("apply_canvas_hooks them", block)

    def test_an_unreadable_settings_file_keeps_the_normal_wording(self):
        from src.utils.canvas_hooks import _qa_will_run
        with mock.patch("src.utils.qa.qa_settings", side_effect=RuntimeError("boom")):
            self.assertTrue(_qa_will_run())


class TheUserHearsAboutItToo(unittest.TestCase):
    """The panel's own lines, independent of whatever the orchestrator writes."""

    def _notices(self, fn, *, enabled, forced_off=False, hooks=(), thread=None):
        from src.utils import agentY_server as srv
        said: list = []
        with mock.patch("src.utils.qa.qa_settings",
                        return_value=_cfg(enabled=enabled, forced_off=forced_off)), \
             mock.patch("src.utils.qa.briefing_from_thread", return_value=thread), \
             mock.patch.object(srv.status_bus, "notify", said.append):
            fx = getattr(srv, fn)
            # `_override_note` returns a suffix for the active-briefing line and
            # takes only the hooks; `_warn_qa_is_off` speaks for itself and needs
            # the thread too.
            out = (fx(list(hooks)) if fn == "_override_note"
                   else fx(list(hooks), "t1"))
        return said, out

    def test_an_override_announces_itself(self):
        # Spending a judge's tokens on a run whose settings say QA is off is
        # right — they wired the node — but it must never be a surprise.
        _said, note = self._notices("_override_note", enabled=False,
                                    hooks=[dict(QA_HOOK)])
        self.assertIn("overrides Settings ▸ qa ▸ enabled", note)
        self.assertIn("bypass or mute", note.lower())

    def test_nothing_is_announced_when_there_was_nothing_to_override(self):
        _said, note = self._notices("_override_note", enabled=True,
                                    hooks=[dict(QA_HOOK)])
        self.assertEqual(note, "")
        _said, note = self._notices("_override_note", enabled=False, hooks=[])
        self.assertEqual(note, "")

    def test_a_force_disabled_canvas_node_is_announced(self):
        said, _ = self._notices("_warn_qa_is_off", enabled=False, forced_off=True,
                                hooks=[dict(QA_HOOK)])
        self.assertEqual(len(said), 1)
        self.assertIn("AGENTY_QA", said[0])
        self.assertIn("will NOT be checked", said[0])

    def test_a_shelved_thread_briefing_is_announced_with_the_way_out(self):
        said, _ = self._notices("_warn_qa_is_off", enabled=False, hooks=[],
                                thread=QaBriefing(criteria="sharp",
                                                  sources=("thread",)))
        self.assertEqual(len(said), 1)
        self.assertIn("Settings ▸ qa ▸ enabled", said[0])
        self.assertIn("wire a QA node into the canvas", said[0])

    def test_no_briefing_no_warning(self):
        said, _ = self._notices("_warn_qa_is_off", enabled=False, hooks=[])
        self.assertEqual(said, [])

    def test_switched_on_says_nothing_here(self):
        said, _ = self._notices("_warn_qa_is_off", enabled=True,
                                hooks=[dict(QA_HOOK)])
        self.assertEqual(said, [])

    def test_a_broken_lookup_never_costs_the_turn(self):
        from src.utils import agentY_server as srv
        with mock.patch("src.utils.qa.qa_settings", side_effect=RuntimeError("boom")), \
             mock.patch.object(srv.status_bus, "notify",
                               side_effect=AssertionError("should not speak")):
            srv._warn_qa_is_off([dict(QA_HOOK)], "t1")     # must not raise
            self.assertEqual(srv._override_note([dict(QA_HOOK)]), "")


if __name__ == "__main__":
    unittest.main()
