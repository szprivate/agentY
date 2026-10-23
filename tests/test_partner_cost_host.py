"""agentY's wiring of the partner-credit checks (logic and most tests: agenty_core).

Offline: schemas and balance are stubs; nothing is submitted to ComfyUI.
Runs under pytest or ``python -m unittest tests.test_partner_cost_host``.
"""
import json
import tempfile
import unittest
from pathlib import Path

from agenty_core import partner_credits as pc
from agenty_core.utils import turn_scope

BADGE = {"engine": "jsonata", "depends_on": {"widgets": [{"name": "n", "type": "INT"}], "inputs": [],
                                             "input_groups": []},
         "expr": '{"type":"usd","usd": widgets.n * 0.5}'}
INFO = {"PaidNode": {"price_badge": BADGE}, "KSampler": {"price_badge": None}}


def wf(n):
    return {"1": {"class_type": "PaidNode", "inputs": {"n": n}}}


def write(w):
    f = Path(tempfile.mkdtemp()) / "w.json"
    f.write_text(json.dumps(w), encoding="utf-8")
    return str(f)


class Host(unittest.TestCase):
    def setUp(self):
        pc.clear_caches()
        self.cents = 10000
        self.balance_calls = 0
        pc._info_provider = lambda ct: INFO.get(ct)
        pc._http_get = self._balance
        pc._sleep = lambda s: None
        self._orig_log, pc._log = pc._log, lambda e: None
        pc._confirmed_paths.clear()
        pc._inflight.clear()
        pc._watermark = None
        from src.utils import partner_cost_host
        self.pch = partner_cost_host

    def _balance(self, url, headers):
        self.balance_calls += 1
        return {"effective_balance_cents": self.cents, "prepaid_balance_cents": self.cents}

    def tearDown(self):
        pc._http_get = pc._info_provider = pc._threshold_provider = None
        pc._sleep = __import__("time").sleep
        pc._log = self._orig_log
        pc._listeners.clear()
        pc.clear_caches()
        self.pch._registered = False

    def test_threshold_comes_from_settings(self):
        self.pch.ensure_registered()
        from src.utils.settings import load_settings
        want = float(load_settings()["partner_credits"]["confirm_above_usd"])
        self.assertEqual(want, 1.0)
        self.assertEqual(pc.confirm_threshold(), want)

    def test_gate_asks_then_remembers_the_go_ahead(self):
        p = write(wf(4))                                      # $2.00
        refusal = self.pch.gate([p])
        self.assertEqual(refusal["error_kind"], "confirm_cost")
        self.assertIsNone(self.pch.gate([p], confirm=True))
        self.assertIsNone(self.pch.gate([p]))                 # remembered for the file
        pc.gate_submission(wf(4), p)                          # and the executor's own gate agrees

    def test_gate_refuses_what_the_balance_cannot_cover(self):
        self.cents = 100
        refusal = self.pch.gate([write(wf(4))], confirm=True)
        self.assertEqual(refusal["error_kind"], "insufficient_credits")

    def test_cheap_and_local_runs_pass_and_local_costs_no_call(self):
        self.assertIsNone(self.pch.gate([write(wf(1))]))      # $0.50
        before = self.balance_calls
        self.assertIsNone(self.pch.gate([write({"1": {"class_type": "KSampler", "inputs": {}}})]))
        self.assertEqual(self.balance_calls, before)
        self.assertIsNone(self.pch.estimate_of([write({"1": {"class_type": "KSampler", "inputs": {}}})]))
        self.assertEqual(self.pch.estimate_of([write(wf(1))]), {"min": 0.5, "max": 0.5})

    def test_spend_lands_in_the_turn_usage(self):
        from src.utils import turn_usage
        scope = turn_scope.Scope("req-1", "thr-1")
        tok = turn_scope._current.set(scope)
        try:
            self.pch.ensure_registered()
            turn_usage.add(100, 20, calls=1)
            pc.track("pp", pc.estimate_workflow(wf(1)))
            self.cents = 10000 - 50
            self.assertEqual(pc.settle("pp"), 0.5)
            snap = turn_usage.snapshot(scope)
            self.assertEqual(snap["partner_cost"], 0.5)
            self.assertIsNone(snap["cost"])                   # LLM cost untouched
            self.assertEqual(turn_usage.take_changed(scope)["partner_cost"], 0.5)
        finally:
            turn_scope._current.reset(tok)

    def test_executor_gate_runs_before_the_post(self):
        from src import executor
        from agenty_core.utils import comfyui_client as C
        posts = []

        class Client:
            api_key = ""

            def post(self, path, json_data=None, **_):
                posts.append(path)
                return {"prompt_id": "pid"}

        orig = C.get_client
        C.get_client = lambda: Client()
        try:
            with self.assertRaises(pc.CostGateError) as cm:
                executor._submit_workflow(write(wf(4)))
            self.assertEqual(cm.exception.error_kind, "confirm_cost")
            self.assertEqual(posts, [])
        finally:
            C.get_client = orig


if __name__ == "__main__":
    unittest.main()
