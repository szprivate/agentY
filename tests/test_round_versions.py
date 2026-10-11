"""One version per round of a stage.

Every graph that was submitted took the next free version. A stage sweeping four
characters submits four graphs, so each picture landed in a version folder of
its own (v185, v186, v187, v188): a folder per file. A round of a stage is one
piece of work, and is now one version.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import conversation_store as cs
from src.utils import output_context as oc
from src.utils import run_ledger as rl

ROOT = Path(__file__).resolve().parent.parent
CTX = {"sequence": "seq010", "shot": "sh020"}


def variant(prompt="a"):
    return {"7": {"class_type": "Gen", "inputs": {"prompt": prompt}},
            "30": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "char"}},
            "40": {"class_type": "bEpicGetPathAYON", "inputs": {"path_key": "pathImages", "Version": 1,
                                                                 "suffix": ""}}}


class Scope:
    """A turn's scope, enough for the output context to keep its slots in."""
    def __init__(self):
        self._slots = {}

    def slot(self, name, make):
        return self._slots.setdefault(name, make())


class Base(unittest.TestCase):
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
        self.root = Path(self._dir.name) / "output"
        self.root.mkdir()
        scope = Scope()
        for target, value in (("src.utils.output_context._scope", lambda: scope),
                              ("src.utils.output_context._output_root", lambda: self.root)):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        oc.set_current(sequence=CTX["sequence"], shot=CTX["shot"])

    def me(self, dry=False):
        obj = types.SimpleNamespace(_dry_run=dry, _verbose=False, _stage_runs={}, _version_named=0)
        obj._ledger_lead = lambda: "lead"
        return obj

    def run_stage(self, obj, n=4, same_round=False, hid="8"):
        prompts = [variant(str(i)) for i in range(n)]
        v = Pipeline._version_stage(obj, hid, prompts, same_round)
        run = obj._stage_runs.get(hid) or {}
        rl.produced("lead", hid, outputs=[f"f{i}.png" for i in range(n)],
                    new_round=bool(run.get("new_round", True)), version=int(run.get("version") or 0))
        return v, prompts


class OneRoundOneVersion(Base):
    def test_every_picture_of_a_sweep_shares_the_version(self):
        v, prompts = self.run_stage(self.me())
        self.assertEqual(v, 1)
        self.assertEqual({p["30"]["inputs"]["filename_prefix"] for p in prompts},
                         {"seq010/sh020/images/v001/sh020_v001_char"})

    def test_the_path_nodes_version_is_set_too(self):
        _v, prompts = self.run_stage(self.me())
        self.assertEqual({p["40"]["inputs"]["Version"] for p in prompts}, {1})

    def test_the_next_round_is_the_next_version(self):
        self.run_stage(self.me())
        v, prompts = self.run_stage(self.me())             # a new turn: the user's notes
        self.assertEqual(v, 2)
        self.assertIn("/v002/sh020_v002_char", prompts[0]["30"]["inputs"]["filename_prefix"])
        self.assertEqual(rl.stage("lead", "8")["round"], 2)

    def test_counted_per_stage(self):
        self.run_stage(self.me(), hid="8")
        self.run_stage(self.me(), hid="8")
        v, _ = self.run_stage(self.me(), hid="118")        # another stage's first round
        self.assertEqual(v, 1)

    def test_the_submission_leaves_a_named_graph_alone(self):
        _v, prompts = self.run_stage(self.me())
        before = prompts[0]["30"]["inputs"]["filename_prefix"]
        oc.stamp(prompts[0])                               # what the executor does later
        self.assertEqual(prompts[0]["30"]["inputs"]["filename_prefix"], before)

    def test_a_dry_run_versions_nothing(self):
        prompts = [variant()]
        self.assertEqual(Pipeline._version_stage(self.me(dry=True), "8", prompts), 0)
        self.assertEqual(prompts[0]["30"]["inputs"]["filename_prefix"], "char")


class StayingInTheRound(Base):
    def test_a_re_run_in_the_same_turn(self):
        obj = self.me()
        self.run_stage(obj)
        v, _ = self.run_stage(obj, n=1)                    # a repair, same turn
        self.assertEqual(v, 1)
        st = rl.stage("lead", "8")
        self.assertEqual((st["round"], len(st["outputs"])), (1, 4))    # f0 again, not a new round

    def test_a_single_picture_redone_later_when_the_call_says_so(self):
        self.run_stage(self.me())
        v, prompts = self.run_stage(self.me(), n=1, same_round=True)
        self.assertEqual(v, 1)
        self.assertIn("/v001/", prompts[0]["30"]["inputs"]["filename_prefix"])
        self.assertEqual(rl.stage("lead", "8")["round"], 1)

    def test_a_loops_next_round_in_the_same_turn_is_a_new_version(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        check = src.split("        async def loop_check(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("self._stage_runs = {}", check)
        obj = self.me()
        self.run_stage(obj)
        obj._stage_runs = {}                               # what loop_check does
        self.assertEqual(self.run_stage(obj)[0], 2)


class AShotWithEarlierWork(Base):
    def setUp(self):
        super().setUp()
        (self.root / "seq010" / "sh020" / "images" / "v188").mkdir(parents=True)
        (self.root / "seq010" / "sh020" / "videos" / "v012").mkdir(parents=True)

    def test_the_run_starts_above_what_is_there(self):
        self.assertEqual(oc.highest_version(oc.current()), 188)
        self.assertEqual(self.run_stage(self.me())[0], 189)

    def test_and_counts_up_from_there_not_from_what_it_makes(self):
        self.run_stage(self.me(), hid="8")                 # v189
        (self.root / "seq010" / "sh020" / "images" / "v189").mkdir()
        self.assertEqual(self.run_stage(self.me(), hid="118")[0], 189)     # its round 1
        self.assertEqual(self.run_stage(self.me(), hid="8")[0], 190)       # round 2

    def test_a_fresh_shot_has_round_and_version_the_same(self):
        oc.set_current(sequence="seq010", shot="sh030")
        self.assertEqual(self.run_stage(self.me())[0], 1)


class AVersionTheUserNamed(Base):
    def test_it_wins(self):
        obj = self.me()
        obj._version_named = 12
        v, prompts = self.run_stage(obj)
        self.assertEqual(v, 12)
        self.assertIn("/v012/sh020_v012_char", prompts[0]["30"]["inputs"]["filename_prefix"])
        self.assertEqual(prompts[0]["40"]["inputs"]["Version"], 12)

    def test_name_outputs_takes_it(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        tool = src.split("        async def name_outputs(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("workflow_path: str = \"\", version: int = 0", tool)
        self.assertIn("self._version_named = max(0, int(version or 0))", tool)
        self.assertIn("_oc.pin_version(self._version_named or None)", tool)

    def test_it_lasts_one_turn(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        start = src.split("self._canvas_keeplive_targets = []", 1)[1].split("self._hook_run_stopped", 1)[0]
        self.assertIn("self._version_named = 0", start)
        self.assertIn("self._stage_runs = {}", start)


class ThePieces(Base):
    def test_only_path_nodes_with_a_version_number_are_touched(self):
        g = {"1": {"class_type": "bEpicGetPath", "inputs": {"Version": 3}},
             "2": {"class_type": "KSampler", "inputs": {"Version": 3}},
             "3": {"class_type": "get_path_ayon", "inputs": {"Version": ["9", 0]}},
             "4": {"class_type": "bepicSetPath", "inputs": {"version": 3}}}
        self.assertEqual(oc.set_versions(g, 7), ["1"])
        self.assertEqual([g[k]["inputs"] for k in "234"],
                         [{"Version": 3}, {"Version": ["9", 0]}, {"version": 3}])

    def test_the_pin_is_lifted(self):
        oc.pin_version(5)
        self.assertEqual(oc.pinned(), 5)
        oc.pin_version(None)
        self.assertIsNone(oc.pinned())
        Pipeline._version_stage(self.me(), "8", [variant()])
        self.assertIsNone(oc.pinned())

    def test_the_ledger_says_which_version_a_round_was_saved_as(self):
        self.run_stage(self.me())
        rl.sync("lead", [{"hook_node_id": "8", "purpose": "set_parameter", "title": "characters"}], {})
        self.assertIn("round 1 (saved as v001)", rl.index("lead"))

    def test_the_stage_run_asks_for_it_before_the_graphs_are_written(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        tool = src.split("        async def apply_canvas_hooks(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("same_round: bool = False", tool)
        self.assertLess(tool.index("_version = self._version_stage("), tool.index("fp.write_text(json.dumps(p)"))


if __name__ == "__main__":
    unittest.main()
