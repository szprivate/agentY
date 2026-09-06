"""`prepare_workflow` has to hand back the graph, not just a path.

The failure this closes: the orchestrator was told `{"status": "ready",
"workflow_path": …}` for a correctly fused two-stage workflow, did not believe
it, and spent six calls and ~297K input tokens re-deriving what the same message
could have carried for a few hundred — a tool it does not hold, a 120-second
permission timeout, a directory listing, and both workflow JSONs re-read.

``Pipeline._attach_built_summary`` is what carries it. It must attach whenever a
workflow exists, whatever the status, and it must never turn a good assembly into
an error by failing to describe it.
"""

import json
import os
import tempfile
import unittest

from src.pipeline import Pipeline


class _Stub:
    """Enough of a Pipeline to call the method under test."""

    _verbose = False
    _attach_built_summary = Pipeline._attach_built_summary


class ItAttachesTheGraph(unittest.TestCase):
    def setUp(self):
        self.stub = _Stub()
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({
                "101": {"class_type": "UNETLoader",
                        "inputs": {"unet_name": "FLUX1\\flux1-dev.safetensors"}},
                "5": {"class_type": "WanImageToVideo",
                      "inputs": {"width": 832, "height": 832}},
                "4": {"class_type": "SaveVideo",
                      "inputs": {"filename_prefix": "agent/videos/out"}},
            }, fh)

    def tearDown(self):
        os.remove(self.path)

    def test_a_ready_result_carries_what_was_built(self):
        result = {"status": "ready", "workflow_path": self.path}
        self.stub._attach_built_summary(result)
        built = result["built"]
        self.assertEqual(built["node_count"], 3)
        # The doubt was "did both stages really end up in one graph" — this is
        # the line that answers it without opening the file.
        self.assertEqual(set(built["nodes"].values()),
                         {"UNETLoader", "WanImageToVideo", "SaveVideo"})
        self.assertEqual(built["resolution"], "832x832")

    def test_a_needs_fix_result_carries_it_too(self):
        # The repair specialist benefits from it as much as the orchestrator.
        result = {"status": "needs_fix", "workflow_path": self.path,
                  "problems": ["something"]}
        self.stub._attach_built_summary(result)
        self.assertIn("built", result)

    def test_a_result_with_no_workflow_is_left_alone(self):
        result = {"status": "blocked", "blockers": ["which image?"]}
        self.stub._attach_built_summary(result)
        self.assertNotIn("built", result)

    def test_an_unreadable_workflow_does_not_spoil_the_result(self):
        result = {"status": "ready", "workflow_path": "no/such/file.json"}
        self.stub._attach_built_summary(result)   # must not raise
        self.assertEqual(result["status"], "ready")
        self.assertNotIn("built", result)

    def test_a_non_dict_result_is_survivable(self):
        self.stub._attach_built_summary("not a dict")  # must not raise

    def test_it_is_small_enough_to_send_on_every_generation(self):
        result = {"status": "ready", "workflow_path": self.path}
        self.stub._attach_built_summary(result)
        self.assertLess(len(json.dumps(result["built"])), 2000)


if __name__ == "__main__":
    unittest.main()
