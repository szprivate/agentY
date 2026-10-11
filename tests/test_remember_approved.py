"""An approved result the canvas does not hold is kept in project memory.

Approved pictures normally sit in the review's collector, an approved text in
its `agentY text` node. When neither is on the canvas - the stop was raised with
no page open, or text nodes are switched off - the approval lived only in the
one run's ledger. It is now also written to project memory, as a pointer and one
line, in the shape a remembered `#tag` is read from.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import review_gate as rg
from src.utils import run_ledger as rl
from src.utils import tag_memory

ROOT = Path(__file__).resolve().parent.parent


class TheEntry(unittest.TestCase):
    def test_a_pointer_and_one_line(self):
        body = rl.memory_body("main location", ["//srv/loc_a.png", "//srv/loc_b.png"], "", "118", "136")
        self.assertEqual(body.splitlines(), [
            '2 approved file(s) of "main location".',
            "path: //srv/loc_a.png",
            "also: //srv/loc_b.png",
            "Approved in an agentY pipeline run (stage 118, review 136)."])

    def test_a_text_is_its_file_and_its_first_line(self):
        body = rl.memory_body("screenplay", [], "//srv/sh_v002_screenplay.md", "4", "131",
                              "INT. KITCHEN - MORNING\n")
        self.assertEqual(body.splitlines()[:2], [
            'Approved text of "screenplay": INT. KITCHEN - MORNING',
            "path: //srv/sh_v002_screenplay.md"])
        self.assertNotIn("\n\n", body)

    def test_it_is_named_by_its_tag_else_by_its_stage(self):
        self.assertEqual(rl.memory_name("Main location", "118", "#appartmentref"), "appartmentref")
        self.assertEqual(rl.memory_name("Main location", "118"), "main_location")
        self.assertEqual(rl.memory_name("", "118"), "stage_118_approved")

    def test_it_reads_back_as_a_remembered_tag(self):
        body = rl.memory_body("main location", ["//srv/loc_a.png"], "", "118", "136")
        entry = types.SimpleNamespace(type="reference", body=body)
        with mock.patch("agenty_core.utils.project_memory.read_entry", return_value=entry):
            ref = tag_memory.remembered_reference("main_location")
        self.assertEqual(ref["path"], "//srv/loc_a.png")
        self.assertEqual(ref["role"], 'Approved file of "main location".')

    def test_nothing_approved_writes_nothing(self):
        with mock.patch("agenty_core.utils.project_memory.write_entry") as write:
            self.assertEqual(rl.remember_approved("x", [], "", "1", "2"), "")
        write.assert_not_called()

    def test_written_as_a_reference(self):
        with mock.patch("agenty_core.utils.project_memory.write_entry", return_value=object()) as write:
            name = rl.remember_approved("main location", ["a.png"], "", "118", "136")
        self.assertEqual(name, "main_location")
        self.assertEqual(write.call_args.args[0], "main_location")
        self.assertEqual(write.call_args.kwargs, {"type": "reference"})


def me(collector=None, graph=None, dry=False):
    obj = types.SimpleNamespace(_dry_run=dry, _verbose=False, _canvas_graph=graph or {})
    obj._ledger_stages_of = lambda rid: {"136": ["118"], "131": ["4"]}[str(rid)]
    obj._ledger_lead = lambda: "lead"
    obj._review_collector = lambda halt=None: collector
    return obj


STAGES = {"118": {"title": "main location", "state": rl.APPROVED},
          "4": {"title": "screenplay", "state": rl.APPROVED, "text": "//srv/s_v002.md"}}
PICTURES = rg.ReviewHalt(hook_node_id="136", collector_key="k", produced=("a.png",))
TEXT = rg.ReviewHalt(hook_node_id="131", text_hooks=("4",))


@mock.patch.object(rl, "stage", lambda lead, sid: dict(STAGES[str(sid)]))
class WhenItIsWritten(unittest.TestCase):
    def remember(self, obj, halt, files, text_nodes=False):
        with mock.patch.object(rl, "remember_approved", return_value="written") as write, \
                mock.patch("src.utils.agentY_server._place_text_nodes_on_canvas",
                           return_value=text_nodes):
            said = Pipeline._remember_approved(obj, halt, files)
        return said, write

    def test_pictures_with_no_collector_on_the_canvas(self):
        said, write = self.remember(me(collector=None), PICTURES, ["a.png"])
        self.assertEqual(said, "written")
        self.assertEqual(write.call_args.args, ("main location", ["a.png"], "", "118", "136", ""))

    def test_not_when_the_collector_holds_them(self):
        said, write = self.remember(me(collector={"node_id": "77", "files": ["a.png"]}), PICTURES, ["a.png"])
        self.assertEqual(said, "")
        write.assert_not_called()

    def test_a_text_when_text_nodes_are_switched_off(self):
        said, write = self.remember(me(), TEXT, [])
        self.assertEqual(write.call_args.args, ("screenplay", [], "//srv/s_v002.md", "4", "131", ""))

    def test_not_when_the_text_node_is_on_the_canvas(self):
        _said, write = self.remember(me(), TEXT, [], text_nodes=True)
        write.assert_not_called()

    def test_under_the_tag_the_user_put_on_it(self):
        graph = {"136": {"class_type": "AgentYReview", "inputs": {}},
                 "50": {"class_type": "AgentYRefNote", "inputs": {"input": ["136", 0], "tag": "appartmentref"}}}
        _said, write = self.remember(me(graph=graph), PICTURES, ["a.png"])
        self.assertEqual(write.call_args.args[-1], "appartmentref")

    def test_never_from_a_dry_run(self):
        _said, write = self.remember(me(dry=True), PICTURES, ["a.png"])
        write.assert_not_called()

    def test_every_approval_passes_through_it(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        body = src.split("    def _ledger_approve(self, halt)", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("self._remember_approved(halt, files)", body)


if __name__ == "__main__":
    unittest.main()
