"""edit_canvas_graph on a node whose inputs live inside a dynamic combo.

ByteDance2ReferenceNodeV2 (Seedance 2.5 reference-to-video) declares only
``model``, ``seed`` and ``watermark`` at the top; prompt, resolution and the
autogrow groups ``reference_images`` / ``reference_videos`` belong to the selected
option and are addressed ``model.prompt``, ``model.reference_images.image_1``.
The edit tool measured names against the top-level inputs only, so the agent
could not find the slots, listed "model, seed, watermark" as what the node has,
and asked the user to place the node by hand.

Now: names of the SELECTED option (nested autogrow slots included) are accepted
and typed, names of another option or none are refused with the valid dotted
names, and a connect carries the slot's type so the page can grow the socket.

    python -m unittest tests.test_canvas_edit_dynamic_combo
"""

import json
import pathlib
import unittest

from src.utils import canvas_edit as ce

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "seedance_ref_v2_object_info.json"
SCHEMAS = json.loads(FIXTURE.read_text(encoding="utf-8"))
SCHEMAS["LoadImage"] = {"input": {"required": {"image": [["a.png"], {}]}},
                        "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"]}
SCHEMAS["LoadVideo"] = {"input": {"required": {"file": [["a.mp4"], {}]}},
                        "output": ["VIDEO"], "output_name": ["VIDEO"]}
NODE = "ByteDance2ReferenceNodeV2"
DRAFT = "Seedance 2.5 Draft"

GRAPH = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
         "2": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}}}


def add_op(**params):
    return {"op": "add", "class_type": NODE, "ref": "v", "params": params}


def connect(src, to, name, output=0):
    return {"op": "connect", "from": src, "output": output, "to": to, "input": name}


class Accepted(unittest.TestCase):

    def test_the_slots_of_the_chosen_option_take_wires(self):
        out = ce.plan(GRAPH, [
            add_op(**{"model": DRAFT, "model.prompt": "a cat", "model.resolution": "480p"}),
            connect("1", "v", "model.reference_images.image_1"),
            connect("1", "v", "model.reference_images.image_2"),
            connect("2", "v", "model.reference_videos.video_1"),
        ], SCHEMAS)
        self.assertTrue(out["ok"], out["errors"])
        node = out["graph"]["3"]["inputs"]
        self.assertEqual(node["model"], DRAFT)
        self.assertEqual(node["model.reference_images.image_1"], ["1", 0])
        self.assertEqual(node["model.reference_images.image_2"], ["1", 0])
        self.assertEqual(node["model.reference_videos.video_1"], ["2", 0])

    def test_a_connect_names_its_type_for_the_page(self):
        out = ce.plan(GRAPH, [add_op(model=DRAFT),
                              connect("1", "v", "model.reference_images.image_1")], SCHEMAS)
        wire = [o for o in out["ops"] if o["op"] == "connect"][0]
        self.assertEqual(wire["slot_type"], "IMAGE")

    def test_the_combo_is_applied_first_whatever_the_order_written(self):
        out = ce.plan(GRAPH, [add_op(**{"model.prompt": "a cat", "model": DRAFT})], SCHEMAS)
        self.assertTrue(out["ok"], out["errors"])
        self.assertEqual(list(out["ops"][0]["params"])[0], "model")

    def test_a_value_is_checked_against_the_options_own_menu(self):
        out = ce.plan(GRAPH, [add_op(**{"model": DRAFT, "model.resolution": "1080p"})], SCHEMAS)
        self.assertFalse(out["ok"])
        self.assertIn("model.resolution", " ".join(out["errors"]))

    def test_a_node_already_on_the_canvas_is_judged_by_its_own_option(self):
        graph = dict(GRAPH, **{"5": {"class_type": NODE, "inputs": {"model": DRAFT}}})
        out = ce.plan(graph, [connect("1", "5", "model.reference_images.image_3")], SCHEMAS)
        self.assertTrue(out["ok"], out["errors"])


class Refused(unittest.TestCase):

    def test_an_unknown_slot_lists_the_valid_dotted_names(self):
        out = ce.plan(GRAPH, [add_op(model=DRAFT),
                              connect("1", "v", "model.reference_pictures.image_1")], SCHEMAS)
        self.assertFalse(out["ok"])
        msg = out["errors"][0]
        self.assertIn("model.reference_images.image_1..image_30", msg)
        self.assertIn("model.prompt", msg)
        self.assertNotIn("image_17", msg)

    def test_a_slot_past_the_end_of_its_group_is_refused(self):
        out = ce.plan(GRAPH, [add_op(model=DRAFT),
                              connect("1", "v", "model.reference_images.image_31")], SCHEMAS)
        self.assertFalse(out["ok"])

    def test_the_undotted_name_points_to_the_dotted_one(self):
        out = ce.plan(GRAPH, [add_op(model=DRAFT), connect("1", "v", "prompt")], SCHEMAS)
        self.assertFalse(out["ok"])
        self.assertIn("model.prompt", out["errors"][0])

    def test_a_slot_the_selected_option_does_not_have_is_refused_and_explained(self):
        # Two options that differ: find an input one has and another lacks.
        options = {o["key"]: ce._option_names(o)
                   for o in SCHEMAS[NODE]["input"]["required"]["model"][1]["options"]}
        pair = next(((a, b, n) for a in options for b in options if a != b
                     for n in options[a] - options[b] if "." in n), None)
        self.assertIsNotNone(pair, "the fixture's options do not differ in a nested slot")
        has, lacks, name = pair
        graph = dict(GRAPH, **{"5": {"class_type": NODE, "inputs": {"model": lacks}}})
        out = ce.plan(graph, [connect("1", "5", f"model.{name}")], SCHEMAS)
        self.assertFalse(out["ok"])
        self.assertIn(has, out["errors"][0])
        self.assertIn("set 'model' first", out["errors"][0])

    def test_a_wire_of_the_wrong_type_is_refused(self):
        out = ce.plan(GRAPH, [add_op(model=DRAFT),
                              connect("2", "v", "model.reference_images.image_1")], SCHEMAS)
        self.assertFalse(out["ok"])
        self.assertIn("IMAGE", out["errors"][0])


class Replaced(unittest.TestCase):

    def test_a_swap_keeps_the_dotted_wires_of_the_same_class(self):
        graph = dict(GRAPH, **{"5": {"class_type": NODE, "inputs": {
            "model": DRAFT, "model.reference_images.image_1": ["1", 0]}}})
        out = ce.replacement_ops(graph, "5", NODE, SCHEMAS, params={"model": DRAFT})
        self.assertTrue(out["ok"], out["errors"])
        self.assertTrue(any("model.reference_images.image_1" in w for w in out["wired_in"]),
                        out["wired_in"])


class AppliedToTheTurnsCopy(unittest.TestCase):

    def test_the_patch_replay_writes_the_dotted_keys(self):
        out = ce.plan(GRAPH, [add_op(model=DRAFT),
                              connect("1", "v", "model.reference_images.image_1")], SCHEMAS)
        replay = json.loads(json.dumps(GRAPH))
        ce.apply_patch(replay, {"op": "edit_graph", "ops": out["ops"]})
        self.assertEqual(replay["3"]["inputs"]["model.reference_images.image_1"], ["1", 0])


if __name__ == "__main__":
    unittest.main()
