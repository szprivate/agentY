"""replace_canvas_node: one node swapped for another type, in its place, wires kept.

Asked to "put a Seedream node in place of the GPT Image node", the agent built a
separate workflow and ran it beside the graph with scripts (2026-10-14): a swap
by hand is add + rewire + delete over two nodes' inputs and every consumer's.
"""
import copy
import unittest
from pathlib import Path

from src.utils import canvas_edit as ce

ROOT = Path(__file__).resolve().parent.parent

SCHEMAS = {
    "AgentYHook": {"input": {"required": {"text": ["STRING", {}]}}, "output": ["STRING"]},
    "LoadImage": {"input": {"required": {"image": [["a.png", "b.png"], {}]}}, "output": ["IMAGE", "MASK"]},
    "OpenAIGPTImage1": {
        "input": {"required": {"prompt": ["STRING", {"multiline": True}]},
                  "optional": {"seed": ["INT", {"default": 0, "min": 0, "max": 2147483647}],
                               "size": ["COMBO", {"options": ["auto", "1024x1024"]}],
                               "image": ["IMAGE", {}], "mask": ["MASK", {}],
                               "model": ["COMBO", {"options": ["gpt-image-1", "gpt-image-2"]}]}},
        "output": ["IMAGE"]},
    "ByteDanceSeedreamNode": {
        "input": {"required": {"model": ["COMBO", {"options": ["seedream 5.0 pro", "seedream 5.0 lite"]}],
                               "prompt": ["STRING", {"multiline": True}],
                               "size_preset": ["COMBO", {"options": ["2048x2048 (1:1)", "2560x1440 (16:9)"]}]},
                  "optional": {"image": ["IMAGE", {}],
                               "seed": ["INT", {"default": 0, "min": 0, "max": 2147483647}]}},
        "output": ["IMAGE"]},
    "bEpicSendToViewer": {"input": {"required": {"input": ["*", {}]}}, "output": []},
    "LatentOnly": {"input": {"required": {"samples": ["LATENT", {}]}}, "output": ["LATENT"]},
}


def graph():
    return {
        "8": {"class_type": "AgentYHook", "inputs": {"text": "x"}},
        "29": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "7": {"class_type": "OpenAIGPTImage1",
              "inputs": {"prompt": ["8", 0], "image": ["29", 0], "mask": ["29", 1],
                         "seed": 42, "size": "auto", "model": "gpt-image-2"}},
        "26": {"class_type": "bEpicSendToViewer", "inputs": {"input": ["7", 0]}},
    }


class Replacement(unittest.TestCase):
    def test_a_wire_with_nowhere_to_go_changes_nothing_and_says_where_it_could(self):
        got = ce.replacement_ops(graph(), "7", "ByteDanceSeedreamNode", SCHEMAS)
        self.assertFalse(got["ok"])
        self.assertEqual(got["ops"], [])
        self.assertIn("'mask'", got["errors"][0])
        self.assertIn("size_preset", got["errors"][0], "the new node's inputs are listed")

    def test_the_swap_keeps_place_id_wires_and_the_values_it_can(self):
        g = graph()
        got = ce.replacement_ops(g, "#7", "ByteDanceSeedreamNode", SCHEMAS,
                                 params={"size_preset": "2560x1440 (16:9)"}, wires={"mask": ""})
        self.assertTrue(got["ok"], got["errors"])
        self.assertEqual(got["carried"], {"seed": 42})
        self.assertEqual(sorted(got["dropped_values"]),
                         ["model = 'gpt-image-2' is not an option; e.g. ['seedream 5.0 pro', 'seedream 5.0 lite']",
                          "size"])
        self.assertEqual(got["left_behind"], ["#29 → mask"])
        res = ce.plan(g, got["ops"], SCHEMAS)
        self.assertTrue(res["ok"], res["errors"])
        new = res["graph"]["7"]
        self.assertEqual(new["class_type"], "ByteDanceSeedreamNode")
        self.assertEqual(new["inputs"]["prompt"], ["8", 0])
        self.assertEqual(new["inputs"]["image"], ["29", 0])
        self.assertNotIn("mask", new["inputs"])
        self.assertEqual(new["inputs"]["seed"], 42)
        self.assertEqual(new["inputs"]["model"], "seedream 5.0 pro", "a required menu gets its default")
        self.assertEqual(new["inputs"]["size_preset"], "2560x1440 (16:9)")
        self.assertEqual(res["graph"]["26"]["inputs"]["input"], ["7", 0])
        self.assertEqual(g, graph(), "the graph handed in is not touched")
        add = next(o for o in res["ops"] if o["op"] == "add")
        self.assertEqual((add["node_id"], add["at"]), ("7", "7"))
        self.assertEqual(res["ops"][0], {"op": "remove", "node": "7"})

    def test_the_panel_replay_gives_the_same_graph(self):
        g = graph()
        got = ce.replacement_ops(g, "7", "ByteDanceSeedreamNode", SCHEMAS, wires={"mask": ""})
        res = ce.plan(g, got["ops"], SCHEMAS)
        replay = ce.apply_patch(copy.deepcopy(g), {"op": "edit_graph", "ops": res["ops"]})
        self.assertEqual(replay, res["graph"])

    def test_a_wire_goes_by_type_when_the_name_differs(self):
        g = graph()
        g["7"]["inputs"]["images"] = g["7"]["inputs"].pop("image")
        got = ce.replacement_ops(g, "7", "ByteDanceSeedreamNode", SCHEMAS, wires={"mask": ""})
        self.assertTrue(got["ok"], got["errors"])
        self.assertIn("#29 → image (was images)", got["wired_in"])

    def test_wires_can_name_the_target(self):
        got = ce.replacement_ops(graph(), "7", "ByteDanceSeedreamNode", SCHEMAS,
                                 wires={"mask": "", "image": "image"})
        self.assertTrue(got["ok"], got["errors"])

    def test_a_consumer_that_would_lose_its_feed_refuses(self):
        g = graph()
        g["9"] = {"class_type": "LatentOnly", "inputs": {"samples": ["7", 0]}}
        SCHEMAS["OpenAIGPTImage1"]["output"] = ["LATENT"]
        try:
            got = ce.replacement_ops(g, "7", "ByteDanceSeedreamNode", SCHEMAS, wires={"mask": ""})
        finally:
            SCHEMAS["OpenAIGPTImage1"]["output"] = ["IMAGE"]
        self.assertFalse(got["ok"])
        self.assertTrue(any("#9 LatentOnly.samples takes LATENT" in e for e in got["errors"]))

    def test_unknown_things_are_named(self):
        self.assertIn("not installed", ce.replacement_ops(graph(), "7", "Nope", SCHEMAS)["errors"][0])
        self.assertIn("no node 70", ce.replacement_ops(graph(), "70", "LoadImage", SCHEMAS)["errors"][0])
        bad = ce.replacement_ops(graph(), "7", "ByteDanceSeedreamNode", SCHEMAS,
                                 params={"steps": 3}, wires={"mask": ""})
        self.assertIn("no input 'steps'", bad["errors"][0])

    def test_an_add_cannot_take_an_id_in_use(self):
        res = ce.plan(graph(), [{"op": "add", "class_type": "LoadImage", "id": "7"}], SCHEMAS)
        self.assertFalse(res["ok"])
        self.assertIn("id 7 is taken", res["errors"][0])


class Wiring(unittest.TestCase):
    def test_the_tool_is_offered_with_the_graph_tools(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("async def replace_canvas_node(", src)
        self.assertIn("edit_canvas_graph, replace_canvas_node, change_graph", src)
        from src.tools import tool_packs
        self.assertIn("replace_canvas_node", tool_packs.PACKS["graph"][1])

    def test_the_prompt_says_a_swap_is_this_tool(self):
        nodes = (ROOT / "config" / "system_prompts" / "orchestrator" / "canvas_nodes.md").read_text(encoding="utf-8")
        self.assertIn("**A swap is `replace_canvas_node(node_id, class_type, params)`**", nodes)
        self.assertNotIn("A swap is add + rewire + delete", nodes)

    def test_the_panel_puts_the_new_node_where_the_old_one_stood(self):
        js = ROOT.parent / "agentY-comfyuiConnect" / "web" / "agent_chat.js"
        if not js.is_file():
            self.skipTest("no comfyuiConnect checkout beside this one")
        text = js.read_text(encoding="utf-8")
        edit = text.split("  _editGraph(ev) {", 1)[1][:3000]
        self.assertIn('if (op.op === "remove")', edit)
        self.assertIn("placeOf[String(op.node)]", edit)
        self.assertIn("placeOf[String(op.at)]", edit)


if __name__ == "__main__":
    unittest.main()
