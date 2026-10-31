"""A loader the user's graph already uses is copied for the agent's new chain.

New nodes wired to an image or video loader used to be dropped beside it - where
the user's own graph goes on - so chains piled up behind existing nodes. Now the
host copies a media source that is in use (``canvas_edit.duplicate_sources``):
the new nodes read from the copy, the original and its wires stay as they were.
The page places copy and chain in free space (agentY-comfyuiConnect
tests/test_chain_placement.py), and copies a free-standing loader itself when
there is no room beside it, telling the host in its acknowledgement
(``record_duplicates``).

    python -m unittest tests.test_canvas_source_copy
"""

import asyncio
import copy
import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import canvas_edit as ce
from src.utils.canvas_hooks import hook_scope_ids

SCHEMAS = {
    "LoadImage": {"input": {"required": {"image": [["cat.png", "dog.png"], {"image_upload": True}]}},
                  "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"]},
    "VHS_LoadVideoPath": {"input": {"required": {"video": ["STRING", {}],
                                                 "frame_load_cap": ["INT", {"default": 0}]},
                                    "optional": {"vae": ["VAE"]}},
                          "output": ["IMAGE", "INT", "AUDIO"], "output_name": ["IMAGE", "frame_count", "audio"]},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT", {"default": 512}]}},
                         "output": ["LATENT"], "output_name": ["LATENT"]},
    "ImageScaleBy": {"input": {"required": {"image": ["IMAGE"], "scale_by": ["FLOAT", {"default": 1.0}]}},
                     "output": ["IMAGE"], "output_name": ["IMAGE"]},
    "ImageInvert": {"input": {"required": {"image": ["IMAGE"]}}, "output": ["IMAGE"], "output_name": ["IMAGE"]},
    "PreviewImage": {"input": {"required": {"images": ["IMAGE"]}}, "output": [], "output_node": True},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "x"}]}},
                  "output": [], "output_node": True},
    "AgentYHook": {"input": {"required": {"directive": ["STRING", {}]}, "optional": {"anchor": ["*"]}},
                   "output": ["*"], "output_name": ["out"]},
}


def _graph():
    """A loader in use (#5 -> #6 -> #7), a loader standing alone (#8), a video
    loader in use (#10 -> #11), and an empty latent in use (#12)."""
    return {
        "5": {"class_type": "LoadImage", "inputs": {"image": "cat.png", "upload": "image"}},
        "6": {"class_type": "ImageScaleBy", "inputs": {"image": ["5", 0], "scale_by": 2.0}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "mine"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "dog.png"}},
        "10": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": "D:/clips/a.mp4", "frame_load_cap": 16}},
        "11": {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
        "12": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}},
    }


def _branch(src="5"):
    return [
        {"op": "add", "class_type": "ImageInvert", "ref": "inv"},
        {"op": "add", "class_type": "PreviewImage", "ref": "pv"},
        {"op": "connect", "from": src, "output": 0, "to": "inv", "input": "image"},
        {"op": "connect", "from": "inv", "output": 0, "to": "pv", "input": "images"},
    ]


class WhatIsAMediaSource(unittest.TestCase):

    def test_loaders_are_and_makers_are_not(self):
        g = _graph()
        self.assertTrue(ce.is_media_source(g, "5", SCHEMAS))
        self.assertTrue(ce.is_media_source(g, "10", SCHEMAS))
        self.assertFalse(ce.is_media_source(g, "12", SCHEMAS), "an empty latent loads nothing")
        self.assertFalse(ce.is_media_source(g, "6", SCHEMAS), "something is wired into it")
        self.assertFalse(ce.is_media_source(g, "99", SCHEMAS))

    def test_a_loader_with_a_wire_in_is_not_one(self):
        """A copy would not carry the wire, so it would not be the same source."""
        g = _graph()
        g["10"]["inputs"]["vae"] = ["12", 0]
        self.assertFalse(ce.is_media_source(g, "10", SCHEMAS))


class TheHostCopiesASourceInUse(unittest.TestCase):

    def test_a_used_loader_is_copied_and_the_original_is_untouched(self):
        g = _graph()
        ops, copies = ce.duplicate_sources(g, _branch("5"), SCHEMAS)
        self.assertEqual(list(copies), ["5"])
        first = ops[0]
        self.assertEqual(first["op"], "add")
        self.assertEqual(first["class_type"], "LoadImage")
        self.assertEqual(first["duplicate_of"], "5")
        self.assertEqual(first["params"], {"image": "cat.png"}, "its values - and no key the class lacks")
        res = ce.plan(g, ops, SCHEMAS)
        self.assertTrue(res["ok"], res["errors"])
        new = res["graph"]
        dup = res["added"][copies["5"]]
        self.assertNotEqual(dup, "5")
        self.assertEqual(new[dup]["inputs"]["image"], "cat.png")
        self.assertEqual(new[res["added"]["inv"]]["inputs"]["image"], [dup, 0], "the chain reads the copy")
        self.assertEqual(new["5"], g["5"], "the original is as it was")
        self.assertEqual(new["6"]["inputs"]["image"], ["5", 0], "and so are its wires")
        add = next(o for o in res["ops"] if o["op"] == "add" and o["node_id"] == dup)
        self.assertEqual(add["duplicate_of"], "5", "the page is told what to clone")

    def test_a_video_loader_in_use_is_copied_with_its_frame_range(self):
        g = _graph()
        ops, copies = ce.duplicate_sources(g, _branch("10"), SCHEMAS)
        res = ce.plan(g, ops, SCHEMAS)
        self.assertTrue(res["ok"], res["errors"])
        dup = res["added"][copies["10"]]
        self.assertEqual(res["graph"][dup]["inputs"], {"video": "D:/clips/a.mp4", "frame_load_cap": 16})

    def test_a_loader_standing_alone_is_wired_as_it_is(self):
        """Whether there is room beside it is the page's to judge."""
        ops, copies = ce.duplicate_sources(_graph(), _branch("8"), SCHEMAS)
        self.assertEqual(copies, {})
        self.assertEqual(ops, _branch("8"))

    def test_a_splice_is_not_a_branch(self):
        """New nodes that feed the user's nodes sit between them: nothing to copy."""
        ops = [{"op": "add", "class_type": "ImageInvert", "ref": "inv"},
               {"op": "connect", "from": "5", "output": 0, "to": "inv", "input": "image"},
               {"op": "connect", "from": "inv", "output": 0, "to": "6", "input": "image"}]
        self.assertEqual(ce.duplicate_sources(_graph(), ops, SCHEMAS), (ops, {}))

    def test_a_replacement_is_left_alone(self):
        ops = [{"op": "remove", "node": "6"}, *_branch("5")]
        self.assertEqual(ce.duplicate_sources(_graph(), ops, SCHEMAS)[1], {})

    def test_nothing_made_but_a_latent_is_copied(self):
        ops = [{"op": "add", "class_type": "PreviewImage", "ref": "pv"}]
        self.assertEqual(ce.duplicate_sources(_graph(), ops, SCHEMAS)[1], {})

    def test_a_source_a_hook_reads_is_wired_in_place(self):
        """A hook can sweep its anchor's values; a copy would not follow them."""
        g = _graph()
        g["9"] = {"class_type": "AgentYHook", "inputs": {"directive": "try others", "anchor": ["5", 0]}}
        self.assertEqual(ce.duplicate_sources(g, _branch("5"), SCHEMAS)[1], {})

    def test_the_patch_replays_to_the_same_graph(self):
        g = _graph()
        ops, _ = ce.duplicate_sources(g, _branch("5"), SCHEMAS)
        res = ce.plan(g, ops, SCHEMAS)
        replay = ce.apply_patch(copy.deepcopy(g), {"op": "edit_graph", "ops": res["ops"]})
        self.assertEqual(replay, res["graph"])


class HookReferencesStillResolve(unittest.TestCase):

    def test_a_hook_downstream_of_the_source_keeps_its_nodes(self):
        g = _graph()
        g["9"] = {"class_type": "AgentYHook", "inputs": {"directive": "judge it", "anchor": ["6", 0]}}
        before = hook_scope_ids(g)
        ops, copies = ce.duplicate_sources(g, _branch("5"), SCHEMAS)
        self.assertEqual(list(copies), ["5"])
        res = ce.plan(g, ops, SCHEMAS)
        after = hook_scope_ids(res["graph"])
        self.assertEqual(res["graph"]["9"]["inputs"]["anchor"], ["6", 0])
        self.assertTrue(before <= after, "every node the hook reached is still there, under its id")
        self.assertNotIn(res["added"][copies["5"]], after, "the copy's branch is not the hook's")

    def test_a_hook_added_on_a_used_loader_reads_the_copy(self):
        g = _graph()
        ops = [{"op": "add", "class_type": "AgentYHook", "ref": "h", "params": {"directive": "vary"}},
               {"op": "connect", "from": "5", "output": 0, "to": "h", "input": "anchor"}]
        ops, copies = ce.duplicate_sources(g, ops, SCHEMAS)
        res = ce.plan(g, ops, SCHEMAS)
        self.assertTrue(res["ok"], res["errors"])
        hook = res["added"]["h"]
        dup = res["added"][copies["5"]]
        self.assertEqual(res["graph"][hook]["inputs"]["anchor"], [dup, 0])
        self.assertIn(dup, hook_scope_ids(res["graph"], [hook]))
        self.assertEqual(res["graph"][dup]["inputs"]["image"], "cat.png", "the same picture")


class ThePageCopiesALoaderWithNoRoom(unittest.TestCase):

    def test_its_copy_reaches_the_turns_graph(self):
        g = _graph()
        res = ce.plan(g, _branch("8"), SCHEMAS)
        inv = res["added"]["inv"]
        new = ce.record_duplicates(res["graph"], [{"of": "8", "id": "20", "wires": [[inv, "image", 0]]}])
        self.assertEqual(new["20"], {"class_type": "LoadImage", "inputs": {"image": "dog.png"}})
        self.assertEqual(new[inv]["inputs"]["image"], ["20", 0])
        self.assertEqual(new["8"], g["8"])

    def test_nonsense_is_skipped(self):
        g = _graph()
        before = copy.deepcopy(g)
        ce.record_duplicates(g, [None, {"of": "nope", "id": "21"}, {"of": "8", "id": "5"},
                                 {"of": "8", "id": "22", "wires": [["x"], ["99", "image", 0]]}])
        self.assertEqual({k: v for k, v in g.items() if k != "22"}, before, "an id in use is never taken over")
        self.assertIn("22", g)


class TheTool(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=True))
        self.enterContext(mock.patch("src.utils.preflight._schema", side_effect=lambda c: SCHEMAS.get(c, {})))
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)

    def _call(self, pipe, **kw):
        return json.loads(asyncio.run(tools(pipe)["edit_canvas_graph"](**kw)))

    def _edit(self, on=True):
        from src.utils.canvas_patch import drain
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[])
        with mock.patch("src.utils.agentY_server._duplicate_canvas_sources", return_value=on):
            out = self._call(pipe, ops=_branch("5"), reason="invert")
        patch = next(e for e in drain() if e.get("op") == "edit_graph")
        return pipe, out, patch

    def test_the_agent_is_told_the_copys_id(self):
        pipe, out, patch = self._edit()
        self.assertEqual(out["status"], "applied")
        dup = out["duplicated"]["5"]
        self.assertIn(dup, pipe._canvas_graph)
        self.assertIn(f"#5 (copy #{dup})", out["message"])
        self.assertTrue(patch["duplicate_sources"])
        self.assertEqual(pipe._canvas_graph["6"]["inputs"]["image"], ["5", 0])

    def test_switched_off_it_wires_the_loader_itself(self):
        pipe, out, patch = self._edit(on=False)
        self.assertNotIn("duplicated", out)
        self.assertFalse(patch["duplicate_sources"])
        inv = out["added"]["inv"]
        self.assertEqual(pipe._canvas_graph[inv]["inputs"]["image"], ["5", 0])

    def test_the_setting_is_on_by_default(self):
        from src.utils import agentY_server as srv
        with mock.patch.dict("os.environ", {"AGENTY_CANVAS_DUPLICATE": "", "AGENTY_HEADLESS": "0"}), \
                mock.patch("src.agent._load_settings", return_value={}):
            self.assertTrue(srv._duplicate_canvas_sources())
        with mock.patch.dict("os.environ", {"AGENTY_CANVAS_DUPLICATE": "0", "AGENTY_HEADLESS": "0"}):
            self.assertFalse(srv._duplicate_canvas_sources())


if __name__ == "__main__":
    unittest.main()
