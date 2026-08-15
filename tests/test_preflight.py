"""Read the graph before it runs, and say what cannot work.

Every check here is one afternoon's failure, found the expensive way — after the
prompts were written, the references generated and the API billed:

* a Kling node whose `reference_images` had nothing feeding it, because the hook
  that fed it was spliced out and no anchor could replace it;
* a hook feeding ONE slot of a batch node while its directive spoke of "all of
  the reference images";
* a directive addressing `anchor_1` with nothing wired to that slot.

The hard part is not finding them, it is not crying wolf: a check that fires on a
working graph teaches people to skip the block, which costs more than the check
ever saves. So the last test here is the one that matters most — a healthy graph
produces nothing at all.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import preflight
from src.utils.canvas_hooks import describe_hooks

# What ComfyUI would say about the nodes these graphs use.
_SCHEMAS = {
    "KlingOmniProImageToVideoNode": {"input": {"required": {
        "prompt": ["STRING", {}], "reference_images": ["IMAGE", {}],
        "duration": ["INT", {}]}}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE", {}]}}, "output_node": True},
    "LoadImage": {"input": {"required": {"image": ["COMBO", {}]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}], "clip": ["CLIP", {}]}}},
    "AgentYImageCollector": {"input": {"required": {"files": ["STRING", {}]}}},
    "BatchImagesNode": {"input": {"required": {"images": ["COMFY_AUTOGROW_V3", {}]}}},
}


def _schema(cls):
    return _SCHEMAS.get(cls, {})


class _Base(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(preflight, "_schema", side_effect=_schema))

    @staticmethod
    def _levels(found, level):
        return [f.text for f in found if f.level == level]


class BlockerTest(_Base):
    def test_a_required_connection_with_nothing_feeding_it(self):
        """The Kling case: the hook was spliced out and nothing replaced it."""
        graph = {"283": {"class_type": "KlingOmniProImageToVideoNode",
                         "inputs": {"prompt": "x", "duration": 10}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["283", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "make the video",
                  "targets": [{"node_id": "283", "to_input": "prompt",
                               "to_input_type": "STRING"}]}]
        blockers = self._levels(preflight.check(hooks, graph), "blocker")
        self.assertTrue(any("reference_images" in b for b in blockers), blockers)
        self.assertTrue(any("fails validation before it starts" in b for b in blockers))

    def test_an_input_a_hook_is_about_to_fill_is_not_missing(self):
        graph = {"283": {"class_type": "KlingOmniProImageToVideoNode",
                         "inputs": {"prompt": "x", "duration": 10}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["283", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "make the video",
                  "targets": [{"node_id": "283", "to_input": "prompt",
                               "to_input_type": "STRING"},
                              {"node_id": "283", "to_input": "reference_images",
                               "to_input_type": "IMAGE"}]}]
        self.assertEqual(self._levels(preflight.check(hooks, graph), "blocker"), [])

    def test_a_graph_that_renders_nothing(self):
        graph = {"6": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
                 "20": {"class_type": "CLIPTextEncode",
                        "inputs": {"text": "", "clip": ["6", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "write the prompt",
                  "targets": [{"node_id": "20", "to_input": "text",
                               "to_input_type": "STRING"}]}]
        blockers = self._levels(preflight.check(hooks, graph), "blocker")
        self.assertTrue(any("saves, previews or displays" in b for b in blockers), blockers)

    def test_a_viewer_counts_as_rendering_something(self):
        graph = {"20": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["6", 0]}},
                 "11": {"class_type": "bEpicSendToViewer", "inputs": {"input": ["20", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "x",
                  "targets": [{"node_id": "20", "to_input": "text",
                               "to_input_type": "STRING"}]}]
        self.assertEqual(self._levels(preflight.check(hooks, graph), "blocker"), [])


class NoteTest(_Base):
    def test_a_directive_naming_a_slot_that_is_not_wired(self):
        hooks = [{"hook_node_id": "30",
                  "directive": "prompts in anchor_0, reference images in anchor_1",
                  "anchors": [{"node_id": "75", "to_input": "anchors.anchor0",
                               "from_output_type": "STRING"}],
                  "targets": [{"node_id": "20", "to_input": "text",
                               "to_input_type": "STRING"}]}]
        graph = {"20": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["6", 0]}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["20", 0]}}}
        notes = self._levels(preflight.check(hooks, graph), "note")
        # Named together, not one at a time: "anchor_0 and anchor_1" needs two
        # inputs under every reading of those names, and this hook has one.
        self.assertTrue(any("anchor_1" in n and "1 input(s) wired" in n for n in notes), notes)

    def test_the_first_anchor_can_be_called_anchor_1(self):
        """The slots are `anchor`, `anchor0`, `anchor1` — the first has no number.

        So "use the style guide in anchor_1" with one input wired is a person
        counting from one, not a directive pointing at nothing. Warning about it
        taught people to skip the block, which costs more than the check saved.
        """
        hooks = [{"hook_node_id": "27",
                  "directive": "use the style guide connected to anchor_1",
                  "anchors": [{"node_id": "75", "to_input": "anchors.anchor",
                               "from_output_type": "STRING"}],
                  "targets": [{"node_id": "20", "to_input": "text",
                               "to_input_type": "STRING"}]}]
        graph = {"20": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["6", 0]}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["20", 0]}}}
        notes = self._levels(preflight.check(hooks, graph), "note")
        self.assertFalse([n for n in notes if "anchor_1" in n], notes)

    def test_a_chained_hook_counts_as_wiring_that_slot(self):
        hooks = [{"hook_node_id": "30",
                  "directive": "prompts in anchor_0, reference images in anchor_1",
                  "anchors": [{"node_id": "75", "to_input": "anchors.anchor0",
                               "from_output_type": "STRING"}],
                  "prev_links": [{"from_hook_id": "5", "to_input": "anchors.anchor1"}],
                  "targets": [{"node_id": "20", "to_input": "text",
                               "to_input_type": "STRING"}]}]
        graph = {"20": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["6", 0]}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["20", 0]}}}
        notes = self._levels(preflight.check(hooks, graph), "note")
        self.assertFalse([n for n in notes if "anchor_1" in n], notes)

    def test_one_image_slot_but_several_references_on_the_hook(self):
        """'All of the reference images' through a wire that carries one per run."""
        graph = {"284": {"class_type": "BatchImagesNode",
                         "inputs": {"images.image0": ["7", 0]}},
                 "7": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
                 "8": {"class_type": "LoadImage", "inputs": {"image": "b.png"}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["284", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "wire ALL the references in",
                  "anchors": [{"node_id": "7", "to_input": "anchors.anchor0",
                               "from_output_type": "IMAGE"},
                              {"node_id": "8", "to_input": "anchors.anchor1",
                               "from_output_type": "IMAGE"}],
                  "targets": [{"node_id": "284", "to_input": "images.image0",
                               "to_input_type": "IMAGE"}]}]
        notes = self._levels(preflight.check(hooks, graph), "note")
        self.assertTrue(any("ONE image slot" in n for n in notes), notes)
        self.assertTrue(any("image collector" in n for n in notes))

    def test_a_target_the_hook_has_nothing_to_feed(self):
        graph = {"283": {"class_type": "KlingOmniProImageToVideoNode",
                         "inputs": {"prompt": "x", "duration": 1,
                                    "reference_images": ["7", 0]}},
                 "7": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["283", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "pick a reference",
                  "anchors": [{"node_id": "75", "to_input": "anchors.anchor0",
                               "from_output_type": "STRING"}],
                  "targets": [{"node_id": "283", "to_input": "reference_images",
                               "to_input_type": "IMAGE"}]}]
        notes = self._levels(preflight.check(hooks, graph), "note")
        self.assertTrue(any("produces a IMAGE" in n or "produces a IMAGE" in n
                            for n in notes) or any("IMAGE" in n for n in notes), notes)

    def test_collector_paths_that_do_not_exist(self):
        graph = {"367": {"class_type": "AgentYImageCollector",
                         "inputs": {"files": "images (1).jpg\nt-rex.png"}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["367", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "x",
                  "targets": [{"node_id": "367", "to_input": "files",
                               "to_input_type": "STRING"}]}]
        notes = self._levels(preflight.check(hooks, graph), "note")
        self.assertTrue(any("do not exist" in n and "skips what it cannot find" in n
                            for n in notes), notes)


class QuietTest(_Base):
    """The test that matters most: a healthy graph says nothing at all."""

    @staticmethod
    def _healthy():
        graph = {"7": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
                 "20": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["7", 0]}},
                 "283": {"class_type": "KlingOmniProImageToVideoNode",
                         "inputs": {"prompt": ["20", 0], "duration": 10,
                                    "reference_images": ["7", 0]}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["283", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "write the video prompt",
                  "anchors": [{"node_id": "7", "to_input": "anchors.anchor0",
                               "from_output_type": "IMAGE"}],
                  "targets": [{"node_id": "20", "to_input": "text",
                               "to_input_type": "STRING"}]}]
        return hooks, graph

    def test_nothing_is_reported_for_a_graph_that_works(self):
        hooks, graph = self._healthy()
        self.assertEqual(preflight.check(hooks, graph), [])
        self.assertEqual(preflight.lines(hooks, graph), [])

    def test_the_block_is_absent_from_the_hooks_description(self):
        hooks, graph = self._healthy()
        self.assertNotIn("PRE-FLIGHT", describe_hooks(hooks, graph))

    def test_it_reaches_the_hooks_description_when_there_is_something_to_say(self):
        graph = {"283": {"class_type": "KlingOmniProImageToVideoNode",
                         "inputs": {"prompt": "x", "duration": 10}},
                 "11": {"class_type": "SaveImage", "inputs": {"images": ["283", 0]}}}
        hooks = [{"hook_node_id": "30", "directive": "make the video",
                  "targets": [{"node_id": "283", "to_input": "prompt",
                               "to_input_type": "STRING"}]}]
        block = describe_hooks(hooks, graph)
        self.assertIn("PRE-FLIGHT", block)
        self.assertIn("BLOCKER", block)
        self.assertIn("Do not start the run", block)


class NoComfyUITest(unittest.TestCase):
    """A check that needs ComfyUI must be absent when it isn't there, never wrong."""

    def test_schema_checks_go_quiet_and_the_rest_still_works(self):
        with mock.patch.object(preflight, "_schema", return_value={}):
            graph = {"283": {"class_type": "KlingOmniProImageToVideoNode",
                             "inputs": {"prompt": "x"}},
                     "11": {"class_type": "SaveImage", "inputs": {"images": ["283", 0]}}}
            hooks = [{"hook_node_id": "30", "directive": "uses anchor_2",
                      "anchors": [], "targets": [{"node_id": "283", "to_input": "prompt",
                                                  "to_input_type": "STRING"}]}]
            found = preflight.check(hooks, graph)
        self.assertEqual([f.text for f in found if f.level == "blocker"], [],
                         "no schema, no required-input verdict")
        self.assertTrue(any("anchor_2" in f.text for f in found),
                        "the wiring checks need nothing from ComfyUI")


if __name__ == "__main__":
    unittest.main()
