"""One collector holding both kinds, as the agent sees it.

There were two collector nodes because their OUTPUT types differ — a stacked
IMAGE tensor versus a list of VIDEO objects — and ComfyUI fixes a node's output
types at registration, so no single output could be both. Carrying both outputs
settles that, and the two nodes become one.

What changes on this side is what the agent is told. A collector line used to
name its kind ("agentY image collector — 3 image file(s)"); one node holds both
now, so the split has to come from the paths. It matters: an image is something
to look at with analyze_image, a video needs analyze_video, and a count that
lumps them together leaves the agent to work it out from suffixes.

    python -m unittest tests.test_merged_collector
"""

import unittest

from src.utils.canvas_hooks import _looks_like_video_file, describe_hooks


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _hook(files, anchor_type="AgentYImageCollector"):
    return {"hook_node_id": "5", "purpose": "inline_parameter",
            "directive": "grade every one of these",
            "anchors": [{"node_id": "40", "type": anchor_type,
                         "widgets": {"files": files}}],
            "targets": [{"node_id": "60", "to_input": "prompt",
                         "to_input_type": "STRING", "type": "CLIPTextEncode"}]}


def _line(block):
    return next(l for l in block.splitlines() if "agentY collector" in l)


class CollectorLineTests(unittest.TestCase):
    def test_images_and_videos_are_counted_separately(self):
        block = describe_hooks([_hook("a.png\nb.jpg\nc.mp4")], {})
        line = _line(block)
        self.assertIn("2 image(s)", line)
        self.assertIn("1 video(s)", line)
        for p in ("a.png", "b.jpg", "c.mp4"):
            self.assertIn(p, line, "every path is still listed to bind directly")

    def test_a_single_kind_says_only_that_kind(self):
        images = _line(describe_hooks([_hook("a.png\nb.jpg")], {}))
        self.assertIn("2 image(s)", images)
        self.assertNotIn("video", images)
        videos = _line(describe_hooks([_hook("c.mp4\nd.mov")], {}))
        self.assertIn("2 video(s)", videos)
        self.assertNotIn("image(s)", videos)

    def test_it_is_named_by_what_it_is_not_by_a_kind(self):
        # The node is "agentY collector" now; a line calling it an image collector
        # would be describing a node the user cannot find on their canvas.
        block = describe_hooks([_hook("a.png")], {})
        self.assertIn("agentY collector", block)
        self.assertNotIn("agentY image collector", block)
        self.assertNotIn("agentY video collector", block)

    def test_an_empty_collector_still_says_so(self):
        block = describe_hooks([_hook("")], {})
        self.assertIn("EMPTY (no files added yet)", block)

    def test_the_old_video_node_is_described_the_same_way(self):
        # Deprecated, not gone: saved graphs still carry it, and it must not read
        # as an unknown node when one turns up.
        block = describe_hooks([_hook("c.mp4", anchor_type="AgentYVideoCollector")], {})
        self.assertIn("agentY collector", block)
        self.assertIn("1 video(s)", block)
        self.assertIn("c.mp4", block)


class SuffixTests(unittest.TestCase):
    """The split is made from the path, since the node type no longer says."""

    def test_videos_are_recognised(self):
        for name in ("c.mp4", "C.MOV", "x/y/clip.webm", "a.mkv", "b.m4v"):
            self.assertTrue(_looks_like_video_file(name), name)

    def test_images_and_prose_are_not(self):
        for name in ("a.png", "b.JPEG", "c.webp", "just some words", "", "noext"):
            self.assertFalse(_looks_like_video_file(name), name)


if __name__ == "__main__":
    unittest.main()
