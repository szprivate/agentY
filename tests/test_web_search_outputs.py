"""References found on the web have to reach the canvas, not just the log.

"Search the web for pictures of X" ended with a JSON manifest in the transcript
and an empty canvas. Every step worked: the scout searched, picked, and called
`download_image`, which really does put the file in ComfyUI's input directory and
report where. Nothing ever told anyone to SHOW it.

The gap is that staging and publishing are different things. Generated media
reaches the canvas because a run's outputs are registered as the turn's outputs;
a file a tool wrote mid-turn reaches it the same way (`annotate_image` does this
already, via the output sink). The scout's downloads were doing neither — so the
one flow whose entire product is a picture was the one flow that showed none.

    python -m unittest discover -s tests
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_stub import pipeline_stub, tools
from src.pipeline import staged_reference_paths


class ManifestReadingTest(unittest.TestCase):
    """Which files a scout run actually staged, read off what it answered."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        root = Path(cls._dir.name)
        cls.a = root / "diner.jpg"
        cls.b = root / "neon.png"
        for f in (cls.a, cls.b):
            f.write_bytes(b"\x89PNG\r\n\x1a\n")
        cls.gone = str(root / "never_downloaded.jpg")

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _manifest(self, *refs):
        return json.dumps({"references": list(refs)})

    def test_an_image_reference_yields_its_file(self):
        out = staged_reference_paths(
            self._manifest({"mode": "image", "path": str(self.a),
                            "description": "a diner"}))
        self.assertEqual(out, [str(self.a)])

    def test_a_text_reference_has_no_file_to_show(self):
        out = staged_reference_paths(
            self._manifest({"mode": "text", "description": "1950s americana"}))
        self.assertEqual(out, [])

    def test_several_images_keep_the_scouts_order(self):
        out = staged_reference_paths(
            self._manifest({"mode": "image", "path": str(self.a)},
                           {"mode": "text", "description": "mood"},
                           {"mode": "image", "path": str(self.b)}))
        self.assertEqual(out, [str(self.a), str(self.b)])

    def test_a_path_that_does_not_exist_is_not_offered(self):
        """A download that failed must not become a broken loader node."""
        out = staged_reference_paths(
            self._manifest({"mode": "image", "path": self.gone}))
        self.assertEqual(out, [])

    def test_the_same_file_twice_is_dropped_once(self):
        out = staged_reference_paths(
            self._manifest({"mode": "image", "path": str(self.a)},
                           {"mode": "image", "path": str(self.a)}))
        self.assertEqual(out, [str(self.a)])

    def test_saved_to_is_understood_as_well_as_path(self):
        """`download_image` returns both names; the scout may echo either."""
        out = staged_reference_paths(
            self._manifest({"mode": "image", "saved_to": str(self.a)}))
        self.assertEqual(out, [str(self.a)])

    def test_a_file_with_no_mode_is_still_shown(self):
        """`mode` says how to USE it. The path is the evidence it exists."""
        out = staged_reference_paths(self._manifest({"path": str(self.a)}))
        self.assertEqual(out, [str(self.a)])

    def test_a_fenced_manifest_is_read(self):
        """Models fence their JSON however the prompt is worded."""
        body = self._manifest({"mode": "image", "path": str(self.a)})
        out = staged_reference_paths(f"Here you go:\n```json\n{body}\n```")
        self.assertEqual(out, [str(self.a)])

    def test_nothing_readable_costs_nothing(self):
        for junk in ("", "no references found", "{", None, "[]",
                     '{"references": "not a list"}', '{"other": []}',
                     # Balanced braces but not JSON — this reaches the parser,
                     # where "{" alone never does.
                     '{"references": [oops]}',
                     '{"references": {"a": 1}}',
                     # Not iterable at all: the loop itself would raise.
                     '{"references": 7}'):
            self.assertEqual(staged_reference_paths(junk), [], repr(junk))

    def test_junk_entries_do_not_stop_the_good_ones(self):
        out = staged_reference_paths(json.dumps(
            {"references": ["a string", None, 7,
                            {"mode": "image", "path": str(self.b)}]}))
        self.assertEqual(out, [str(self.b)])

    def test_a_path_that_is_not_a_string_is_ignored(self):
        """A model that answers `"path": 123` must not take the search down."""
        out = staged_reference_paths(json.dumps(
            {"references": [{"mode": "image", "path": 123},
                            {"mode": "image", "path": None},
                            {"mode": "image", "path": ["a", "list"]},
                            {"mode": "image", "path": "   "},
                            {"mode": "image", "path": str(self.a)}]}))
        self.assertEqual(out, [str(self.a)])


class _FakeAgent:
    """Stands in for the Reference Scout: answers with whatever the test set."""

    def __init__(self, answer):
        self._answer = answer
        self.messages = []

    async def invoke_async(self, text):
        return self._answer()


class ThroughTheToolTest(unittest.TestCase):
    """The tool: whatever the scout staged becomes one of this turn's outputs."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.img = Path(self._dir.name) / "diner.jpg"
        self.img.write_bytes(b"\x89PNG\r\n\x1a\n")
        self._answer = "{}"
        self.pipe = pipeline_stub(
            # `_run_specialist` is a closure inside the tool factory, so the seam
            # is the AGENT it invokes — which is the more honest stub anyway: the
            # real path around it (usage accounting, message reset, logging) still
            # runs, so a change there is not silently skipped here.
            _search_web_agent=_FakeAgent(lambda: self._answer),
            _usage_snapshot=lambda agent: {},
            _record_agent_usage=lambda agent, snap: None,
        )

    def tearDown(self):
        self._dir.cleanup()

    def _run(self, manifest):
        self._answer = manifest
        return asyncio.run(tools(self.pipe)["run_web_search"]("a diner"))

    def test_a_staged_reference_becomes_a_turn_output(self):
        out = self._run(json.dumps(
            {"references": [{"mode": "image", "path": str(self.img),
                             "description": "a diner"}]}))
        self.assertIn(str(self.img), self.pipe._session.current_output_paths,
                      "the reference never reached the canvas")
        self.assertIn(str(self.img), self.pipe._chain_output_paths)
        self.assertIn("diner", out, "the manifest must still reach the agent")

    def test_a_text_only_answer_publishes_nothing(self):
        self._run(json.dumps(
            {"references": [{"mode": "text", "description": "1950s americana"}]}))
        self.assertEqual(self.pipe._session.current_output_paths, [])

    def test_the_manifest_is_returned_unchanged(self):
        """The agent still needs the descriptions to pass to a generator."""
        manifest = json.dumps(
            {"references": [{"mode": "image", "path": str(self.img),
                             "description": "chrome and neon"}]})
        self.assertEqual(self._run(manifest), manifest)

    def test_an_unreadable_answer_does_not_break_the_search(self):
        out = self._run("I could not find anything usable.")
        self.assertEqual(self.pipe._session.current_output_paths, [])
        self.assertIn("could not find", out)


class ToolContractTest(unittest.TestCase):

    def test_the_docstring_says_the_images_are_already_shown(self):
        """Or the agent politely re-uploads them, and they land twice."""
        doc = tools(pipeline_stub())["run_web_search"].__doc__ or ""
        self.assertIn("canvas", doc)
        self.assertIn("Do NOT", doc)


if __name__ == "__main__":
    unittest.main()
