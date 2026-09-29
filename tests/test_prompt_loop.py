"""The panel's prompt loop: the agent writes, the user queues.

The loop people actually run — ask for a prompt, queue it yourself, look, ask for
a change — had no state anywhere, so every round re-established which node held the
prompt, what the last one said, and which render came out of it. Now it is per
conversation and on disk, and these are the parts that would break quietly:

* the numbering and the lineage of a go-back, because "back to v2" has to mean
  something three rounds later;
* pairing a render with the version that produced it — the user queues the graph
  themselves, so ComfyUI's history is the only trace of the run;
* the loop writing a prompt and NOT running anything.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import prompt_loop as pl


class Store(dict):
    """The per-thread store, without a database."""

    def get_prompt_loop(self, thread_id):
        return self.get(thread_id)

    def set_prompt_loop(self, thread_id, loop):
        if loop is None:
            self.pop(thread_id, None)
        else:
            self[thread_id] = loop


class Fixture(unittest.TestCase):

    def setUp(self):
        self.store = Store()
        p = mock.patch.object(pl, "_store", return_value=self.store)
        p.start()
        self.addCleanup(p.stop)


class Versions(Fixture):

    def test_a_loop_starts_off_and_empty(self):
        self.assertIsNone(pl.state("t1"))
        self.assertIsNone(pl.active("t1"))
        self.assertEqual(pl.versions("t1"), [])

    def test_versions_are_numbered_from_one(self):
        pl.start("t1", "6")
        self.assertEqual(pl.add_version("t1", "a neon alley")["v"], 1)
        self.assertEqual(pl.add_version("t1", "a neon alley, wet")["v"], 2)
        self.assertEqual([e["v"] for e in pl.versions("t1")], [1, 2])
        self.assertEqual(pl.current("t1")["text"], "a neon alley, wet")

    def test_a_go_back_records_where_it_came_from(self):
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.add_version("t1", "two")
        entry = pl.add_version("t1", "one, warmer", based_on=1)
        self.assertEqual(entry["v"], 3)
        self.assertEqual(entry["from"], 1)
        self.assertEqual(pl.version_text("t1", "1"), "one")
        self.assertEqual(pl.version_text("t1", "v2"), "two")
        self.assertIsNone(pl.version_text("t1", "9"))

    def test_switching_off_keeps_the_history(self):
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.stop("t1")
        self.assertIsNone(pl.active("t1"), "off means off for the turn block")
        self.assertEqual(len(pl.versions("t1")), 1)
        pl.start("t1")
        self.assertIsNotNone(pl.active("t1"))
        self.assertEqual(pl.add_version("t1", "two")["v"], 2, "numbering carries on")

    def test_clearing_is_the_only_thing_that_forgets(self):
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.clear("t1")
        self.assertEqual(pl.versions("t1"), [])

    def test_two_conversations_are_two_loops(self):
        pl.start("t1", "6")
        pl.add_version("t1", "alley")
        pl.start("t2", "11")
        pl.add_version("t2", "portrait")
        self.assertEqual(pl.target("t1"), ("6", ""))
        self.assertEqual(pl.target("t2"), ("11", ""))
        self.assertEqual(len(pl.versions("t1")), 1)
        self.assertEqual(pl.current("t2")["text"], "portrait")

    def test_a_long_loop_does_not_grow_without_bound(self):
        pl.start("t1", "6")
        for i in range(pl._MAX_VERSIONS + 5):
            pl.add_version("t1", f"take {i}")
        self.assertEqual(len(pl.versions("t1")), pl._MAX_VERSIONS)
        self.assertEqual(pl.current("t1")["v"], pl._MAX_VERSIONS + 5,
                         "numbers keep counting even when old entries drop")


class PairingARender(Fixture):

    def test_the_render_lands_on_the_version_that_was_live(self):
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.pair_output("t1", "W:/out/a_00001_.png")
        self.assertEqual(pl.current("t1")["output"], "W:/out/a_00001_.png")

    def test_a_second_render_does_not_overwrite_the_first(self):
        """The render a version is judged on is the first one it got."""
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.pair_output("t1", "W:/out/a_00001_.png")
        pl.pair_output("t1", "W:/out/a_00002_.png")
        self.assertEqual(pl.current("t1")["output"], "W:/out/a_00001_.png")

    def test_a_new_version_starts_unrendered(self):
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.pair_output("t1", "W:/out/a_00001_.png")
        pl.add_version("t1", "two")
        self.assertEqual(pl.current("t1")["output"], "")

    def test_nothing_to_pair_to_is_not_an_error(self):
        self.assertIsNone(pl.pair_output("t1", "W:/out/a.png"))
        pl.start("t1", "6")
        self.assertIsNone(pl.pair_output("t1", ""))


class TheNewestRender(Fixture):
    """ComfyUI's history is the only trace of a run the user queued themselves."""

    def _history(self, payload):
        client = mock.Mock(get=mock.Mock(return_value=payload))
        return mock.patch("agenty_core.utils.comfyui_client.get_client",
                          return_value=client)

    def _entry(self, filename, when_ms):
        return {"status": {"messages": [["execution_success", {"timestamp": when_ms}]]},
                "outputs": {"9": {"images": [{"filename": filename, "subfolder": "",
                                              "type": "output"}]}}}

    def test_the_most_recent_one_wins(self):
        with self._history({"p1": self._entry("a.png", 1000),
                            "p2": self._entry("b.png", 3000)}), \
             mock.patch.object(pl, "_resolve", side_effect=lambda r: "W:/o/" + r["filename"]):
            self.assertEqual(pl.newest_output(), "W:/o/b.png")

    def test_a_render_older_than_the_version_is_not_its_render(self):
        with self._history({"p1": self._entry("old.png", 1000)}), \
             mock.patch.object(pl, "_resolve", side_effect=lambda r: "W:/o/" + r["filename"]):
            self.assertEqual(pl.newest_output(since=2.0), "")

    def test_a_temp_preview_is_not_a_render(self):
        entry = self._entry("t.png", 3000)
        entry["outputs"]["9"]["images"][0]["type"] = "temp"
        with self._history({"p1": entry}), \
             mock.patch.object(pl, "_resolve", side_effect=lambda r: "W:/o/" + r["filename"]):
            self.assertEqual(pl.newest_output(), "")

    def test_no_comfyui_is_simply_no_render(self):
        client = mock.Mock(get=mock.Mock(side_effect=OSError("connection refused")))
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
            self.assertEqual(pl.newest_output(), "")


class TheTurnBlock(Fixture):

    def test_a_loop_that_is_off_says_nothing(self):
        pl.start("t1", "6")
        pl.add_version("t1", "one")
        pl.stop("t1")
        self.assertEqual(pl.block("t1"), "")
        self.assertEqual(pl.block(""), "")

    def test_it_names_the_target_the_versions_and_the_render(self):
        pl.start("t1", "6", "text")
        pl.add_version("t1", "a neon alley")
        pl.pair_output("t1", "W:/out/a_00001_.png")
        pl.add_version("t1", "a neon alley, brighter", based_on=1)
        block = pl.block("t1")
        self.assertIn("node 6", block)
        self.assertIn("`text`", block)
        self.assertIn("v1: a neon alley", block)
        self.assertIn("W:/out/a_00001_.png", block)
        self.assertIn("(from v1)", block)
        self.assertIn("v2 has no render yet", block)

    def test_a_loop_with_no_target_says_what_to_do_about_it(self):
        pl.start("t1")
        self.assertIn("NOT SET", pl.block("t1"))
        self.assertIn("node_id", pl.block("t1"))

    def test_an_old_version_is_summarised_not_dumped(self):
        pl.start("t1", "6")
        pl.add_version("t1", "x" * 2000)
        pl.add_version("t1", "the current one")
        block = pl.block("t1")
        self.assertLess(len(block), 1200, "an old prompt is truncated in the block")
        self.assertIn("the current one", block)

    def test_only_the_last_few_versions_are_shown(self):
        pl.start("t1", "6")
        for i in range(pl._BLOCK_VERSIONS + 4):
            pl.add_version("t1", f"take {i}")
        block = pl.block("t1")
        self.assertIn("earlier version(s) not shown", block)
        self.assertIn(f"take {pl._BLOCK_VERSIONS + 3}", block)
        self.assertNotIn("take 0", block)


class TheTool(Fixture):
    """revise_prompt — the write, the version, and what it must never do."""

    def _pipe(self, **over):
        graph = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
                 "7": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        base = dict(_canvas_graph=graph, _canvas_base_prompt=graph,
                    _session=mock.Mock(session_id="t1", current_output_paths=[]),
                    _canvas_selection=[{"id": "6", "type": "CLIPTextEncode",
                                        "title": "positive", "widgets": {"text": "old"}}])
        base.update(over)
        return pipeline_stub(**base)

    def _call(self, pipe, **kw):
        import asyncio
        return json.loads(asyncio.run(tools(pipe)["revise_prompt"](**kw)))

    def setUp(self):
        super().setUp()
        from src.utils import canvas_patch
        canvas_patch.clear()
        self.addCleanup(canvas_patch.clear)
        self.patches = canvas_patch

    def test_it_writes_the_widget_and_records_a_version(self):
        out = self._call(self._pipe(), text="a neon alley", node_id="6")
        self.assertEqual(out["status"], "written")
        self.assertEqual(out["version"], 1)
        self.assertEqual(out["input"], "text")
        events = self.patches.drain()
        write = next(e for e in events if e.get("node_id") == "6" and "params" in e)
        self.assertEqual(write["params"], {"text": "a neon alley"})
        strip = next(e for e in events if e.get("op") == "prompt_version")
        self.assertEqual((strip["v"], strip["text"]), (1, "a neon alley"))

    def test_the_node_is_remembered_for_the_next_turn(self):
        pipe = self._pipe()
        self._call(pipe, text="one", node_id="6")
        out = self._call(pipe, text="two")          # no node_id this time
        self.assertEqual(out["version"], 2)
        self.assertEqual(out["node_id"], "6")

    def test_with_no_target_it_says_what_it_needs(self):
        out = self._call(self._pipe(), text="one")
        self.assertIn("no target node", out["error"])
        self.assertIn("CANVAS GRAPH", out["fix"])
        self.assertEqual(pl.versions("t1"), [], "nothing recorded for a refused write")

    def test_from_version_alone_puts_that_prompt_back(self):
        pipe = self._pipe()
        self._call(pipe, text="one", node_id="6")
        self._call(pipe, text="two")
        out = self._call(pipe, from_version="1")
        self.assertEqual(out["version"], 3)
        self.assertEqual(out["based_on"], 1)
        write = [e for e in self.patches.drain() if e.get("params")][-1]
        self.assertEqual(write["params"], {"text": "one"})

    def test_an_unknown_version_is_refused_with_the_ones_that_exist(self):
        pipe = self._pipe()
        self._call(pipe, text="one", node_id="6")
        out = self._call(pipe, from_version="7")
        self.assertIn("no version '7'", out["error"])
        self.assertIn("v1", out["error"])

    def test_an_empty_prompt_is_refused(self):
        out = self._call(self._pipe(), text="   ", node_id="6")
        self.assertIn("empty", out["error"])

    def test_a_node_that_is_not_there_does_not_become_a_version(self):
        out = self._call(self._pipe(), text="one", node_id="404")
        self.assertIn("error", out)
        self.assertEqual(pl.versions("t1"), [])

    def test_it_runs_nothing(self):
        """The whole point: they queue it. A tool that generates here spends their
        GPU on a prompt they have not read."""
        import inspect
        from src.pipeline import Pipeline
        src = inspect.getsource(Pipeline._build_delegation_tools)
        start = src.index("async def revise_prompt(")
        body = src[start:src.index("async def run_python_node(", start)]
        for forbidden in ("execute_workflow", "submit_prompt", "signal_workflow_ready",
                          "_run_canvas_batch", "run_now=True"):
            self.assertNotIn(forbidden, body, forbidden)


class ThePromptSlot(unittest.TestCase):
    """Which widget holds the prompt — asked, not assumed."""

    def _slot(self, widgets, node_id="6", cls="CLIPTextEncode"):
        pipe = pipeline_stub(_canvas_selection=[{"id": node_id, "type": cls,
                                                 "title": "p", "widgets": widgets}])
        return pipe._prompt_slot_of(node_id)

    def test_text_is_the_usual_one(self):
        self.assertEqual(self._slot({"text": "x", "speak": "y"}), "text")

    def test_an_api_node_calls_it_prompt(self):
        """Assuming `text` on a partner node is a mistake this codebase has made."""
        self.assertEqual(self._slot({"prompt": "x", "model": "v1"}), "prompt")

    def test_nothing_recognisable_falls_back_to_the_longest_string(self):
        self.assertEqual(self._slot({"seed": 1, "caption_here": "a long sentence",
                                     "mode": "x"}), "caption_here")

    def test_an_unreadable_node_still_answers(self):
        pipe = pipeline_stub(_canvas_selection=[], _canvas_graph={})
        self.assertEqual(pipe._prompt_slot_of("6"), "text")


class TheRetiredPurpose(unittest.TestCase):
    """`iterate` is gone; a canvas saved with one must be told, not ignored."""

    def test_a_saved_iterate_hook_is_recognised_as_retired(self):
        from src.utils.canvas_hooks import _is_retired
        self.assertTrue(_is_retired({"purpose": "iterate"}))
        self.assertTrue(_is_retired({"purpose": "Iterative_Refine"}))
        self.assertFalse(_is_retired({"purpose": "general_request"}))

    def test_the_hook_block_says_where_the_feature_went(self):
        from src.utils.canvas_hooks import describe_hooks
        block = describe_hooks([{"hook_node_id": "9", "purpose": "iterate",
                                 "directive": "refine it", "anchors": [], "targets": []}],
                               {"1": {"class_type": "KSampler", "inputs": {}}})
        self.assertIn("RETIRED", block)
        self.assertIn("PROMPT LOOP", block)
        self.assertIn("revise_prompt", block)
        self.assertNotIn("iterate_step", block)

    def test_the_tool_is_gone_from_the_toolset(self):
        names = set(tools(pipeline_stub()))
        self.assertNotIn("iterate_step", names)
        self.assertIn("revise_prompt", names)


if __name__ == "__main__":
    unittest.main()
