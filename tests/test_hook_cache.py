"""A memorizing hook answers once, and keeps answering until something changes.

The case is a hook that reads an image and writes a description, wired into a
graph the user iterates on all afternoon: the same vision call, the same answer,
twenty times, for a picture that never moved. With ``memorize`` on, the value is
stored against a fingerprint of everything feeding the hook and put straight back
into the graph next time.

Which makes the fingerprint the whole feature. It has to move when the inputs move
— a different image, a rewire, an edit three nodes upstream, a changed directive —
and it has to stay still when nothing that matters changed, including where the
value is delivered. Both halves are tested here, because a cache that invalidates
too eagerly is just a slower run, and one that invalidates too rarely is wrong.

    python -m unittest discover -s tests
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub
from src.pipeline import Pipeline
from src.utils import hook_cache as hc
from src.utils.canvas_hooks import describe_hooks


def _graph():
    """A hook reading one loaded image, writing into a prompt node."""
    return {
        "10": {"class_type": "LoadImage", "inputs": {"image": "hero.png"}},
        "11": {"class_type": "ImageScale", "inputs": {"image": ["10", 0], "width": 1024}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "21": {"class_type": "SaveImage", "inputs": {"images": ["20", 0],
                                                     "filename_prefix": "out"}},
    }


def _hook(memorize=True, directive="Describe the STYLE of the wired image."):
    """A hook as the current panel sends it: ONE keep switch, named `remember`."""
    return {
        "hook_node_id": "30", "purpose": "text", "directive": directive,
        "remember": memorize,
        "anchors": [{"node_id": "11", "to_input": "anchors.anchor0", "from_output_slot": 0}],
        "targets": [{"node_id": "20", "to_input": "text", "to_input_type": "STRING"}],
    }


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        # Never touch the real project store, and never call a live ComfyUI.
        self.enterContext(mock.patch("src.utils.hook_cache._file_stamp", return_value=""))

    def test_the_same_hook_and_graph_key_the_same(self):
        self.assertEqual(hc.fingerprint(_hook(), _graph()),
                         hc.fingerprint(_hook(), _graph()))

    def test_a_different_image_upstream_releases_it(self):
        g = _graph()
        g["10"]["inputs"]["image"] = "villain.png"
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()),
                            hc.fingerprint(_hook(), g),
                            "the answer was about the other picture")

    def test_an_edit_further_upstream_releases_it_too(self):
        g = _graph()
        g["11"]["inputs"]["width"] = 512      # two nodes back from the hook
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(_hook(), g))

    def test_rewiring_the_anchor_releases_it(self):
        h = _hook()
        h["anchors"] = [{"node_id": "10", "to_input": "anchors.anchor0",
                         "from_output_slot": 0}]
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(h, _graph()))

    def test_changing_the_prompt_releases_it(self):
        self.assertNotEqual(
            hc.fingerprint(_hook(), _graph()),
            hc.fingerprint(_hook(directive="Describe the COLOUR instead."), _graph()))

    def test_changing_a_setting_releases_it(self):
        h = _hook()
        h["purpose"] = "general_request"
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(h, _graph()))

    def test_no_spelling_of_the_keep_switch_is_in_the_key(self):
        """It has to be out, or a hindsight flip lands on a key nothing wrote to.

        `bake` and `freeze` used to be hashed as two components and were one bit
        — the frontend sent `freeze: bake` — so a single switch moved the key
        twice, and turning it on after a good run found nothing there.
        """
        base = hc.fingerprint(_hook(), _graph())
        for field in ("remember", "memorize", "bake", "freeze"):
            for value in (True, False):
                h = _hook()
                h[field] = value
                with self.subTest(field=field, value=value):
                    self.assertEqual(base, hc.fingerprint(h, _graph()))

    def test_moving_the_output_elsewhere_releases_it(self):
        h = _hook()
        h["targets"] = [{"node_id": "22", "to_input": "text"}]
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(h, _graph()))

    def test_a_downstream_change_does_NOT_release_it(self):
        """Renaming the save prefix doesn't change what the picture looks like."""
        g = _graph()
        g["21"]["inputs"]["filename_prefix"] = "something_else"
        self.assertEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(_hook(), g))

    def test_the_memorize_toggle_itself_is_not_in_the_key(self):
        """Off has to resolve to the key On wrote under, or it could never forget."""
        self.assertEqual(hc.fingerprint(_hook(memorize=True), _graph()),
                         hc.fingerprint(_hook(memorize=False), _graph()))

    def test_a_cycle_does_not_hang(self):
        g = {"1": {"class_type": "A", "inputs": {"x": ["2", 0]}},
             "2": {"class_type": "B", "inputs": {"y": ["1", 0]}}}
        h = _hook()
        h["anchors"] = [{"node_id": "1"}]
        self.assertTrue(hc.fingerprint(h, g))

    def test_a_file_that_changed_behind_its_name_releases_it(self):
        with mock.patch("src.utils.hook_cache._file_stamp",
                        side_effect=["100:1", "250:9"]):   # same name, new bytes
            before = hc.fingerprint(_hook(), _graph())
            after = hc.fingerprint(_hook(), _graph())
        self.assertNotEqual(before, after,
                            "ComfyUI's input dir is where hero.png gets overwritten")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.enterContext(mock.patch.object(hc, "cache_dir",
                                            side_effect=lambda create=False: self.tmp))

    def test_a_value_survives_the_round_trip(self):
        hc.write("abc", "a warm, low-contrast alley at night", hook="30")
        got = hc.read("abc")
        self.assertEqual(got["value"], "a warm, low-contrast alley at night")
        self.assertEqual(got["hook"], "30")
        self.assertIn("when", got)

    def test_a_miss_is_just_none(self):
        self.assertIsNone(hc.read("nothing-here"))

    def test_forgetting_removes_it(self):
        hc.write("abc", "x")
        self.assertTrue(hc.forget("abc"))
        self.assertIsNone(hc.read("abc"))
        self.assertFalse(hc.forget("abc"), "forgetting twice is not an error")

    def test_an_empty_value_is_not_stored(self):
        self.assertFalse(hc.write("abc", "   "))

    def test_no_project_store_means_no_cache_and_no_crash(self):
        with mock.patch.object(hc, "cache_dir", return_value=None):
            self.assertFalse(hc.write("abc", "x"))
            self.assertIsNone(hc.read("abc"))
            self.assertFalse(hc.forget("abc"))

    def test_the_switch_is_read_the_way_the_frontend_sends_it(self):
        for truthy in (True, "true", "True", "1", "on", "yes"):
            with self.subTest(v=truthy):
                self.assertTrue(hc.remembering({"remember": truthy}))
        for falsy in (False, "false", "0", "", None):
            with self.subTest(v=falsy):
                self.assertFalse(hc.remembering({"remember": falsy}))
        self.assertFalse(hc.remembering({}), "a hook that says nothing keeps nothing")

    def test_a_canvas_saved_before_the_merge_still_reads_correctly(self):
        """Two switches on the wire, resolved the way the node resolved them then.

        `bake` was what make_workflow looked at; `memorize` was what every other
        purpose looked at. Reading the wrong one of the two would silently turn
        memorizing on for a hook the user only asked to bake, and vice versa.
        """
        self.assertTrue(hc.remembering({"purpose": "make_workflow", "bake": True,
                                        "memorize": False}))
        self.assertFalse(hc.remembering({"purpose": "make_workflow", "bake": False,
                                         "memorize": True}))
        self.assertTrue(hc.remembering({"purpose": "text", "bake": False,
                                        "memorize": True}))
        self.assertFalse(hc.remembering({"purpose": "text", "bake": True,
                                         "memorize": False}))

    def test_the_merged_field_wins_over_the_legacy_pair(self):
        self.assertTrue(hc.remembering({"purpose": "text", "remember": True,
                                        "memorize": False}))
        self.assertFalse(hc.remembering({"purpose": "text", "remember": False,
                                         "memorize": True}))

    def test_memorizing_is_still_the_name_the_pipeline_calls_it_by(self):
        self.assertIs(hc.memorizing, hc.remembering)


class MediaTest(unittest.TestCase):
    """A hook remembers everything it produced, not only what it wrote.

    Text was the easy half. The files a hook generates are the expensive half —
    a video re-rendered because nobody wrote down that it already existed is the
    whole cost this is meant to avoid — and they are remembered by PATH, because
    the file is already sitting in the output directory and copying it would only
    create a second truth about which one is real.
    """

    def setUp(self):
        self.out = Path(tempfile.mkdtemp())          # stands in for ComfyUI's output dir
        self.store = self.out / "agent" / "memory"
        self.store.mkdir(parents=True)
        self.enterContext(mock.patch.object(hc, "output_dir", return_value=self.out))

    def _file(self, rel):
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n")
        return str(p)

    def test_the_store_sits_next_to_the_outputs_it_describes(self):
        self.assertEqual(hc.cache_dir(), self.out / "agent" / "memory")

    def test_a_produced_file_survives_the_round_trip(self):
        shot = self._file("agent/images/ref_00042_.png")
        hc.write("k", "", outputs=[shot])
        got = hc.recall("k")
        self.assertEqual([o["path"] for o in got["outputs"]], [shot])
        self.assertEqual(got["outputs"][0]["kind"], "image")

    def test_a_path_under_the_output_dir_is_stored_relative(self):
        """So moving or copying the folder does not strand every entry in it."""
        hc.write("k", "", outputs=[self._file("agent/videos/shot_01.mp4")])
        raw = json.loads((self.store / "k.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["outputs"][0]["path"], "agent/videos/shot_01.mp4")
        self.assertEqual(raw["outputs"][0]["kind"], "video")

    def test_a_path_outside_it_is_stored_absolute(self):
        outside = Path(tempfile.mkdtemp()) / "elsewhere.png"
        outside.write_bytes(b"x")
        hc.write("k", "", outputs=[str(outside)])
        raw = json.loads((self.store / "k.json").read_text(encoding="utf-8"))
        self.assertTrue(Path(raw["outputs"][0]["path"]).is_absolute(),
                        "nothing sensible to make it relative to")
        got = Path(hc.recall("k")["outputs"][0]["path"])
        self.assertEqual(got, outside.resolve())
        self.assertTrue(got.is_file())

    def test_a_relative_entry_follows_the_output_dir_when_it_moves(self):
        hc.write("k", "", outputs=[self._file("agent/images/a.png")])
        moved = Path(tempfile.mkdtemp())
        (moved / "agent" / "images").mkdir(parents=True)
        (moved / "agent" / "images" / "a.png").write_bytes(b"x")
        with mock.patch.object(hc, "output_dir", return_value=moved):
            (moved / "agent" / "memory").mkdir(parents=True)
            (moved / "agent" / "memory" / "k.json").write_text(
                (self.store / "k.json").read_text(encoding="utf-8"), encoding="utf-8")
            got = hc.recall("k")
        self.assertEqual([o["path"] for o in got["outputs"]],
                         [str(moved / "agent" / "images" / "a.png")])

    def test_a_remembered_file_that_is_gone_is_a_MISS_not_a_broken_graph(self):
        shot = self._file("agent/images/ref.png")
        hc.write("k", "some text too", outputs=[shot])
        Path(shot).unlink()                          # the user tidied the folder
        self.assertIsNone(hc.recall("k"),
                          "a re-run is right; a graph pointing at nothing is not")
        self.assertIsNotNone(hc.read("k"), "the record itself is not destroyed")

    def test_one_missing_file_invalidates_the_whole_set(self):
        """Five reference frames were produced as a SET; replaying four is wrong."""
        files = [self._file(f"agent/images/ref_{i}.png") for i in range(5)]
        hc.write("k", "", outputs=files)
        Path(files[2]).unlink()
        self.assertIsNone(hc.recall("k"))

    def test_text_and_files_are_remembered_together(self):
        hc.write("k", "a warm alley at night",
                 outputs=[self._file("agent/images/a.png")])
        got = hc.recall("k")
        self.assertEqual(got["value"], "a warm alley at night")
        self.assertEqual(len(got["outputs"]), 1)

    def test_an_entry_with_neither_is_not_worth_a_file(self):
        self.assertFalse(hc.write("k", "   ", outputs=[]))
        self.assertIsNone(hc.read("k"))

    def test_kinds_are_read_off_the_extension(self):
        for name, kind in (("a.png", "image"), ("a.mp4", "video"), ("a.wav", "audio"),
                           ("a.glb", "model"), ("a.py", "script"), ("a.txt", "file")):
            with self.subTest(name=name):
                self.assertEqual(hc.kind_of(name), kind)

    def test_duplicates_collapse(self):
        shot = self._file("agent/images/a.png")
        hc.write("k", "", outputs=[shot, shot])
        self.assertEqual(len(hc.recall("k")["outputs"]), 1)


class PruneTest(unittest.TestCase):
    """The journal is written every turn, so it needs a bound. Kept entries don't."""

    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        (self.out / "agent" / "memory").mkdir(parents=True)
        self.enterContext(mock.patch.object(hc, "output_dir", return_value=self.out))

    def _age(self, key, days):
        f = hc.cache_dir() / f"{key}.json"
        old = time.time() - days * 86400
        os.utime(f, (old, old))

    def test_a_stale_journal_entry_is_dropped(self):
        hc.write("old", "x")
        self._age("old", 30)
        self.assertEqual(hc.prune(ttl_days=14), 1)
        self.assertIsNone(hc.read("old"))

    def test_a_kept_entry_is_never_pruned_however_old(self):
        hc.write("keeper", "x", kept=True)
        self._age("keeper", 3650)
        self.assertEqual(hc.prune(ttl_days=14), 0)
        self.assertIsNotNone(hc.read("keeper"))

    def test_a_fresh_journal_entry_is_left_alone(self):
        hc.write("fresh", "x")
        self.assertEqual(hc.prune(ttl_days=14), 0)

    def test_the_journal_is_capped_oldest_first(self):
        for i in range(6):
            hc.write(f"j{i}", "x")
            self._age(f"j{i}", 6 - i)          # j0 oldest, j5 newest
        hc.prune(ttl_days=999, max_entries=3)
        self.assertEqual([hc.read(f"j{i}") is not None for i in range(6)],
                         [False, False, False, True, True, True])


class TurnTest(unittest.TestCase):
    """What the pipeline does with it at the start of a turn."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.enterContext(mock.patch.object(hc, "cache_dir",
                                            side_effect=lambda create=False: self.tmp))
        self.enterContext(mock.patch("src.utils.hook_cache._file_stamp", return_value=""))

    @staticmethod
    def _pipe(hooks, graph=None):
        return pipeline_stub(_canvas_base_prompt=graph or _graph(), _canvas_hooks=hooks)

    def _apply(self, pipe):
        Pipeline._apply_hook_cache(pipe)

    def test_a_hit_is_put_back_into_the_graph_without_asking_anyone(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        self._apply(pipe)                                    # miss: nothing stored yet
        self.assertNotIn("_cached", hooks[0])
        hc.write(hooks[0]["_cache_key"], "a warm, low-contrast alley")

        hooks2 = [_hook()]
        pipe2 = self._pipe(hooks2)
        self._apply(pipe2)
        self.assertEqual(hooks2[0]["_cached"]["value"], "a warm, low-contrast alley")
        self.assertEqual(hooks2[0]["_cached"]["targets"], ["20"])
        self.assertEqual(pipe2._canvas_base_prompt["20"]["inputs"]["text"],
                         "a warm, low-contrast alley",
                         "the value has to be IN the graph, not just reported")

    def test_a_changed_input_is_a_miss(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        hc.write(hooks[0]["_cache_key"], "the old answer")

        g = _graph()
        g["10"]["inputs"]["image"] = "villain.png"
        hooks2 = [_hook()]
        self._apply(self._pipe(hooks2, g))
        self.assertNotIn("_cached", hooks2[0])

    def test_switching_the_toggle_off_releases_what_was_kept(self):
        hooks = [_hook()]
        self._apply(self._pipe(hooks))
        key = hooks[0]["_cache_key"]
        hc.write(key, "the answer", kept=True)

        self._apply(self._pipe([_hook(memorize=False)]))
        self.assertIsNone(hc.read(key), "off is how the user forces a fresh result")

    def test_switching_it_off_does_NOT_delete_the_journal(self):
        """Deleting it here is what would force the decision to be made upfront.

        A journalled entry is the raw material of a hindsight decision: the run
        happened, nobody has said yet whether it was worth keeping. Dropping it
        because the switch is currently off means the switch could only ever be
        set BEFORE seeing the result, which is the thing this is meant to fix.
        """
        hooks = [_hook(memorize=False)]
        self._apply(self._pipe(hooks))
        key = hooks[0]["_cache_key"]
        hc.write(key, "what that run produced")          # journalled, not kept

        self._apply(self._pipe([_hook(memorize=False)]))
        self.assertEqual((hc.read(key) or {})["value"], "what that run produced")

    def test_a_switch_flipped_AFTER_the_run_finds_what_that_run_produced(self):
        """The whole point: decide in hindsight, when you have seen the result."""
        off = [_hook(memorize=False)]
        self._apply(self._pipe(off))                     # run with the switch OFF
        hc.write(off[0]["_cache_key"], "a warm, low-contrast alley")

        on = [_hook(memorize=True)]                      # user flips it on afterwards
        pipe = self._pipe(on)
        self._apply(pipe)
        self.assertEqual(on[0]["_cached"]["value"], "a warm, low-contrast alley")
        self.assertEqual(pipe._canvas_base_prompt["20"]["inputs"]["text"],
                         "a warm, low-contrast alley")
        self.assertTrue(hc.kept(hc.read(on[0]["_cache_key"])),
                        "and it is durable from here on, not journal any more")

    def test_the_value_is_stored_when_the_agent_places_it(self):
        import asyncio
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        tool = {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}
        out = json.loads(asyncio.run(tool["place_canvas_text"](
            hook_node_id="30", text="a warm, low-contrast alley")))
        # "placed" or "injected" — the text-node switch decides which, and the
        # journalling this test is about happens either way.
        self.assertNotIn("error", out)
        self.assertEqual(hc.read(hooks[0]["_cache_key"])["value"],
                         "a warm, low-contrast alley")

    def test_a_hook_that_does_not_memorize_journals_but_does_not_keep(self):
        """It is written either way — `kept` is what the switch actually decides."""
        import asyncio
        hooks = [_hook(memorize=False)]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        tool = {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}
        asyncio.run(tool["place_canvas_text"](hook_node_id="30", text="an answer"))
        entry = hc.read(hooks[0]["_cache_key"])
        self.assertEqual(entry["value"], "an answer")
        self.assertFalse(hc.kept(entry), "nobody has said this was worth keeping")

    def test_placing_a_value_on_a_remembering_hook_keeps_it_outright(self):
        import asyncio
        hooks = [_hook(memorize=True)]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        tool = {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}
        asyncio.run(tool["place_canvas_text"](hook_node_id="30", text="an answer"))
        self.assertTrue(hc.kept(hc.read(hooks[0]["_cache_key"])))


class MediaTurnTest(unittest.TestCase):
    """Replaying remembered media through the same door a fresh run delivers by.

    A remembered image has to reach the panel, the gallery, ComfyUI's input dir
    and the canvas exactly as a generated one does — otherwise a cache hit and a
    real run diverge downstream, and the divergence surfaces weeks later in
    something that looks nothing like a caching bug.
    """

    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        (self.out / "agent" / "memory").mkdir(parents=True)
        self.enterContext(mock.patch.object(hc, "output_dir", return_value=self.out))
        self.enterContext(mock.patch("src.utils.hook_cache._file_stamp", return_value=""))

    def _file(self, rel):
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n")
        return str(p)

    @staticmethod
    def _pipe(hooks):
        return pipeline_stub(_canvas_base_prompt=_graph(), _canvas_hooks=hooks)

    def test_remembered_files_are_delivered_as_this_turn_s_outputs(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        Pipeline._apply_hook_cache(pipe)
        shots = [self._file("agent/images/a.png"), self._file("agent/images/b.png")]
        hc.write(hooks[0]["_cache_key"], "", outputs=shots)

        hooks2 = [_hook()]
        pipe2 = self._pipe(hooks2)
        Pipeline._apply_hook_cache(pipe2)
        self.assertEqual(hooks2[0]["_cached"]["outputs"], shots)
        self.assertEqual(list(pipe2._session.current_output_paths), shots,
                         "the panel and the gallery read this list")
        self.assertEqual(list(pipe2._chain_output_paths), shots,
                         "and this one is what survives to reach the canvas")

    def test_a_deleted_file_makes_it_a_miss_so_the_hook_runs_again(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        Pipeline._apply_hook_cache(pipe)
        shot = self._file("agent/images/a.png")
        hc.write(hooks[0]["_cache_key"], "", outputs=[shot])
        Path(shot).unlink()

        hooks2 = [_hook()]
        pipe2 = self._pipe(hooks2)
        Pipeline._apply_hook_cache(pipe2)
        self.assertNotIn("_cached", hooks2[0], "it must be offered as work again")
        self.assertEqual(list(pipe2._session.current_output_paths), [])

    def test_the_files_a_hook_produced_are_journalled_at_the_end_of_the_turn(self):
        hooks = [_hook(memorize=False)]
        pipe = self._pipe(hooks)
        Pipeline._apply_hook_cache(pipe)
        shot = self._file("agent/images/a.png")
        pipe._session.current_output_paths = [shot]

        with mock.patch("src.utils.output_tags.meta_for", return_value={"hook": "30"}):
            Pipeline._journal_hook_outputs(pipe)
        entry = hc.read(hooks[0]["_cache_key"])
        self.assertEqual(entry["outputs"][0]["path"], "agent/images/a.png")
        self.assertFalse(hc.kept(entry), "journalled — the switch was off")

    def test_a_file_belonging_to_another_hook_is_not_journalled_here(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        Pipeline._apply_hook_cache(pipe)
        pipe._session.current_output_paths = [self._file("agent/images/a.png")]

        with mock.patch("src.utils.output_tags.meta_for", return_value={"hook": "99"}):
            Pipeline._journal_hook_outputs(pipe)
        self.assertIsNone(hc.read(hooks[0]["_cache_key"]))

    def test_a_dry_run_journals_nothing(self):
        """A stand-in is not a fact, and a remembered one is served silently later."""
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        Pipeline._apply_hook_cache(pipe)
        pipe._dry_run = True
        pipe._session.current_output_paths = [self._file("agent/images/a.png")]

        with mock.patch("src.utils.output_tags.meta_for", return_value={"hook": "30"}):
            Pipeline._journal_hook_outputs(pipe)
        self.assertIsNone(hc.read(hooks[0]["_cache_key"]))

    def test_journalling_files_keeps_the_text_the_hook_already_wrote(self):
        """A make_workflow hook writes a prompt AND produces files, at different moments."""
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        Pipeline._apply_hook_cache(pipe)
        key = hooks[0]["_cache_key"]
        hc.write(key, "a warm alley at night", kept=True)
        pipe._session.current_output_paths = [self._file("agent/images/a.png")]

        with mock.patch("src.utils.output_tags.meta_for", return_value={"hook": "30"}):
            Pipeline._journal_hook_outputs(pipe)
        entry = hc.read(key)
        self.assertEqual(entry["value"], "a warm alley at night")
        self.assertEqual(len(entry["outputs"]), 1)
        self.assertTrue(hc.kept(entry), "it was kept before; adding files doesn't undo that")


class BlockTest(unittest.TestCase):
    """How a cached hook reads to the agent — it must not be offered as work."""

    def test_a_cached_hook_is_reported_as_done_not_assigned(self):
        h = _hook()
        h["_cached"] = {"value": "a warm, low-contrast alley", "targets": ["20"],
                        "when": "2026-08-14T10:00:00"}
        block = describe_hooks([h], _graph())
        self.assertIn("ALREADY DONE", block)
        self.assertIn("a warm, low-contrast alley", block)
        self.assertIn("filled node(s) 20", block)
        self.assertNotIn("TEXT hook 30", block, "it must not also be listed as work")
        self.assertIn("apply_canvas_hooks(resolutions=[])", block,
                      "with every hook cached, running the graph is all that's left")

    def test_a_consumer_is_handed_the_remembered_value_not_a_promise(self):
        producer = _hook()
        producer["_cached"] = {"value": "warm sodium light", "targets": [], "when": ""}
        consumer = {
            "hook_node_id": "31", "purpose": "text",
            "directive": "Write a prompt using the style from the previous hook.",
            "anchors": [{"node_id": "30"}], "targets": [{"node_id": "20",
                                                         "to_input": "text"}],
        }
        block = describe_hooks([producer, consumer], _graph())
        self.assertIn('the remembered value of hook 30: "warm sodium light"', block)
        self.assertNotIn("the value you produce for hook 30", block)

    def test_an_ordinary_graph_gains_nothing(self):
        self.assertNotIn("ALREADY DONE", describe_hooks([_hook()], _graph()))


if __name__ == "__main__":
    unittest.main()
