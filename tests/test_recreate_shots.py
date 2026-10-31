"""Recreate selected shots with changes - agentY's side.

The tools are agentY-core's (agenty_core.tools.regenerate); agentY adds:

* provenance on every output a run makes (src/utils/provenance_capture.py, from the run's
  own /history entry, on a background thread), kept by the output-tag sidecar writer and
  copied with a staged copy;
* ``regenerate_from_source`` run through ``run_workflow_now`` (src/tools/recreate_host.py),
  the conversation's output lists pointed at each shot's new versioned file;
* the "recreate" tool pack and the recreate_shots guide, only when clips are selected in
  the viewer's Edit panel or the message asks to recreate shots;
* takes kept by the Edit stage's timeline (src/utils/edit_timeline.py).

THE TOKEN RULE, enforced below: the graph never enters a prompt or a tool result; the
``[VIEWER]`` selection line is at most 40 tokens and is never kept in the history; a tool
result is at most 400 tokens per shot; the tool pack loads only on such a turn.
"""
from __future__ import annotations

import asyncio
import copy
import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agenty_core.tools import regenerate as rg
from agenty_core.utils import provenance as pv
from agenty_core.utils import turn_scope

SENTINEL = "UNIQUE_GRAPH_SENTINEL_d41d8cd9"
LONG = ("a lighthouse on a cliff at dusk, waves breaking on black rocks, volumetric fog, "
        "cinematic wide shot, 35mm film grain, " + SENTINEL + ", ") * 6


def graph(seed=912334, text=LONG):
    return {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 28, "cfg": 6.5,
                                                   "sampler_name": "dpmpp_2m", "scheduler": "karras",
                                                   "denoise": 1.0, "model": ["10", 0], "positive": ["6", 0],
                                                   "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl_base_1.0.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1344, "height": 768, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["10", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, lowres, watermark", "clip": ["10", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "shot"}},
        "10": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["4", 1],
                                                      "lora_name": "film_look_v3.safetensors",
                                                      "strength_model": 0.8, "strength_clip": 0.8}},
    }


def tok(text: str) -> int:
    return len(text) // 4


def history_entry(prompt_id, g, files, node="9"):
    return {prompt_id: {"prompt": [1, prompt_id, g, {"extra_pnginfo": {"workflow": {}},
                                                     "api_key_comfy_org": "comfyui-SECRETKEY"}, ["9"]],
                        "outputs": {node: {"images": [{"filename": Path(f).name, "subfolder": "",
                                                        "type": "output"} for f in files]}}}}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        net = mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError("no net"))
        net.start()
        self.addCleanup(net.stop)

    def shot(self, name="shot_03.png", g=None, prompt_id="pid0"):
        f = self.tmp / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\x89PNG old")
        pv.write(f, g or graph(), prompt_id=prompt_id, output_node="9", output_index=0)
        return str(f)


# ── provenance on every output ───────────────────────────────────────────────

class Capture(Tmp):
    def test_outputs_get_the_exact_graph_from_history_never_extra_data(self):
        from src.utils import provenance_capture as pc
        a, b = self.tmp / "out_00001_.png", self.tmp / "out_00002_.png"
        for f in (a, b):
            f.write_bytes(b"x")
        t = pc.capture([a, b], history_entry("p1", graph(), [a, b]), "p1", "D:/wf/w.json")
        self.assertIsNotNone(t)
        pc.wait()
        got = pv.provenance_of(b)
        self.assertEqual((got["source"], got["confidence"], got["prompt_id"]), ("sidecar", "exact", "p1"))
        self.assertEqual((got["output_node"], got["output_index"]), ("9", 1))
        self.assertEqual(got["graph"]["3"]["inputs"]["seed"], 912334)
        for f in (a, b):
            blob = gzip.decompress(Path(str(f) + ".prov.json.gz").read_bytes()).decode("utf-8")
            self.assertNotIn("SECRETKEY", blob)
            self.assertNotIn("SECRETKEY", Path(str(f) + ".agenty.json").read_text(encoding="utf-8"))

    def test_the_run_is_not_held_up_and_nothing_breaks_without_a_graph(self):
        from src.utils import provenance_capture as pc
        f = self.tmp / "x.png"
        f.write_bytes(b"x")
        self.assertIsNone(pc.capture([f], {}, "p"))
        self.assertIsNone(pc.capture([f], {"p": {"prompt": "junk"}}, "p"))
        self.assertEqual(pc.capture([f], history_entry("p", graph(), [f]), "p", background=False), 1)
        gate = __import__("threading").Event()
        real = pv.write
        with mock.patch.object(pv, "write", side_effect=lambda *a, **k: (gate.wait(5), real(*a, **k))[1]):
            t = pc.capture([f], history_entry("p", graph(), [f]), "p")
            self.assertTrue(t.is_alive())               # capture() returned while the write waits
            gate.set()
            pc.wait()

    def test_executor_captures_after_resolving_outputs(self):
        import inspect
        from src import executor
        src = inspect.getsource(executor._process_completed_job)
        self.assertIn("provenance_capture.capture(saved_paths, history, prompt_id, workflow_path)", src)
        self.assertLess(src.index("provenance_capture.capture"), src.index("Output QA"))

    def test_the_role_record_keeps_the_provenance(self):
        from src.utils import output_tags
        f = self.shot("a.png")
        self.assertTrue(output_tags.write_sidecar(f, "hero shot", hook="12"))
        doc = json.loads(Path(f + ".agenty.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["role"], "hero shot")
        self.assertEqual(doc["provenance"]["prompt_id"], "pid0")
        self.assertEqual(pv.provenance_of(f)["source"], "sidecar")
        # a later role refresh keeps it too, and a meta key cannot overwrite it
        output_tags.write_sidecar(f, "other", provenance={"bogus": 1})
        doc = json.loads(Path(f + ".agenty.json").read_text(encoding="utf-8"))
        self.assertEqual((doc["role"], doc["provenance"]["prompt_id"]), ("other", "pid0"))

    def test_a_staged_copy_keeps_the_provenance_and_the_graph(self):
        from src.utils import agentY_server as srv
        from src.utils import output_tags
        f = self.shot("render.png")
        output_tags.write_sidecar(f, "the hero")
        in_dir = self.tmp / "input"
        (in_dir / "agentY").mkdir(parents=True)
        staged = in_dir / "agentY" / "render_1.png"
        shutil.copyfile(f, staged)
        with mock.patch.object(srv, "_comfy_input_dir", return_value=in_dir):
            srv._copy_sidecar(f, "agentY/render_1.png")
        doc = json.loads(Path(str(staged) + ".agenty.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["role"], "the hero")
        self.assertEqual(doc["provenance"]["graph_file"], "render_1.png.prov.json.gz")
        got = pv.provenance_of(staged)
        self.assertEqual((got["source"], got["graph"]["3"]["inputs"]["seed"]), ("sidecar", 912334))

    def test_a_staged_copy_without_a_role_still_keeps_the_provenance(self):
        from src.utils import agentY_server as srv
        f = self.shot("plain.png")
        in_dir = self.tmp / "input"
        in_dir.mkdir()
        shutil.copyfile(f, in_dir / "plain.png")
        with mock.patch.object(srv, "_comfy_input_dir", return_value=in_dir):
            srv._copy_sidecar(f, "plain.png")
        self.assertEqual(pv.provenance_of(in_dir / "plain.png")["source"], "sidecar")


# ── the tools, the pack, the guide ───────────────────────────────────────────

class Wiring(unittest.TestCase):
    def test_shared_core_names_and_the_orchestrator_list(self):
        from src import tools
        from src.tools import tool_packs as tp
        for name in ("recreatable_params", "regenerate_from_source", "swap_in_takes"):
            self.assertIn(name, tools._SHARED_CORE_TOOLS)
            self.assertEqual(tp._PACK_OF[name], "recreate")
        names = {tp.name_of(t) for t in tools.ORCHESTRATOR_TOOLS}
        self.assertTrue({"recreatable_params", "swap_in_takes"} <= names)
        # regenerate_from_source is the pipeline's closure (it runs through run_workflow_now)
        self.assertNotIn("regenerate_from_source", names)

    def test_the_pack_is_held_out_of_an_ordinary_turn(self):
        from src import tools
        from src.tools import tool_packs as tp
        always, held = tp.split(list(tools.ORCHESTRATOR_TOOLS))
        self.assertIn("recreate", held)
        self.assertFalse({"recreatable_params", "swap_in_takes"} & {tp.name_of(t) for t in always})

    def test_recreate_wanted(self):
        from src.tools.tool_packs import recreate_wanted
        sel = {"open": True, "edit": {"selected": [{"name": "shot_03", "file": "D:/o/shot_03.mp4"}]}}
        self.assertTrue(recreate_wanted("make it warmer", sel))
        for text in ("recreate shot 3 with warmer light", "Regenerate the selected clips",
                     "redo these two shots with a new seed", "give me another take of shot 5",
                     "warmer\n\n(Recreate with changes, from the bEpic viewer's Edit panel - ...) Clips: a = b"):
            self.assertTrue(recreate_wanted(text, None), text)
        for text, state in (("a cat on a sofa", None), ("recreate the look of this painting", None),
                            ("make it warmer", {"open": True, "edit": {"selected": []}}),
                            ("make it warmer", {"open": True, "edit": {"selected": {"kind": "text"}}}),
                            ("make it warmer", {"open": False, "edit": {"selected": [{"name": "x"}]}})):
            self.assertFalse(recreate_wanted(text, state), text)

    def test_the_guide_is_a_partial_named_in_the_turn_guides(self):
        from src import pipeline as pl
        self.assertIn("recreate_shots", pl._TURN_GUIDES)
        text = pl._orch_partial("recreate_shots")
        for step in ("recreatable_params", "dry_run=true", "swap_in_takes", "confirm_cost", "Announce the plan"):
            self.assertIn(step, text)
        self.assertLess(tok(text), 700)
        import inspect
        src = inspect.getsource(pl.Pipeline)
        self.assertNotIn("Recreating shots with changes", src)       # never hardcoded


class Pins(unittest.TestCase):
    def pins(self, text="", viewer=None):
        from tests.pipeline_stub import pipeline_stub
        p = pipeline_stub()
        p._orchestrator_agent = SimpleNamespace(messages=[])
        p._history_selection = lambda: {}
        tok_ = turn_scope.enter(turn_scope.Scope("r", "t"))
        try:
            if viewer is not None:
                turn_scope.current().set("viewer_state", viewer)
            p._pin_tool_packs(False, text)
        finally:
            turn_scope.leave(tok_)
        return p._orchestrator_agent._tool_packs_pinned

    def test_pinned_with_a_selection_or_when_asked_and_not_otherwise(self):
        sel = {"open": True, "edit": {"open": True, "selected": [{"name": "a", "file": "D:/a.mp4"}]}}
        self.assertTrue({"recreate", "viewer"} <= self.pins("warmer please", sel))
        self.assertIn("recreate", self.pins("regenerate shot 4 with more fog"))
        self.assertNotIn("recreate", self.pins("a portrait of a fox"))
        self.assertNotIn("recreate", self.pins("a portrait of a fox", {"open": True, "edit": {"selected": []}}))


# ── the [VIEWER] selection line ──────────────────────────────────────────────

class SelectionLine(unittest.TestCase):
    def clips(self, n):
        return [{"name": f"forest_walk_establishing_shot_{i:02d}_final", "file": f"D:/out/shot_{i:02d}.mp4",
                 "in": 0, "out": 48, "track": "V1", "id": f"V1:{i}"} for i in range(n)]

    def state(self, n):
        from agenty_core.tools.tests.test_viewer_state import QUIET
        return dict(copy.deepcopy(QUIET), edit={"open": True, "cut": "trailer", "clips": 12,
                                                "selected": self.clips(n), "otio": "D:/out/trailer.otio"})

    def test_at_most_forty_tokens_however_many_are_selected(self):
        from agenty_core.tools.viewer_state import describe_lines
        for n in (1, 2, 5, 50):
            (line,) = [ln for ln in describe_lines(self.state(n)).splitlines() if "clips selected" in ln
                       or "clip selected" in ln]
            self.assertLessEqual(tok(line), 40, line)
            self.assertNotIn("D:/out", line)           # names, never the paths

    def test_never_kept_in_the_history(self):
        from tests.test_viewer_state_context import count, stub, turn
        p = stub()
        turn(p, "recreate these with warmer light", viewer=self.state(3))
        self.assertEqual(count(p, "clips selected"), 1)
        turn(p, "and now 5 of them", viewer=self.state(5))
        self.assertEqual(count(p, "3 clips selected"), 0)
        self.assertEqual(count(p, "clips selected"), 1)
        turn(p, "thanks", viewer={"open": False})
        p._ensure_orch_clean_history()
        self.assertEqual(count(p, "clips selected"), 0)

    def test_the_turn_gets_the_guide_in_the_prompt_not_in_the_message(self):
        from src import pipeline as pl
        from tests.test_viewer_state_context import stub, turn
        p = stub()
        seen = []
        p._set_turn_guides = lambda names: (seen.append(list(names)), True)[1]
        got = turn(p, "make them warmer", viewer=self.state(2))
        self.assertIn("recreate_shots", seen[-1])
        self.assertNotIn(pl._orch_partial("recreate_shots")[:60], got)
        turn(p, "a portrait of a fox")
        self.assertNotIn("recreate_shots", seen[-1])


# ── regenerate_from_source through run_workflow_now ─────────────────────────

class Host(Tmp):
    def make(self, pipeline, calls, *, produce=True, answer=None):
        from src.utils import provenance_capture as pc
        out_dir = self.tmp / "comfy_out"
        out_dir.mkdir(exist_ok=True)

        async def run_workflow_now(workflow_path, confirm_cost=False, rerun_same=False):
            calls.append((workflow_path, confirm_cost))
            if answer is not None:
                return json.dumps(answer)
            g = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
            f = out_dir / f"shot_{len(calls):05d}_.png"
            f.write_bytes(b"new")
            pipeline._session.current_output_paths.append(str(f))
            pipeline._chain_output_paths.append(str(f))
            pc.capture([f], history_entry(f"newpid{len(calls)}", g, [f]), f"newpid{len(calls)}", workflow_path)
            return json.dumps({"status": "done", "outputs": [str(f)] if produce else []})

        from strands import tool
        from src.tools.recreate_host import make_tool
        return make_tool(pipeline, run_workflow_now, tool)

    def call(self, t, **kw):
        with mock.patch.object(rg, "_validation_error", return_value=""):
            return json.loads(asyncio.run(t(**kw)))

    def pipe(self):
        return SimpleNamespace(_session=SimpleNamespace(current_output_paths=[]), _chain_output_paths=[])

    def test_the_tool_has_cores_name_and_description(self):
        t = self.make(self.pipe(), [])
        self.assertEqual(t.tool_name, "regenerate_from_source")
        self.assertIn("EXACTLY", t.tool_spec["description"])

    def test_dry_run_runs_nothing(self):
        calls = []
        t = self.make(self.pipe(), calls)
        a = self.shot()
        out = self.call(t, paths=[a], changes=[{"param": "positive", "append": "warm light"}], dry_run=True)
        self.assertEqual((out["status"], out["dry_run"], calls), ("ok", True, []))
        self.assertIn("warm light", out["shots"][0]["diff"][0])

    def test_a_shot_runs_through_run_workflow_now_and_the_lists_follow_the_new_version(self):
        calls, p = [], self.pipe()
        t = self.make(p, calls)
        a, b = self.shot("cut/shot_03.png"), self.shot("cut/shot_05.png")
        out = self.call(t, paths=[a, b], changes=[{"param": "3.seed", "value": 7}], dry_run=False,
                        confirm_cost=True)
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual([c[1] for c in calls], [True, True])
        news = [r["new_file"] for r in out["results"]]
        self.assertEqual([Path(n).name for n in news], ["shot_03_v002.png", "shot_05_v002.png"])
        self.assertEqual(p._session.current_output_paths, news)          # not the moved-away paths
        self.assertEqual(p._chain_output_paths, news)
        self.assertEqual([r["prompt_id"] for r in out["results"]], ["newpid1", "newpid2"])
        got = pv.provenance_of(news[0])
        self.assertEqual((got["source"], got["graph"]["3"]["inputs"]["seed"]), ("sidecar", 7))
        doc = json.loads(Path(news[0] + ".agenty.json").read_text(encoding="utf-8"))["provenance"]
        self.assertEqual((doc["source"], doc["derived_from"]["file"]), ("regenerate", a))
        self.assertEqual(out["takes"][0], {"from_file": a, "file": news[0], "version": 2})
        self.assertIsNone(rg._runner)                         # the host's runner is not left behind

    def test_refusals_from_run_workflow_now_reach_the_shot(self):
        a = self.shot()
        for answer, want in (({"status": "error", "error_kind": "confirm_cost", "error": "costs $2.40"}, "confirm_cost"),
                             ({"status": "cached", "outputs": ["x"]}, "cache"),
                             ({"status": "dry_run", "message": "dry run"}, "dry-run"),
                             ({"error": "the plan is not approved yet"}, "not approved")):
            out = self.call(self.make(self.pipe(), [], answer=answer), paths=[a],
                            changes=[{"param": "steps", "value": 30}], dry_run=False)
            r = out["results"][0]
            self.assertEqual(r["status"], "error", answer)
            self.assertIn(want, json.dumps(r), answer)
            self.assertFalse(Path(self.tmp / "shot_03_v002.png").exists())


# ── the token rule ───────────────────────────────────────────────────────────

class TokenRule(Tmp):
    def test_no_graph_and_at_most_400_tokens_per_shot(self):
        heavy = graph()
        for i in range(30):          # a big graph: many loras, long texts
            heavy[str(100 + i)] = {"class_type": "LoraLoader", "inputs": {
                "model": ["10", 0], "clip": ["10", 1], "lora_name": f"style_{i:02d}_{'x' * 40}.safetensors",
                "strength_model": 0.5, "strength_clip": 0.5}}
        heavy["7"]["inputs"]["text"] = LONG
        paths = [self.shot(f"s/shot_{i:02d}.png", heavy) for i in range(4)]
        for p in paths:
            raw = rg.recreatable_params(p)
            self.assertLessEqual(tok(raw), 400)
            self.assertNotIn(SENTINEL, raw)
            self.assertNotIn('"class_type"', raw)
        raw = rg.regenerate_from_source(paths, [{"param": "positive", "replace": ["lighthouse", "windmill"]},
                                                {"param": "negative", "append": "oversaturated"}],
                                        new_seed=True)
        out = json.loads(raw)
        self.assertEqual(len(out["shots"]), 4)
        for s in out["shots"]:
            self.assertLessEqual(tok(json.dumps(s, ensure_ascii=False)), 400)
        self.assertNotIn(SENTINEL, raw)
        self.assertNotIn('"class_type"', raw)
        print(f"\n  token sizes: recreatable_params={tok(rg.recreatable_params(paths[0]))}, "
              f"dry-run shot={max(tok(json.dumps(s, ensure_ascii=False)) for s in out['shots'])}")

    def test_the_pack_definitions_cost_nothing_on_an_ordinary_turn(self):
        from src import tools
        from src.tools import tool_packs as tp
        always, held = tp.split(list(tools.ORCHESTRATOR_TOOLS))
        spec = json.dumps([t.tool_spec for t in held["recreate"]])
        self.assertGreater(tok(spec), 100)       # what an ordinary turn does not send
        self.assertLess(tok(tp.catalogue({"recreate": held["recreate"]})), 200)


# ── takes on the Edit stage's timeline ───────────────────────────────────────

class TimelineTakes(unittest.TestCase):
    def doc(self):
        from src.utils import edit_timeline as et
        tl = et.new(24.0)
        tl["clips"] = [{"id": "c1", "src": "D:/o/shot_03_v002.mp4", "in": 0.0, "out": 2.0, "transition": 0.0,
                        "label": "shot_03"},
                       {"id": "c2", "src": "D:/o/shot_05.mp4", "in": 0.0, "out": 1.5, "transition": 0.0,
                        "label": "shot_05"}]
        doc = et.to_otio(tl)
        doc["tracks"]["children"][0]["children"][0]["metadata"]["bepic"] = {"takes": [
            {"file": "D:/o/shot_03.mp4", "version": 1, "active": False, "note": "original", "created": ""},
            {"file": "D:/o/shot_03_v002.mp4", "version": 2, "active": True, "note": "warmer", "created": "x"}]}
        return doc

    def test_from_otio_keeps_the_takes_and_plays_the_active_one(self):
        from src.utils import edit_timeline as et
        tl = et.from_otio(self.doc())
        c1, c2 = tl["clips"]
        self.assertEqual(c1["src"], "D:/o/shot_03_v002.mp4")
        self.assertEqual([(t["version"], t["active"]) for t in c1["takes"]], [(1, False), (2, True)])
        self.assertNotIn("takes", c2)
        rows = et.summary(tl)["clips"]
        self.assertEqual(rows[0]["take"], "v2 of 2")
        self.assertNotIn("take", rows[1])

    def test_to_otio_writes_them_back_and_a_replaced_clip_loses_them(self):
        from src.utils import edit_timeline as et
        tl = et.from_otio(self.doc())
        back = et.to_otio(tl)["tracks"]["children"][0]["children"][0]
        self.assertEqual(back["media_reference"]["target_url"], "file:///D:/o/shot_03_v002.mp4")
        self.assertEqual([t["file"] for t in back["metadata"]["bepic"]["takes"]],
                         ["D:/o/shot_03.mp4", "D:/o/shot_03_v002.mp4"])
        # switched back to the original (the viewer's Takes menu) - the src says which is active
        tl2 = copy.deepcopy(tl)
        tl2["clips"][0]["src"] = "D:/o/shot_03.mp4"
        takes = et.to_otio(tl2)["tracks"]["children"][0]["children"][0]["metadata"]["bepic"]["takes"]
        self.assertEqual([t["active"] for t in takes], [True, False])
        # another shot in its place: the old shot's versions do not apply
        tl = et.apply(tl, [{"op": "replace", "clip": "c1", "src": "D:/o/other.mp4", "length": 2}])
        self.assertNotIn("bepic", et.to_otio(tl)["tracks"]["children"][0]["children"][0]["metadata"])


if __name__ == "__main__":
    unittest.main()
