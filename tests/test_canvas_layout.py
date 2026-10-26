"""Tidying the canvas: arrange, group, split into chunks, save.

The agent grouped the graphs it BUILT (inside the deterministic assembly) but had
no way to do it for a graph that was already open. These are the orchestrator's
tools for that, checked from the host side: what is proposed, what is pushed at
the panel, what is refused. The geometry is the panel's (connect's
test_canvas_tidy); the tools only say what to lay out.
"""

import asyncio
import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.tools import tool_packs
from src.utils import canvas_layout as cl
from src.utils.canvas_patch import clear, drain
from test_canvas_edit import SCHEMAS, _graph


EXTRA = {"VAEEncode": {"input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}},
                       "output": ["LATENT"], "output_name": ["LATENT"]}}


class Schemas:
    def get(self, cls, default=None):
        return {**SCHEMAS, **EXTRA}.get(cls, default)


def _two_stage():
    g = _graph()
    # a second generator fed by the first one's picture
    g.update({
        "20": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "b.safetensors"}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": "then", "clip": ["20", 1]}},
        "22": {"class_type": "KSampler", "inputs": {"model": ["20", 0], "positive": ["21", 0], "negative": ["21", 0],
                                                    "latent_image": ["25", 0], "seed": 2}},
        "25": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["20", 2]}},
        "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["20", 2]}},
        "24": {"class_type": "PreviewImage", "inputs": {"images": ["23", 0]}},
    })
    g["7"]["inputs"]["images"] = ["6", 0]
    return g


class Chunks(unittest.TestCase):

    def test_a_plain_graph_is_split_by_what_each_part_does(self):
        got = {c["title"]: c["node_ids"] for c in cl.propose(_graph(), Schemas())}
        self.assertEqual(got["Load models"], ["1"])
        self.assertEqual(got["Prompt"], ["2"])
        self.assertEqual(sorted(got["Sampling"]), ["4", "5"], "the empty latent goes with its sampler")
        self.assertEqual(got["Decode & save"], ["6", "7"])

    def test_chunks_come_in_the_order_the_data_flows(self):
        titles = [c["title"] for c in cl.propose(_graph(), Schemas())]
        self.assertEqual(titles, ["Load models", "Prompt", "Sampling", "Decode & save"])

    def test_a_chunk_follows_the_chunks_that_feed_it(self):
        # the role order would put Prompt before Sampling; here a sampler feeds the prompt
        g = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a"}},
            "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0], "clip": ["1", 1]}},
            "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        }
        chunks = cl.propose(g, Schemas())
        where = {n: i for i, c in enumerate(chunks) for n in c["node_ids"]}
        for k, n in g.items():
            for v in n["inputs"].values():
                if isinstance(v, list):
                    self.assertLessEqual(where[v[0]], where[k], (v[0], k))

    def test_a_cycle_between_chunks_still_returns_every_chunk(self):
        g = {
            "1": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0]}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        }
        ids = [n for c in cl.propose(g, Schemas()) for n in c["node_ids"]]
        self.assertEqual(sorted(ids), ["1", "2", "3", "4"])

    def test_every_node_is_in_exactly_one_chunk(self):
        ids = [n for c in cl.propose(_graph(), Schemas()) for n in c["node_ids"]]
        self.assertEqual(sorted(ids), sorted(_graph()))

    def test_the_same_kind_of_chunk_is_the_same_colour(self):
        for c in cl.propose(_graph(), Schemas()):
            self.assertEqual(c["color"], cl.ROLE_COLORS[c["role"]])

    def test_two_generating_stages_are_cut_per_stage(self):
        chunks = cl.propose(_two_stage(), Schemas())
        titles = [c["title"] for c in chunks]
        self.assertTrue(all(" · " in t for t in titles), titles)
        self.assertEqual(len({t.split(" · ")[0] for t in titles}), 2, "two stages, two prefixes")
        stage_of = {n: c["title"].split(" · ")[0] for c in chunks for n in c["node_ids"]}
        self.assertEqual(stage_of["1"], stage_of["5"])
        self.assertEqual(stage_of["20"], stage_of["22"])
        self.assertNotEqual(stage_of["5"], stage_of["22"])

    def test_a_graph_too_small_to_need_boxes_gets_none(self):
        small = {k: v for k, v in _graph().items() if k in ("1", "2", "4")}
        self.assertEqual(cl.propose(small, Schemas()), [])
        self.assertEqual(cl.propose({}, Schemas()), [])

    def test_hook_nodes_are_their_own_chunk(self):
        g = _graph()
        g["30"] = {"class_type": "AgentYHook", "inputs": {"directive": "x", "images": ["6", 0]}}
        got = {c["title"]: c["node_ids"] for c in cl.propose(g, Schemas())}
        self.assertEqual(got["Hooks"], ["30"])

    def test_the_levels_survive_a_cycle(self):
        g = {"1": {"class_type": "A", "inputs": {"x": ["2", 0]}}, "2": {"class_type": "B", "inputs": {"x": ["1", 0]}}}
        self.assertEqual(set(cl.levels(g)), {"1", "2"})


class GroupSpecs(unittest.TestCase):

    def test_a_good_spec_is_cleaned(self):
        specs, errors = cl.clean_group_specs([{"title": " Load models ", "node_ids": ["#1", 1, "2"], "color": "models"}])
        self.assertEqual(errors, [])
        self.assertEqual(specs, [{"title": "Load models", "node_ids": ["1", "2"], "color": cl.ROLE_COLORS["models"]}])

    def test_what_cannot_be_drawn_is_said(self):
        _, errors = cl.clean_group_specs([{"title": "", "node_ids": []}, {"title": "A", "node_ids": ["1"], "color": "red-ish"},
                                          {"title": "a", "node_ids": ["2"]}])
        self.assertEqual(len(errors), 4, errors)

    def test_a_rename_needs_no_nodes(self):
        specs, errors = cl.clean_group_specs([{"title": "New", "rename_from": "Old"}])
        self.assertEqual(errors, [])
        self.assertEqual(specs[0]["rename_from"], "Old")


class Tools(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=True))
        self.enterContext(mock.patch("src.agent._load_settings", return_value={}))
        self.enterContext(mock.patch("src.pipeline._CanvasSchemas", Schemas))
        clear()
        self.addCleanup(clear)

    def _call(self, tool_name, pipe=None, **kw):
        pipe = pipe or pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        return json.loads(asyncio.run(tools(pipe)[tool_name](**kw))), pipe

    def _layout_patches(self):
        return [e for e in drain() if e.get("op") == "layout"]

    def test_arrange_pushes_one_layout_patch_and_records_the_edit(self):
        out, pipe = self._call("arrange_canvas", reason="it overlaps")
        self.assertEqual(out["status"], "applied")
        (patch,) = self._layout_patches()
        self.assertEqual(patch["arrange"], {"node_ids": None})
        self.assertEqual(patch["groups"], [])
        self.assertTrue(pipe._canvas_edits, "the turn's checkpoint knows the canvas was changed")

    def test_arrange_can_be_limited_to_some_nodes(self):
        self._call("arrange_canvas", node_ids=["#5", "6", "5"])
        (patch,) = self._layout_patches()
        self.assertEqual(patch["arrange"]["node_ids"], ["5", "6"])

    def test_group_pushes_the_groups_and_arranges_them_by_default(self):
        out, _ = self._call("group_canvas_nodes", groups=[{"title": "Load models", "node_ids": ["1"], "color": "models"}])
        self.assertEqual(out["status"], "applied")
        (patch,) = self._layout_patches()
        self.assertEqual(patch["groups"][0]["title"], "Load models")
        self.assertTrue(patch["arrange"]["only_groups"])

    def test_group_without_arrange_leaves_the_nodes_alone(self):
        self._call("group_canvas_nodes", groups=[{"title": "T", "node_ids": ["1"]}], arrange=False)
        (patch,) = self._layout_patches()
        self.assertIsNone(patch["arrange"])

    def test_a_bad_group_changes_nothing(self):
        out, _ = self._call("group_canvas_nodes", groups=[{"title": "", "node_ids": ["1"]}])
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(self._layout_patches(), [])

    def test_ids_not_on_the_canvas_are_refused(self):
        out, _ = self._call("group_canvas_nodes", groups=[{"title": "T", "node_ids": ["99"]}])
        self.assertEqual(out["status"], "rejected")

    def test_a_rename_goes_through(self):
        out, _ = self._call("group_canvas_nodes", groups=[{"title": "Base render", "rename_from": "Group"}])
        self.assertEqual(out["status"], "applied")
        self.assertEqual(self._layout_patches()[0]["groups"][0]["rename_from"], "Group")

    def test_proposing_changes_nothing(self):
        out, _ = self._call("auto_group_canvas")
        self.assertEqual(out["status"], "proposed")
        self.assertEqual([c["title"] for c in out["chunks"]][0], "Load models")
        self.assertEqual(drain(), [])

    def test_applying_draws_the_proposal_and_arranges(self):
        out, pipe = self._call("auto_group_canvas", apply=True, reason="split it up")
        self.assertEqual(out["status"], "applied")
        (patch,) = self._layout_patches()
        self.assertEqual([g["title"] for g in patch["groups"]], [c["title"] for c in out["chunks"]])
        self.assertTrue(pipe._canvas_edits)

    def test_an_unconfirmed_layout_is_not_applied_and_not_recorded(self):
        from agenty_core.utils import turn_scope
        from src.utils import canvas_ack
        tok = turn_scope.enter(turn_scope.Scope("rq", "A"))
        self.addCleanup(turn_scope.leave, tok)
        turn_scope.current().set("canvas_client", "page-1")
        with mock.patch.object(canvas_ack, "wait", lambda pid, t: {"ok": False, "timeout": True}):
            out, pipe = self._call("arrange_canvas")
            grouped, _ = self._call("group_canvas_nodes", groups=[{"title": "T", "node_ids": ["1"]}])
        for res in (out, grouped):
            self.assertEqual(res["status"], "not_applied")
            self.assertIn("NOT applied to any canvas", res["error"])
        self.assertFalse(pipe._canvas_edits, "an edit that did not land is not recorded")

    def test_a_confirmed_layout_is_applied(self):
        from agenty_core.utils import turn_scope
        from src.utils import canvas_ack
        tok = turn_scope.enter(turn_scope.Scope("rq", "A"))
        self.addCleanup(turn_scope.leave, tok)
        turn_scope.current().set("canvas_client", "page-1")
        with mock.patch.object(canvas_ack, "wait", lambda pid, t: {"ok": True}):
            out, _ = self._call("arrange_canvas")
        self.assertEqual(out["status"], "applied")
        (patch,) = self._layout_patches()
        self.assertTrue(patch["patch_id"])

    def test_a_tiny_graph_is_not_split(self):
        pipe = pipeline_stub(_canvas_graph={"1": _graph()["1"]}, _canvas_selection=[], _canvas_edits=[])
        out, _ = self._call("auto_group_canvas", pipe=pipe, apply=True)
        self.assertEqual(out["status"], "nothing_to_split")
        self.assertEqual(drain(), [])

    def test_no_graph_no_layout(self):
        pipe = pipeline_stub(_canvas_graph={}, _canvas_selection=[], _canvas_edits=[])
        for name, kw in (("arrange_canvas", {}), ("auto_group_canvas", {}),
                         ("group_canvas_nodes", {"groups": [{"title": "T", "node_ids": ["1"]}]})):
            out, _ = self._call(name, pipe=pipe, **kw)
            self.assertIn("error", out, name)
        self.assertEqual(drain(), [])

    def test_a_branch_of_a_pipeline_does_not_move_the_shared_canvas(self):
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        refusal = json.dumps({"status": "not_in_a_branch"})
        pipe._canvas_lease_refusal = lambda structural=True: refusal
        for name, kw in (("arrange_canvas", {}), ("group_canvas_nodes", {"groups": [{"title": "T", "node_ids": ["1"]}]})):
            out, _ = self._call(name, pipe=pipe, **kw)
            self.assertEqual(out["status"], "not_in_a_branch")
        self.assertEqual(drain(), [])

    def test_selection_only_mode_keeps_to_the_selection(self):
        with mock.patch("src.utils.canvas_view.full_graph_visible", return_value=False):
            pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[{"id": 5}], _canvas_edits=[])
            out, _ = self._call("arrange_canvas", pipe=pipe, node_ids=["6"])
            self.assertIn("not in the current canvas selection", out["error"])
            out, _ = self._call("arrange_canvas", pipe=pipe)
            self.assertEqual(out["arranged"], ["5"])

    def test_saving_asks_the_page_and_passes_its_answer_on(self):
        calls = []

        def ask(kind, payload, timeout):
            calls.append((kind, payload))
            return {"status": "needs_overwrite", "action": "refuse", "path": "workflows/a.json"}
        with mock.patch("src.utils.canvas_probe.request", ask):
            out, _ = self._call("save_canvas_workflow", name=" flow ")
        self.assertEqual(calls, [("save_workflow", {"name": "flow", "overwrite": False})])
        self.assertEqual(out["status"], "needs_overwrite", "the host never decides to overwrite")

    def test_saving_with_the_page_closed_says_so(self):
        with mock.patch("src.utils.canvas_probe.request", lambda *a, **k: {"error": "no answer", "timeout": True}):
            out, _ = self._call("save_canvas_workflow", name="x")
        self.assertEqual(out["status"], "failed")


class Wiring(unittest.TestCase):

    def test_the_tools_are_in_the_orchestrators_list_and_a_pack(self):
        pack = tool_packs.PACKS["layout"][1]
        self.assertEqual(set(pack), {"arrange_canvas", "group_canvas_nodes", "auto_group_canvas", "save_canvas_workflow"})
        with mock.patch("src.utils.canvas_view.full_graph_visible", return_value=True), \
                mock.patch("src.agent._load_settings", return_value={}):
            names = set(tools(pipeline_stub(_canvas_graph=_graph())))
        self.assertTrue(set(pack) <= names)

    def test_the_orchestrator_has_the_skill_and_the_researchers_do_not(self):
        from src import agent
        self.assertIn("canvas-layout", agent._ORCH_SKILL_NAMES)
        self.assertNotIn("canvas-layout", agent._ASSEMBLY_SKILL_NAMES)
        self.assertTrue(any(s.endswith("canvas-layout") for s in agent._orchestrator_skill_sources()))

    def test_the_skill_names_the_tools_it_uses(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent / "skills" / "canvas-layout" / "SKILL.md").read_text(encoding="utf-8")
        for name in ("arrange_canvas", "group_canvas_nodes", "auto_group_canvas", "save_canvas_workflow"):
            self.assertIn(name, text)
        self.assertTrue(text.startswith("---\nname: canvas-layout\n"))

    def test_the_prompt_points_at_the_skill(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent / "config" / "system_prompts"
                / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertIn("canvas-layout", text)


if __name__ == "__main__":
    unittest.main()
