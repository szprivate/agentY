"""Running one stage runs that stage, and nothing of the stages after it.

A pipeline wired the natural way joins its stages with DATA as well as with the
execution wire: a review reads the stage's save node, the next hook reads the
review, and feeds its own generator. Passing those through as wires in the graph
that is queued put the LATER generators behind the first stage's save node - so
a character sweep ran the location, background and final-shot generators too,
once per variant, on whatever was in their prompts.

The canvas this is modelled on (data wires only; hooks 118, 125, 142 and 8 each
write the prompt of a generator, 125 also its first image input):

    118 -> [26 gen] -> [31 save] -> review 136 -> hook 125 -> [124 gen] -> [127 save]
                                                -> review 140 -> hook 142 -> [144 gen] -> [145 save]
    18 -> hook 8 -> [7 gen] -> [30 save] -> review 135 ------------------^
"""
import unittest

from src.utils import canvas_hooks as ch


def graph():
    gen = "OpenAIGPTImageNodeV2"
    return {
        "4": {"class_type": "AgentYHook", "inputs": {"directive": "screenplay"}},
        "131": {"class_type": "AgentYReview", "inputs": {"anchors.anchor0": ["4", 0]}},
        "18": {"class_type": "AgentYHook", "inputs": {"anchors.anchor0": ["131", 0]}},
        "8": {"class_type": "AgentYHook", "inputs": {"anchors.anchor0": ["18", 0]}},
        "7": {"class_type": gen, "inputs": {"prompt": ["8", 0]}},
        "30": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": ["8", 0]}},
        "135": {"class_type": "AgentYReview", "inputs": {"anchors.anchor0": ["30", 0]}},
        "118": {"class_type": "AgentYHook", "inputs": {"directive": "main location"}},
        "26": {"class_type": gen, "inputs": {"prompt": ["118", 0]}},
        "31": {"class_type": "SaveImage", "inputs": {"images": ["26", 0]}},
        "136": {"class_type": "AgentYReview", "inputs": {"anchors.anchor0": ["31", 0]}},
        "125": {"class_type": "AgentYHook",
                "inputs": {"anchors.anchor0": ["136", 0], "anchors.anchor1": ["131", 0]}},
        "124": {"class_type": gen,
                "inputs": {"prompt": ["125", 0], "model.images.image_1": ["125", 0]}},
        "127": {"class_type": "SaveImage", "inputs": {"images": ["124", 0]}},
        "140": {"class_type": "AgentYReview",
                "inputs": {"anchors.anchor0": ["127", 0], "references.reference0": ["136", 0]}},
        "142": {"class_type": "AgentYHook",
                "inputs": {"anchors.anchor0": ["135", 0], "anchors.anchor1": ["140", 0],
                           "anchors.anchor2": ["131", 0]}},
        "144": {"class_type": gen, "inputs": {"prompt": ["142", 0]}},
        "145": {"class_type": "SaveImage", "inputs": {"images": ["144", 0]}},
    }


def hook(hid, targets):
    return {"hook_node_id": hid, "purpose": "set_parameter",
            "targets": [{"node_id": n, "to_input": i, "to_input_type": t} for n, i, t in targets]}


HOOKS = [
    hook("118", [("26", "prompt", "STRING")]),
    hook("8", [("7", "prompt", "STRING"), ("30", "filename_prefix", "STRING")]),
    hook("125", [("124", "prompt", "STRING"), ("124", "model.images.image_1", "IMAGE")]),
    hook("142", [("144", "prompt", "STRING")]),
]


class Spliced(unittest.TestCase):
    def setUp(self):
        self.clean, self.removed = ch.splice_hook_nodes(graph(), HOOKS)

    def test_every_stage_node_is_gone(self):
        self.assertEqual(sorted(self.removed, key=int),
                         ["4", "8", "18", "118", "125", "131", "135", "136", "140", "142"])

    def test_a_later_generator_is_not_wired_to_an_earlier_stages_output(self):
        # 124's image came from hook 125, which read review 136, which read save 31.
        # That is what stage 118 made - for the agent to hand over, not a wire.
        self.assertNotIn("model.images.image_1", self.clean["124"]["inputs"])
        for nid, node in self.clean.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2:
                    self.assertIn(str(value[0]), self.clean, f"{nid} reads a node that is gone")

    def test_a_prompt_a_hook_writes_is_left_for_its_value(self):
        for nid in ("7", "26", "124", "144"):
            self.assertNotIn("prompt", self.clean[nid]["inputs"])

    def test_a_hook_reading_a_real_node_still_passes_it_through(self):
        g = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
             "2": {"class_type": "AgentYHook", "inputs": {"anchors.anchor0": ["1", 0]}},
             "3": {"class_type": "Upscale", "inputs": {"image": ["2", 0]}}}
        clean, _ = ch.splice_hook_nodes(g)
        self.assertEqual(clean["3"]["inputs"]["image"], ["1", 0])

    def test_a_review_reading_a_real_node_still_passes_it_through(self):
        g = {"1": {"class_type": "SaveImage", "inputs": {}},
             "2": {"class_type": "AgentYReview", "inputs": {"anchors.anchor0": ["1", 0]}},
             "3": {"class_type": "Next", "inputs": {"image": ["2", 0]}}}
        clean, _ = ch.splice_hook_nodes(g)
        self.assertEqual(clean["3"]["inputs"]["image"], ["1", 0])


class OneStagePerRun(unittest.TestCase):
    def setUp(self):
        self.clean, _ = ch.splice_hook_nodes(graph(), HOOKS)
        self.by_id = {h["hook_node_id"]: h for h in HOOKS}

    def kept(self, hid):
        scoped, _dropped = ch.scope_to_hook(self.clean, self.by_id[hid])
        return sorted(scoped, key=int)

    def test_the_location_stage_runs_only_its_own_generator(self):
        self.assertEqual(self.kept("118"), ["26", "31"])

    def test_the_character_stage_runs_only_its_own_generator(self):
        self.assertEqual(self.kept("8"), ["7", "30"])

    def test_the_background_stage_runs_only_its_own_generator(self):
        self.assertEqual(self.kept("125"), ["124", "127"])

    def test_no_stage_reaches_the_final_generator(self):
        for hid in ("118", "8", "125"):
            self.assertNotIn("144", self.kept(hid))


class ACallThatNamesNodes(unittest.TestCase):
    """With no single hook to scope to, the nodes the call names are the scope."""

    def setUp(self):
        self.clean, _ = ch.splice_hook_nodes(graph(), HOOKS)

    def test_it_is_cut_to_what_those_nodes_drive(self):
        scoped, dropped = ch.scope_to_nodes(self.clean, ["7"])
        self.assertEqual(sorted(scoped, key=int), ["7", "30"])
        self.assertIn("144", dropped)

    def test_naming_nothing_on_the_graph_leaves_it_alone(self):
        scoped, dropped = ch.scope_to_nodes(self.clean, ["999"])
        self.assertEqual((len(scoped), dropped), (len(self.clean), []))

    def test_the_pipeline_never_leaves_a_variant_whole_when_it_names_nodes(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        trim = src.split("    def _trim_variants(self,", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("cand, dropped = scope_to_nodes(kept, wanted)", trim)


class AWrittenValueRunsItsOwnStage(unittest.TestCase):
    """A value written into a generator queues a run at the end of the turn."""

    def prompt_for(self, targets):
        import types
        from src.pipeline import Pipeline
        clean, _ = ch.splice_hook_nodes(graph(), HOOKS)
        obj = types.SimpleNamespace(_canvas_base_prompt=clean,
                                    _canvas_keeplive_targets=list(targets))
        return clean, Pipeline._keeplive_prompt(obj)

    def test_it_is_that_stage_and_not_the_canvas(self):
        _clean, queued = self.prompt_for(["26"])
        self.assertEqual(sorted(queued, key=int), ["26", "31"])

    def test_no_generator_is_queued_without_its_prompt(self):
        # What ComfyUI refused: 7, 124 and 144 queued beside the stage that ran.
        _clean, queued = self.prompt_for(["26"])
        for nid in ("7", "30", "124", "144"):
            self.assertNotIn(nid, queued)

    def test_two_values_written_in_one_turn_run_both_stages(self):
        _clean, queued = self.prompt_for(["26", "7"])
        self.assertEqual(sorted(queued, key=int), ["7", "26", "30", "31"])

    def test_a_canvas_of_one_stage_is_queued_whole(self):
        clean, queued = self.prompt_for([])
        self.assertEqual(queued, clean)

    def test_the_turn_end_run_asks_for_it(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("json.dumps(self._keeplive_prompt())", src)
        self.assertNotIn("_kp.write_text(json.dumps(self._canvas_base_prompt)", src)
        self.assertEqual(src.count("self._canvas_keeplive_targets = []"), 1)


class ALoopOnOneNode(unittest.TestCase):
    """refine_canvas_until and prompt_autoloop change one value and run again."""

    def cut(self, hooks):
        import types
        from src.pipeline import Pipeline
        clean, _ = ch.splice_hook_nodes(graph(), HOOKS)
        return clean, Pipeline._stage_graph(types.SimpleNamespace(_canvas_hooks=hooks), clean, ["124"])

    def test_on_a_pipeline_it_runs_that_nodes_stage(self):
        _clean, queued = self.cut(HOOKS)
        self.assertEqual(sorted(queued, key=int), ["124", "127"])

    def test_on_an_ordinary_canvas_it_runs_the_graph_that_is_open(self):
        clean, queued = self.cut(None)
        self.assertEqual(queued, clean)

    def test_both_loops_cut_before_they_queue(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("graph = self._stage_graph(graph, [tid])", src)
        self.assertIn("graph = self._stage_graph(graph, [str(node_id)])", src)


class OneCallOneStage(unittest.TestCase):
    def test_resolutions_for_two_stages_are_refused(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        tool = src.split("        async def apply_canvas_hooks(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("if len(set(_worked)) > 1:", tool)
        self.assertIn("One call\n                             \"runs ONE stage.\"".replace("\n                             \"", " "),
                      tool.replace("\"\n                             \"", ""))
        self.assertLess(tool.index("if len(set(_worked)) > 1:"), tool.index("_build_batch("))


class BranchesOfSweeps(unittest.TestCase):
    """A branch made of on-canvas sweeps is work worth a conversation."""

    def test_the_parallel_instructions_are_given_for_them(self):
        from src.utils import hook_flow as hf

        def stage(hid, purpose, after=(), **more):
            return {"hook_node_id": hid, "purpose": purpose, "exec_wired": True,
                    "via_hook_ids": list(after), "exec_prev_ids": list(after),
                    "prev_hook_ids": [], "prev_hook_id": None, "anchors": [], "targets": [], **more}
        hooks = [stage("4", "text_only", directive="story"),
                 stage("8", "set_parameter", after=["4"], directive="characters",
                       targets=[{"node_id": "7", "to_input": "prompt"}]),
                 stage("118", "set_parameter", after=["4"], directive="places",
                       targets=[{"node_id": "26", "to_input": "prompt"}]),
                 stage("9", "loop_start")]
        flow = hf.plan(hooks)
        block = ch.describe_hooks(flow.hooks, {}, flow=flow, into_canvas=True)
        self.assertIn("PARALLEL BRANCHES — the execution wire splits into 2 branches", block)

    def test_two_branches_of_text_alone_are_not_worth_conversations(self):
        from src.utils import hook_flow as hf

        def stage(hid, after=()):
            return {"hook_node_id": hid, "purpose": "text_only", "exec_wired": True,
                    "via_hook_ids": list(after), "prev_hook_ids": [], "anchors": [],
                    "targets": [], "directive": "write"}
        hooks = [stage("1"), stage("2", ["1"]), stage("3", ["1"]),
                 {"hook_node_id": "9", "purpose": "loop_start", "via_hook_ids": []}]
        flow = hf.plan(hooks)
        self.assertNotIn("PARALLEL BRANCHES", ch.describe_hooks(flow.hooks, {}, flow=flow))


if __name__ == "__main__":
    unittest.main()
