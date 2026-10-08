"""The agent can draw a hook pipeline itself: the skill, and the tool it uses.

The skill's worked example is run through the same planner the tool uses, against
node definitions shaped like the extension's - so an example that would be
refused on a real canvas fails here first.
"""
import json
import re
import unittest
from pathlib import Path

from src.utils import canvas_edit as ce

ROOT = Path(__file__).resolve().parent.parent
SKILL = (ROOT / "skills" / "hook-pipeline" / "SKILL.md").read_text(encoding="utf-8")

EXEC = "AGENTY_EXEC"
_exec_in = {"optional": {"exec": [EXEC, {}]}}
_anchors = ["COMFY_AUTOGROW_V3", {"template": {"input": {"required": {"anchor": ["*", {}]}}}}]
PURPOSES = ["set / sweep parameter", "make workflow", "text only"]
# What ComfyUI's /object_info says of these nodes, cut down to what the planner reads.
NODES = {
    "AgentYHook": {"input": {"required": {"directive": ["STRING", {"multiline": True}],
                                          "purpose": [PURPOSES, {}],
                                          "remember": ["BOOLEAN", {"default": False}],
                                          "anchors": _anchors}, **_exec_in},
                   "output": ["*", EXEC], "output_name": ["out", "exec"]},
    "AgentYReview": {"input": {"required": {"reviewer": [["human", "agent"], {}],
                                            "anchors": _anchors,
                                            "notes": ["STRING", {"multiline": True}]}, **_exec_in},
                     "output": ["*", EXEC], "output_name": ["out", "exec"]},
    "AgentYLoopStart": {"input": {"required": {}, **_exec_in},
                        "output": [EXEC], "output_name": ["exec"]},
    "AgentYLoopBreak": {"input": {"required": {"condition": ["STRING", {}],
                                               "max_rounds": ["INT", {"default": 3}],
                                               "forward": [["best", "all that pass", "all"], {}]},
                                  **_exec_in},
                        "output": [EXEC], "output_name": ["exec"]},
    "AgentYJoin": {"input": {"required": {"execs": ["COMFY_AUTOGROW_V3",
                                                    {"template": {"input": {"required": {"exec": [EXEC, {}]}}}}]}},
                   "output": [EXEC], "output_name": ["exec"]},
    "KSampler": {"input": {"required": {"seed": ["INT", {}], "model": ["MODEL", {}]}},
                 "output": ["LATENT"], "output_name": ["LATENT"]},
}


def example_ops() -> list:
    block = re.search(r"```json\n(\[.*?\])\n```", SKILL, re.S)
    assert block, "the skill has no worked example"
    return json.loads(block.group(1))


class TheSkill(unittest.TestCase):
    def test_it_is_one_the_orchestrator_can_activate(self):
        from src import agent
        self.assertIn("hook-pipeline", agent._ORCH_SKILL_NAMES)
        self.assertTrue(any(s.replace("\\", "/").endswith("skills/hook-pipeline")
                            for s in agent._skill_sources(agent._ORCH_SKILL_NAMES)))

    def test_its_header_is_well_formed(self):
        head = SKILL.split("---", 2)[1]
        self.assertIn("name: hook-pipeline", head)
        tools = head.split("allowed-tools:", 1)[1].strip().split(",")
        self.assertEqual([t.strip() for t in tools],
                         ["edit_canvas_graph", "set_canvas_node_params", "delete_canvas_nodes",
                          "get_canvas_node"])

    def test_it_names_the_purposes_as_the_node_offers_them(self):
        for purpose in PURPOSES:
            self.assertIn(f"`{purpose}`", SKILL)
        for old in ("inline_parameter", "general_request", "text_only", "make_workflow",
                    "human_review"):
            self.assertNotIn(old, SKILL)

    def test_it_builds_and_does_not_run(self):
        self.assertIn("Do not run a pipeline in the turn you build it", SKILL)
        self.assertNotIn("apply_canvas_hooks", SKILL)

    def test_the_agent_is_pointed_at_it(self):
        prompt = (ROOT / "config" / "system_prompts" / "system_prompt.orchestrator.md"
                  ).read_text(encoding="utf-8")
        self.assertIn("**`hook-pipeline`** skill", prompt)
        hooks = (ROOT / "config" / "system_prompts" / "orchestrator" / "canvas_hooks.md"
                 ).read_text(encoding="utf-8")
        self.assertIn("`hook-pipeline` skill", hooks)


class TheWorkedExample(unittest.TestCase):
    """What the skill tells the agent to send has to be something the tool accepts."""

    def setUp(self):
        self.res = ce.plan({}, example_ops(), NODES)

    def test_it_is_accepted_on_an_empty_canvas(self):
        self.assertTrue(self.res["ok"], self.res.get("errors"))

    def test_every_node_is_on_the_execution_wire(self):
        g = self.res["graph"]
        on_wire = {nid for nid, n in g.items()
                   if any(k == "exec" or k.startswith("execs.") for k in n["inputs"]
                          if isinstance(n["inputs"][k], list))}
        fed = {str(v[0]) for n in g.values() for k, v in n["inputs"].items()
               if isinstance(v, list) and (k == "exec" or k.startswith("execs."))}
        self.assertEqual(on_wire | fed, set(g))

    def test_the_join_has_a_wire_per_branch(self):
        join = next(n for n in self.res["graph"].values() if n["class_type"] == "AgentYJoin")
        self.assertEqual(sorted(k for k in join["inputs"] if k.startswith("execs.")),
                         ["execs.exec0", "execs.exec1"])

    def test_a_stage_reads_what_it_uses_through_an_anchor(self):
        frames = next(n for n in self.res["graph"].values()
                      if "start frame" in str(n["inputs"].get("directive", "")))
        anchors = [k for k, v in frames["inputs"].items()
                   if k.startswith("anchors.") and isinstance(v, list)]
        self.assertEqual(len(anchors), 3)


class TheExecWireOnlyTakesExec(unittest.TestCase):
    def add(self, *more):
        return [{"op": "add", "class_type": "AgentYHook", "ref": "a", "params": {"directive": "x"}},
                {"op": "add", "class_type": "AgentYHook", "ref": "b", "params": {"directive": "y"}},
                *more]

    def test_a_value_cannot_be_wired_into_exec(self):
        res = ce.plan({}, self.add({"op": "connect", "from": "a", "output": "out",
                                    "to": "b", "input": "exec"}), NODES)
        self.assertFalse(res["ok"])
        self.assertIn("takes AGENTY_EXEC", " ".join(res["errors"]))

    def test_the_wire_cannot_be_plugged_into_a_data_input(self):
        res = ce.plan({}, self.add({"op": "connect", "from": "a", "output": "exec",
                                    "to": "b", "input": "anchors.anchor0"}), NODES)
        self.assertFalse(res["ok"])

    def test_exec_to_exec_and_out_to_anchor_are_fine(self):
        res = ce.plan({}, self.add(
            {"op": "connect", "from": "a", "output": "exec", "to": "b", "input": "exec"},
            {"op": "connect", "from": "a", "output": "out", "to": "b", "input": "anchors.anchor0"}),
            NODES)
        self.assertTrue(res["ok"], res.get("errors"))

    def test_ordinary_wildcards_still_connect(self):
        self.assertTrue(ce._compatible("*", "IMAGE"))
        self.assertTrue(ce._compatible("LATENT", "*"))
        self.assertFalse(ce._compatible("*", "AGENTY_EXEC"))
        self.assertTrue(ce._compatible("AGENTY_EXEC", "AGENTY_EXEC"))


class BuildingFromNothing(unittest.TestCase):
    """edit_canvas_graph refuses an empty canvas - except for a pipeline of agentY's own nodes."""

    def setUp(self):
        self.src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.tool = self.src.split("        async def edit_canvas_graph(", 1)[1].split("        @_tool", 1)[0]

    def test_a_pipeline_may_start_on_an_empty_canvas(self):
        self.assertIn("if not isinstance(graph, dict) or (not graph and not _from_nothing):", self.tool)
        for cls in ("AgentYHook", "AgentYReview", "AgentYLoopStart", "AgentYLoopBreak", "AgentYJoin"):
            self.assertIn(f'"{cls}"', self.tool)

    def test_anything_else_still_needs_a_graph(self):
        rule = self.tool.split("_from_nothing = bool(ops) and all(", 1)[1].split("for o in ops)", 1)[0]
        self.assertIn('o.get("op") == "add" and str(o.get("class_type")) in _pipeline_nodes', rule)
        self.assertNotIn("delete", rule)


if __name__ == "__main__":
    unittest.main()
