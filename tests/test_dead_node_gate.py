"""A build does not finish while part of it will never run.

ComfyUI executes a graph backwards from its output nodes, so a node whose output
nothing reads is skipped in silence — it fails no validation, local or server-side.
Three live from-scratch builds produced two such graphs, both reported `valid:
true`: an LTX-2 video carrying a dead `GetImageSize` (the resolution was hardcoded,
so "match the input image" was quietly dropped), a duplicate `VAEDecodeTiled`, and
an unused `LTXVAddGuide`.

So `Pipeline._settle_dead_nodes` hands the exact list back to the builder once, and
removes whatever is still dead afterwards — safe by definition, and said out loud,
because a dead node is usually the trace of an intention that went missing.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest
from unittest import mock

from src.pipeline import Pipeline

# The shape of the real LTX-2 build, trimmed to what matters: a live chain into
# SaveVideo, plus the three nodes nothing read.
GRAPH = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
    "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx.safetensors"}},
    "9": {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]},
          "_meta": {"title": "Get Image Size"}},
    "16": {"class_type": "LTXVImgToVideoInplace",
           "inputs": {"image": ["1", 0], "vae": ["2", 2], "width": 1280, "height": 704}},
    "35": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["2", 2]}},
    "37": {"class_type": "CreateVideo", "inputs": {"images": ["35", 0], "fps": 25.0}},
    "38": {"class_type": "SaveVideo", "inputs": {"video": ["37", 0]}},
    "39": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["16", 0], "vae": ["2", 2]},
           "_meta": {"title": "VAE Decode (tiled)"}},
    "40": {"class_type": "LTXVAddGuide", "inputs": {"latent": ["16", 0], "image": ["1", 0]},
           "_meta": {"title": "LTXV Add Guide"}},
}

DEAD = [{"node_id": n, "class_type": GRAPH[n]["class_type"],
         "title": (GRAPH[n].get("_meta") or {}).get("title", ""),
         "problem": "nothing reads this node's output …"} for n in ("9", "39", "40")]


class Builder:
    """A generate_new_workflow agent that answers a handback in a stated way."""

    def __init__(self, fixes: bool = True, raises=None, hangs: bool = False):
        self.fixes, self.raises, self.hangs = fixes, raises, hangs
        self.prompts: list = []

    async def invoke_async(self, prompt):
        self.prompts.append(str(prompt))
        if self.raises is not None:
            raise self.raises
        if self.hangs:
            await asyncio.sleep(5)
        return "workflow_path: wf.json"


class Gate(unittest.TestCase):
    """_settle_dead_nodes, with the detector and update_workflow stubbed."""

    def run_gate(self, agent, rounds: list, timeout: float = 30.0):
        """*rounds* is what each successive dead-node check reports."""
        answers = list(rounds)
        removed: list = []

        def check(_path):
            found = answers.pop(0) if answers else []
            return found, [d["problem"] for d in found]

        def update(_path, _patches="[]", _adds="[]", remove_nodes="[]"):
            removed.extend(json.loads(remove_nodes))
            return json.dumps({"status": "ok"})

        pipe = Pipeline.__new__(Pipeline)
        pipe._verbose = False
        pipe._FIX_ASSEMBLY_TIMEOUT = timeout
        with mock.patch("agenty_core.tools.assembly_deterministic.dead_nodes_in_file",
                        check), \
             mock.patch("agenty_core.tools.comfyui.update_workflow", update):
            res = asyncio.run(pipe._settle_dead_nodes(agent, "wf.json"))
        return res, removed

    def test_a_clean_build_is_left_alone(self):
        agent = Builder()
        res, removed = self.run_gate(agent, [[]])
        self.assertEqual(res, {"dead": 0, "handed_back": 0, "pruned": []})
        self.assertEqual(agent.prompts, [])   # never bothered
        self.assertEqual(removed, [])

    def test_the_builder_is_handed_the_exact_list(self):
        agent = Builder()
        res, removed = self.run_gate(agent, [DEAD, []])
        self.assertEqual(res["dead"], 0)
        self.assertEqual(res["handed_back"], 1)
        self.assertEqual(removed, [])         # it fixed them, nothing to remove
        handback = agent.prompts[0]
        for nid, cls in (("9", "GetImageSize"), ("39", "VAEDecodeTiled"),
                         ("40", "LTXVAddGuide")):
            self.assertIn(nid, handback)
            self.assertIn(cls, handback)
        self.assertIn("wf.json", handback)
        # It must say WHY, or the builder re-learns nothing for the next build.
        self.assertIn("backwards", handback.lower())

    def test_what_survives_the_handback_is_removed(self):
        agent = Builder(fixes=False)
        res, removed = self.run_gate(agent, [DEAD, DEAD])
        self.assertEqual(res["dead"], 3)
        self.assertEqual(sorted(removed), ["39", "40", "9"])
        self.assertEqual(res["pruned"], ["9", "39", "40"])

    def test_it_hands_back_once(self):
        agent = Builder(fixes=False)
        self.run_gate(agent, [DEAD, DEAD])
        self.assertEqual(len(agent.prompts), 1)
        self.assertEqual(Pipeline._MAX_DEAD_NODE_ROUNDS, 1)

    def test_a_builder_that_throws_does_not_lose_the_build(self):
        agent = Builder(raises=RuntimeError("model refused"))
        res, removed = self.run_gate(agent, [DEAD])
        self.assertEqual(res["dead"], 3)
        self.assertEqual(sorted(removed), ["39", "40", "9"])

    def test_a_builder_that_hangs_is_given_up_on(self):
        agent = Builder(hangs=True)
        res, removed = self.run_gate(agent, [DEAD], timeout=0.05)
        self.assertEqual(res["dead"], 3)
        self.assertEqual(sorted(removed), ["39", "40", "9"])

    def test_a_validate_that_explodes_never_fails_the_build(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe._verbose = False
        pipe._FIX_ASSEMBLY_TIMEOUT = 30.0
        agent = Builder()
        with mock.patch("agenty_core.tools.assembly_deterministic.dead_nodes_in_file",
                        side_effect=OSError("comfyui is down")):
            res = asyncio.run(pipe._settle_dead_nodes(agent, "wf.json"))
        self.assertEqual(res, {"dead": 0, "handed_back": 0, "pruned": []})
        self.assertEqual(agent.prompts, [])

    def test_a_failed_removal_says_so_instead_of_claiming_a_clean_graph(self):
        def check(_path):
            return DEAD, [d["problem"] for d in DEAD]

        def update(*_a, **_k):
            raise OSError("read-only file")

        pipe = Pipeline.__new__(Pipeline)
        pipe._verbose = False
        pipe._FIX_ASSEMBLY_TIMEOUT = 30.0
        with mock.patch("agenty_core.tools.assembly_deterministic.dead_nodes_in_file",
                        check), \
             mock.patch("agenty_core.tools.comfyui.update_workflow", update):
            res = asyncio.run(pipe._settle_dead_nodes(Builder(fixes=False), "wf.json"))
        self.assertEqual(res["dead"], 3)
        self.assertEqual(res["pruned"], [])


class Wiring(unittest.TestCase):

    def test_the_build_path_runs_the_gate_before_reporting_ready(self):
        import inspect
        src = inspect.getsource(Pipeline._run_generate_new_workflow)
        self.assertIn("_settle_dead_nodes", src)
        self.assertLess(src.index("_settle_dead_nodes"),
                        src.index('return {"status": "ready"'))

    def test_the_detector_agrees_with_the_real_graph(self):
        # The gate reads dead_nodes off the shared detector; this is the other half
        # of that contract — the detector on the graph that produced this fixture.
        from agenty_core.tools.assembly_deterministic import dead_nodes
        info = {c: {"output": ["IMAGE"]} for c in
                ("LoadImage", "CheckpointLoaderSimple", "GetImageSize",
                 "LTXVImgToVideoInplace", "VAEDecode", "VAEDecodeTiled",
                 "CreateVideo", "LTXVAddGuide")}
        info["SaveVideo"] = {"output": [], "output_node": True}
        self.assertEqual(sorted(d["node_id"] for d in dead_nodes(GRAPH, info)),
                         ["39", "40", "9"])


if __name__ == "__main__":
    unittest.main()
