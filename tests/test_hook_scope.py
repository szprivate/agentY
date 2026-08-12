"""Tests for canvas-hook graph scoping (canvas_hooks.hook_scope_ids / prune_to_hooks).

A hook on one branch of a large canvas must run only that branch: everything the
hook reaches downstream, plus whatever those nodes need as input. Unrelated output
chains are left out instead of being executed and written into every workflow
generated for the hook.

Self-contained: hand-built API-format prompts, no ComfyUI and no corpus.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import hook_scope_ids, prune_to_hooks, splice_hook_nodes


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _two_branch(hook_inputs, hook_consumer=None):
    """Two sampler branches sharing one checkpoint loader; hook lives on branch A.

    ckpt -> pos/neg/latent -> KSampler 5 -> VAEDecode 6 -> SaveImage 7   (branch A)
    ckpt -> pos2/latent2   -> KSampler 10 -> VAEDecode 11 -> SaveImage 12 (branch B)
    """
    g = {
        "1": _node("CheckpointLoaderSimple", ckpt_name="sd.safetensors"),
        "2": _node("CLIPTextEncode", clip=["1", 1], text="a cat"),
        "3": _node("CLIPTextEncode", clip=["1", 1], text="blurry"),
        "4": _node("EmptyLatentImage", width=512, height=512),
        "5": _node("KSampler", model=["1", 0], positive=["2", 0], negative=["3", 0],
                   latent_image=["4", 0], seed=1),
        "6": _node("VAEDecode", samples=["5", 0], vae=["1", 2]),
        "7": _node("SaveImage", images=["6", 0]),
        "8": _node("CLIPTextEncode", clip=["1", 1], text="a dog"),
        "9": _node("EmptyLatentImage", width=768, height=768),
        "10": _node("KSampler", model=["1", 0], positive=["8", 0], negative=["3", 0],
                    latent_image=["9", 0], seed=2),
        "11": _node("VAEDecode", samples=["10", 0], vae=["1", 2]),
        "12": _node("SaveImage", images=["11", 0]),
        "99": _node("AgentYHook", **hook_inputs),
    }
    if hook_consumer:
        nid, field = hook_consumer
        g[nid]["inputs"][field] = ["99", 0]
    return g


BRANCH_A = {"1", "2", "4", "5", "6", "7"}
BRANCH_B_ONLY = {"8", "9", "10", "11", "12"}


class HookScopeTests(unittest.TestCase):
    def test_producer_hook_drops_unrelated_branch(self):
        g = _two_branch({"anchor": ["2", 0], "directive": "rewrite"}, ("2", "text"))
        keep = hook_scope_ids(g, ["99"])
        self.assertTrue(BRANCH_A <= keep)
        self.assertFalse(BRANCH_B_ONLY & keep)

    def test_prune_reports_dropped_ids(self):
        g = _two_branch({"anchor": ["2", 0], "directive": "rewrite"}, ("2", "text"))
        scoped, dropped = prune_to_hooks(g, ["99"])
        self.assertEqual(set(dropped), BRANCH_B_ONLY)
        self.assertFalse(BRANCH_B_ONLY & set(scoped))

    def test_scoped_graph_is_still_runnable(self):
        """Every surviving input reference must resolve, or ComfyUI rejects the run."""
        g = _two_branch({"anchor": ["2", 0], "directive": "rewrite"}, ("2", "text"))
        scoped, _ = prune_to_hooks(g, ["99"])
        clean, removed = splice_hook_nodes(scoped)
        dangling = [(nid, k, v[0])
                    for nid, n in clean.items()
                    for k, v in (n.get("inputs") or {}).items()
                    if isinstance(v, list) and len(v) == 2 and str(v[0]) not in clean]
        self.assertEqual(dangling, [])
        self.assertEqual(removed, ["99"])
        # The kept KSampler's model/latent are ancestors, not descendants, of the
        # hook — the ancestor closure is what keeps them.
        self.assertEqual(clean["5"]["inputs"]["model"], ["1", 0])
        self.assertEqual(clean["5"]["inputs"]["latent_image"], ["4", 0])

    def test_inline_parameter_hook_with_unwired_output(self):
        """A seed sweep anchors on a node and wires nothing; its branch must survive."""
        g = _two_branch({"anchor": ["5", 0], "directive": "sweep the seed"})
        keep = hook_scope_ids(g, ["99"])
        self.assertTrue({"5", "6", "7"} <= keep)      # anchor's downstream
        self.assertTrue({"1", "2", "3", "4"} <= keep)  # anchor's own inputs
        self.assertFalse({"8", "9", "10", "12"} & keep)

    def test_unexecuted_hook_does_not_widen_scope(self):
        """A bypassed/muted hook isn't collected, so its branch isn't run either."""
        g = _two_branch({"anchor": ["2", 0], "directive": "rewrite"}, ("2", "text"))
        g["98"] = _node("AgentYHook", anchor=["8", 0], directive="disabled")
        keep = hook_scope_ids(g, ["99"])
        self.assertFalse({"10", "11", "12"} & keep)

    def test_two_executed_hooks_keep_both_branches(self):
        g = _two_branch({"anchor": ["2", 0], "directive": "a"}, ("2", "text"))
        g["98"] = _node("AgentYHook", anchor=["8", 0], directive="b")
        g["8"]["inputs"]["text"] = ["98", 0]
        keep = hook_scope_ids(g, ["99", "98"])
        self.assertTrue({"5", "7", "10", "12"} <= keep)

    def test_no_hooks_means_no_scoping(self):
        g = _two_branch({"anchor": ["2", 0], "directive": "x"})
        del g["99"]
        self.assertIsNone(hook_scope_ids(g, []))
        scoped, dropped = prune_to_hooks(g, [])
        self.assertIs(scoped, g)
        self.assertEqual(dropped, [])

    def test_single_branch_canvas_is_untouched(self):
        g = {
            "1": _node("CheckpointLoaderSimple"),
            "2": _node("CLIPTextEncode", clip=["1", 1], text=["99", 0]),
            "3": _node("KSampler", model=["1", 0], positive=["2", 0]),
            "4": _node("SaveImage", images=["3", 0]),
            "99": _node("AgentYHook", anchor=["1", 0], directive="x"),
        }
        scoped, dropped = prune_to_hooks(g, ["99"])
        self.assertIs(scoped, g)
        self.assertEqual(dropped, [])

    def test_autogrow_anchor_names_are_recognised(self):
        """Anchors arrive as anchor/anchor0/… or group-prefixed (anchors.anchor0)."""
        g = _two_branch({"anchors.anchor0": ["5", 0], "directive": "sweep"})
        keep = hook_scope_ids(g, ["99"])
        self.assertTrue({"5", "6", "7"} <= keep)
        self.assertFalse({"10", "12"} & keep)

    def test_empty_and_malformed_prompts(self):
        self.assertEqual(prune_to_hooks({}, ["1"]), ({}, []))
        self.assertEqual(prune_to_hooks(None, ["1"]), (None, []))


if __name__ == "__main__":
    unittest.main()
