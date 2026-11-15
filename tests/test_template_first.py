"""Templates come before models: until prepare_workflow is called in a turn, the
first model/template lookup is handed back once."""

import unittest
from types import SimpleNamespace

from src.utils import template_first as tf

PINNED = ("[CANVAS DISPLAY] Auto-graphing of generated workflows onto the ComfyUI canvas is OFF. "
          "Build and run workflows as usual.")


def user(text):
    return [{"role": "user", "content": [{"text": text}]}]


class Turn:
    def __init__(self, text, tools=("prepare_workflow", "find_local_models", "hf_search")):
        self.agent = SimpleNamespace(tool_registry=SimpleNamespace(registry={t: object() for t in tools}),
                                     messages=[])
        self.hook = tf.TemplateFirstHook()
        self.hook._on_turn(SimpleNamespace(agent=self.agent, messages=user(text)))

    def call(self, name, **inp):
        ev = SimpleNamespace(agent=self.agent, tool_use={"name": name, "input": inp}, cancel_tool=False)
        self.hook._on_tool(ev)
        return ev.cancel_tool


class WhatCountsAsALookup(unittest.TestCase):

    def test_model_and_template_lookups(self):
        for name in ("find_local_models", "check_model", "hf_search", "hf_repo", "find_hf_file",
                     "find_workflows", "inspect_workflow_file", "download_hf_model"):
            self.assertTrue(tf.is_lookup(name, {}), name)

    def test_loading_the_lookup_packs(self):
        self.assertTrue(tf.is_lookup("load_tools", {"pack": "models"}))
        self.assertTrue(tf.is_lookup("load_tools", {"pack": "inspect"}))
        self.assertFalse(tf.is_lookup("load_tools", {"pack": "edit"}))

    def test_a_web_search_only_when_it_looks_for_a_workflow(self):
        self.assertTrue(tf.is_lookup("web_search", {"query": "LTX-2.5 SDR to HDR workflow ComfyUI"}))
        self.assertFalse(tf.is_lookup("web_search", {"query": "1950s American diner interior"}))

    def test_everything_else_is_left_alone(self):
        for name in ("analyze_image", "upload_image", "memory_read", "run_canvas", "partner_run"):
            self.assertFalse(tf.is_lookup(name, {}), name)


class WhenTheTurnIsArmed(unittest.TestCase):

    def test_a_workflow_request_with_pinned_blocks(self):
        self.assertTrue(tf.armed_for(PINNED + "\n\ncan you get an ltx 2.5 sdr-to-hdr workflow for me?"))

    def test_a_question_about_models_is_not(self):
        for q in ("which LTX models do I have?", "is flux installed", "download the missing files",
                  "do I have Z-Image Turbo?", "find the model on hugging face"):
            self.assertFalse(tf.armed_for(PINNED + "\n\n" + q), q)

    def test_only_the_users_own_words_decide(self):
        # a pinned block that talks about models does not disarm a workflow request
        self.assertTrue(tf.armed_for("[HARD CONSTRAINTS] which models the user named: LTX\n\nmake a video of a car"))

    def test_hook_and_open_graph_turns_are_not(self):
        self.assertFalse(tf.armed_for("[CANVAS HOOKS — REFERENCE ...]\n\nrun it"))
        self.assertFalse(tf.armed_for("[OPEN GRAPH MODE] This conversation works IN ...\n\nadd an upscaler"))

    def test_nothing_said(self):
        self.assertFalse(tf.armed_for(""))


class TheRedirect(unittest.TestCase):

    def test_the_first_lookup_is_handed_back_with_prepare_workflow(self):
        t = Turn(PINNED + "\n\ncan you build an ltx 2.5 sdr-to-hdr workflow for me?")
        why = t.call("find_local_models", pattern="ltx-2.5")
        self.assertIn("NOT RUN", why)
        self.assertIn("prepare_workflow(request, staged_inputs)", why)
        self.assertIn("find_local_models", why)

    def test_only_once_per_turn(self):
        t = Turn("make me a z-image turbo picture of a blue car")
        self.assertTrue(t.call("find_local_models", pattern="z-image"))
        self.assertFalse(t.call("find_local_models", pattern="z-image"))
        self.assertFalse(t.call("hf_search", query="z-image"))

    def test_after_prepare_workflow_lookups_run(self):
        t = Turn("make me a z-image turbo picture of a blue car")
        self.assertFalse(t.call("prepare_workflow", request="...", staged_inputs=[]))
        self.assertFalse(t.call("check_model", model_name="z_image_turbo.safetensors"))

    def test_other_calls_first_do_not_use_up_the_redirect(self):
        t = Turn("make me a video from this picture")
        self.assertFalse(t.call("analyze_image", file_paths=["a.png"]))
        self.assertTrue(t.call("hf_search", query="ltx"))

    def test_a_new_turn_arms_again(self):
        t = Turn("make me a z-image turbo picture of a blue car")
        t.call("find_local_models")
        t.hook._on_turn(SimpleNamespace(agent=t.agent, messages=user("now the same as a video")))
        self.assertTrue(t.call("find_local_models"))

    def test_an_agent_without_prepare_workflow_is_left_alone(self):
        t = Turn("make me a picture of a blue car", tools=("find_local_models",))
        self.assertFalse(t.call("find_local_models"))

    def test_a_model_question_is_answered(self):
        t = Turn("which LTX models are installed?")
        self.assertFalse(t.call("find_local_models", pattern="ltx"))

    def test_a_call_another_hook_cancelled_is_not_touched(self):
        t = Turn("make me a picture of a blue car")
        ev = SimpleNamespace(agent=t.agent, tool_use={"name": "find_local_models", "input": {}},
                             cancel_tool="held by another hook")
        t.hook._on_tool(ev)
        self.assertEqual(ev.cancel_tool, "held by another hook")


if __name__ == "__main__":
    unittest.main()
