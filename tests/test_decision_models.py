"""Decision models: local judges, asked only on the local path.

No Ollama is reached here; the HTTP call is stood in for.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import decision_models as dm

GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a red fox in a snowy forest"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1],
                                                     "text": "blurry, low quality, distorted, deformed, watermark, text"}},
    "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0]}},
}
FLUX = {
    "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "an old fisherman mending nets"}},
    "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
    "10": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["5", 0]}},
    "6": {"class_type": "KSampler", "inputs": {"positive": ["10", 0], "negative": ["7", 0]}},
}


def _models(*names):
    return mock.patch.object(dm, "installed", side_effect=lambda m, host=None: m in names)


def _local(yes=True):
    return mock.patch.object(dm, "orchestrator_is_local", return_value=yes)


class WhoIsAsked(unittest.TestCase):
    """A hosted setup is never asked; nothing is downloaded."""

    def test_auto_needs_a_local_orchestrator_and_the_model_installed(self):
        with _models("nimble", "clef-flash"), _local(True):
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS)), "nimble")
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS)), "clef-flash")

    def test_a_hosted_orchestrator_means_no_model_even_when_installed(self):
        with _models("nimble", "clef-flash"), _local(False):
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS)), "")
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS)), "")

    def test_a_model_that_is_not_installed_is_not_asked_for(self):
        with _models(), _local(True):
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS)), "")

    def test_off_is_off_and_a_named_model_is_used_when_it_is_there(self):
        with _models("nimble", "clef"), _local(False):
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS, router="off")), "")
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS, prompt_check="clef")), "clef")
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS, prompt_check="clef-flash")), "")

    def test_the_defaults_are_auto(self):
        self.assertEqual((dm.DEFAULTS["router"], dm.DEFAULTS["prompt_check"]), ("auto", "auto"))

    def test_no_ollama_means_nothing_is_installed(self):
        dm._installed.update(at=0.0, host="", names=())
        with mock.patch.object(dm, "_call", side_effect=OSError("refused")):
            self.assertFalse(dm.installed("nimble", "http://127.0.0.1:1"))


class TheRouter(unittest.TestCase):

    def _ask(self, value=None, error=None):
        def ask(model, state, questions, images=None, timeout=0):
            if error:
                raise error
            self.sent = (model, state, questions)
            return {"change": {"type": "noul", "noul": value}}
        return mock.patch.object(dm, "ask", side_effect=ask)

    def test_the_request_and_what_is_open_are_put_to_the_model(self):
        with _models("nimble"), _local(), self._ask(0.93):
            got = dm.change_probability("add a canny ControlNet to this workflow", GRAPH)
        self.assertEqual(got, 0.93)
        self.assertEqual(self.sent[0], "nimble")
        self.assertIn("KSampler", self.sent[1]["canvas"])
        self.assertIn("CLIPTextEncode x2", self.sent[1]["canvas"])

    def test_no_model_or_no_answer_is_no_opinion(self):
        with _models(), _local():
            self.assertIsNone(dm.change_probability("add a ControlNet", GRAPH))
        with _models("nimble"), _local(), self._ask(error=TimeoutError()):
            self.assertIsNone(dm.change_probability("add a ControlNet", GRAPH))
        with _models("nimble"), _local(), self._ask(None):
            self.assertIsNone(dm.change_probability("add a ControlNet", GRAPH))

    def test_in_the_pipeline_a_sure_new_build_skips_the_question(self):
        from src.pipeline import Pipeline
        pipe = Pipeline.__new__(Pipeline)
        pipe._verbose = False
        pipe._canvas_graph = dict(GRAPH)
        pipe._canvas_edits, pipe._canvas_hooks, pipe._autoloop = [], None, None
        pipe._canvas_edit_first_said = False
        pipe._session = SimpleNamespace(open_graph_mode=True)
        cfg = {"edit_open_graph_first": True}
        with mock.patch("src.utils.settings.load_settings", return_value=cfg), \
                mock.patch.object(dm, "change_probability", return_value=0.04):
            self.assertIsNone(pipe._canvas_edit_first_refusal("build a new LTX-2 video workflow"))
        self.assertFalse(pipe._canvas_edit_first_said)
        with mock.patch("src.utils.settings.load_settings", return_value=cfg), \
                mock.patch.object(dm, "change_probability", return_value=0.96):
            self.assertEqual(pipe._canvas_edit_first_refusal("add a ControlNet")["status"], "graph_is_open")
        pipe._canvas_edit_first_said = False
        with mock.patch("src.utils.settings.load_settings", return_value=cfg), \
                mock.patch.object(dm, "change_probability", return_value=None):
            self.assertEqual(pipe._canvas_edit_first_refusal("anything")["status"], "graph_is_open")


class ThePrompt(unittest.TestCase):

    def test_it_is_what_the_sampler_was_given_not_the_longest_text(self):
        self.assertEqual(dm.positive_prompt(GRAPH), "a red fox in a snowy forest")

    def test_it_is_found_through_the_nodes_in_between(self):
        self.assertEqual(dm.positive_prompt(FLUX), "an old fisherman mending nets")

    def test_a_graph_without_one_gives_nothing(self):
        self.assertEqual(dm.positive_prompt({"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}}), "")
        self.assertEqual(dm.positive_prompt(None), "")


class ThePromptCheck(unittest.TestCase):

    def _run(self, answers, paths=("a.png", "b.png", "clip.mp4"), prompts=None):
        prompts = prompts or {"a.png": "a lighthouse keeper", "b.png": "a red fox in snow"}
        asked = []

        def ask(model, state, questions, images=None, timeout=0):
            asked.append(questions["shows"]["instructions"])
            return {"shows": {"type": "noul", "noul": answers[len(asked) - 1]}}
        with _models("clef-flash"), _local(), mock.patch.object(dm, "ask", side_effect=ask), \
                mock.patch.object(dm, "prompt_of_file", side_effect=lambda p: prompts.get(p, "")), \
                mock.patch.object(dm, "_encoded", return_value="QUJD"):
            return dm.prompt_check(list(paths)), asked

    def test_a_picture_that_does_not_show_its_prompt_is_named(self):
        found, asked = self._run([0.04, 0.91])
        self.assertEqual([f["path"] for f in found], ["a.png"])
        self.assertEqual(found[0]["probability"], 0.04)
        self.assertEqual(len(asked), 2)                       # the video is not asked about
        text = dm.prompt_check_text(found)
        self.assertIn("a.png", text)
        self.assertIn("Nothing was re-rendered", text)

    def test_all_fine_says_nothing(self):
        found, _ = self._run([0.8, 0.9])
        self.assertEqual(found, [])
        self.assertEqual(dm.prompt_check_text(found), "")

    def test_a_picture_without_a_stored_prompt_is_skipped(self):
        found, asked = self._run([0.01], prompts={"b.png": "a red fox"})
        self.assertEqual(len(asked), 1)
        self.assertEqual([f["path"] for f in found], ["b.png"])

    def test_no_model_means_no_check(self):
        with _models(), _local():
            self.assertEqual(dm.prompt_check(["a.png"]), [])
        with _models("clef-flash"), _local(False):
            self.assertEqual(dm.prompt_check(["a.png"]), [])


if __name__ == "__main__":
    unittest.main()
