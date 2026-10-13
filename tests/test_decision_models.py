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


def _models(*names, up=True):
    """Ollama answers (or not) and has exactly *names*."""
    return mock.patch.multiple(dm, installed=mock.Mock(side_effect=lambda m, host=None: m in names),
                               ollama_up=mock.Mock(return_value=up))


def _local(yes=True):
    return mock.patch.object(dm, "orchestrator_is_local", return_value=yes)


def _no_fetch():
    return mock.patch.object(dm, "fetch_in_background", return_value=False)


class WhoIsAsked(unittest.TestCase):
    """Never without Ollama; the router only for a local orchestrator."""

    def test_the_router_needs_a_local_orchestrator(self):
        with _models("nimble", "clef-flash"), _local(True), _no_fetch():
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS)), "nimble")
        with _models("nimble", "clef-flash"), _local(False), _no_fetch():
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS)), "")

    def test_the_prompt_check_runs_beside_a_hosted_orchestrator_too(self):
        for local in (True, False):
            with _models("clef-flash"), _local(local), _no_fetch():
                self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS)), "clef-flash")

    def test_without_ollama_nothing_is_asked_and_nothing_is_fetched(self):
        for use in ("router", "prompt_check"):
            with _models("nimble", "clef-flash", up=False), _local(True), \
                    mock.patch.object(dm, "fetch_in_background") as fetch:
                self.assertEqual(dm.model_for(use, dict(dm.DEFAULTS)), "")
                self.assertEqual(dm.model_for(use, dict(dm.DEFAULTS, **{use: "clef"})), "")
            fetch.assert_not_called()

    def test_off_is_off_and_a_named_model_is_used_when_it_is_there(self):
        with _models("nimble", "clef"), _local(False), _no_fetch():
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS, router="off")), "")
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS, prompt_check="clef")), "clef")
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS, router="nimble")), "nimble")

    def test_the_defaults_are_auto(self):
        self.assertEqual((dm.DEFAULTS["router"], dm.DEFAULTS["prompt_check"]), ("auto", "auto"))

    def test_no_ollama_means_nothing_is_installed(self):
        dm._installed.update(at=0.0, host="", names=(), up=True)
        with mock.patch.object(dm, "_call", side_effect=OSError("refused")):
            self.assertFalse(dm.installed("nimble", "http://127.0.0.1:1"))
            self.assertFalse(dm.ollama_up("http://127.0.0.1:1"))
        dm._installed.update(at=0.0)


class TheWeights(unittest.TestCase):
    """Fetched on first use, in the background, with a bar - never in the way."""

    def test_a_model_that_is_missing_is_fetched_and_the_use_waits_for_it(self):
        with _models(), _local(True), mock.patch.object(dm, "fetch_in_background") as fetch:
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS)), "")
            self.assertEqual(dm.model_for("router", dict(dm.DEFAULTS)), "")
        self.assertEqual([c.args[0] for c in fetch.call_args_list], ["clef-flash", "nimble"])

    def test_the_router_is_not_fetched_for_a_hosted_orchestrator(self):
        with _models(), _local(False), mock.patch.object(dm, "fetch_in_background") as fetch:
            dm.model_for("router", dict(dm.DEFAULTS))
        fetch.assert_not_called()

    def test_download_off_fetches_nothing(self):
        with _models(), _local(True), mock.patch.object(dm, "fetch_in_background") as fetch:
            self.assertEqual(dm.model_for("prompt_check", dict(dm.DEFAULTS, download=False)), "")
        fetch.assert_not_called()

    def test_the_bar_is_one_the_run_card_rewrites_in_place(self):
        line = dm.progress_line("clef-flash", 3_800_000_000, 10_000_000_000, "prompt check")
        self.assertIn("\u2588", line)
        self.assertIn("\u2591", line)
        self.assertIn("38 %", line)
        self.assertIn("3.8 / 10.0 GB", line)
        self.assertEqual(dm.bar(0, 0), "\u2591" * 16)
        self.assertEqual(dm.bar(10, 10), "\u2588" * 16)

    def _stream(self, messages):
        body = [json.dumps(m).encode() + b"\n" for m in messages]

        class Reply:
            def __enter__(self_inner):
                return iter(body)

            def __exit__(self_inner, *a):
                return False
        return mock.patch.object(dm.urllib.request, "urlopen", return_value=Reply())

    def test_the_download_reports_progress_and_says_when_it_is_ready(self):
        said, told = [], []
        layers = [{"status": "pulling manifest"},
                  {"digest": "a", "total": 8_000, "completed": 2_000},
                  {"digest": "b", "total": 2_000, "completed": 0},
                  {"digest": "a", "total": 8_000, "completed": 8_000},
                  {"digest": "b", "total": 2_000, "completed": 2_000},
                  {"status": "success"}]
        with self._stream(layers), mock.patch.object(dm, "installed", return_value=True):
            ok = dm._pull("nimble", "routing", "http://h", said.append, lambda text, **kw: told.append(text))
        self.assertTrue(ok)
        self.assertTrue(said and "100 %" in said[-1])
        self.assertTrue(all("nimble" in line for line in said))
        self.assertIn("nimble is ready", told[-1])

    def test_a_failed_download_is_said_with_what_to_do(self):
        told = []
        with self._stream([{"error": "pull model manifest: file does not exist"}]):
            ok = dm._pull("nimble", "routing", "http://h", lambda line: None, lambda text, **kw: told.append(text))
        self.assertFalse(ok)
        self.assertIn("ollama pull nimble", told[-1])

    def test_one_download_at_a_time_per_model(self):
        import threading
        release = threading.Event()
        started = []

        def slow(model, why, host, say, notify):
            started.append(model)
            release.wait(5)
            return True
        dm._pulling.clear()
        with mock.patch.object(dm, "_pull", side_effect=slow), \
                mock.patch("src.utils.status_bus.notify"):
            self.assertTrue(dm.fetch_in_background("nimble", "routing", "http://h"))
            self.assertFalse(dm.fetch_in_background("nimble", "routing", "http://h"))
            release.set()
            dm._pulling["nimble"].join(5)
        self.assertEqual(started, ["nimble"])
        dm._pulling.clear()


class TheRouter(unittest.TestCase):

    def _ask(self, value=None, error=None):
        def ask(model, state, questions, images=None, timeout=0):
            if error:
                raise error
            self.sent = (model, state, questions)
            probs = None if value is None else {"change": value, "new": 1 - value}
            return {"what": {"type": "choice", "choice": "change", "probabilities": probs}}
        return mock.patch.object(dm, "ask", side_effect=ask)

    def test_it_is_asked_about_the_wording_alone_as_a_choice(self):
        """Told that a graph is open, the model said "change" to nearly every
        build request (10 of 40 right); asked about the wording, 40 of 40."""
        with _models("nimble"), _local(), self._ask(0.98):
            got = dm.change_probability("add a canny ControlNet to this workflow", GRAPH)
        self.assertEqual(got, 0.98)
        model, state, questions = self.sent
        self.assertEqual(model, "nimble")
        self.assertEqual(list(state), ["request"])                 # nothing about the canvas
        self.assertEqual(questions["what"]["type"], "choice")
        self.assertEqual(set(questions["what"]["criteria"]), {"change", "new"})

    def test_no_model_or_no_answer_is_no_opinion(self):
        with _models(), _local(), _no_fetch():
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

    def test_a_guider_with_one_conditioning_is_followed_too(self):
        redux = {
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": "a portrait of a lighthouse keeper"}},
            "26": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": 3.5}},
            "41": {"class_type": "StyleModelApply", "inputs": {"conditioning": ["26", 0], "style_model": ["42", 0]}},
            "22": {"class_type": "BasicGuider", "inputs": {"model": ["30", 0], "conditioning": ["41", 0]}},
            "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["22", 0], "latent_image": ["27", 0]}},
        }
        self.assertEqual(dm.positive_prompt(redux), "a portrait of a lighthouse keeper")

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
        with _models(), _local(), _no_fetch():
            self.assertEqual(dm.prompt_check(["a.png"]), [])
        with _models("clef-flash", up=False), _local(True), _no_fetch():
            self.assertEqual(dm.prompt_check(["a.png"]), [])


if __name__ == "__main__":
    unittest.main()
