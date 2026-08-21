"""The closed refine loop: run the user's own graph until its output meets a condition.

Panel mode's standing rule is that agentY edits the canvas and the *user* presses
Queue. A loop is the one thing that suspends that rule, so what it does with their
graph has to be exactly what they asked for and nothing more. The checks here are
on the three ways that goes wrong:

  - it varies the **wrong value** — a checkpoint name, a negative prompt, a wired
    input — and the user's graph comes back changed in a way they did not ask for;
  - it **claims a match nobody made**, because the judge passes on doubt by design
    and a loop reads a pass as "stop, you're done";
  - it **spends the budget on the same picture**, re-running a value already
    judged, or looping past the cap the user set.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import canvas_loop as loop
from src.utils.qa import QaResult

# A small but realistic txt2img graph: one positive prompt, one negative (known
# only from the wire into the sampler), a checkpoint, and a loaded reference.
GRAPH = {
    "4": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "a woman standing in a field at dawn", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "blurry, watermark", "clip": ["4", 1]}},
    "9": {"class_type": "LoadImage", "inputs": {"image": "original_frame.png [input]"}},
    "3": {"class_type": "KSampler",
          "inputs": {"seed": 42, "sampler_name": "euler", "scheduler": "normal",
                     "positive": ["6", 0], "negative": ["7", 0], "model": ["4", 0]}},
    "10": {"class_type": "SaveImage",
           "inputs": {"filename_prefix": "ComfyUI", "images": ["3", 0]}},
}


def _pipe(**over):
    base = dict(_canvas_base_prompt=dict(GRAPH), _canvas_hooks=[],
                _session=SimpleNamespace(current_output_paths=[]))
    base.update(over)
    return pipeline_stub(**base)


class TextTargetTests(unittest.TestCase):
    """Which widgets the loop is willing to consider varying at all."""

    def setUp(self):
        self.found = {(t["node_id"], t["param"]): t for t in loop.text_targets(GRAPH)}

    def test_a_prompt_is_a_target(self):
        self.assertIn(("6", "text"), self.found)

    def test_a_checkpoint_name_is_not(self):
        """It is a string like any other. Varying it swaps the user's model out."""
        self.assertNotIn(("4", "ckpt_name"), self.found)

    def test_a_sampler_enum_is_not(self):
        self.assertNotIn(("3", "sampler_name"), self.found)
        self.assertNotIn(("3", "scheduler"), self.found)

    def test_a_filename_prefix_is_not(self):
        self.assertNotIn(("10", "filename_prefix"), self.found)

    def test_a_loaded_image_is_not(self):
        self.assertNotIn(("9", "image"), self.found)

    def test_a_wired_input_is_not(self):
        """`positive` is a link. There is no value there to rewrite."""
        self.assertNotIn(("3", "positive"), self.found)

    def test_a_number_is_not(self):
        self.assertNotIn(("3", "seed"), self.found)

    def test_the_negative_is_found_but_marked(self):
        """It is a real prompt — it just must never be picked unasked."""
        self.assertEqual(self.found[("7", "text")]["role"], "negative")
        self.assertEqual(self.found[("6", "text")]["role"], "positive")

    def test_negative_is_read_off_the_wire_not_the_node(self):
        """A CLIPTextEncode cannot know which of the two it is — the sampler can."""
        both = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat on a mat"}},
                "3": {"class_type": "KSampler", "inputs": {"negative": ["6", 0]}}}
        self.assertEqual(loop.text_targets(both)[0]["role"], "negative")

    def test_a_widget_named_negative_is_marked_too(self):
        one = {"5": {"class_type": "FluxPrompt",
                     "inputs": {"positive": "a lit street", "negative": "rain"}}}
        roles = {t["param"]: t["role"] for t in loop.text_targets(one)}
        self.assertEqual(roles, {"positive": "positive", "negative": "negative"})

    def test_an_empty_prompt_box_is_still_a_prompt_box(self):
        empty = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
        self.assertEqual(loop.text_targets(empty)[0]["param"], "text")

    def test_prose_in_a_widget_that_is_never_a_prompt_is_still_not_one(self):
        """The suffix and prose rules acquit most of these by accident. A save
        prefix with a space in it is caught by nothing but the name."""
        named = {"10": {"class_type": "SaveImage",
                        "inputs": {"filename_prefix": "renders/woman in a field"}},
                 "12": {"class_type": "OpenAIGPTImage",
                        "inputs": {"model": "gpt image 1 large",
                                   "background": "opaque or transparent"}}}
        self.assertEqual(loop.text_targets(named), [])

    def test_an_unnamed_widget_needs_to_look_like_prose(self):
        odd = {"1": {"class_type": "Thing",
                     "inputs": {"a": "fp8_e4m3fn", "b": "a long line of actual words"}}}
        self.assertEqual([t["param"] for t in loop.text_targets(odd)], ["b"])

    def test_the_longest_positive_comes_first(self):
        """The refusal list is only readable if the likely answer is at the top."""
        order = [(t["node_id"], t["param"]) for t in loop.text_targets(GRAPH)]
        self.assertEqual(order[0], ("6", "text"))
        self.assertEqual(order[-1], ("7", "text"))


class ChooseTargetTests(unittest.TestCase):
    """Picking the one value to vary — or refusing, with the candidates."""

    def test_the_single_positive_is_picked_unaided(self):
        target, err = loop.choose_target(GRAPH)
        self.assertIsNone(err)
        self.assertEqual((target["node_id"], target["param"]), ("6", "text"))

    def test_a_negative_is_never_picked_on_its_own(self):
        """Refining "what I don't want" against a goal phrased as "what I do want"
        inverts the loop, so it is never the automatic answer."""
        only_negative = {"7": {"class_type": "CLIPTextEncode",
                               "inputs": {"text": "blurry, watermark"}},
                         "3": {"class_type": "KSampler", "inputs": {"negative": ["7", 0]}}}
        target, err = loop.choose_target(only_negative)
        self.assertIsNone(target)
        self.assertIn("node_id", err["error"])

    def test_two_prompts_is_a_refusal_that_lists_them(self):
        two = dict(GRAPH)
        two["11"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "a second prompt here"}}
        target, err = loop.choose_target(two)
        self.assertIsNone(target)
        self.assertEqual(len(err["candidates"]), 2)
        self.assertTrue(any("6.text" in c for c in err["candidates"]))

    def test_a_named_node_with_one_text_widget_needs_no_param(self):
        target, err = loop.choose_target(GRAPH, node_id="7")
        self.assertIsNone(err)
        self.assertEqual(target["param"], "text")

    def test_a_named_node_with_two_asks_which(self):
        graph = {"5": {"class_type": "FluxPrompt",
                       "inputs": {"positive": "a lit street", "prompt": "another one"}}}
        target, err = loop.choose_target(graph, node_id="5")
        self.assertIsNone(target)
        self.assertIn("param", err["error"])
        self.assertEqual(len(err["choices"]), 2)

    def test_naming_a_wired_input_says_so(self):
        target, err = loop.choose_target(GRAPH, node_id="3", param="positive")
        self.assertIsNone(target)
        self.assertIn("WIRED", err["error"])

    def test_naming_a_number_says_what_it_holds(self):
        target, err = loop.choose_target(GRAPH, node_id="3", param="seed")
        self.assertIsNone(target)
        self.assertIn("42", err["error"])

    def test_an_unknown_node_comes_back_with_the_canvas(self):
        target, err = loop.choose_target(GRAPH, node_id="999")
        self.assertIsNone(target)
        self.assertTrue(any("6.text" in c for c in err["text_widgets_on_the_canvas"]))

    def test_a_node_with_no_text_at_all(self):
        target, err = loop.choose_target(GRAPH, node_id="4")
        self.assertIsNone(target)
        self.assertIn("no text widget", err["error"])

    def test_a_graph_with_nothing_to_vary(self):
        target, err = loop.choose_target({"4": GRAPH["4"]})
        self.assertIsNone(target)
        self.assertIn("name the node", err["error"])


class BudgetTests(unittest.TestCase):

    def setUp(self):
        patch = mock.patch.object(loop, "max_runs_cap", return_value=4)
        patch.start()
        self.addCleanup(patch.stop)

    def test_asking_for_fewer_gets_fewer(self):
        self.assertEqual(loop.clamp_runs(2), (2, 4))

    def test_asking_for_more_gets_the_cap(self):
        self.assertEqual(loop.clamp_runs(10), (4, 4))

    def test_asking_for_nothing_gets_the_cap(self):
        self.assertEqual(loop.clamp_runs(0), (4, 4))
        self.assertEqual(loop.clamp_runs(None), (4, 4))
        self.assertEqual(loop.clamp_runs("many"), (4, 4))


class ReferenceTests(unittest.TestCase):
    """"Matches the original frame" names an image already on their canvas."""

    def test_a_loaded_image_is_found_and_resolved(self):
        seen = []

        def resolver(value, kind=""):
            seen.append(value)
            return "W:/frames/original_frame.png"

        self.assertEqual(loop.graph_reference_images(GRAPH, resolver),
                         ["W:/frames/original_frame.png"])
        self.assertEqual(seen, ["original_frame.png [input]"],
                         "the resolver gets the annotated value ComfyUI wrote")

    def test_a_checkpoint_is_not_a_reference(self):
        self.assertEqual(loop.graph_reference_images({"4": GRAPH["4"]}, lambda v, k="": v), [])

    def test_what_the_resolver_cannot_find_is_left_out(self):
        self.assertEqual(loop.graph_reference_images(GRAPH, lambda v, k="": None), [])

    def test_the_same_file_twice_is_listed_once(self):
        twice = {"9": GRAPH["9"], "10": dict(GRAPH["9"])}
        self.assertEqual(len(loop.graph_reference_images(twice, lambda v, k="": "/a.png")), 1)

    def test_the_limit_holds(self):
        many = {str(i): {"class_type": "LoadImage", "inputs": {"image": f"f{i}.png"}}
                for i in range(9)}
        self.assertEqual(len(loop.graph_reference_images(many, lambda v, k="": v, limit=3)), 3)


class VerdictTests(unittest.TestCase):
    """A judge that cannot be read must not be able to declare victory."""

    def test_a_pass_is_a_match(self):
        status, _s, failures = loop.verdict_of(QaResult(path="a.png", passed=True,
                                                        summary="she is in place"))
        self.assertEqual(status, "matched")
        self.assertEqual(failures, [])

    def test_a_fail_carries_its_criteria(self):
        result = QaResult(path="a.png", passed=False, summary="drifted left",
                          checks=[{"criterion": "position matches", "result": "fail",
                                   "note": "she is a third of a frame left"}])
        status, summary, failures = loop.verdict_of(result)
        self.assertEqual(status, "missed")
        self.assertEqual(summary, "drifted left")
        self.assertEqual(len(failures), 1)
        self.assertIn("third of a frame", failures[0])

    def test_an_unreadable_judge_is_neither_pass_nor_fail(self):
        """qa.check_output passes on doubt so it can never condemn the user's work.
        Read here as a pass, that doubt would end the loop in a success nobody
        verified."""
        result = QaResult(path="a.png", passed=True, error="the QA model returned nothing")
        status, summary, failures = loop.verdict_of(result)
        self.assertEqual(status, "unjudged")
        self.assertIn("returned nothing", summary)
        self.assertEqual(failures, [])

    def test_a_fail_with_no_checks_still_says_something(self):
        status, _s, failures = loop.verdict_of(
            QaResult(path="a.png", passed=False, summary="she is on the right"))
        self.assertEqual(status, "missed")
        self.assertEqual(failures, ["she is on the right"])


class RevisionTests(unittest.TestCase):

    def test_the_prompt_file_has_both_sections(self):
        prompts = loop.load_refine_prompts()
        self.assertTrue(prompts.get("system", "").strip())
        self.assertTrue(prompts.get("user", "").strip())

    def test_every_placeholder_is_filled(self):
        messages = loop.revision_messages(
            "she stands where she does in the frame",
            {"node_id": "6", "param": "text", "title": "Positive", "class_type": "CLIPTextEncode"},
            "a woman in a field",
            ["position matches — she drifted left"],
            [{"run": 1, "value": "a distant figure on the horizon", "status": "missed",
              "failures": ["position matches — she was barely in shot"]}])
        body = messages[1]["content"]
        self.assertNotIn("{{", body, "an unfilled placeholder reaches the model as literal")
        self.assertIn("she stands where she does in the frame", body)
        self.assertIn("Positive (#6)", body)
        self.assertIn("a woman in a field", body, "the value being rewritten")
        self.assertIn("drifted left", body, "what this run missed")
        # Deliberately nowhere else in the message: an earlier attempt and its own
        # verdict can only have arrived through {{ATTEMPTS}}. Without them the
        # reviser oscillates, walking back a phrase it already failed on.
        self.assertIn("a distant figure on the horizon", body)
        self.assertIn("barely in shot", body)

    def test_the_reviser_is_shown_what_was_already_tried(self):
        """One rejection at a time makes it oscillate: it walks back the phrase it
        added two runs ago, because nothing told it that had failed too."""
        rendered = loop.render_attempts([
            {"run": 1, "value": "a woman in a field", "status": "missed",
             "failures": ["she drifted left"]},
            {"run": 2, "value": "a woman at the right edge", "status": "missed",
             "failures": ["now she is too far right"]}])
        self.assertIn("a woman in a field", rendered)
        self.assertIn("a woman at the right edge", rendered)
        self.assertIn("too far right", rendered)

    def test_nothing_tried_yet_says_so(self):
        self.assertIn("first run", loop.render_attempts([]))

    def test_a_fenced_reply_is_unwrapped(self):
        self.assertEqual(loop.clean_revision("```\na woman at the left edge\n```"),
                         "a woman at the left edge")

    def test_a_labelled_reply_is_unwrapped(self):
        self.assertEqual(loop.clean_revision('New prompt: "a woman at the left edge"'),
                         "a woman at the left edge")

    def test_a_plain_reply_survives_intact(self):
        text = "a woman standing at the left edge, facing right"
        self.assertEqual(loop.clean_revision(text), text)

    def test_an_apostrophe_is_not_mistaken_for_a_wrapping_quote(self):
        self.assertEqual(loop.clean_revision("a woman's coat"), "a woman's coat")


class AlreadyTriedTests(unittest.TestCase):

    HISTORY = [{"run": 1, "value": "A Woman  in a field"}]

    def test_the_same_value_again(self):
        self.assertTrue(loop.already_tried("a woman in a field", self.HISTORY))

    def test_a_genuinely_new_value(self):
        self.assertFalse(loop.already_tried("a woman at the left edge", self.HISTORY))

    def test_an_empty_revision_counts_as_nothing_new(self):
        self.assertTrue(loop.already_tried("", self.HISTORY))
        self.assertTrue(loop.already_tried("   ", []))


# ── the tool itself ─────────────────────────────────────────────────────────────

def _exec_stub(paths):
    """Stand in for the executor: each run appends the next path it was given."""
    made = iter(list(paths))

    async def _execute(wf, brief, user_message="", verbose=False,
                       collected_paths=None, qa_briefing=None, **kw):
        yield "▶ queued"
        nxt = next(made, None)
        if nxt is not None and collected_paths is not None:
            collected_paths.append(nxt)
        yield "✔ done"

    return _execute


def _judge_stub(results):
    """Stand in for the QA judge: one verdict per run, in order."""
    verdicts = iter(list(results))

    def _check(path, briefing, request="", agent=None):
        return next(verdicts)

    return _check


def _reviser_stub(replies):
    """Stand in for the value reviser."""
    answers = iter(list(replies))

    class _LLM:
        @classmethod
        def from_settings(cls):
            return cls()

        async def chat(self, messages, **kw):
            return next(answers, "")

    return _LLM


class RefineToolTests(unittest.TestCase):
    """``refine_canvas_until`` end to end: the loop, its stops, and what it queues."""

    def _run(self, pipe, *, outputs=(), verdicts=(), revisions=(), cap=4, **kw):
        self.patched = []
        with mock.patch("src.executor.execute_workflow", _exec_stub(outputs)), \
             mock.patch("src.utils.qa.check_output", _judge_stub(verdicts)), \
             mock.patch("src.utils.llm_functions.LLMFunctions", _reviser_stub(revisions)), \
             mock.patch.object(loop, "max_runs_cap", return_value=cap), \
             mock.patch("src.utils.canvas_patch.push", self.patched.append):
            return json.loads(asyncio.run(tools(pipe)["refine_canvas_until"](**kw)))

    def test_it_stops_the_run_it_matches_on(self):
        """Budget for four, match on two, and the last two must never be spent.

        The stubs are deliberately stocked for all four runs: a loop that failed to
        stop would find a verdict and a revision waiting for it, and report four.
        """
        pipe = _pipe()
        out = self._run(
            pipe, cap=4,
            outputs=["W:/out/%d.png" % n for n in (1, 2, 3, 4)],
            verdicts=[QaResult(path="1.png", passed=False, summary="she drifted left",
                               checks=[{"criterion": "position", "result": "fail",
                                        "note": "left of where she should be"}]),
                      QaResult(path="2.png", passed=True, summary="in position"),
                      QaResult(path="3.png", passed=True), QaResult(path="4.png", passed=True)],
            revisions=["a woman standing at the right of frame",
                       "a woman further right again", "a woman right at the edge"],
            condition="her position matches the original frame")
        self.assertEqual(out["status"], "matched")
        self.assertEqual(out["runs"], 2, "it must not keep spending after a match")
        self.assertEqual(len(out["outputs"]), 2)
        self.assertEqual(out["varied"], "6.text")
        self.assertEqual([h["value"] for h in out["history"]],
                         ["a woman standing in a field at dawn",
                          "a woman standing at the right of frame"])

    def test_the_user_watches_it_happen_on_their_own_canvas(self):
        pipe = _pipe()
        self._run(pipe,
                  outputs=["W:/out/1.png", "W:/out/2.png"],
                  verdicts=[QaResult(path="1.png", passed=False, summary="no"),
                            QaResult(path="2.png", passed=True)],
                  revisions=["a woman at the right of frame"],
                  condition="her position matches")
        self.assertEqual([(p["node_id"], p["params"]["text"]) for p in self.patched],
                         [("6", "a woman standing in a field at dawn"),
                          ("6", "a woman at the right of frame")],
                         "each value goes onto the live canvas before it runs")

    def test_the_graph_it_queues_is_theirs_with_one_value_changed(self):
        pipe = _pipe()
        seen = {}

        async def _spy(wf, *a, **kw):
            seen["graph"] = json.loads(Path(wf).read_text(encoding="utf-8"))
            kw["collected_paths"].append("W:/out/1.png")
            yield "done"

        with mock.patch("src.executor.execute_workflow", _spy), \
             mock.patch("src.utils.qa.check_output",
                        _judge_stub([QaResult(path="1.png", passed=True)])), \
             mock.patch.object(loop, "max_runs_cap", return_value=2), \
             mock.patch("src.utils.canvas_patch.push", lambda p: None):
            asyncio.run(tools(pipe)["refine_canvas_until"](
                condition="anything", start_value="a new prompt"))
        self.assertEqual(set(seen["graph"]), set(GRAPH), "no node added or dropped")
        self.assertEqual(seen["graph"]["6"]["inputs"]["text"], "a new prompt")
        self.assertEqual(seen["graph"]["7"]["inputs"]["text"], "blurry, watermark",
                         "the negative is untouched")
        self.assertEqual(seen["graph"]["3"]["inputs"]["seed"], 42,
                         "the seed holds unless vary_seed was asked for")

    def test_vary_seed_rerolls(self):
        pipe = _pipe()
        seeds = []

        async def _spy(wf, *a, **kw):
            graph = json.loads(Path(wf).read_text(encoding="utf-8"))
            seeds.append(graph["3"]["inputs"]["seed"])
            kw["collected_paths"].append("W:/out/x.png")
            yield "done"

        with mock.patch("src.executor.execute_workflow", _spy), \
             mock.patch("src.utils.qa.check_output",
                        _judge_stub([QaResult(path="x.png", passed=True)])), \
             mock.patch.object(loop, "max_runs_cap", return_value=2), \
             mock.patch("src.utils.canvas_patch.push", lambda p: None):
            asyncio.run(tools(pipe)["refine_canvas_until"](
                condition="anything", vary_seed=True))
        self.assertNotEqual(seeds[0], 42)

    def test_a_spent_budget_reports_the_misses_and_keeps_the_original(self):
        pipe = _pipe()
        out = self._run(
            pipe, cap=2,
            outputs=["W:/out/1.png", "W:/out/2.png"],
            verdicts=[QaResult(path="1.png", passed=False, summary="drifted left"),
                      QaResult(path="2.png", passed=False, summary="drifted right")],
            revisions=["a woman at the right of frame"],
            condition="her position matches")
        self.assertEqual(out["status"], "missed")
        self.assertEqual(out["runs"], 2)
        self.assertEqual(out["original_value"], "a woman standing in a field at dawn")
        self.assertEqual(out["value_on_canvas"], "a woman at the right of frame")
        self.assertIn("drifted right", json.dumps(out["history"]))

    def test_an_unreadable_judge_stops_it_without_claiming_a_match(self):
        pipe = _pipe()
        out = self._run(
            pipe, cap=4,
            outputs=["W:/out/1.png", "W:/out/2.png"],
            verdicts=[QaResult(path="1.png", passed=True, error="QA model unreachable")],
            condition="her position matches")
        self.assertEqual(out["status"], "unjudged")
        self.assertEqual(out["runs"], 1, "spending the budget on an unreadable judge is waste")
        self.assertIn("unreachable", out["stopped_because"])
        self.assertIn("never judged", out["message"])

    def test_a_repeated_value_stalls_instead_of_re_rendering_it(self):
        pipe = _pipe()
        out = self._run(
            pipe, cap=4,
            outputs=["W:/out/1.png", "W:/out/2.png"],
            verdicts=[QaResult(path="1.png", passed=False, summary="drifted left")],
            revisions=["  a woman standing in a FIELD at dawn  "],
            condition="her position matches")
        self.assertEqual(out["status"], "stalled")
        self.assertEqual(out["runs"], 1)
        self.assertIn("already tried", out["stopped_because"])

    def test_a_word_from_the_user_ends_it_at_the_next_run(self):
        """An interjection only reaches the model at a tool boundary, and this tool
        can hold the turn for several minutes. Without the check, "stop" typed at
        run 2 of 6 is read four generations too late."""
        pipe = _pipe()
        with mock.patch("src.utils.interject_bus.pending_count", return_value=1):
            out = self._run(
                pipe, cap=4, outputs=["W:/out/1.png"],
                verdicts=[QaResult(path="1.png", passed=True)],
                condition="her position matches")
        self.assertEqual(out["status"], "interrupted")
        self.assertEqual(out["runs"], 0)

    def test_no_output_is_reported_as_nothing_to_judge(self):
        pipe = _pipe()
        out = self._run(pipe, outputs=[], verdicts=[], condition="her position matches")
        self.assertIn("error", out)
        self.assertIn("save_to_output", out["error"])
        self.assertNotIn("status", out)

    def test_an_ambiguous_target_refuses_before_anything_runs(self):
        graph = dict(GRAPH)
        graph["11"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "a second prompt here"}}
        pipe = _pipe(_canvas_base_prompt=graph)
        ran = []
        with mock.patch("src.executor.execute_workflow", lambda *a, **k: ran.append(1)):
            out = json.loads(asyncio.run(
                tools(pipe)["refine_canvas_until"](condition="her position matches")))
        self.assertEqual(ran, [], "nothing may be queued before the target is settled")
        self.assertIn("candidates", out)

    def test_no_condition_is_a_refusal_and_nothing_runs(self):
        ran = []
        with mock.patch("src.executor.execute_workflow", lambda *a, **k: ran.append(1)):
            out = json.loads(asyncio.run(
                tools(_pipe())["refine_canvas_until"](condition="  ")))
        self.assertEqual(ran, [], "there is nothing to loop toward, so nothing may run")
        self.assertIn("judged against", out["error"])

    def test_no_open_graph_is_a_refusal(self):
        pipe = _pipe(_canvas_base_prompt=None)
        out = json.loads(asyncio.run(
            tools(pipe)["refine_canvas_until"](condition="her position matches")))
        self.assertIn("no graph is open", out["error"])

    def test_a_dry_run_will_not_pretend(self):
        """Every iteration exists to be looked at; a stand-in would loop on nothing
        and report a verdict it invented."""
        pipe = _pipe(_dry_run=True)
        out = json.loads(asyncio.run(
            tools(pipe)["refine_canvas_until"](condition="her position matches")))
        self.assertIn("DRY RUN", out["error"])

    def test_asking_past_the_cap_says_where_the_cap_lives(self):
        pipe = _pipe()
        out = self._run(
            pipe, cap=2, max_runs=10,
            outputs=["W:/out/1.png"],
            verdicts=[QaResult(path="1.png", passed=True)],
            condition="her position matches")
        self.assertEqual(out["budget"], 2)
        self.assertIn("refine", out["budget_note"])

    def test_it_tells_the_caller_not_to_queue_the_graph_again(self):
        """The loop already ran the graph. A signal on top of it renders it once
        more at the end of the turn, for nothing."""
        pipe = _pipe()
        out = self._run(pipe, outputs=["W:/out/1.png"],
                        verdicts=[QaResult(path="1.png", passed=True)],
                        condition="her position matches")
        self.assertIn("signal_workflow_ready", out["message"])

    def test_the_produced_files_survive_the_end_of_turn_reset(self):
        pipe = _pipe()
        self._run(pipe, outputs=["W:/out/1.png", "W:/out/2.png"],
                  verdicts=[QaResult(path="1.png", passed=False, summary="no"),
                            QaResult(path="2.png", passed=True)],
                  revisions=["a woman at the right of frame"],
                  condition="her position matches")
        self.assertEqual(pipe._chain_output_paths, ["W:/out/1.png", "W:/out/2.png"])

    def test_the_condition_is_what_each_output_is_judged_against(self):
        pipe = _pipe()
        seen = {}

        def _check(path, briefing, request="", agent=None):
            seen["criteria"] = briefing.criteria
            seen["refs"] = list(briefing.reference_paths)
            return QaResult(path=path, passed=True)

        with mock.patch("src.executor.execute_workflow", _exec_stub(["W:/out/1.png"])), \
             mock.patch("src.utils.qa.check_output", _check), \
             mock.patch.object(loop, "max_runs_cap", return_value=2), \
             mock.patch("src.utils.canvas_patch.push", lambda p: None):
            asyncio.run(tools(pipe)["refine_canvas_until"](
                condition="her position matches the original frame",
                references=[__file__]))
        self.assertEqual(seen["criteria"], "her position matches the original frame")
        self.assertEqual(seen["refs"], [__file__])

    def test_a_reference_that_is_not_on_disk_is_named_not_swallowed(self):
        pipe = _pipe()
        out = self._run(pipe, outputs=["W:/out/1.png"],
                        verdicts=[QaResult(path="1.png", passed=True)],
                        condition="her position matches",
                        references=["W:/nope/missing.png"])
        self.assertEqual(out["references_not_found"], ["W:/nope/missing.png"])


if __name__ == "__main__":
    unittest.main()
