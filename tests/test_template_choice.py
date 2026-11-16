"""A local and a partner-API workflow of the same kind: the user picks first.

prepare_workflow answers ``choose_template`` (nothing built, installed or
downloaded) when the picked template has a twin on the other side of local /
API and the request says neither; the choice is a ``choices`` block that the
panel draws as cards and Slack as numbers (src/utils/choices.py).
"""
import json
import unittest

from src.pipeline import Pipeline
from src.utils import choices

TASKS = [
    {"task": "Image to Video", "models": [
        {"model": "LTX", "uses_api_nodes": False, "member_files": ["video_ltx2_5_i2v.json", "video_ltx2_i2v"]},
        {"model": "Generic", "uses_api_nodes": False, "member_files": ["generic_i2v"]},
    ]},
    {"task": "API / Partner Nodes - Image to Video", "models": [
        {"model": "LTX", "uses_api_nodes": True, "member_files": ["api_ltxv_image_to_video"]},
        {"model": "Kling", "uses_api_nodes": True, "member_files": ["api_kling_i2v"]},
    ]},
]


class _Stub:
    _LOCAL_RE = Pipeline._LOCAL_RE
    _API_RE = Pipeline._API_RE
    _SUPPRESSED_TEMPLATES = frozenset()
    _norm_request = staticmethod(Pipeline._norm_request)
    _task_base = staticmethod(Pipeline._task_base)
    _template_twins = Pipeline._template_twins
    _template_choice = Pipeline._template_choice

    def _load_recipe_tasks(self):
        return TASKS


class TemplateChoice(unittest.TestCase):
    def setUp(self):
        self.p = _Stub()

    def test_a_local_template_with_an_api_twin_asks(self):
        out = self.p._template_choice("video_ltx2_5_i2v", "animate this picture with ltx")
        self.assertEqual(out["status"], "choose_template")
        kinds = [o["kind"] for o in out["options"]]
        self.assertEqual(kinds[0], "local")
        self.assertIn("api", kinds)
        self.assertIn("api_ltxv_image_to_video", [o["template"] for o in out["options"]])
        self.assertNotIn("api_kling_i2v", [o["template"] for o in out["options"]])
        c = choices.find(out["ask"])
        self.assertEqual(len(c["options"]), len(out["options"]))
        self.assertIn("Nothing was built", out["guidance"])

    def test_the_api_side_asks_too(self):
        out = self.p._template_choice("api_ltxv_image_to_video", "animate this picture")
        self.assertEqual(out["options"][0]["kind"], "local")
        self.assertIn("api_ltxv_image_to_video", [o["template"] for o in out["options"]])

    def test_saying_local_or_api_asks_nothing(self):
        self.assertIsNone(self.p._template_choice("video_ltx2_5_i2v", "animate it locally"))
        self.assertIsNone(self.p._template_choice("video_ltx2_5_i2v", "animate it with the API"))
        self.assertIsNone(self.p._template_choice("video_ltx2_5_i2v", "use partner nodes"))

    def test_no_twin_or_generic_asks_nothing(self):
        self.assertIsNone(self.p._template_choice("api_kling_i2v", "animate it"))
        self.assertIsNone(self.p._template_choice("generic_i2v", "animate it"))
        self.assertIsNone(self.p._template_choice("not_in_the_db", "animate it"))

    def test_the_answer_is_json(self):
        json.dumps(self.p._template_choice("video_ltx2_5_i2v", "animate it"))


class Choices(unittest.TestCase):
    ANSWER = ("Both kinds exist.\n\n```choices\nWhich workflow should I use?\n"
              "1. Local - ltx | runs here\n2. API - ltx api | credits\n```\n")

    def test_parse(self):
        c = choices.find(self.ANSWER)
        self.assertEqual(c["question"], "Which workflow should I use?")
        self.assertEqual([o["label"] for o in c["options"]], ["Local - ltx", "API - ltx api"])
        self.assertEqual(c["options"][1]["detail"], "credits")

    def test_one_option_is_no_choice(self):
        self.assertIsNone(choices.find("```choices\nq\n1. only\n```"))

    def test_a_number_is_spelled_out(self):
        for reply in ("2", "2.", " option 2 ", "b", "#2"):
            out = choices.resolve_reply(self.ANSWER, reply)
            self.assertTrue(out.startswith("2. API - ltx api (credits)"), (reply, out))
        self.assertEqual(choices.resolve_reply(self.ANSWER, "3"), "3")
        self.assertEqual(choices.resolve_reply(self.ANSWER, "the local one please"), "the local one please")
        self.assertEqual(choices.resolve_reply("no block here", "2"), "2")

    def test_plain_text_for_slack(self):
        from src.utils.slack_render import to_mrkdwn
        out = to_mrkdwn(self.ANSWER)
        self.assertNotIn("```", out)
        self.assertIn("1. Local - ltx - runs here", out)
        self.assertIn("2. API - ltx api - credits", out)
        self.assertIn("Reply with the number", out)


if __name__ == "__main__":
    unittest.main()
