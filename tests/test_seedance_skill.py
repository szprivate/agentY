"""Tests for the Seedance 2.0/2.5 prompting fragment + node skill.

Three things here are worth pinning, because all three fail silently:

* the **vendored** copy of ByteDance's `sd25-pe` skill must keep its attribution and
  must NOT keep the section that told the agent to run `npx skills@latest update`
  on first use each session — a prompt that rewrites itself from the network mid-run
  is a prompt nobody can reproduce;
* the hot-path fragment must stay small, since it loads on every Seedance prompt
  (the whole reason `prompting` was split into per-family fragments);
* the node facts in the skill must match the ComfyUI nodes agentY actually drives —
  ByteDance's own docs describe the ModelArk REST API, which differs (mov output,
  `duration: -1` editing), and copying those into the skill would have the agent
  reach for widgets that do not exist.

The last group talks to ComfyUI and skips when it isn't running.

    python -m unittest discover -s tests
"""

import json
import unittest
import urllib.request
from pathlib import Path

from src.agent import _ASSEMBLY_SKILL_NAMES, _skill_sources

SKILLS = Path(__file__).resolve().parent.parent / "skills"
FRAGMENT = SKILLS / "prompting" / "references" / "seedance.md"
VENDORED = SKILLS / "prompting" / "references" / "seedance-2.5-full.md"
NODE_SKILL = SKILLS / "seedance-reference" / "SKILL.md"

COMFY = "http://127.0.0.1:8188"


def _object_info(node: str):
    try:
        with urllib.request.urlopen(f"{COMFY}/object_info/{node}", timeout=6) as r:
            return json.load(r)[node]
    except Exception:
        return None


def _model_options(node_info) -> dict:
    """{option key -> its nested required inputs} for a dynamic-combo model widget."""
    spec = node_info["input"]["required"]["model"][1]
    return {o["key"]: (o.get("inputs") or {}).get("required", {}) for o in spec["options"]}


def _autogrow_slots(inputs: dict, key: str) -> int:
    """How many slots an autogrow reference input offers (names live on the template)."""
    info = inputs[key][1]
    names = info.get("names") or (info.get("template") or {}).get("names") or []
    return len(names)


class VendoredSkillTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(VENDORED.is_file(), "the vendored sd25-pe copy is missing")
        self.text = VENDORED.read_text(encoding="utf-8")
        # agentY's attribution header, then the vendored body. The header quotes the
        # install/update commands on purpose; the ban below applies to the body, which
        # is what the model reads as instructions.
        self.header, _, self.body = self.text.partition("\n---\n")
        self.assertTrue(self.body.strip(), "header/body separator missing")
        # Line-wrap- and blockquote-insensitive: the header is a `>` block, so a
        # sentence can break across lines with a marker in the middle of it.
        self.flat = " ".join(" ".join(l.lstrip("> ") for l in self.text.splitlines()).split())

    def test_the_self_update_instruction_is_gone(self):
        # The original opened with: run `npx --yes skills@latest update sd25-pe -y`
        # the first time the skill is triggered in a session. Vendored content is
        # pinned content; if this ever comes back, the researcher's own prompt
        # becomes whatever a remote served that day.
        for banned in ("npx", "skills@latest", "Self-update", "self-update"):
            self.assertNotIn(banned, self.body, f"{banned!r} must not survive vendoring")
        # ...and the header has to say it was removed, or the diff from upstream is
        # invisible to anyone comparing the two.
        self.assertIn("Self-update", self.header)

    def test_it_credits_its_source(self):
        for expected in ("sd25-pe", "ByteDance", "0.1.1", "Retrieved"):
            self.assertIn(expected, self.text, f"attribution must name {expected}")
        # Licence status is a fact a reader needs, and it was absent upstream.
        self.assertIn("No licence, copyright, or terms accompanied the file", self.text)

    def test_it_says_which_side_wins_against_the_node_contract(self):
        # Its parameter rules are REST-API-shaped; without this the agent has two
        # contradictory sources and no rule for choosing.
        self.assertIn("the node wins", self.flat)
        self.assertIn("this file wins", self.flat)

    def test_the_procedure_itself_survived(self):
        # Verbatim body minus one section — spot-check the parts worth vendoring for.
        for section in ("Select One Primary Task", "Aspect Ratio and Duration Compatibility Gate",
                        "Unused Assets", "Final Checklist"):
            self.assertIn(section, self.text)


class FragmentTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(FRAGMENT.is_file())
        self.text = FRAGMENT.read_text(encoding="utf-8")

    def test_it_stays_a_fragment(self):
        # It loads on every Seedance prompt. The vendored original is ~23k tokens;
        # this must stay in the range of its neighbours (flux/wan/kling are 1-3 KB).
        size = FRAGMENT.stat().st_size
        self.assertLess(size, 8_000, f"the hot-path fragment has grown to {size} B")

    def test_it_points_at_the_deep_reference_by_a_name_that_exists(self):
        self.assertIn(VENDORED.name, self.text)
        self.assertTrue((FRAGMENT.parent / VENDORED.name).is_file())

    def test_it_separates_2_0_from_2_5(self):
        # Getting this wrong is a wasted generation: 2.0 ignores timestamps entirely.
        self.assertIn("4–30 s", self.text)
        self.assertIn("4–15 s", self.text)
        self.assertRegex(self.text, r"(?s)timestamps.*ignored")

    def test_it_carries_the_binding_rule(self):
        self.assertIn("@Image1", self.text)
        self.assertRegex(self.text, r"(?i)upload order")

    def test_it_is_listed_in_the_model_index(self):
        index = (SKILLS / "prompting" / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("references/seedance.md", index)
        self.assertIn("2.5", index)

    def test_parameters_are_kept_out_of_the_prompt(self):
        self.assertRegex(self.text, r"(?i)never write output parameters into the prompt")


class NodeSkillTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NODE_SKILL.is_file())
        self.text = NODE_SKILL.read_text(encoding="utf-8")

    def test_the_builders_can_actually_see_it(self):
        # A skill missing from the allowlist is invisible to every agent.
        self.assertIn("seedance-reference", _ASSEMBLY_SKILL_NAMES)
        self.assertTrue(any(s.endswith("seedance-reference") for s in _skill_sources(_ASSEMBLY_SKILL_NAMES)))

    def test_frontmatter_is_well_formed(self):
        self.assertTrue(self.text.startswith("---\n"))
        fm = self.text.split("---", 2)[1]
        self.assertIn("name: seedance-reference", fm)
        self.assertIn("description:", fm)

    def test_it_names_all_three_nodes(self):
        for node in ("ByteDance2TextToVideoNode", "ByteDance2ReferenceNode",
                     "ByteDance2FirstLastFrameNode"):
            self.assertIn(node, self.text)

    def test_it_does_not_repeat_the_api_only_advice(self):
        # ByteDance recommends mov for edit/extend continuity; the node has no such
        # option, and `duration: -1` / content roles are REST-only concepts.
        self.assertNotIn("duration: -1", self.text)
        self.assertRegex(self.text, r"(?i)`mp4` is the only option")


class LiveNodeContractTest(unittest.TestCase):
    """The claims in the skill, checked against the running ComfyUI."""

    @classmethod
    def setUpClass(cls):
        cls.ref = _object_info("ByteDance2ReferenceNode")
        cls.flf = _object_info("ByteDance2FirstLastFrameNode")
        if cls.ref is None or cls.flf is None:
            raise unittest.SkipTest("ComfyUI is not running on :8188")

    def test_seedance_2_5_is_offered(self):
        self.assertIn("Seedance 2.5", _model_options(self.ref))

    def test_the_documented_durations_are_real(self):
        opts = _model_options(self.ref)
        self.assertEqual((opts["Seedance 2.5"]["duration"][1]["min"],
                          opts["Seedance 2.5"]["duration"][1]["max"]), (4, 30))
        self.assertEqual((opts["Seedance 2.0"]["duration"][1]["min"],
                          opts["Seedance 2.0"]["duration"][1]["max"]), (4, 15))

    def test_the_documented_reference_slot_counts_are_real(self):
        o25, o20 = _model_options(self.ref)["Seedance 2.5"], _model_options(self.ref)["Seedance 2.0"]
        kinds = ("images", "videos", "audios")
        self.assertEqual([_autogrow_slots(o25, f"reference_{k}") for k in kinds], [30, 10, 10])
        self.assertEqual([_autogrow_slots(o20, f"reference_{k}") for k in kinds], [9, 3, 3])

    def test_mp4_really_is_the_only_output_format(self):
        self.assertEqual(_model_options(self.ref)["Seedance 2.5"]["output_format"][1]["options"],
                         ["mp4"])

    def test_video_editing_is_a_2_5_only_boolean(self):
        opts = _model_options(self.ref)
        self.assertIn("video_editing", opts["Seedance 2.5"])
        self.assertNotIn("video_editing", opts["Seedance 2.0"])

    def test_the_first_last_frame_node_has_no_ratio_to_set(self):
        # The skill says the first frame's aspect ratio wins because there is no
        # widget to override it. If one appears, that guidance needs revisiting.
        self.assertNotIn("ratio", _model_options(self.flf)["Seedance 2.5"])


if __name__ == "__main__":
    unittest.main()
