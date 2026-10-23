"""The dcc tools reach the orchestrator: wrapped as shared core tools, held in the `dcc` pack, named in the skill."""

import unittest
from pathlib import Path

import src.tools as T
from src.tools import tool_packs as tp

SKILL = Path(__file__).resolve().parents[1] / "skills" / "model-from-reference" / "SKILL.md"


class Wiring(unittest.TestCase):
    def test_shared_core_and_wrapped(self):
        for name in ("dcc_build", "dcc_silhouette_check"):
            self.assertIn(name, T._SHARED_CORE_TOOLS)
            self.assertEqual(getattr(T, name).tool_name, name)

    def test_in_the_dcc_pack(self):
        self.assertEqual(tp._PACK_OF["dcc_build"], "dcc")
        self.assertEqual(tp._PACK_OF["dcc_silhouette_check"], "dcc")

    def test_skill_allows_and_requires_them(self):
        text = SKILL.read_text(encoding="utf-8")
        head = text.split("---")[1]
        self.assertIn("dcc_build", head)
        self.assertIn("dcc_silhouette_check", head)
        self.assertIn("not passed until `dcc_silhouette_check`", text)


if __name__ == "__main__":
    unittest.main()
