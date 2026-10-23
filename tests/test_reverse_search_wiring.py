"""reverse_image_search reaches the orchestrator: wrapped as a shared core tool, held in the web_more pack."""

import unittest

import src.tools as T
from src.tools import tool_packs as tp


class Wiring(unittest.TestCase):
    def test_shared_core_and_wrapped(self):
        self.assertIn("reverse_image_search", T._SHARED_CORE_TOOLS)
        self.assertEqual(T.reverse_image_search.tool_name, "reverse_image_search")

    def test_in_a_pack_and_the_orchestrator(self):
        self.assertEqual(tp._PACK_OF["reverse_image_search"], "web_more")
        self.assertTrue(any(getattr(t, "tool_name", None) == "reverse_image_search" for t in T.ORCHESTRATOR_TOOLS))


if __name__ == "__main__":
    unittest.main()
