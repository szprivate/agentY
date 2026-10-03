"""The agent the user talks to can fetch a missing model itself.

The HuggingFace tools lived only on the repair and build specialists, which run
inside prepare_workflow. "Download the missing models" is a request users make
directly, and it had no route to them: the orchestrator improvised with run_script
and hand-written scripts, spent a turn hunting for the model-paths config, and
downloaded nothing.

    python -m unittest discover -s tests
"""

import pathlib
import unittest

from src.tools import ORCHESTRATOR_TOOLS

PROMPT = (pathlib.Path(__file__).resolve().parent.parent
          / "config" / "system_prompts" / "system_prompt.orchestrator.md")


def _names(tools):
    names = set()
    for tool in tools:
        spec = getattr(tool, "tool_spec", None)
        names.add(spec["name"] if spec else getattr(tool, "__name__", ""))
    return names


class TheOrchestratorCanDownload(unittest.TestCase):

    def test_it_carries_the_model_tools(self):
        wanted = {"check_model", "find_hf_file", "download_hf_model",
                  # model research (agenty_core.tools.model_research)
                  "hf_search", "hf_repo", "hf_file", "find_local_models",
                  "inspect_workflow_file"}
        self.assertEqual(wanted - _names(ORCHESTRATOR_TOOLS), set())

    def test_it_carries_the_looking_around_tools(self):
        # What run_script was mostly used for in the logs (node source, folders,
        # workflow files, media, frames, web pages) — one tool each.
        wanted = {"get_node_source", "list_files", "find_workflows", "media_info",
                  "video_frames", "read_web_page"}
        self.assertEqual(wanted - _names(ORCHESTRATOR_TOOLS), set())
        prompt = PROMPT.read_text(encoding="utf-8")
        for name in wanted:
            self.assertIn(f"`{name}`", prompt)
        self.assertIn("not for looking things up", prompt)

    def test_its_prompt_no_longer_says_it_cannot(self):
        prompt = PROMPT.read_text(encoding="utf-8")
        self.assertNotIn("model-download tools", prompt)
        self.assertIn("download_hf_model", prompt)

    def test_its_prompt_steers_away_from_scripted_downloads(self):
        prompt = PROMPT.read_text(encoding="utf-8")
        self.assertIn("Never use `run_script` for any of this", prompt)
        self.assertIn("no scripted downloads", prompt)


if __name__ == "__main__":
    unittest.main()
