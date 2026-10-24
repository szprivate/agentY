"""LoRA training wiring in agentY: the tools are wrapped and in a pack, the skill is offered to the orchestrator, the
host lends the captioner, and a detached run's events become notifications."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class Wiring(unittest.TestCase):
    NAMES = ("lora_dataset_prepare", "lora_dataset_set_captions", "lora_dataset_review", "train_lora",
             "lora_training_status", "lora_training_stop")

    def test_tools_are_strands_tools_in_the_orchestrator_and_a_pack(self):
        import src.tools as T
        from src.tools import tool_packs
        from strands.tools.decorator import DecoratedFunctionTool
        names = {tool_packs.name_of(t) for t in T.ORCHESTRATOR_TOOLS}
        for n in self.NAMES:
            self.assertIsInstance(getattr(T, n), DecoratedFunctionTool, n)
            self.assertIn(n, names)
            self.assertIn(n, tool_packs.PACKS["lora"][1])
            self.assertIn(n, T._SHARED_CORE_TOOLS)

    def test_skill_is_offered_and_exists(self):
        import src.agent as A
        self.assertIn("lora-training", A._ORCH_SKILL_NAMES)
        self.assertNotIn("lora-training", A._ASSEMBLY_SKILL_NAMES)
        self.assertTrue((Path(A._SKILLS_DIR) / "lora-training" / "SKILL.md").is_file())
        self.assertTrue(any(s.endswith("lora-training") for s in A._orchestrator_skill_sources()))

    def test_captioner_is_lent(self):
        import src.tools  # noqa: F401  (installs)
        from agenty_core.tools.lora_training import get_captioner
        from src.tools import lora_host
        self.assertIs(get_captioner(), lora_host.caption_with_vision)

    def test_events_become_notifications_without_replaying_old_runs(self):
        from agenty_core.tools.lora_training import formats as F
        from src.tools import lora_host
        from src.utils import notify_bus
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            old = runs / "old_v1"
            old.mkdir()
            F.write_json(old / "progress.json", {"status": "done"})
            F.event(old, "finished", "old run finished")
            got = []
            with mock.patch.object(lora_host, "_runs_root", return_value=runs), \
                    mock.patch.object(notify_bus, "emit", side_effect=lambda e: got.append(e)), \
                    mock.patch.dict(lora_host._seen, {}, clear=True):
                lora_host._scan(first=True)
                self.assertEqual(got, [])
                new = runs / "new_v1"
                new.mkdir()
                F.write_json(new / "progress.json", {"status": "running"})
                F.event(new, "started", "training new_v1 started")
                F.event(new, "failed", "out of memory")
                lora_host._scan(first=False)
                lora_host._scan(first=False)                    # nothing twice
        self.assertEqual([g["toast"]["body"] for g in got], ["training new_v1 started", "out of memory"])
        self.assertEqual(got[1]["toast"]["level"], "error")
        self.assertIn("new_v1", got[0]["toast"]["title"])

    def test_caption_with_vision_picks_the_agent_by_media(self):
        from src.tools import lora_host
        with mock.patch("src.tools.image_handling.analyze_image", return_value={"status": "success", "content": [{"text": "a dog"}]}) as ai, \
                mock.patch("src.tools.video_handling.analyze_video", return_value={"status": "success", "content": [{"text": "a run"}]}) as av:
            self.assertEqual(lora_host.caption_with_vision("x.jpg", "q"), "a dog")
            self.assertEqual(lora_host.caption_with_vision("x.mp4", "q"), "a run")
            ai.assert_called_once()
            av.assert_called_once()


if __name__ == "__main__":
    unittest.main()
