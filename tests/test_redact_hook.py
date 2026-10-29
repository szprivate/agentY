"""Every agent masks secrets in tool results before the model sees them."""
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.utils.redact_hook import RedactSecretsHook, redact_tool_result

SECRET = "hf_" + "Q" * 34


class RedactHook(unittest.TestCase):

    def test_text_and_json_blocks_are_masked(self):
        res = {"toolUseId": "1", "status": "success",
               "content": [{"text": f"HF_TOKEN={SECRET}"}, {"json": {"k": SECRET}},
                           {"image": {"format": "png", "source": {"bytes": b"x"}}}]}
        out = redact_tool_result(res)
        self.assertNotIn(SECRET, str(out))
        self.assertEqual(out["content"][2], res["content"][2])

    def test_a_clean_result_is_the_same_object(self):
        res = {"toolUseId": "1", "status": "success", "content": [{"text": "ok"}]}
        self.assertIs(redact_tool_result(res), res)

    def test_the_hook_replaces_the_events_result(self):
        ev = SimpleNamespace(result={"toolUseId": "1", "status": "success",
                                     "content": [{"text": SECRET}]},
                             tool_use={"name": "run_script"})
        RedactSecretsHook()._on_after(ev)
        self.assertNotIn(SECRET, str(ev.result))

    def test_the_agent_factory_installs_it_last(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "agent.py").read_text(encoding="utf-8")
        at = src.index("RedactSecretsHook())")
        self.assertLess(at, src.index('agent_kwargs["hooks"] = _hooks'))
        self.assertGreater(at, src.index("SpecialistInterjectHook(role))"))


if __name__ == "__main__":
    unittest.main()
