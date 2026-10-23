"""The services "Add API key" offers: a catalog the settings payload carries."""
import re
import unittest

from scripts import mcp_presets
from src.utils import agentY_server as srv


class ServicesCatalog(unittest.TestCase):

    def setUp(self):
        self.services = srv._services()

    def test_present_with_the_shape_the_page_reads(self):
        self.assertTrue(self.services)
        for s in self.services:
            for k in ("id", "label", "key_env", "url_env", "default_url", "help_url", "hint", "fields"):
                self.assertIn(k, s)
            self.assertTrue(s["label"])

    def test_ids_and_env_names_are_unique(self):
        ids = [s["id"] for s in self.services]
        self.assertEqual(len(ids), len(set(ids)))
        names = [s["key_env"] for s in self.services] + [f["env"] for s in self.services for f in s["fields"]]
        self.assertEqual(len(names), len(set(names)))

    def test_env_names_are_valid_and_known_or_from_a_preset(self):
        preset_keys = {k["key"] for p in mcp_presets.load_catalog() for k in p["env_keys"]}
        for s in self.services:
            for name in [s["key_env"]] + [f["env"] for f in s["fields"]]:
                self.assertRegex(name, r"^[A-Za-z_][A-Za-z0-9_]*$")
                self.assertTrue(name in srv._KNOWN_ENV_KEYS or name in preset_keys, name)
            if s["url_env"]:
                self.assertIn(s["url_env"], srv._KNOWN_ENV_KEYS)
                self.assertTrue(s["default_url"].startswith("https://"))

    def test_the_named_services_are_there(self):
        by = {s["key_env"]: s for s in self.services}
        for key in ("DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                    "COMFYUI_API_KEY", "HF_TOKEN", "WORLDLABS_API_KEY", "SLACK_BOT_TOKEN"):
            self.assertIn(key, by)
        self.assertEqual(by["DASHSCOPE_API_KEY"]["url_env"], "DASHSCOPE_BASE_URL")
        self.assertEqual(by["DASHSCOPE_API_KEY"]["default_url"],
                         "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.assertEqual([f["env"] for f in by["SLACK_BOT_TOKEN"]["fields"]],
                         ["SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"])

    def test_every_preset_env_key_is_offered(self):
        offered = {s["key_env"] for s in self.services} | {f["env"] for s in self.services for f in s["fields"]}
        for p in mcp_presets.load_catalog():
            for k in p["env_keys"]:
                self.assertIn(k["key"], offered)
        labels = {s["label"] for s in self.services}
        self.assertIn("ElevenLabs", labels)

    def test_the_payload_carries_them(self):
        payload = srv.settings_payload(offline=True)
        self.assertEqual(payload["services"], self.services)
        self.assertIsInstance(payload["env_file_keys"], list)


if __name__ == "__main__":
    unittest.main()
