"""Custom OpenAI-compatible LLM providers: registry, factory, model menus, vision."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import agentY_server as srv
from src.utils import custom_providers as cp
from src.utils import settings as st

ENTRY = {"label": "Mercury", "key_env": "MERCURY_API_KEY", "url_env": "MERCURY_BASE_URL", "vision": True}


class Isolated(unittest.TestCase):
    """settings.local.json in a temp dir; the environment restored afterwards."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(st, "_LOCAL_PATH", Path(self.tmp.name) / "settings.local.json")
        p.start()
        self.addCleanup(p.stop)
        st.invalidate()
        self.addCleanup(st.invalidate)
        e = mock.patch.dict(os.environ, {}, clear=False)
        e.start()
        self.addCleanup(e.stop)
        for k in ("MERCURY_API_KEY", "MERCURY_BASE_URL"):
            os.environ.pop(k, None)
        srv._MODEL_CACHE.clear()
        self.addCleanup(srv._MODEL_CACHE.clear)


class Registry(Isolated):

    def test_save_and_load(self):
        out = srv._save_custom_providers({"add": {"mercury": ENTRY}})
        self.assertEqual(out["custom_providers_added"], ["mercury"])
        self.assertEqual(cp.load()["mercury"], ENTRY)
        self.assertIn("custom_providers", st.load_local()["llm"])

    def test_settings_save_reports_it(self):
        res = srv.settings_save({"custom_providers": {"add": {"mercury": ENTRY}}})
        self.assertIn("llm.custom_providers", res["settings_updated"])

    def test_remove_keeps_the_others(self):
        srv._save_custom_providers({"add": {"mercury": ENTRY, "other": dict(ENTRY, label="O", key_env="O_API_KEY", url_env="O_BASE_URL")}})
        out = srv._save_custom_providers({"remove": ["mercury"]})
        self.assertEqual(out["custom_providers_removed"], ["mercury"])
        self.assertEqual(list(cp.load()), ["other"])

    def test_clashes_with_built_in_names_are_rejected(self):
        for name in ("anthropic", "claude", "dashscope", "qwen", "alibaba", "modelstudio", "openai", "gpt",
                     "google", "gemini", "ollama", "bedrock"):
            with self.subTest(name):
                out = srv._save_custom_providers({"add": {name: ENTRY}})
                self.assertIn(name, out["custom_providers_rejected"])
                self.assertNotIn(name, cp.load())

    def test_bad_ids_and_env_names_are_rejected(self):
        self.assertTrue(cp.validate("9x", ENTRY))
        self.assertTrue(cp.validate("mercury", dict(ENTRY, key_env="not a name")))
        self.assertEqual(cp.validate("mercury", ENTRY), "")

    def test_slug(self):
        self.assertEqual(cp.slug(" Open-Router.ai "), "open_router_ai")

    def test_env_changes_of_its_variables_rebuild_agents(self):
        from src.utils import model_reload
        self.assertFalse(model_reload.env_affects_clients(["MERCURY_API_KEY"]))
        srv._save_custom_providers({"add": {"mercury": ENTRY}})
        self.assertTrue(model_reload.env_affects_clients(["MERCURY_API_KEY"]))


class Factory(Isolated):

    def build(self, llm="mercury", model="m-1"):
        from src import agent
        return agent._build_model(role="t", llm=llm, system_prompt="x", dashscope_model=model)

    def setUp(self):
        super().setUp()
        srv._save_custom_providers({"add": {"mercury": ENTRY}})

    def test_missing_url_names_the_variable(self):
        os.environ["MERCURY_API_KEY"] = "k"
        with self.assertRaisesRegex(RuntimeError, "MERCURY_BASE_URL"):
            self.build()

    def test_missing_key_names_the_variable(self):
        os.environ["MERCURY_BASE_URL"] = "http://x/v1"
        with self.assertRaisesRegex(RuntimeError, "MERCURY_API_KEY"):
            self.build()

    def test_builds_an_openai_compatible_model(self):
        os.environ["MERCURY_API_KEY"] = "k"
        os.environ["MERCURY_BASE_URL"] = "http://x/v1"
        model, model_id = self.build()
        self.assertEqual(model_id, "m-1")
        cfg = model.get_config()
        self.assertEqual(cfg["model_id"], "m-1")
        self.assertNotIn("extra_body", cfg.get("params", {}))
        self.assertEqual(model.client_args["base_url"], "http://x/v1")


class Menus(Isolated):

    def setUp(self):
        super().setUp()
        srv._save_custom_providers({"add": {"mercury": ENTRY}})
        os.environ["MERCURY_API_KEY"] = "k"
        os.environ["MERCURY_BASE_URL"] = "http://x/v1/"

    def _group(self, get):
        with mock.patch("requests.get", get):
            return srv._list_models_now()["Mercury"]

    def test_models_come_from_get_models(self):
        seen = {}

        def get(url, headers=None, timeout=None, **kw):
            if url.startswith("http://x/"):
                seen.update(url=url, headers=headers, timeout=timeout)
            r = mock.Mock()
            r.raise_for_status.return_value = None
            r.json.return_value = {"data": [{"id": "b"}, {"id": "a"}]}
            return r

        self.assertEqual(self._group(get), [["mercury,a", "a"], ["mercury,b", "b"]])
        self.assertEqual(seen["url"], "http://x/v1/models")
        self.assertEqual(seen["headers"], {"Authorization": "Bearer k"})
        self.assertLessEqual(seen["timeout"], 8)

    def test_a_failure_still_offers_the_provider(self):
        def get(*a, **kw):
            raise ConnectionError("down")
        self.assertEqual(self._group(get), [])

    def test_the_test_button(self):
        def get(*a, **kw):
            r = mock.Mock()
            r.raise_for_status.return_value = None
            r.json.return_value = {"data": [{"id": "a"}]}
            return r
        with mock.patch("requests.get", get):
            self.assertEqual(srv.test_custom_provider("mercury"), {"ok": True, "count": 1, "error": ""})
        with mock.patch("requests.get", side_effect=ConnectionError("down")):
            res = srv.test_custom_provider("mercury")
        self.assertFalse(res["ok"])
        self.assertIn("down", res["error"])
        os.environ.pop("MERCURY_BASE_URL")
        self.assertIn("MERCURY_BASE_URL", srv.test_custom_provider("mercury")["error"])

    def test_the_payload_lists_them(self):
        ids = [p["id"] for p in srv._custom_provider_list()]
        self.assertEqual(ids, ["mercury"])


class Vision(Isolated):

    def _supports(self, spec):
        with mock.patch("src.agent.role_model", return_value=spec):
            return srv._orchestrator_supports_vision()

    def test_vision_flag_decides(self):
        srv._save_custom_providers({"add": {"mercury": ENTRY}})
        self.assertTrue(self._supports("mercury,plain-model"))
        srv._save_custom_providers({"add": {"mercury": dict(ENTRY, vision=False)}})
        self.assertFalse(self._supports("mercury,plain-model"))


if __name__ == "__main__":
    unittest.main()
