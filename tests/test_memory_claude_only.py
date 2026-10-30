"""Memory works on a machine that has only Anthropic credentials.

Anthropic has no embeddings API, and the memory LLM inherits the llm_functions
role - Claude there. mem0 builds its LLM when Memory starts, before any search,
and it used to build an OpenAI client with no key: "Missing credentials", and
every search failed with it. Search needs only the embedder; the LLM is used by
infer=True writes alone.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import memory as mem

_NO_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY",
            "DASHSCOPE_PAYG_API_KEY", "ALIBABA_API_KEY", "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class _FakeEmbedder:
    def __init__(self, dims=768):
        self.dims = dims

    def embed(self, text, action=None):
        v = [float((hash(text) >> i) & 1) + 0.01 for i in range(self.dims)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


class _Env(unittest.TestCase):
    """A Claude-only machine: llm_functions = Claude, no OpenAI/DashScope/Gemini key."""

    preset = "ollama"

    def setUp(self):
        self.store = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.store, True)
        env = {k: "" for k in _NO_KEYS}
        env.update({"MEMORY_EMBEDDER_PRESET": self.preset, "MEMORY_LLM_PROVIDER": "ollama",
                    "MEMORY_LLM_MODEL": "", "MEMORY_STORE_DIR": str(self.store),
                    "MEM0_TELEMETRY": "False", "MEMORY_ENABLED": "true"})
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        for k in _NO_KEYS:
            os.environ.pop(k, None)
        real_build = mem._build_config

        def _build():
            cfg = real_build()
            cfg["history_db_path"] = str(self.store / "history.db")   # not the user's
            return cfg
        self.created: list = []

        def _create(provider, config, vs_config=None):
            self.created.append(provider)
            return _FakeEmbedder(int(config.get("embedding_dims") or 768))
        for target, value in (
            ("src.agent.role_model", mock.Mock(return_value="claude,claude-haiku-4-5")),
            ("src.utils.memory._build_config", _build),
            ("src.utils.memory._ensure_model", mock.Mock()),
            ("src.utils.memory._installed_ollama_chat_model", mock.Mock(return_value="")),
            ("src.utils.memory._anthropic_auto_credentials", mock.Mock(return_value=False)),
            ("mem0.utils.factory.EmbedderFactory.create", _create),
        ):
            p = mock.patch(target, value)
            p.start()
            self.addCleanup(p.stop)
        mem._notified.clear()
        mem.reset_client()
        self.addCleanup(mem.reset_client)

    def _add_and_search(self):
        with mock.patch("builtins.print") as printed:
            added = mem.memory_add("User prefers 1024x1024 portraits.")
            hits = mem.memory_search("User prefers 1024x1024 portraits.")
        errors = [c for c in printed.call_args_list if "error" in str(c) or "credentials" in str(c)]
        self.assertEqual(errors, [])
        self.assertIsNotNone(added)
        self.assertEqual([h["memory"] for h in hits], ["User prefers 1024x1024 portraits."])


class OllamaEmbedderClaudeLlm(_Env):

    def test_no_key_at_all_searches_and_skips_extraction(self):
        with mock.patch("src.utils.memory._ollama_reachable", return_value=True):
            cfg = mem._build_config()
            self.assertEqual(cfg["embedder"]["provider"], "ollama")      # a working Ollama is kept
            self.assertNotEqual(cfg["llm"]["provider"], "openai")
            self.assertTrue(mem._llm_unavailable)
            self._add_and_search()
            # an infer write is stored as written instead of reaching a missing LLM
            with mock.patch("mem0.memory.main.Memory.add", return_value={"results": []}) as add:
                mem.memory_add("long text", infer=True)
            self.assertFalse(add.call_args.kwargs["infer"])

    def test_with_an_anthropic_key_claude_extracts_facts(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        with mock.patch("src.utils.memory._ollama_reachable", return_value=True):
            cfg = mem._build_config()
            self.assertEqual(cfg["llm"], {"provider": "anthropic", "config": {
                "model": "claude-haiku-4-5", "temperature": 0.1, "api_key": "sk-ant-test"}})
            self.assertEqual(mem._llm_unavailable, "")
            self._add_and_search()
            self.assertEqual(type(mem.mem0_client().llm).__name__, "AnthropicLLM")

    def test_a_subscription_auth_token_is_used_as_the_agents_use_it(self):
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "token"
        with mock.patch("src.utils.memory._ollama_reachable", return_value=True):
            cfg = mem._build_config()
        self.assertEqual(cfg["llm"]["provider"], "anthropic")
        self.assertNotIn("api_key", cfg["llm"]["config"])            # the SDK reads the token

    def test_with_ollama_running_an_installed_chat_model_stands_in(self):
        with mock.patch("src.utils.memory._ollama_reachable", return_value=True), \
             mock.patch("src.utils.memory._installed_ollama_chat_model", return_value="qwen3:4b"):
            cfg = mem._build_config()
        self.assertEqual(cfg["llm"]["provider"], "ollama")
        self.assertEqual(cfg["llm"]["config"]["model"], "qwen3:4b")
        self.assertEqual(mem._llm_unavailable, "")


class EmbedderThatCannotWork(_Env):

    def test_ollama_not_running_falls_back_to_the_local_embedder(self):
        with mock.patch("src.utils.memory._ollama_reachable", return_value=False), \
             mock.patch("src.utils.memory._fastembed_installed", return_value=True):
            cfg = mem._build_config()
            self.assertEqual(cfg["embedder"]["provider"], "fastembed")
            self.assertEqual(cfg["embedder"]["config"]["embedding_dims"], 768)
            self.assertEqual(cfg["vector_store"]["config"]["embedding_model_dims"], 768)
            self._add_and_search()


class OpenAIEmbedderWithoutKey(_Env):

    preset = "openai"

    def test_a_missing_key_falls_back_to_local_and_re_embeds(self):
        # an index the OpenAI embedder built (1536) is re-embedded, not mixed
        from mem0.vector_stores.faiss import FAISS
        old = FAISS(collection_name="agenty_memory", path=str(self.store), embedding_model_dims=1536)
        old.insert(vectors=[[0.1] * 1536], payloads=[{"data": "old fact", "user_id": mem.MEMORY_NAMESPACE}],
                   ids=["id-a"])
        with mock.patch("src.utils.memory._ollama_reachable", return_value=False), \
             mock.patch("src.utils.memory._fastembed_installed", return_value=True), \
             mock.patch("src.utils.memory._reembed_store", wraps=mem._reembed_store) as reembed:
            cfg = mem._build_config()
            self.assertEqual(cfg["embedder"]["provider"], "fastembed")
            hits = mem.memory_search("old fact")
        reembed.assert_called_once()
        self.assertEqual([h["memory"] for h in hits], ["old fact"])


if __name__ == "__main__":
    unittest.main()
