"""Every Ollama call asks for the configured context size.

Without it Ollama loads the model at the size in the model file (262k for a
current Qwen: 35 GB, partly on the CPU) and reloads it whenever a helper and an
agent take turns on one model.
"""
import asyncio
import os
import unittest
from unittest import mock

from src.utils import llm_functions
from src.utils.llm_functions import LLMFunctions


class _Reply:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "ok"}}


class _Client:
    sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, timeout=None):
        _Client.sent.append(json)
        return _Reply()


class OllamaContextTest(unittest.TestCase):
    def setUp(self):
        _Client.sent = []
        self.fn = LLMFunctions.__new__(LLMFunctions)
        self.fn.model, self.fn.host, self.fn.provider = "qwen3.6:27b", "http://127.0.0.1:11434", "ollama"

    def _settings(self, num_ctx):
        return mock.patch.object(llm_functions, "_load_settings",
                                 return_value={"llm": {"ollama": {"num_ctx": num_ctx}}})

    def test_a_chat_asks_for_the_configured_context(self):
        with self._settings(65536), mock.patch.object(llm_functions.httpx, "AsyncClient", _Client), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_NUM_CTX", None)
            asyncio.run(self.fn._ollama_chat([{"role": "user", "content": "hi"}], json_format=False))
        self.assertEqual(_Client.sent[0]["options"], {"num_ctx": 65536})

    def test_a_picture_call_asks_for_it_too(self):
        with self._settings(40000), mock.patch.object(llm_functions.httpx, "AsyncClient", _Client):
            os.environ.pop("OLLAMA_NUM_CTX", None)
            asyncio.run(self.fn._ollama_vision("what is this", [b"x"], system=""))
        self.assertEqual(_Client.sent[0]["options"], {"num_ctx": 40000})

    def test_the_environment_wins_and_nonsense_falls_back(self):
        with self._settings("many"), mock.patch.dict(os.environ, {"OLLAMA_NUM_CTX": "8192"}):
            self.assertEqual(LLMFunctions._ollama_options(), {"num_ctx": 8192})
        # Nothing usable in the setting: what the machine works out (src.utils.ollama_window).
        from src.utils import ollama_window
        with self._settings("many"), mock.patch.object(ollama_window, "auto", return_value={"window": 98304, "why": ""}):
            os.environ.pop("OLLAMA_NUM_CTX", None)
            self.assertEqual(LLMFunctions._ollama_options(), {"num_ctx": 98304})
        with self._settings(0), mock.patch.object(ollama_window, "auto", return_value={"window": 49152, "why": ""}):
            os.environ.pop("OLLAMA_NUM_CTX", None)
            self.assertEqual(LLMFunctions._ollama_options(), {"num_ctx": 49152})


if __name__ == "__main__":
    unittest.main()
