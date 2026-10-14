"""The small helper calls to Model Studio ask for no thinking.

A Qwen model thinks unless told not to. On a compaction summary that was 171 s
and 8,849 reasoning tokens against 15 s without, so every summary ran into the
60 s timeout and fell back to the digest — 21 of 21 in one conversation.

    python -m unittest tests.test_llm_functions_thinking
"""

import asyncio
import unittest
from unittest import mock

from src.utils import llm_functions as lf


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class _Client:
    sent: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None, timeout=None):
        _Client.sent.append(json)
        return _Resp()


class NoThinking(unittest.TestCase):

    def setUp(self):
        _Client.sent = []
        self.enterContext(mock.patch.object(lf.httpx, "AsyncClient", _Client))
        self.llm = lf.LLMFunctions("dashscope", "qwen3.8-flash", host="", base_url="https://x", api_key="k")

    def test_chat(self):
        asyncio.run(self.llm.chat([{"role": "user", "content": "hi"}]))
        self.assertIs(_Client.sent[0]["enable_thinking"], False)

    def test_vision(self):
        asyncio.run(self.llm.vision_chat("what is it", b"\x89PNG\r\n\x1a\n"))
        self.assertIs(_Client.sent[0]["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
