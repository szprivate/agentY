"""Gemini 3 thought signatures survive a tool round trip (no network)."""
import asyncio
import json
import unittest

from openai.types.chat import ChatCompletionChunk

from src.utils import brain_memory
from src.utils.gemini_compat import DUMMY_SIGNATURE, GeminiCompatModel
from src.utils.compaction import fit_to_window

SIG = "CiQBjz1rX5Zk-signature=="


def _chunk(delta, finish=None):
    return ChatCompletionChunk.model_validate({
        "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gemini-3-flash-preview",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


def _gemini_stream():
    """What the OpenAI-compatible endpoint streams for one function call."""
    return [
        _chunk({"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "find_local_models", "arguments": '{"kind": "checkpoint"}'},
            "extra_content": {"google": {"thought_signature": SIG}}}]}),
        _chunk({}, finish="tool_calls"),
    ]


class _FakeStream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeClient:
    def __init__(self, chunks):
        self.chat = self
        self.completions = self
        self._chunks = chunks
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeStream(self._chunks)


def _run_model(model, messages):
    from strands.event_loop.streaming import process_stream

    async def go():
        out = None
        async for ev in process_stream(model.stream(messages)):
            if "stop" in ev:
                out = ev["stop"]
        return out
    return asyncio.run(go())


class ThoughtSignatureTest(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient(_gemini_stream())
        self.model = GeminiCompatModel(model_id="gemini-3-flash-preview", client_args={})
        self.model._get_client = self._get_client

    def _get_client(self):
        import contextlib

        @contextlib.asynccontextmanager
        async def cm():
            yield self.client
        return cm()

    def _assistant_message(self):
        _, message, *_ = _run_model(self.model, [{"role": "user", "content": [{"text": "hi"}]}])
        return message

    def _next_request_calls(self, history):
        msgs = history + [{"role": "user", "content": [
            {"toolResult": {"toolUseId": "call_1", "status": "success", "content": [{"text": "ok"}]}}]}]
        req = self.model.format_request(msgs)
        return [c for m in req["messages"] for c in m.get("tool_calls", [])]

    def test_signature_is_captured_on_the_tool_use_block(self):
        use = self._assistant_message()["content"][0]["toolUse"]
        self.assertEqual(use["name"], "find_local_models")
        self.assertEqual(use["reasoningSignature"], SIG)

    def test_signature_goes_back_on_the_function_call(self):
        calls = self._next_request_calls([self._assistant_message()])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"], SIG)
        json.dumps(calls)

    def test_signature_survives_save_and_load(self):
        saved = json.loads(json.dumps(brain_memory.serialize_messages([self._assistant_message()])))
        calls = self._next_request_calls(saved)
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"], SIG)

    def test_signature_survives_compaction(self):
        big = "x" * 20000
        history = [
            self._assistant_message(),
            {"role": "user", "content": [{"toolResult": {
                "toolUseId": "call_1", "status": "success", "content": [{"text": big}]}}]},
            {"role": "assistant", "content": [{"text": "done"}]},
            {"role": "user", "content": [{"text": "again"}]},
        ]
        out, _ = fit_to_window(history, 10, keep_recent=1, tool_chars=100, tool_head=50)
        use = out[0]["content"][0]["toolUse"]
        self.assertEqual(use["reasoningSignature"], SIG)

    def test_a_call_without_a_signature_gets_geminis_skip_marker(self):
        history = [{"role": "assistant", "content": [{"toolUse": {
            "toolUseId": "call_1", "name": "t", "input": {}}}]}]
        calls = self._next_request_calls(history)
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"], DUMMY_SIGNATURE)


if __name__ == "__main__":
    unittest.main()
