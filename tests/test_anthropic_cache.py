"""The Anthropic request is marked for prompt caching: tools, the stable system
prompt and the conversation so far, in at most four breakpoints, with a prefix
that stays the same from one call of a tool loop to the next."""

import asyncio
import copy
import json
import os
import unittest
from unittest import mock

from src.utils import anthropic_cache as ac

GUIDES = ac.TURN_GUIDES_HEAD + "Canvas nodes: do this, this turn."
SKILLS = "<available_skills><skill>writing</skill></available_skills>"
MAIN = "You are the orchestrator."
TOOLS = [{"name": n, "description": n, "inputSchema": {"json": {"type": "object", "properties": {}}}}
         for n in ("read", "write", "run")]


def _model(ttl=None):
    from src.agent import AnthropicModel
    m = AnthropicModel(model_id="claude-haiku-4-5", max_tokens=1024, client_args={"api_key": "x"})
    if ttl:
        m._cache_ttl = ttl
    return m


def _prompt(blocks):
    from src.agent import _CallPrompt
    p = _CallPrompt("\n".join(b["text"] for b in blocks))
    p.blocks = blocks
    return p


def _user(text):
    return {"role": "user", "content": [{"text": text}]}


def _tool_round(i, n_calls=1):
    """One round of a tool loop: the model's tool calls and their results."""
    uses = [{"toolUse": {"toolUseId": f"t{i}_{k}", "name": "read", "input": {"k": k}}} for k in range(n_calls)]
    results = [{"toolResult": {"toolUseId": f"t{i}_{k}", "status": "success",
                               "content": [{"text": f"result {i}.{k}"}]}} for k in range(n_calls)]
    return [{"role": "assistant", "content": [{"text": "checking"}] + uses},
            {"role": "user", "content": results}]


def _marks(req):
    """Where the breakpoints are: ("tools", i) / ("system", i) / ("messages", i, j)."""
    out = [("tools", i) for i, t in enumerate(req.get("tools") or []) if t.get("cache_control")]
    out += [("system", i) for i, b in enumerate(req.get("system") or []) if b.get("cache_control")]
    for i, m in enumerate(req["messages"]):
        out += [("messages", i, j) for j, b in enumerate(m["content"]) if b.get("cache_control")]
    return out


def _strip(obj):
    """The request without its cache markers: what the cache key is made of."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


class TheRequest(unittest.TestCase):

    def test_tools_stable_prompt_and_the_conversation_each_get_a_breakpoint(self):
        msgs = [_user("make a cat")] + _tool_round(1)
        req = _model().format_request(msgs, TOOLS, _prompt([{"text": MAIN}, {"text": GUIDES}, {"text": SKILLS}]))
        marks = _marks(req)
        self.assertLessEqual(len(marks), 4)
        self.assertIn(("tools", 2), marks)
        # The turn's guidance goes last, behind the breakpoint: a new turn with
        # other guidance still reads the prompt and the skills list.
        self.assertEqual([b["text"] for b in req["system"]], [MAIN, SKILLS, GUIDES])
        self.assertIn(("system", 1), marks)
        self.assertNotIn(("system", 2), marks)
        self.assertIn(("messages", 2, 0), marks)       # the last message
        self.assertIn(("messages", 0, 0), marks)       # where the previous call wrote

    def test_never_more_than_four(self):
        msgs = [_user("a")] + _tool_round(1) + _tool_round(2)
        msgs[0]["content"].append({"cachePoint": {"type": "default"}})   # one of Strands' own
        req = _model().format_request(msgs, TOOLS, _prompt([{"text": MAIN}, {"text": SKILLS}]))
        self.assertEqual(len(_marks(req)), 4)

    def test_two_calls_of_a_loop_share_the_prefix_and_the_second_reads_the_first(self):
        model = _model("1h")
        prompt = _prompt([{"text": MAIN}, {"text": GUIDES}, {"text": SKILLS}])
        msgs = [_user("make a cat")] + _tool_round(1, n_calls=3)
        first = model.format_request(copy.deepcopy(msgs), TOOLS, prompt)
        msgs += _tool_round(2, n_calls=12)        # more blocks than the API looks back
        second = model.format_request(copy.deepcopy(msgs), TOOLS, prompt)
        n = len(first["messages"])
        self.assertEqual(json.dumps(_strip([first["tools"], first["system"], first["messages"]])),
                         json.dumps(_strip([second["tools"], second["system"], second["messages"][:n]])))
        # The second call's breakpoint sits exactly where the first wrote its last.
        last_of_first = [m for m in _marks(first) if m[0] == "messages" and m[1] == n - 1]
        self.assertTrue(last_of_first)
        self.assertIn(last_of_first[0], _marks(second))
        self.assertIn(("messages", len(msgs) - 1, 11), _marks(second))
        self.assertLessEqual(len(_marks(second)), 4)

    def test_the_static_prefix_takes_the_hour_and_the_conversation_five_minutes(self):
        req = _model("1h").format_request([_user("a")] + _tool_round(1), TOOLS, _prompt([{"text": MAIN}]))
        self.assertEqual(req["tools"][-1]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(req["system"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(req["messages"][-1]["content"][-1]["cache_control"], {"type": "ephemeral"})

    def test_the_agents_history_is_not_marked(self):
        msgs = [_user("a")] + _tool_round(1)
        before = copy.deepcopy(msgs)
        _model().format_request(msgs, TOOLS, _prompt([{"text": MAIN}]))
        self.assertEqual(msgs, before)

    def test_no_breakpoint_on_a_thinking_block_or_empty_text(self):
        msgs = [_user("a"), {"role": "assistant", "content": [
            {"text": "answer"}, {"reasoningContent": {"reasoningText": {"text": "hm", "signature": "s"}}}]},
            {"role": "user", "content": [{"text": "next"}, {"text": " "}]}]
        req = _model().format_request(msgs, None, _prompt([{"text": MAIN}]))
        self.assertIn(("messages", 2, 0), _marks(req))
        self.assertNotIn(("messages", 2, 1), _marks(req))
        self.assertNotIn(("messages", 1, 1), _marks(req))


class TheStream(unittest.TestCase):

    def test_the_prompt_reaches_the_request_as_its_blocks(self):
        sent = {}

        class Stop(Exception):
            pass

        class _Ctx:
            def __init__(self, **req):
                sent.update(req)

            async def __aenter__(self):
                raise Stop

            async def __aexit__(self, *a):
                return False

        model = _model()
        model.client = mock.Mock()
        model.client.messages.stream = lambda **req: _Ctx(**req)
        blocks = [{"text": MAIN}, {"text": GUIDES}, {"text": SKILLS}]

        async def run():
            async for _ in model.stream([_user("a")], TOOLS, "\n".join(b["text"] for b in blocks),
                                        system_prompt_content=blocks):
                pass

        with self.assertRaises(Stop):
            asyncio.run(run())
        self.assertEqual([b["text"] for b in sent["system"]], [MAIN, SKILLS, GUIDES])


class TheBuiltModel(unittest.TestCase):

    def test_takes_the_ttl_from_the_settings(self):
        from src import agent as A
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}):
            os.environ.pop("ANTHROPIC_CACHE_TTL", None)
            model, _ = A._build_model(role="orchestrator", llm="claude", system_prompt="s",
                                      anthropic_model="claude-haiku-4-5")
            self.assertEqual(model._cache_ttl, "1h")
            with mock.patch.dict(os.environ, {"ANTHROPIC_CACHE_TTL": "5m"}):
                model, _ = A._build_model(role="orchestrator", llm="claude", system_prompt="s",
                                          anthropic_model="claude-haiku-4-5")
            self.assertEqual(model._cache_ttl, "5m")


if __name__ == "__main__":
    unittest.main()
