"""Code the agent makes is sent to the viewer's Agent code panel.

Tool arguments are read at the one hook every agent has; the reply's fenced
blocks are read from the turn's event stream. The viewer transport is a stub.

    python -m unittest tests.test_code_sink
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import code_sink as cs

PY = "import os\n\ndef run(x):\n    return os.path.join(x, 'a')\n"
TSX = ("import React from 'react';\nimport {AbsoluteFill} from 'remotion';\n"
       "export const Main = () => <AbsoluteFill><h1>Hi</h1></AbsoluteFill>;\n")


class Fresh(unittest.TestCase):
    def setUp(self):
        cs._sent.clear()
        cs._texts.clear()
        cs._off_until = 0.0
        self.posts = []
        client = mock.Mock()
        client.post.side_effect = lambda path, json_data=None: (self.posts.append((path, json_data)), {"ok": True})[1]
        for p in (mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client),
                  mock.patch.object(cs, "_enabled", return_value=True)):
            p.start()
            self.addCleanup(p.stop)

    def sent(self):
        return [b for _, b in self.posts]

    def send_sync(self, snip, **ctx):
        return cs.send(snip, wait=True, **ctx)


class ToolArguments(unittest.TestCase):
    def test_run_python_node(self):
        s = cs.snippets_for_tool("run_python_node", {"code": PY, "title": "paths"})
        self.assertEqual([(x["title"], x["language"], x["source"]) for x in s],
                         [("paths", "python", "run_python_node")])

    def test_write_text_file_only_for_code_extensions(self):
        self.assertEqual(cs.snippets_for_tool("write_text_file", {"path": "D:/x/notes.txt", "content": PY}), [])
        s = cs.snippets_for_tool("write_text_file", {"path": "D:/x/nodes.py", "content": PY})
        self.assertEqual((s[0]["title"], s[0]["language"]), ("nodes.py", "python"))

    def test_run_script(self):
        s = cs.snippets_for_tool("run_script", {"command": 'python -c "print(1)\nprint(2)"'})
        self.assertEqual((s[0]["language"], s[0]["code"]), ("python", "print(1)\nprint(2)"))
        self.assertEqual(cs.snippets_for_tool("run_script", {"command": "dir"}), [])
        self.assertEqual(cs.snippets_for_tool("run_script", {"command": "a\nb"})[0]["language"], "shell")

    def test_remotion_composition_is_tsx(self):
        s = cs.snippets_for_tool("remotion__update_composition", {"id": "Main", "code": TSX})
        self.assertEqual(len(s), 1)
        self.assertEqual((s[0]["language"], s[0]["source"]), ("tsx", "remotion__update_composition"))

    def test_mcp_arguments_that_are_not_code_are_left_alone(self):
        self.assertEqual(cs.snippets_for_tool("magnific__generate", {"prompt": "a red fox\nin snow, cinematic"}), [])
        self.assertEqual(cs.snippets_for_tool("x__y", {"text": "hello there"}), [])
        self.assertEqual(cs.snippets_for_tool("view_image", {"code": PY}), [])


class Fences(unittest.TestCase):
    def test_extract(self):
        text = "Here:\n```py\nprint(1)\n```\nand\n```json\n{\"a\": 1}\n```\n```\nplain\n```\n"
        self.assertEqual(cs.extract_fences(text),
                         [("python", "print(1)"), ("json", '{"a": 1}'), ("text", "plain")])

    def test_open_block_is_not_extracted(self):
        self.assertEqual(cs.extract_fences("```python\nprint(1)\n"), [])

    def test_guess_untagged(self):
        self.assertEqual(cs.guess_language(PY), "python")
        self.assertEqual(cs.guess_language(TSX), "tsx")


class Delivery(Fresh):
    def test_dedupes_identical_snippets(self):
        s = {"title": "a", "language": "python", "code": PY, "source": "run_python_node"}
        self.assertTrue(self.send_sync(s, conversation="c1", turn="t1"))
        self.assertFalse(self.send_sync(dict(s, title="b"), conversation="c1", turn="t2"))
        self.assertEqual(len(self.posts), 1)
        path, body = self.posts[0]
        self.assertEqual(path, "/bepic/code")
        self.assertEqual((body["source"], body["conversation"], body["turn"]), ("run_python_node", "c1", "t1"))

    def test_failed_delivery_may_be_retried(self):
        client = mock.Mock()
        client.post.side_effect = RuntimeError("boom")
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
            self.assertFalse(self.send_sync({"language": "python", "code": PY}))
        self.assertTrue(self.send_sync({"language": "python", "code": PY}))

    def test_old_viewer_without_route_is_left_alone_for_a_while(self):
        client = mock.Mock()
        client.post.side_effect = RuntimeError("404 Not Found")
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
            self.send_sync({"language": "python", "code": PY})
            self.assertFalse(self.send_sync({"language": "python", "code": PY + "#"}))
        self.assertEqual(client.post.call_count, 1)

    def test_hook_reads_arguments_before_the_call(self):
        with mock.patch.object(cs.threading, "Thread") as th:
            th.side_effect = lambda target, **kw: SimpleNamespace(start=target)
            event = SimpleNamespace(tool_use={"name": "write_text_file",
                                              "input": {"path": "n.py", "content": PY}})
            cs.CodeSinkHookProvider()._before(event)
        self.assertEqual(self.sent()[0]["title"], "n.py")

    def test_reply_fences_are_sent_once_when_they_close(self):
        turn = SimpleNamespace(request_id="req12345678", thread_id="thr")
        with mock.patch.object(cs.threading, "Thread") as th:
            th.side_effect = lambda target, **kw: SimpleNamespace(start=target)
            for chunk in ("Try this:\n```pyth", "on\nprint(1)\n", "```\nand more", " text", "\n"):
                cs.observe_turn({"type": "text", "data": chunk}, turn)
            self.assertEqual(len(self.sent()), 1)
            cs.observe_turn({"type": "done"}, turn)
        body = self.sent()[0]
        self.assertEqual((body["language"], body["code"], body["source"], body["conversation"], body["turn"]),
                         ("python", "print(1)", "reply", "thr", "req12345"))

    def test_unclosed_block_is_sent_at_the_end(self):
        turn = SimpleNamespace(request_id="r", thread_id="t")
        with mock.patch.object(cs.threading, "Thread") as th:
            th.side_effect = lambda target, **kw: SimpleNamespace(start=target)
            cs.observe_turn({"type": "text", "data": "```python\nx = 1\n"}, turn)
            self.assertEqual(self.sent(), [])
            cs.observe_turn({"type": "done"}, turn)
        self.assertEqual(self.sent()[0]["code"], "x = 1")


if __name__ == "__main__":
    unittest.main()
