"""Long-term-memory writes can be paused for a while.

A benchmark night wrote 244 entries into the user's memory, and the agent
recalled them in later cases - notes about the benchmark's own checks, and a
wrong "fix" that failed a case four times. A run now pauses the writes; the
pause expires on its own, so a run that dies cannot leave memory off.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import memory as M


class FakeMem0:
    def __init__(self):
        self.added = []

    def add(self, content, user_id=None, metadata=None, infer=False):
        self.added.append(content)
        return {"results": [{"id": "1", "event": "ADD"}]}


class MemoryPauseTest(unittest.TestCase):

    def setUp(self):
        self.fake = FakeMem0()
        for name, value in (("mem0_client", lambda: self.fake), ("_is_enabled", lambda: True)):
            patcher = mock.patch.object(M, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        M.pause_writes(0)
        self.addCleanup(M.pause_writes, 0)

    def test_a_paused_write_stores_nothing_and_a_resumed_one_does(self):
        self.assertGreater(M.pause_writes(60), 59)
        self.assertIsNone(M.memory_add("a lesson from a benchmark turn"))
        self.assertEqual(self.fake.added, [])
        self.assertEqual(M.pause_writes(0), 0)
        self.assertIsNotNone(M.memory_add("something the user said to remember"))
        self.assertEqual(self.fake.added, ["something the user said to remember"])

    def test_the_pause_expires_on_its_own(self):
        M.pause_writes(60)
        with mock.patch.object(M.time, "time", return_value=M.time.time() + 61):
            self.assertEqual(M.writes_paused(), 0)
            self.assertIsNotNone(M.memory_add("written after the run died"))

    def test_it_cannot_be_paused_for_days(self):
        self.assertLessEqual(M.pause_writes(10 ** 9), 6 * 3600)

    def test_the_agents_own_save_says_it_was_not_saved(self):
        from src.tools import memory_tools as T
        save = getattr(T.memory_write, "func", getattr(T.memory_write, "_tool_func", T.memory_write))
        M.pause_writes(60)
        out = save("remember this")
        self.assertIn("paused", out)
        self.assertIn("NOT saved", out)
        self.assertEqual(self.fake.added, [])

    def test_the_route_sets_and_reports_it(self):
        from src.utils import agentY_server as S
        from route_client import authorised_client
        client = authorised_client(S._build_app())
        self.assertFalse(client.get("/agentY/memory/writes").get_json()["paused"])
        out = client.post("/agentY/memory/writes", json={"pause_seconds": 120}).get_json()
        self.assertTrue(out["paused"])
        self.assertGreater(out["seconds_left"], 100)
        self.assertEqual(client.post("/agentY/memory/writes", json={"pause_seconds": "x"}).status_code, 400)
        self.assertFalse(client.post("/agentY/memory/writes", json={"pause_seconds": 0}).get_json()["paused"])


if __name__ == "__main__":
    unittest.main()
