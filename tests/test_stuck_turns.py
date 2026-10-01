"""The agent "stuck after 'Orchestrator finished'", until the host is restarted.

The turn log showed the chain. A turn ended with a stream still open; closing
the turn's event loop waited for that stream's cleanup forever (seven turns end
at ``post:close_loop`` and never END); so whatever the stream held was never
let go — a researcher lease slot, an agent's invocation lock — and the NEXT turn
waited on it, printing nothing, the console still showing the previous turn's
"Orchestrator finished". Stop ended that waiting turn but not the leak; only a
restart did.

    python -m unittest discover -s tests
"""

import asyncio
import importlib
import os
import threading
import time
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub
from src import pipeline as P
from src.pipeline import Pipeline


def _finished_task():
    async def nothing():
        return None
    loop = asyncio.new_event_loop()
    try:
        t = loop.create_task(nothing())
        loop.run_until_complete(t)
    finally:
        loop.close()
    return t


class FakeAgent:
    def __init__(self):
        self._invocation_lock = threading.Lock()
        self.messages = []


class LeaseTest(unittest.TestCase):

    def _lease(self, pipe):
        async def take():
            async with Pipeline._researcher_lease(pipe) as agent:
                return agent
        return asyncio.run(asyncio.wait_for(take(), 5))

    def test_slots_left_by_finished_turns_are_freed(self):
        """Four leaked slots made every later prepare_workflow wait forever."""
        primary = FakeAgent()
        dead = [[FakeAgent(), _finished_task()] for _ in range(Pipeline._MAX_PARALLEL_RESEARCH)]
        for s in dead:
            s[0]._invocation_lock.acquire()          # their streams were never closed
        pipe = pipeline_stub(_researcher=primary, _researchers_busy=list(dead),
                             _researcher_spares=[], _verbose=False,
                             _MAX_PARALLEL_RESEARCH=Pipeline._MAX_PARALLEL_RESEARCH)
        with mock.patch.object(P, "_push_progress", lambda *a, **k: None):
            got = self._lease(pipe)
        self.assertIs(got, primary)
        self.assertEqual(pipe._researchers_busy, [], "the lease returned its own slot too")
        self.assertFalse(any(s[0]._invocation_lock.locked() for s in dead),
                         "a freed slot's agent must not stay marked as running")

    def test_live_slots_are_respected(self):
        """A slot whose turn is still running is not stolen: the lease waits."""
        async def scenario():
            primary = FakeAgent()
            blocker = asyncio.Event()

            async def holder():
                await blocker.wait()
            busy = [[FakeAgent(), asyncio.get_running_loop().create_task(holder())]
                    for _ in range(Pipeline._MAX_PARALLEL_RESEARCH)]
            pipe = pipeline_stub(_researcher=primary, _researchers_busy=busy,
                                 _researcher_spares=[], _verbose=False,
                             _MAX_PARALLEL_RESEARCH=Pipeline._MAX_PARALLEL_RESEARCH)

            async def take():
                async with Pipeline._researcher_lease(pipe):
                    return "got"
            waiting = asyncio.ensure_future(take())
            await asyncio.sleep(0.6)
            self.assertFalse(waiting.done(), "took a slot that was still in use")
            blocker.set()
            await asyncio.sleep(0.6)                 # holders finish -> slots dead -> freed
            self.assertEqual(await asyncio.wait_for(waiting, 3), "got")
        with mock.patch.object(P, "_push_progress", lambda *a, **k: None):
            asyncio.run(scenario())


class StaleLockTest(unittest.TestCase):

    def test_every_held_lock_is_released_at_a_turn_with_none_in_flight(self):
        orch, info, spare = FakeAgent(), FakeAgent(), FakeAgent()
        for a in (orch, info, spare):
            a._invocation_lock.acquire()
        pipe = pipeline_stub(_orchestrator_agent=orch, _info_agent=info,
                             _researcher_spares=[(None, spare)], _researchers_busy=[["x", None]])
        n = Pipeline.release_stale_locks(pipe)
        self.assertEqual(n, 3)
        self.assertFalse(orch._invocation_lock.locked())
        self.assertEqual(pipe._researchers_busy, [])

    def test_the_server_only_does_it_with_no_turn_in_flight(self):
        from src.utils import agentY_server as S
        src = open(S.__file__, encoding="utf-8").read()
        self.assertIn("if not _turn_running():\n        try:\n            n = pipeline.release_stale_locks()", src)


class CloseLoopTest(unittest.TestCase):

    def test_a_cleanup_that_never_finishes_cannot_hold_the_turn(self):
        os.environ["AGENTY_CLOSE_LOOP_TIMEOUT"] = "1"
        from src.utils import agentY_server as S
        S = importlib.reload(S)
        try:
            loop = asyncio.new_event_loop()
            released = []

            async def stream():
                try:
                    yield 1
                    yield 2
                finally:
                    try:
                        await asyncio.Event().wait()      # a close nobody answers
                    finally:
                        released.append(True)             # what a lock release looks like

            async def turn():
                gen = stream()
                await gen.__anext__()                     # abandoned mid-stream
                return gen
            gen = loop.run_until_complete(turn())
            t0 = time.monotonic()
            S._close_loop(loop, "")
            self.assertLess(time.monotonic() - t0, 5)
            self.assertEqual(released, [True], "the rest of the cleanup ran")
            self.assertTrue(loop.is_closed())
            del gen
        finally:
            os.environ.pop("AGENTY_CLOSE_LOOP_TIMEOUT", None)
            importlib.reload(S)


class StopReachesDownloadsTest(unittest.TestCase):

    def test_stop_cancels_downloads(self):
        """A download runs in a worker thread that cancelling the turn cannot
        reach; Stop has to tell it directly."""
        from src.utils import agentY_server as S
        from route_client import authorised_client
        client = authorised_client(S._build_app())
        with mock.patch("agenty_core.tools.huggingface.cancel_downloads", return_value=1) as cancel, \
                mock.patch.object(S, "_interrupt_comfy", return_value={}):
            out = client.post("/agentY/stop", json={"thread_id": "nope"}).get_json()
        cancel.assert_called_once()
        self.assertEqual(out["downloads_stopped"], 1)


class TaskDumpTest(unittest.TestCase):

    def test_a_quiet_turn_says_what_it_waits_on(self):
        from src.utils import turn_watchdog as wd
        written = []
        with mock.patch.object(wd, "_write", written.append):
            wd.begin("req-quiet", "t")
            loop = asyncio.new_event_loop()
            wd.attach_loop("req-quiet", loop)

            async def wait_for_comfyui():
                await asyncio.sleep(3600)
            task = loop.create_task(wait_for_comfyui(), name="the-turn")
            loop.run_until_complete(asyncio.sleep(0))
            wd.dump_tasks("req-quiet", "no event for 120s")
            task.cancel()
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            loop.close()
            wd.end("req-quiet")
        dump = "\n".join(written)
        self.assertIn("the-turn", dump)
        self.assertIn("wait_for_comfyui", dump)


if __name__ == "__main__":
    unittest.main()
