"""The shared Vision/Video agents must survive a turn that analyses several files.

The orchestrator routinely emits one assistant turn with N ``analyze_image`` calls.
Strands runs those concurrently (``ConcurrentToolExecutor``) and each sync tool
lands in its own thread (``asyncio.to_thread``), so they all arrive at the single
shared agent registered by ``set_vision_agent`` at the same moment.

A Strands ``Agent`` refuses that: ``stream_async`` takes ``_invocation_lock``
non-blocking and raises ``ConcurrencyException`` on every caller but the first.
``analyze_image`` caught the exception and fell through to ``mode='full'``, which
for a text-only orchestrator returns "[The image itself is not shown ...]" — so a
turn asking about 11 images came back with exactly one real description and ten
placeholders, and the model then invented the rest.

The fakes below reproduce that contract: re-entrant invocation raises, and
``messages`` is mutated the way a real agent mutates it.

    python -m unittest tests.test_vision_concurrency
"""

import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

import src.agent as agent_mod
import src.tools.image_handling as ih
import src.tools.video_handling as vh


class FakeAgent:
    """Stand-in for a Strands Agent with the same re-entrancy contract."""

    def __init__(self, work_s: float = 0.05):
        self.messages: list = []
        self._lock = threading.Lock()
        self._work_s = work_s
        self.calls = 0
        self.concurrent_rejections = 0
        self.max_in_flight = 0
        self._in_flight = 0

    def __call__(self, user_message):
        # Strands: `self._invocation_lock.acquire(blocking=False)` → raise if held.
        if not self._lock.acquire(blocking=False):
            self.concurrent_rejections += 1
            raise RuntimeError(
                "Agent is already processing a request. "
                "Concurrent invocations are not supported."
            )
        try:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.calls += 1
            self.messages.append({"role": "user", "content": user_message})
            time.sleep(self._work_s)
            # A cleared history mid-flight is the other half of the collision.
            if not self.messages:
                raise RuntimeError("history was wiped by a concurrent call")
            self.messages.append({"role": "assistant", "content": [{"text": "ok"}]})
            return "a description of the image"
        finally:
            self._in_flight -= 1
            self._lock.release()


def _png(path: Path, size=(64, 64)) -> str:
    Image.new("RGB", size, (120, 90, 60)).save(path, "PNG")
    return str(path)


class VisionConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self._prev = ih._vision_agent
        self.agent = FakeAgent()
        ih.set_vision_agent(self.agent)
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.files = [_png(Path(self._tmp.name) / f"ref_{i:05d}.png") for i in range(6)]

    def tearDown(self):
        ih.set_vision_agent(self._prev) if self._prev else setattr(ih, "_vision_agent", None)

    def _describe(self, path):
        return ih.analyze_image(file_path=path, question="What is this?", mode="describe")

    def test_every_image_in_a_parallel_batch_gets_a_description(self):
        # This is the reported failure: 11 images in, 1 description out.
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            results = list(pool.map(self._describe, self.files))

        described = [r for r in results
                     if "Image analysis for" in r["content"][0]["text"]]
        self.assertEqual(len(described), len(self.files),
                         "every concurrently-requested image must be described")
        self.assertEqual(self.agent.concurrent_rejections, 0,
                         "calls must queue on the shared agent, not collide")
        self.assertEqual(self.agent.calls, len(self.files))

    def test_a_single_agent_is_never_entered_twice(self):
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            list(pool.map(self._describe, self.files))
        self.assertEqual(self.agent.max_in_flight, 1,
                         "one agent serves one call at a time")
        self.assertEqual(self.agent.concurrent_rejections, 0,
                         "a caller that has to wait must wait, not be turned away")

    def test_no_placeholder_text_survives(self):
        # The placeholder reads as success, which is how the wrong data got
        # downstream: the model treated "not shown" as an answer.
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            results = list(pool.map(self._describe, self.files))
        for r in results:
            self.assertEqual(r["status"], "success")
            self.assertNotIn("The image itself is not shown", r["content"][0]["text"])
            self.assertIn("a description of the image", r["content"][0]["text"])

    def test_history_is_still_wiped_between_calls(self):
        # Serialising must not turn the stateless vision agent stateful.
        for f in self.files[:3]:
            self._describe(f)
        self.assertEqual(len(self.agent.messages), 2, "each call starts from empty history")


class VisionPoolTest(unittest.TestCase):
    """With a factory, describes actually overlap instead of queueing."""

    def setUp(self):
        self._prev = ih._vision_agent
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.files = [_png(Path(self._tmp.name) / f"r{i}.png") for i in range(8)]
        self.built: list[FakeAgent] = []

        def factory():
            a = FakeAgent(work_s=0.12)
            self.built.append(a)
            return a

        self.primary = FakeAgent(work_s=0.12)
        self.built.append(self.primary)
        ih.set_vision_agent(self.primary, factory=factory, max_parallel=4)

    def tearDown(self):
        ih._vision_agent, ih._vision_pool = self._prev, None

    def _describe(self, path):
        return ih.analyze_image(file_path=path, question="?", mode="describe")

    def test_all_of_them_still_get_described(self):
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            results = list(pool.map(self._describe, self.files))
        self.assertTrue(all(r["status"] == "success" for r in results))
        self.assertEqual(sum(a.concurrent_rejections for a in self.built), 0)

    def test_the_work_actually_overlaps(self):
        # 8 files x 0.12s is ~0.96s serial; with 4 agents it should be ~0.24-0.5s.
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            list(pool.map(self._describe, self.files))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.75, f"describes did not run in parallel ({elapsed:.2f}s)")

    def test_it_never_exceeds_the_cap(self):
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            list(pool.map(self._describe, self.files))
        self.assertLessEqual(len(self.built), 4, "the pool grew past max_parallel")
        # A local GPU cap of 1 has to mean 1, or the setting is a lie.
        for a in self.built:
            self.assertEqual(a.max_in_flight, 1)

    def test_it_grows_lazily(self):
        # A lone describe must not pay for extra model handshakes.
        self._describe(self.files[0])
        self.assertEqual(len(self.built), 1, "a single call should not grow the pool")

    def test_every_instance_is_visible_for_cost_accounting(self):
        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            list(pool.map(self._describe, self.files))
        # Tokens burnt by a grown instance must not be invisible to the turn cost.
        self.assertEqual(len(ih.vision_agents()), len(self.built))

    def test_size_one_serialises(self):
        ih.set_vision_agent(FakeAgent(work_s=0.01), factory=lambda: FakeAgent(), max_parallel=1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self._describe, self.files[:4]))
        self.assertTrue(all(r["status"] == "success" for r in results))
        self.assertEqual(len(ih.vision_agents()), 1)


class ParallelismSettingTest(unittest.TestCase):
    """The cap follows the backend: a local GPU is not a cloud endpoint."""

    def setUp(self):
        self._env = {k: os.environ.get(k) for k in
                     ("VISION_MAX_PARALLEL", "VIDEO_MAX_PARALLEL",
                      "VISION_AGENT_MODEL", "VIDEO_AGENT_MODEL")}
        for k in self._env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env.items():
            os.environ[k] = v if v is not None else ""
            if v is None:
                os.environ.pop(k, None)

    def test_a_local_ollama_vision_model_stays_serial(self):
        os.environ["VISION_AGENT_MODEL"] = "ollama,gemma4:26b"
        self.assertEqual(agent_mod.vision_parallelism(), 1)

    def test_a_hosted_vision_model_runs_several(self):
        os.environ["VISION_AGENT_MODEL"] = "dashscope,qwen3.7-flash"
        self.assertGreater(agent_mod.vision_parallelism(), 1)

    def test_an_explicit_setting_wins_over_the_provider_guess(self):
        os.environ["VISION_AGENT_MODEL"] = "ollama,gemma4:26b"
        os.environ["VISION_MAX_PARALLEL"] = "3"
        self.assertEqual(agent_mod.vision_parallelism(), 3)

    def test_video_is_capped_lower_than_vision(self):
        os.environ["VISION_AGENT_MODEL"] = "dashscope,qwen3.7-flash"
        os.environ["VIDEO_AGENT_MODEL"] = "dashscope,qwen2.5-vl-72b-instruct"
        self.assertLess(agent_mod.video_parallelism(), agent_mod.vision_parallelism())


class VisionFailureReportingTest(unittest.TestCase):
    """A failed describe must not come back looking like a successful read."""

    def setUp(self):
        self._prev = ih._vision_agent
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.file = _png(Path(self._tmp.name) / "x.png")

        class Boom:
            messages: list = []

            def __call__(self, _msg):
                raise RuntimeError("vision backend exploded")

        ih.set_vision_agent(Boom())
        # Pretend the orchestrator is text-only, as qwen-flash is.
        self._prev_supports = getattr(ih, "_orchestrator_supports_vision", None)
        import src.utils.agentY_server as srv
        self._srv = srv
        self._prev_srv = srv._orchestrator_supports_vision
        srv._orchestrator_supports_vision = lambda: False

    def tearDown(self):
        self._srv._orchestrator_supports_vision = self._prev_srv
        ih._vision_agent = self._prev

    def test_it_reports_an_error_not_a_success(self):
        r = ih.analyze_image(file_path=self.file, question="what?", mode="describe")
        self.assertEqual(r["status"], "error")

    def test_it_names_the_real_cause_and_forbids_guessing(self):
        text = ih.analyze_image(file_path=self.file, mode="describe")["content"][0]["text"]
        self.assertIn("vision backend exploded", text)
        self.assertIn("do NOT guess", text)
        # The old text told the model to do the very thing it had just done.
        self.assertNotIn("Call analyze_image(mode='describe')", text)


class VideoConcurrencyTest(unittest.TestCase):
    """Same singleton shape in video_handling — pin it so only one path gets fixed."""

    def setUp(self):
        self._prev = vh._video_agent
        vh._video_agent, vh._video_pool = None, None

    def tearDown(self):
        vh._video_agent, vh._video_pool = self._prev, None

    def test_it_borrows_from_a_pool_rather_than_reusing_one_agent(self):
        primary = FakeAgent()
        vh.set_video_agent(primary, factory=FakeAgent, max_parallel=3)
        pool = vh._ensure_video_pool()
        self.assertEqual(pool.size, 3)
        with pool.borrow() as a, pool.borrow() as b:
            self.assertIsNot(a, b, "two concurrent clips need two agents")

    def test_a_bare_registration_still_works(self):
        # Older callers pass just the agent; that must mean "serial", not "crash".
        vh.set_video_agent(FakeAgent())
        self.assertEqual(vh._ensure_video_pool().size, 1)
        self.assertEqual(len(vh.video_agents()), 1)


if __name__ == "__main__":
    unittest.main()
