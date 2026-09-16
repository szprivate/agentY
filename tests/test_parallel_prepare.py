"""prepare_workflow can run several times at once.

A Strands agent takes one invocation at a time, so parallel prepare_workflow
calls on the single researcher failed with "Agent is already processing a
request" and the orchestrator fell back to one-at-a-time. And every assembly
from one template patched the same cached file, so a bedroom and a kitchen
prepared from the same template both ended up as the kitchen.

    python -m unittest discover -s tests
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.pipeline as pl
from src.pipeline import Pipeline


class _Agent:
    def __init__(self, name):
        self.name = name
        self.messages = [f"history of {name}"]


def _pipeline(primary):
    p = Pipeline.__new__(Pipeline)
    p._researcher = primary
    p._verbose = False
    return p


class LeaseTest(unittest.TestCase):

    def setUp(self):
        self.built = []

        def build():
            agent = _Agent(f"spare{len(self.built)}")
            self.built.append(agent)
            return agent

        self.enterContext(mock.patch.object(pl, "create_query_templates_agent", build))

    def test_one_call_gets_the_primary_with_its_history(self):
        primary = _Agent("primary")
        p = _pipeline(primary)

        async def go():
            async with p._researcher_lease() as a:
                return a
        self.assertIs(asyncio.run(go()), primary)
        self.assertEqual(primary.messages, ["history of primary"])
        self.assertEqual(self.built, [])

    def test_parallel_calls_each_get_their_own_agent(self):
        p = _pipeline(_Agent("primary"))
        seen = []

        async def one():
            async with p._researcher_lease() as a:
                seen.append(a)
                self.assertEqual(a.messages if a is not p._researcher else [], [])
                await asyncio.sleep(0.01)

        async def go():
            await asyncio.gather(one(), one(), one())
        asyncio.run(go())
        self.assertEqual(len({id(a) for a in seen}), 3)
        self.assertIs(seen[0], p._researcher)

    def test_spares_are_reused(self):
        p = _pipeline(_Agent("primary"))

        async def pair():
            async with p._researcher_lease():
                async with p._researcher_lease() as b:
                    return b

        async def go():
            return await pair(), await pair()
        first, second = asyncio.run(go())
        self.assertIs(first, second)
        self.assertEqual(len(self.built), 1)

    def test_a_model_switch_drops_the_spares(self):
        p = _pipeline(_Agent("old"))

        async def pair():
            async with p._researcher_lease():
                async with p._researcher_lease() as b:
                    return b

        async def go():
            first = await pair()
            p._researcher = _Agent("new")  # what model_reload does
            return first, await pair()
        first, second = asyncio.run(go())
        self.assertIsNot(first, second)
        self.assertEqual(len(self.built), 2)

    def test_past_the_cap_a_call_waits_its_turn(self):
        p = _pipeline(_Agent("primary"))
        p._MAX_PARALLEL_RESEARCH = 2
        running, peak = [0], [0]

        async def one():
            async with p._researcher_lease():
                running[0] += 1
                peak[0] = max(peak[0], running[0])
                await asyncio.sleep(0.05)
                running[0] -= 1

        async def go():
            await asyncio.gather(*(one() for _ in range(4)))
        asyncio.run(go())
        self.assertEqual(peak[0], 2)


class OwnCopyTest(unittest.TestCase):

    def test_each_assembly_patches_its_own_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpl = Path(tmp) / "image_edit.json"
            tpl.write_text('{"1": {}}', encoding="utf-8")
            a, b = Pipeline._own_copy(str(tpl)), Pipeline._own_copy(str(tpl))
            self.assertNotEqual(a, b)
            for path in (a, b):
                self.assertTrue(Path(path).name.startswith("image_edit_"))
                self.assertEqual(Path(path).read_text(encoding="utf-8"), '{"1": {}}')

    def test_a_missing_file_is_passed_through(self):
        self.assertEqual(Pipeline._own_copy("Z:/nope/x.json"), "Z:/nope/x.json")


if __name__ == "__main__":
    unittest.main()
