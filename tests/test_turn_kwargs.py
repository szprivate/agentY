"""Every kwarg the server sends must survive the whole way to the turn.

The bug this exists for: `open_workflows` was added to the route, to both server
runner functions, and to `_astream_orchestrator` — but `stream_async` sits between
the last two as a plain passthrough, and it was missed. Nothing failed until a
real turn ran, and then EVERY turn failed with

    Pipeline.stream_async() got an unexpected keyword argument 'open_workflows'

which is not a hook bug, not a canvas bug, and reads like neither.

No test caught it because the tests drive the tools directly (pipeline_stub) or
the units underneath, and nothing walks the chain the panel actually uses:

    POST /agentY/chat
      -> _run_pipeline_stream
        -> _run_pipeline_turn
          -> Pipeline.stream_async          <- the link that was missing
            -> Pipeline._astream_orchestrator

Signatures are the cheapest place to check that, and they catch it at import time
rather than on the user's first hook run.

    python -m unittest discover -s tests
"""

import inspect
import unittest

from src.pipeline import Pipeline
from src.utils import agentY_server


def kwargs_of(fn) -> set:
    """Named parameters *fn* accepts (ignoring self/varargs)."""
    out = set()
    for name, p in inspect.signature(fn).parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.add(name)
    return out


# Everything the panel sends that has to reach the turn. Named explicitly rather
# than derived, so ADDING one to the route without threading it through is what
# fails here — which is exactly the mistake this is about.
FROM_THE_PANEL = {
    "canvas_prompt",
    "canvas_hooks",
    "canvas_selection",
    "open_workflows",
    "dry_run",
}

CHAIN = (
    ("_run_pipeline_stream", agentY_server._run_pipeline_stream),
    ("_run_pipeline_turn", agentY_server._run_pipeline_turn),
    ("stream_async", Pipeline.stream_async),
    ("_astream_orchestrator", Pipeline._astream_orchestrator),
)


class TurnKwargChainTest(unittest.TestCase):

    def test_every_link_accepts_what_the_panel_sends(self):
        for name, fn in CHAIN:
            missing = FROM_THE_PANEL - kwargs_of(fn)
            # `qa_briefing` is resolved mid-chain rather than sent, so the two
            # server-side functions legitimately do not take it; everything in
            # FROM_THE_PANEL travels the whole way.
            self.assertFalse(
                missing,
                f"{name}() cannot accept {sorted(missing)} — a turn carrying "
                f"it dies at this link with 'unexpected keyword argument'")

    def test_the_route_passes_exactly_what_the_runner_takes(self):
        """A kwarg the route invents is as fatal as one a link forgot."""
        import re
        src = inspect.getsource(agentY_server)
        # The kwargs dict handed to the runner thread at the /chat route.
        m = re.search(r"kwargs=\{([^}]*)\},\s*\n\s*daemon=True", src)
        self.assertIsNotNone(m, "the runner's kwargs dict moved — update this test")
        sent = set(re.findall(r'"(\w+)":', m.group(1)))
        self.assertTrue(
            FROM_THE_PANEL <= sent,
            f"the route stopped sending {sorted(FROM_THE_PANEL - sent)}")
        accepted = kwargs_of(agentY_server._run_pipeline_stream)
        self.assertFalse(
            sent - accepted,
            f"the route sends {sorted(sent - accepted)}, which the runner does "
            "not accept")

    def test_the_turn_stores_what_it_is_given(self):
        """Accepting a kwarg and dropping it is the quieter version of the bug."""
        src = inspect.getsource(Pipeline._astream_orchestrator)
        for name in ("canvas_selection", "open_workflows"):
            self.assertIn(f"self._{name}", src,
                          f"{name} is accepted but never stored on the turn")


if __name__ == "__main__":
    unittest.main()
