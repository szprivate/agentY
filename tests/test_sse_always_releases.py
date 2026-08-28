"""The panel must always be told the turn ended, whatever happened to the stream.

The reported symptom, several times over: the CLI says the orchestrator finished,
and the panel never reacts to anything typed afterwards.

The watchdog trace named it exactly. Four turns showed:

    PHASE req=… post:emit_done -> post:close_loop
    NOTE  req=… sse generator closed
    END   req=… last_phase=post:close_loop runner exited

`post:emit_done` means the runner really did put `{"type":"done"}` on the queue.
`sse generator closed` with no `sse yielding done` in between, and no client
disconnect, means the generator died with that `done` still sitting behind it.

Reproduced exactly: `_sse` is `json.dumps`, so ONE event carrying anything json
cannot render (a Path, bytes, a set, an exception object) raises inside the
generator. That is not `GeneratorExit`, so it was not caught; the generator tore
down, and the panel — which listens for exactly one thing — stayed streaming and
deaf until the tab was reloaded. The turn itself had completed perfectly, which
is why it reads as "went quiet" rather than as a crash.

Two defences, because either alone leaves a hole: `_sse` no longer raises, and
the generator releases the panel even when something else does.

    python -m unittest discover -s tests
"""

import json
import queue
import unittest
from pathlib import Path
from unittest import mock

from src.utils import agentY_server as srv


class FrameEncodingTest(unittest.TestCase):
    """`_sse` is called inside the generator, so raising there ends the turn."""

    def test_an_ordinary_event_is_unchanged(self):
        frame = srv._sse({"type": "text", "data": "hello"})
        self.assertEqual(frame, 'data: {"type": "text", "data": "hello"}\n\n')

    def test_unicode_is_not_escaped(self):
        self.assertIn("wüst — ok", srv._sse({"type": "text", "data": "wüst — ok"}))

    def test_something_json_cannot_render_still_produces_a_frame(self):
        for value in (Path("C:/x/y.png"), {1, 2}, b"bytes", object(),
                      RuntimeError("boom")):
            frame = srv._sse({"type": "output", "path": value})
            self.assertTrue(frame.startswith("data: "), repr(value))
            self.assertTrue(frame.endswith("\n\n"), repr(value))
            json.loads(frame[len("data: "):])          # still valid JSON

    def test_the_event_survives_even_when_one_field_cannot(self):
        """Losing a field beats losing the end of the turn."""
        frame = srv._sse({"type": "output", "kind": "image", "path": Path("a.png")})
        body = json.loads(frame[len("data: "):])
        self.assertEqual(body["type"], "output")
        self.assertEqual(body["kind"], "image")

    def test_even_an_unrenderable_key_yields_a_usable_frame(self):
        """`default=` only rescues values; a bad KEY needs the last resort."""
        frame = srv._sse({object(): "x"})
        body = json.loads(frame[len("data: "):])
        self.assertEqual(body["type"], "error")


class StreamAlwaysEndsTest(unittest.TestCase):
    """Whatever happens, the last frame the panel sees is `done`."""

    def _drive(self, items, in_flight=True):
        """Run the real generator over *items*, collecting frames and notes."""
        q: queue.Queue = queue.Queue()
        for it in items:
            q.put(it)
        notes: list = []
        frames = []
        with mock.patch.object(srv._wd, "note", lambda rid, msg: notes.append(msg)), \
             mock.patch.object(srv._wd, "is_in_flight", return_value=in_flight):
            # `poll` is the queue wait; the real 15s is a keep-alive cadence, not
            # part of what is being tested.
            for frame in srv._stream_turn(q, "req1", "thread1", poll=0.05):
                frames.append(frame)
        return frames, notes

    def _types(self, frames):
        out = []
        for f in frames:
            if not f.startswith("data: "):
                continue
            out.append(json.loads(f[len("data: "):]).get("type"))
        return out

    def test_a_normal_turn_ends_with_done(self):
        frames, notes = self._drive([{"type": "text", "data": "hi"},
                                     {"type": "done"}, None])
        self.assertEqual(self._types(frames)[-1], "done")
        self.assertIn("sse yielding done", notes)
        self.assertEqual(self._types(frames).count("done"), 1,
                         "the panel was sent done twice")

    def test_an_unserialisable_event_does_not_cost_the_done(self):
        """The exact failure from the trace."""
        frames, notes = self._drive([{"type": "output", "path": Path("a.png")},
                                     {"type": "done"}, None])
        self.assertEqual(self._types(frames)[-1], "done")
        self.assertIn("sse yielding done", notes)

    def test_a_queue_that_just_stops_still_releases_the_panel(self):
        """`None` with no `done` in front of it — the runner died mid-turn."""
        frames, notes = self._drive([{"type": "text", "data": "hi"}, None])
        self.assertEqual(self._types(frames)[-1], "done")
        self.assertIn("releasing the panel", " ".join(notes))

    def test_a_runner_that_vanished_releases_the_panel_once(self):
        """Nothing on the queue and the watchdog no longer tracks the turn."""
        frames, notes = self._drive([], in_flight=False)
        types = self._types(frames)
        self.assertEqual(types[-1], "done")
        self.assertEqual(types.count("done"), 1, "released twice")
        self.assertIn("runner gone without done", " ".join(notes))

    def test_a_stream_error_is_named_in_the_trace(self):
        """A type name is actionable; "it went quiet again" is not."""
        real, calls = srv._sse, {"n": 0}

        def flaky(obj):
            # The first two frames are the header, emitted before the loop. Fail
            # on the third — inside it, with the turn's `done` still queued.
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("kaboom")
            return real(obj)

        q: queue.Queue = queue.Queue()
        q.put({"type": "text", "data": "hi"})
        q.put({"type": "done"})
        q.put(None)
        notes: list = []
        with mock.patch.object(srv._wd, "note", lambda rid, m: notes.append(m)), \
             mock.patch.object(srv._wd, "is_in_flight", return_value=True), \
             mock.patch.object(srv, "_sse", flaky):
            frames = list(srv._stream_turn(q, "req1", "thread1", poll=0.05))
        joined = " ".join(notes)
        self.assertIn("sse stream failed: RuntimeError: kaboom", joined)
        self.assertIn("releasing the panel", joined)
        self.assertIn("sse generator closed", joined)
        self.assertEqual(self._types(frames)[-1], "done",
                         "the panel was left streaming after a stream error")

    def test_the_generator_is_always_recorded_as_closed(self):
        _, notes = self._drive([{"type": "done"}, None])
        self.assertEqual(notes[-1], "sse generator closed")


if __name__ == "__main__":
    unittest.main()
