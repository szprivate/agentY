"""What ``split_video_into_shots`` actually wrote, not what it meant to write.

The bug this pins: with ``fast=true`` the tool stream-copied, and a stream copy
cannot start anywhere but a keyframe — so ffmpeg silently seeks BACKWARDS to the
previous one, with a zero exit code and nothing on stderr. Generated video is
normally written with a single keyframe at the start, so every shot came back
running from frame 0 and the last one was the source file again, byte for byte.
The tool reported a clean split throughout, because it printed the boundaries the
DETECTOR found and never looked at the files.

So the contract under test is: measure the output, and when a copy landed
somewhere other than where it was asked to, cut it properly instead.

    python -m unittest tests.test_shot_splitting
"""

import unittest
from unittest import mock

from agenty_core.tools import video as V


class ProbeTests(unittest.TestCase):
    """Reading a written file's real length back off the container."""

    def _probe(self, stderr):
        proc = mock.Mock(stderr=stderr, returncode=1)
        with mock.patch.object(V.subprocess, "run", return_value=proc):
            return V._probe_duration("ffmpeg", V.Path("x.mp4"))

    def test_it_reads_the_duration_ffmpeg_prints(self):
        self.assertAlmostEqual(
            self._probe("  Duration: 00:00:04.08, start: 0.080000, bitrate: 12 kb/s"),
            4.08, places=3)

    def test_hours_and_minutes_carry(self):
        self.assertAlmostEqual(self._probe("Duration: 01:02:03.50, start: 0"),
                               3723.5, places=3)

    def test_no_duration_line_is_unknown_not_zero(self):
        # Zero would read as "an empty file was written" and fail a good shot.
        self.assertIsNone(self._probe("some other ffmpeg noise"))

    def test_a_probe_that_blows_up_is_unknown(self):
        with mock.patch.object(V.subprocess, "run", side_effect=OSError("boom")):
            self.assertIsNone(V._probe_duration("ffmpeg", V.Path("x.mp4")))


class _Recorder:
    """Stands in for ffmpeg: records every cut asked for, returns a set length.

    ``durations`` maps ``(shot_start, fast)`` to the length that would land on
    disk, which is how a keyframe snap is expressed — ask for 2s from 6s in and
    get 8s of film back.
    """

    def __init__(self, durations, default=None):
        self.durations = durations
        self.default = default
        self.calls = []

    def __call__(self, exe, source, dest, start, duration, fast):
        self.calls.append({"start": start, "duration": duration, "fast": fast,
                           "dest": str(dest)})
        actual = self.durations.get((round(start, 3), fast),
                                    self.default if self.default is not None else duration)
        return True, "", actual


def _run(recorder, shots, fast=True, suffix=".mp4", removed=None):
    """Drive the tool over *shots* (a list of ``(start, end)``) with ffmpeg faked.

    *removed* collects every path the tool deleted, so a test can check that a
    superseded file was cleaned up rather than only that unlink was survivable.
    """
    rows = [(a, b, "tc", "tc") for a, b in shots]
    meta = {"fps": 25.0, "duration_s": shots[-1][1], "detector": "content",
            "threshold": 27.0, "cuts": len(shots) - 1}
    sink = [] if removed is None else removed

    def _unlink(self, *a, **k):
        sink.append(str(self))

    with mock.patch.object(V, "_resolve_video", return_value=f"/src/clip{suffix}"), \
         mock.patch.object(V, "_detect_shots", return_value=(rows, meta)), \
         mock.patch.object(V, "_ffmpeg_exe", return_value="ffmpeg"), \
         mock.patch.object(V, "_shots_dir", return_value=V.Path("/out")), \
         mock.patch.object(V, "_cut_one", recorder), \
         mock.patch.object(V.Path, "unlink", _unlink):
        return V.split_video_into_shots(file_path=f"clip{suffix}", fast=fast)


SHOTS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]


class KeyframeSnapTests(unittest.TestCase):
    def test_a_copy_that_ran_long_is_cut_again_properly(self):
        # Every copy snaps back to frame 0: asked for 2s at 2.0, got 4s of film.
        rec = _Recorder({(0.0, True): 2.0, (2.0, True): 4.0,
                         (4.0, True): 6.0, (6.0, True): 8.0})
        out = _run(rec, SHOTS)
        redone = [c for c in rec.calls if not c["fast"]]
        self.assertEqual([c["start"] for c in redone], [2.0, 4.0, 6.0],
                         "every shot after the bad copy must be cut for real")
        for r in out["shots"]:
            self.assertAlmostEqual(r["written_duration_s"], r["duration_s"], places=3)

    def test_it_stops_copying_once_a_copy_has_proved_wrong(self):
        # The source's keyframe spacing does not change part-way through, so
        # re-testing the copy per shot only buys wasted encodes.
        rec = _Recorder({(0.0, True): 2.0, (2.0, True): 4.0}, default=2.0)
        _run(rec, SHOTS)
        attempted_fast = [c["start"] for c in rec.calls if c["fast"]]
        self.assertEqual(attempted_fast, [0.0, 2.0],
                         "shots 3 and 4 must not be stream-copied again")

    def test_an_accurate_copy_is_kept(self):
        # A keyframe-dense source: the copy is right, so it must not be thrown
        # away — that is the whole point of asking for fast.
        rec = _Recorder({}, default=None)
        _run(rec, SHOTS)
        self.assertTrue(all(c["fast"] for c in rec.calls))
        self.assertEqual(len(rec.calls), 4, "no shot should be cut twice")

    def test_a_frame_of_overshoot_is_tolerated(self):
        # Container rounding runs a copy a frame or two long. Re-encoding for
        # that would mean fast never survives anywhere.
        rec = _Recorder({}, default=2.08)
        rec2 = _Recorder({}, default=2.0 + V._CUT_TOLERANCE + 0.05)
        _run(rec, SHOTS)
        self.assertTrue(all(c["fast"] for c in rec.calls), "0.08s over is rounding")
        _run(rec2, SHOTS)
        self.assertTrue(any(not c["fast"] for c in rec2.calls),
                        "past the tolerance it is a snap, not rounding")

    def test_a_short_shot_is_not_re_encoded(self):
        # The last shot often runs a little short at EOF. That is not the bug.
        rec = _Recorder({}, default=1.5)
        _run(rec, SHOTS)
        self.assertTrue(all(c["fast"] for c in rec.calls))

    def test_an_unmeasurable_shot_is_left_alone(self):
        # No duration back means "could not tell", not "wrong". Re-encoding on an
        # unreadable probe would punish every source the probe cannot parse.
        rec = _Recorder({}, default=None)
        rec.durations = {(0.0, True): None, (2.0, True): None,
                         (4.0, True): None, (6.0, True): None}
        _run(rec, SHOTS)
        self.assertTrue(all(c["fast"] for c in rec.calls))

    def test_the_re_cut_lands_on_the_same_shot_not_a_new_file(self):
        # A fallback that wrote clip_shot_002.mp4 beside a stale clip_shot_002.mkv
        # would leave two files for one shot, and the wrong one is the one that
        # sorts first. The re-cut is an .mp4 and the copy it replaces is deleted.
        rec = _Recorder({(0.0, True): 2.0, (2.0, True): 4.0}, default=2.0)
        removed = []
        out = _run(rec, SHOTS, suffix=".mkv", removed=removed)
        redone = [c for c in rec.calls if not c["fast"]]
        self.assertTrue(redone)
        for c in redone:
            self.assertTrue(c["dest"].endswith(".mp4"))
        self.assertIn("shot_002.mkv", " ".join(removed),
                      "the superseded stream copy must not be left on disk")
        kept = [r["path"] for r in out["shots"]]
        self.assertEqual(len(kept), len(set(kept)), "one file per shot")

    def test_nothing_is_deleted_when_the_copy_was_good(self):
        # Cleanup that fires on the happy path would delete the shot just written.
        rec = _Recorder({}, default=2.0)
        removed = []
        _run(rec, SHOTS, suffix=".mkv", removed=removed)
        self.assertEqual(removed, [])

    def test_fast_off_never_stream_copies(self):
        rec = _Recorder({}, default=2.0)
        _run(rec, SHOTS, fast=False)
        self.assertTrue(all(not c["fast"] for c in rec.calls))


class ReportTests(unittest.TestCase):
    """The text the agent reads has to match the files on disk."""

    def test_a_fallback_is_declared(self):
        rec = _Recorder({(0.0, True): 2.0, (2.0, True): 4.0}, default=2.0)
        text = _run(rec, SHOTS)["content"][0]["text"]
        self.assertIn("fast=true was ignored", text)
        self.assertNotIn("Stream-copied:", text,
                         "claiming a stream copy after abandoning it is the old lie")

    def test_a_real_stream_copy_still_warns_about_its_edges(self):
        rec = _Recorder({}, default=2.08)
        text = _run(rec, SHOTS)["content"][0]["text"]
        self.assertIn("Stream-copied:", text)
        self.assertNotIn("fast=true was ignored", text)

    def test_a_plain_re_encode_claims_neither(self):
        rec = _Recorder({}, default=2.0)
        text = _run(rec, SHOTS, fast=False)["content"][0]["text"]
        self.assertNotIn("Stream-copied:", text)
        self.assertNotIn("fast=true was ignored", text)


if __name__ == "__main__":
    unittest.main()
