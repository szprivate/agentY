"""The rating page's backend: sibling slates, opaque ids, picks into the log.

The fitness weights are fitted from choices, and none had ever been recorded.
These cover what makes a pick worth fitting on: the slate is siblings from ONE
folder, the page can only ever fetch images it was shown (by id, never by path),
and a pick lands in the preference log in the same slate form as a review.
"""

import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.utils import preference_log, rating


def _png(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (shade, shade // 2, 255 - shade)).save(path)


class Fixture(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agenty_rating_"))
        for i in range(4):
            _png(self.tmp / "stadium" / "v002" / f"a_{i:05d}_.png", 40 * i + 10)
        for i in range(2):                        # too few to be a choice
            _png(self.tmp / "lonely" / f"b_{i}.png", 90)
        for i in range(3):                        # ComfyUI's scratch: never rated
            _png(self.tmp / "temp" / f"c_{i}.png", 120)
        self.log = self.tmp / "preferences.jsonl"
        for target, value in ((rating, "_ids"), (rating, "_rated")):
            p = mock.patch.object(target, value, type(getattr(target, value))())
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(preference_log, "LOG_PATH", self.log)
        p.start()
        self.addCleanup(p.stop)
        folders = rating.scan_folders([self.tmp])
        p = mock.patch.dict(rating._state, {"folders": folders, "scanned": True,
                                            "scanning": False, "error": ""})
        p.start()
        self.addCleanup(p.stop)
        # Registered last, so it runs FIRST: a pick is written from a background
        # thread, and one still running when LOG_PATH is un-patched writes into
        # the user's real preference log (it did, twice).
        self.addCleanup(self._join_writers)

    @staticmethod
    def _join_writers():
        import threading
        for t in threading.enumerate():
            if t.name == "agentY-rating-log":
                t.join(timeout=10)


class TheScan(Fixture):

    def test_only_folders_with_enough_siblings_count(self):
        names = [Path(f["folder"]).name for f in rating._state["folders"]]
        self.assertEqual(names, ["v002"])

    def test_the_slate_is_siblings_from_one_folder(self):
        slate = rating.next_slate(random.Random(1))
        self.assertEqual(len(slate["items"]), 4)
        folders = {Path(rating.path_of(it["id"])).parent for it in slate["items"]}
        self.assertEqual(len(folders), 1)
        self.assertEqual(slate["label"].split("/")[-1], "v002")


class TheIds(Fixture):

    def test_the_page_never_sees_a_path(self):
        slate = rating.next_slate(random.Random(1))
        text = json.dumps(slate["items"])
        self.assertNotIn(str(self.tmp), text)
        self.assertNotIn("\\\\", text)

    def test_an_unknown_id_is_not_served(self):
        self.assertIsNone(rating.preview("../../etc/passwd"))

    def test_a_shown_image_is_served_as_a_small_jpeg(self):
        slate = rating.next_slate(random.Random(1))
        data = rating.preview(slate["items"][0]["id"])
        self.assertTrue(data.startswith(b"\xff\xd8"))


class ThePick(Fixture):

    def _wait_for_log(self):
        for _ in range(100):
            if preference_log.read_events():
                return preference_log.read_events()
            time.sleep(0.05)
        self.fail("nothing was logged")

    def test_a_pick_is_one_slate_in_the_preference_log(self):
        slate = rating.next_slate(random.Random(1))
        ids = [it["id"] for it in slate["items"]]
        out = rating.record_pick(ids[2], ids, slate["folder"])
        self.assertEqual(out, {"ok": True, "pairs": 3})
        ev = self._wait_for_log()[0]
        self.assertEqual(ev["source"], "rating")
        self.assertEqual(len(ev["chosen"]), 1)
        self.assertEqual(len(ev["rejected"]), 3)
        self.assertEqual(ev["chosen"][0]["path"], rating.path_of(ids[2]))
        self.assertEqual(len(preference_log.slates()), 1, "the fit can read it")

    def test_rated_images_are_not_shown_again(self):
        slate = rating.next_slate(random.Random(1))
        ids = [it["id"] for it in slate["items"]]
        rating.record_pick(ids[0], ids)
        self.assertIsNone(rating.next_slate(random.Random(2)), "the folder is used up")

    def test_a_skip_logs_nothing_and_moves_on(self):
        slate = rating.next_slate(random.Random(1))
        rating.skip([it["id"] for it in slate["items"]])
        time.sleep(0.1)
        self.assertEqual(preference_log.read_events(), [])
        self.assertIsNone(rating.next_slate(random.Random(2)))

    def test_a_pick_of_something_never_shown_is_refused(self):
        self.assertFalse(rating.record_pick("nope", ["nope", "also"])["ok"])


class ThePage(unittest.TestCase):

    def test_the_page_is_reachable_like_the_other_viewers(self):
        from src.utils import api_guard
        self.assertIn("/agentY/rate", api_guard.PUBLIC_PATHS)
        # ... and only the page: the data routes need the token.
        self.assertNotIn("/agentY/rate/pick", api_guard.PUBLIC_PATHS)

    def test_the_page_exists_and_fetches_images_through_fetch(self):
        page = Path(__file__).resolve().parent.parent / "scripts" / "rating.html"
        html = page.read_text(encoding="utf-8")
        self.assertIn("/agentY/rate/image/", html)
        self.assertIn("createObjectURL", html, "an <img src> could not carry the token")


if __name__ == "__main__":
    unittest.main()
