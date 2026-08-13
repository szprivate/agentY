"""Per-project memory: the store that switches when the project switches.

The whole design rests on one fact — ComfyUI reports its own ``--user-directory``,
and the production pipeline (Ayon) changes that directory when it changes project.
So these tests drive the store by moving that reported path around, which is
exactly what a project switch looks like from agentY's side.

    python -m unittest tests.test_project_memory
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import project_memory as pm


class StoreTestCase(unittest.TestCase):
    """Base: point the store at a temp dir by faking what ComfyUI reports."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="agenty-projmem-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self._user_dir = self.root / "projA" / "user"
        patcher = mock.patch("src.tools.comfyui.get_comfyui_dirs",
                             side_effect=lambda: json.dumps({"user_dir": str(self._user_dir)}))
        self.dirs = patcher.start()
        self.addCleanup(patcher.stop)

    def switch_project(self, name):
        """What Ayon does: same agentY, ComfyUI now reports a different user dir."""
        self._user_dir = self.root / name / "user"


class RoundTripTests(StoreTestCase):
    def test_a_fact_written_now_reads_back_later(self):
        pm.write_entry("hero", "grizzled dockworker, mid-50s\nsalt-and-pepper beard",
                       type="character")
        got = pm.read_entry("hero")
        self.assertIsNotNone(got)
        self.assertEqual(got.type, "character")
        self.assertIn("dockworker", got.body)
        self.assertEqual(got.summary, "grizzled dockworker, mid-50s",
                         "the first line is what every later turn sees")

    def test_the_name_is_forgiving(self):
        pm.write_entry("Aspect Ratio", "2.39:1", type="technical")
        for asked in ["aspect ratio", "Aspect-Ratio", "  aspect   ratio  ", "aspect-ratio"]:
            with self.subTest(asked=asked):
                self.assertIsNotNone(pm.read_entry(asked), asked)

    def test_writing_the_same_name_replaces_rather_than_appends(self):
        pm.write_entry("hero", "clean-shaven", type="character")
        pm.write_entry("hero", "full beard", type="character")
        self.assertEqual(pm.read_entry("hero").body, "full beard")
        self.assertEqual(len([e for e in pm.list_entries() if e.name == "hero"]), 1)

    def test_refiling_under_another_type_leaves_no_second_copy(self):
        # Otherwise the old file keeps answering with the old fact, from a folder
        # nobody thinks to look in.
        pm.write_entry("alley", "the location", type="note")
        pm.write_entry("alley", "the location", type="reference")
        entries = [e for e in pm.list_entries() if e.name == "alley"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].type, "reference")

    def test_forgetting_a_fact(self):
        pm.write_entry("hero", "someone", type="character")
        self.assertTrue(pm.delete_entry("hero"))
        self.assertIsNone(pm.read_entry("hero"))
        self.assertFalse(pm.delete_entry("hero"), "already gone")

    def test_empties_are_refused(self):
        self.assertIsNone(pm.write_entry("", "something"))
        self.assertIsNone(pm.write_entry("hero", "   "))
        self.assertEqual(pm.list_entries(), [])


class ProjectSwitchTests(StoreTestCase):
    """The point of the whole exercise."""

    def test_memory_follows_the_project(self):
        pm.write_entry("hero", "the dockworker", type="character")
        self.switch_project("projB")
        self.assertIsNone(pm.read_entry("hero"),
                          "project B must not see project A's cast")
        pm.write_entry("hero", "a child on a bicycle", type="character")
        self.assertEqual(pm.read_entry("hero").body, "a child on a bicycle")
        self.switch_project("projA")
        self.assertEqual(pm.read_entry("hero").body, "the dockworker",
                         "and switching back finds project A's again")

    def test_nothing_is_cached_across_the_switch(self):
        # A cache here would be invisible until the day someone switched project
        # mid-session and got the previous project's characters.
        pm.write_entry("grade", "bleach bypass", type="style")
        pm.render_context()               # a plausible moment to have cached
        self.switch_project("projB")
        self.assertEqual(pm.list_entries(), [])

    def test_no_comfyui_is_not_an_error(self):
        with mock.patch("src.tools.comfyui.get_comfyui_dirs",
                        side_effect=lambda: json.dumps({"error": "connection refused"})):
            self.assertIsNone(pm.store_dir())
            self.assertEqual(pm.list_entries(), [])
            self.assertEqual(pm.render_context(), "")
            self.assertIsNone(pm.write_entry("hero", "someone"))

    def test_an_unknown_user_dir_is_not_treated_as_a_path(self):
        with mock.patch("src.tools.comfyui.get_comfyui_dirs",
                        side_effect=lambda: json.dumps({"user_dir": "unknown"})):
            self.assertIsNone(pm.store_dir())


class InjectedBlockTests(StoreTestCase):
    def test_an_empty_project_injects_nothing(self):
        self.assertEqual(pm.render_context(), "")

    def test_technical_settings_arrive_in_full_and_the_rest_by_name(self):
        pm.write_entry("aspect-ratio", "2.39:1, 3840×1608 on every hero shot",
                       type="technical")
        pm.write_entry("hero", "grizzled dockworker, mid-50s\n\nWears a "
                               "salt-stained donkey jacket that " + "x" * 400,
                       type="character")
        block = pm.render_context()
        self.assertIn("IN FORCE", block)
        self.assertIn("3840×1608", block, "a delivery spec is no use paraphrased")
        self.assertIn("hero (character) — grizzled dockworker, mid-50s", block)
        self.assertNotIn("donkey jacket", block,
                         "the body is read on demand, not injected")
        self.assertIn('project_memory_read("<name>")', block)

    def test_a_long_summary_is_clipped(self):
        pm.write_entry("hero", "x" * 400, type="character")
        line = next(l for l in pm.render_context().splitlines() if "hero" in l)
        self.assertLess(len(line), 200)
        self.assertIn("…", line)

    def test_the_block_does_not_grow_without_bound(self):
        for i in range(pm._MAX_LISTED + 15):
            pm.write_entry(f"char-{i:03d}", f"character number {i}", type="character")
        listed = [l for l in pm.render_context().splitlines() if l.strip().startswith("- char-")]
        self.assertEqual(len(listed), pm._MAX_LISTED)


class OnDiskShapeTests(StoreTestCase):
    """The files are the product: a person opens this folder and edits them."""

    def test_one_fact_per_file_under_its_type(self):
        e = pm.write_entry("hero", "the dockworker", type="character")
        self.assertEqual(e.path.name, "hero.md")
        self.assertEqual(e.path.parent.name, "character")
        self.assertEqual(e.path.read_text(encoding="utf-8").strip(), "the dockworker")

    def test_a_hand_edited_file_is_read_as_written(self):
        # No frontmatter, no index lookup: whatever the file says is the fact.
        d = pm.store_dir(create=True) / "style"
        d.mkdir(parents=True, exist_ok=True)
        (d / "grade.md").write_text("bleach-bypass, teal shadows\n", encoding="utf-8")
        self.assertEqual(pm.read_entry("grade").body, "bleach-bypass, teal shadows")

    def test_a_stale_index_cannot_hide_a_fact(self):
        pm.write_entry("hero", "the dockworker", type="character")
        (pm.store_dir() / pm._INDEX_NAME).write_text("# Project memory\n", encoding="utf-8")
        self.assertIsNotNone(pm.read_entry("hero"))

    def test_the_index_is_regenerated_for_whoever_opens_the_folder(self):
        pm.write_entry("hero", "the dockworker", type="character")
        pm.write_entry("grade", "bleach bypass", type="style")
        index = (pm.store_dir() / pm._INDEX_NAME).read_text(encoding="utf-8")
        self.assertIn("**hero** (character) — the dockworker", index)
        self.assertIn("**grade** (style) — bleach bypass", index)
        pm.delete_entry("hero")
        self.assertNotIn("**hero**", (pm.store_dir() / pm._INDEX_NAME).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
