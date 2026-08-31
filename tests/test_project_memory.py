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
        pm.forget_miss()
        self.addCleanup(pm.forget_miss)
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

    def test_a_down_server_is_asked_once_not_once_per_call(self):
        # Each failed lookup is a two-second connection timeout, and one write does
        # several — on a host that cannot generate anything anyway.
        calls = []

        def refused():
            calls.append(1)
            return json.dumps({"error": "connection refused"})

        with mock.patch("src.tools.comfyui.get_comfyui_dirs", side_effect=refused):
            for _ in range(5):
                pm.write_entry("hero", "someone")
                pm.render_context()
            self.assertEqual(len(calls), 1, "asked once, then remembered")

    def test_the_note_is_droppable_when_the_server_comes_back(self):
        with mock.patch("src.tools.comfyui.get_comfyui_dirs",
                        side_effect=lambda: json.dumps({"error": "refused"})):
            self.assertIsNone(pm.store_dir())
        pm.forget_miss()
        self.assertIsNotNone(pm.store_dir(), "the fake dirs are answering again")

    def test_an_unknown_user_dir_is_not_treated_as_a_path(self):
        with mock.patch("src.tools.comfyui.get_comfyui_dirs",
                        side_effect=lambda: json.dumps({"user_dir": "unknown"})):
            self.assertIsNone(pm.store_dir())


class WhereComfyUIKeepsItsFiles(unittest.TestCase):
    """How the user directory is resolved — the step everything above assumes.

    Every test in this file mocks that answer, which is right for testing the
    store and wrong as the only coverage: the answer itself was broken, and no
    test here could have noticed. agentY wrote project memory to
    <agentY>/user/agentY/project while the ComfyUI nodes read
    <ComfyUI>/user/agentY/project, so the load-item node reported "nothing stored
    yet" about a store that existed and had two references in it.

    The cause was one `.resolve()`. `python main.py` reports argv[0] as
    "./main.py", and resolving a relative path uses the CALLER's working
    directory — agentY's, not ComfyUI's.
    """

    def setUp(self):
        from agenty_core.tools import comfyui
        self.mod = comfyui
        comfyui._tool_dirs_result = None
        self.addCleanup(setattr, comfyui, "_tool_dirs_result", None)

    def _dirs(self, argv, route=None):
        """get_comfyui_dirs() with ComfyUI reporting *argv*, and the extension's
        /agent/comfy_dirs answering *route* (None = absent, as on an older build)."""
        class Client:
            def get(_self, path, *a, **k):
                if path == "/system_stats":
                    return {"system": {"argv": list(argv)}}
                if path == "/agent/comfy_dirs":
                    if route is None:
                        raise RuntimeError("404 Not Found")
                    return route
                raise AssertionError(path)

        with mock.patch.object(self.mod, "get_client", lambda: Client()):
            fn = getattr(self.mod.get_comfyui_dirs, "func", self.mod.get_comfyui_dirs)
            return json.loads(fn())

    def test_a_relative_argv0_is_not_resolved_against_our_own_directory(self):
        """The regression. "./main.py" says nothing about where ComfyUI lives, and
        the old code turned it into an absolute path under whatever directory
        agentY happened to be running from."""
        got = self._dirs(["./main.py"])
        self.assertEqual(got["user_dir"], "unknown")
        self.assertNotIn("agentY", got["user_dir"])

    def test_a_bare_filename_is_not_resolved_either(self):
        # `python main.py` from inside the ComfyUI folder.
        self.assertEqual(self._dirs(["main.py"])["user_dir"], "unknown")

    def test_an_absolute_argv0_still_names_the_root(self):
        got = self._dirs(["/opt/ComfyUI/main.py"])
        self.assertEqual(got["user_dir"], str(Path("/opt/ComfyUI/user")))
        self.assertEqual(got["input_dir"], str(Path("/opt/ComfyUI/input")))

    def test_an_explicit_flag_still_wins(self):
        got = self._dirs(["./main.py", "--user-directory", "/srv/projectA/user"])
        self.assertEqual(got["user_dir"], "/srv/projectA/user")

    def test_comfyui_is_asked_when_argv_cannot_say(self):
        """The fix: folder_paths knows, so ask the process that has it."""
        got = self._dirs(["./main.py"], route={
            "ok": True, "user_dir": "/real/ComfyUI/user",
            "input_dir": "/real/ComfyUI/input", "output_dir": "/real/ComfyUI/output"})
        self.assertEqual(got["user_dir"], "/real/ComfyUI/user")
        self.assertEqual(got["source"], "argv+comfyui")

    def test_what_comfyui_says_beats_what_argv0_implies(self):
        """A ComfyUI started by absolute path but with a moved user directory —
        the project switch this whole store depends on."""
        got = self._dirs(["/opt/ComfyUI/main.py"], route={
            "ok": True, "user_dir": "/srv/projectB/user"})
        self.assertEqual(got["user_dir"], "/srv/projectB/user")

    def test_an_extension_that_fails_is_not_fatal(self):
        """An install without the sidebar is a supported install."""
        got = self._dirs(["/opt/ComfyUI/main.py"], route={"ok": False, "error": "no"})
        self.assertEqual(got["user_dir"], str(Path("/opt/ComfyUI/user")))

    def test_unknown_survives_to_the_store_as_no_memory(self):
        """"unknown" must reach project_memory as None, never as a directory named
        "unknown" quietly created next to the agent."""
        with mock.patch("src.tools.comfyui.get_comfyui_dirs",
                        side_effect=lambda: json.dumps({"user_dir": "unknown"})):
            pm.forget_miss()
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
