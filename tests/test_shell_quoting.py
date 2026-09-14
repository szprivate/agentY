"""`python -c "…"` runs the code it is given, on Windows too.

On Windows the sandbox split commands with non-POSIX shlex, which keeps quotes: the
program received `"print('hello')"` with the double quotes still on. Python took
that as a string literal, evaluated it, printed nothing and exited 0. Every inline
probe the agent ran came back empty with no error, so it decided the shell was
broken and never downloaded the models it had been asked for.

    python -m unittest discover -s tests
"""

import json
import shlex
import unittest

from agenty_core import sandbox
from agenty_core.tools.shell import run_script


class SplittingOnWindows(unittest.TestCase):

    def split(self, command):
        return sandbox.split_command(command, windows=True)

    def test_inline_python_gets_its_code_without_the_quotes(self):
        self.assertEqual(self.split('python -c "print(\'hello\')"'),
                         ["python", "-c", "print('hello')"])

    def test_single_quotes_come_off_too(self):
        self.assertEqual(self.split("python -c 'print(1)'"), ["python", "-c", "print(1)"])

    def test_a_quoted_path_with_spaces_is_one_argument(self):
        self.assertEqual(self.split('python "C:/My Folder/run.py" --fast'),
                         ["python", "C:/My Folder/run.py", "--fast"])

    def test_backslashes_in_windows_paths_survive(self):
        self.assertEqual(self.split(r"python C:\Users\me\probe.py"),
                         ["python", r"C:\Users\me\probe.py"])

    def test_quotes_inside_the_code_are_kept(self):
        self.assertEqual(self.split('python -c "print(\'a b\')"')[2], "print('a b')")

    def test_unquoted_arguments_are_untouched(self):
        self.assertEqual(self.split("ffmpeg -i in.mp4 -crf 18 out.mp4"),
                         ["ffmpeg", "-i", "in.mp4", "-crf", "18", "out.mp4"])

    def test_a_lone_quote_character_is_not_stripped_to_nothing(self):
        self.assertEqual(sandbox._unquote('"'), '"')


class SplittingElsewhere(unittest.TestCase):

    def test_posix_rules_are_unchanged(self):
        command = 'python -c "print(\'hello\')" --flag'
        self.assertEqual(sandbox.split_command(command, windows=False),
                         shlex.split(command, posix=True))


class ThroughRunScript(unittest.TestCase):
    """The end-to-end check that would have caught it: the code actually runs."""

    def test_inline_python_prints(self):
        out = json.loads(run_script('python -c "print(\'agenty\')"'))
        self.assertEqual(out.get("exit_code"), 0, out)
        self.assertEqual(out.get("stdout"), "agenty")

    def test_inline_python_errors_are_reported(self):
        out = json.loads(run_script('python -c "raise SystemExit(3)"'))
        self.assertEqual(out.get("exit_code"), 3, out)


if __name__ == "__main__":
    unittest.main()
