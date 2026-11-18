"""The viewer's packages go into ComfyUI's Python, found from the ComfyUI folder.

    python -m unittest tests.test_viewer_deps
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("viewer_deps", ROOT / "scripts" / "viewer_deps.py")
vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vd)


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class LayoutTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.comfy = self.base / "ComfyUI"

    def test_viewer_under_any_folder_name(self):
        nodes = self.comfy / "custom_nodes"
        _touch(nodes / "other" / "requirements.txt", "numpy\n")
        _touch(nodes / "comfyui-imageviewer" / "requirements.txt", "usd-core\n")
        _touch(nodes / "comfyui-imageviewer" / "pyproject.toml", 'Repository = "https://github.com/szprivate/ComfyUI-ImageViewer"\n')
        self.assertEqual(vd.find_viewer(nodes), nodes / "comfyui-imageviewer")

    def test_python_inside_comfy_venv(self):
        py = _touch(self.comfy / ".venv" / "bin" / "python")
        self.assertEqual(vd.find_comfy_python(self.comfy), py)

    def test_windows_portable(self):
        py = _touch(self.base / "python_embeded" / "python.exe")
        self.assertEqual(vd.find_comfy_python(self.comfy), py)

    def test_mac_standalone(self):
        py = _touch(self.base / "standalone-env" / "bin" / "python")
        self.assertEqual(vd.find_comfy_python(self.comfy), py)

    def test_none(self):
        self.assertIsNone(vd.find_comfy_python(self.comfy))


class MissingTest(unittest.TestCase):
    def test_requirements_skip_comments_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = _touch(Path(tmp) / "requirements.txt", "# c\nPillow\n-e .\nusd-core>=24 # USD\n\n")
            self.assertEqual(vd.requirements(req), ["Pillow", "usd-core>=24"])

    def test_import_names(self):
        probed = []

        def run(cmd, capture_output):
            probed.append(cmd[-1])
            return mock.Mock(returncode=0 if cmd[-1] == "import PIL" else 1)

        with mock.patch.object(vd.subprocess, "run", run):
            gone = vd.missing(Path(sys.executable), ["Pillow", "usd-core>=24", "trimesh"])
        self.assertEqual(probed, ["import PIL", "import pxr", "import trimesh"])
        self.assertEqual(gone, ["usd-core>=24", "trimesh"])


if __name__ == "__main__":
    unittest.main()
