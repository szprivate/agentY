"""3D model results come back from a run like images and videos do.

run_workflow_now read only the images / gifs / videos / audio keys of ComfyUI's
history. SaveGLB reports its mesh under "3d", Preview3D as a bare path in
"result", and some mesh exporters report nothing, so a finished .glb looked like
"no outputs" and the agent ran the workflow again (four times, in one case).

    python -m unittest tests.test_model_outputs
"""
import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src import executor
from src.executor import _extract_output_files, _process_completed_job


def hist(outputs, start_ms=None):
    msgs = [["execution_start", {"prompt_id": "p1", "timestamp": start_ms}]] if start_ms else []
    return {"p1": {"status": {"status_str": "success", "completed": True, "messages": msgs},
                   "outputs": outputs}}


def run(history, workflow_path=""):
    paths: list = []

    async def go():
        return [line async for line in _process_completed_job(
            history, "p1", {}, workflow_path=workflow_path, verbose=False,
            collected_paths=paths)]
    return asyncio.run(go()), paths


class ExtractTest(unittest.TestCase):
    def test_saveglb_3d_key(self):
        files = _extract_output_files(hist({"9": {"3d": [
            {"filename": "mesh_00001_.glb", "subfolder": "3d", "type": "output"}]}}))
        self.assertEqual([(f["filename"], f["subfolder"], f["node_id"]) for f in files],
                         [("mesh_00001_.glb", "3d", "9")])

    def test_any_key_with_file_items(self):
        files = _extract_output_files(hist({"4": {"mesh": [{"filename": "a.ply", "type": "output"}]}}))
        self.assertEqual([f["filename"] for f in files], ["a.ply"])

    def test_preview3d_result_path(self):
        files = _extract_output_files(hist({"5": {"result": ["3d/mesh_00001_.glb", {"position": 1}]}}))
        self.assertEqual([(f["filename"], f["subfolder"]) for f in files], [("mesh_00001_.glb", "3d")])

    def test_annotated_and_input_paths(self):
        files = _extract_output_files(hist({"5": {"result": ["mesh.glb [output]"]},
                                            "6": {"result": ["loaded.glb [input]"]}}))
        self.assertEqual([f["filename"] for f in files], ["mesh.glb"])

    def test_absolute_path(self):
        files = _extract_output_files(hist({"5": {"result": ["D:\\out\\m.fbx"]}}))
        self.assertEqual(files[0]["abs_path"], "D:/out/m.fbx")

    def test_non_model_strings_ignored(self):
        self.assertEqual(_extract_output_files(hist({"5": {"text": ["hello"], "animated": [True]}})), [])

    def test_images_unchanged(self):
        files = _extract_output_files(hist({"1": {"images": [{"filename": "a.png", "type": "output"}]}}))
        self.assertEqual([f["filename"] for f in files], ["a.png"])


class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        p = mock.patch.object(executor, "_get_comfyui_output_dir", return_value=self.out)
        p.start()
        self.addCleanup(p.stop)

    def test_saveglb_and_preview_collected_once(self):
        (self.out / "3d").mkdir()
        glb = self.out / "3d" / "mesh_00001_.glb"
        glb.write_bytes(b"glTF")
        lines, paths = run(hist({"9": {"3d": [{"filename": glb.name, "subfolder": "3d", "type": "output"}]},
                                 "10": {"result": ["3d/mesh_00001_.glb", {}]}}))
        self.assertEqual(paths, [str(glb)])
        self.assertTrue(any("Done" in str(line) for line in lines), lines)

    def test_unreported_export_found_by_prefix(self):
        (self.out / "3D").mkdir()
        start = time.time()
        glb = self.out / "3D" / "Hy3D_00001_.glb"
        glb.write_bytes(b"glTF")
        old = self.out / "3D" / "Hy3D_old.glb"
        old.write_bytes(b"glTF")
        os.utime(old, (start - 3600, start - 3600))
        wf = Path(tempfile.mkdtemp()) / "w.json"
        wf.write_text(json.dumps({"7": {"class_type": "Hy3DExportMesh",
                                        "inputs": {"filename_prefix": "3D/Hy3D", "file_format": "glb"}}}))
        lines, paths = run(hist({}, start_ms=int(start * 1000)), workflow_path=str(wf))
        self.assertEqual(paths, [str(glb)])

    def test_nothing_written_still_says_no_outputs(self):
        wf = Path(tempfile.mkdtemp()) / "w.json"
        wf.write_text(json.dumps({"7": {"class_type": "Hy3DExportMesh",
                                        "inputs": {"filename_prefix": "3D/Hy3D"}}}))
        lines, paths = run(hist({}, start_ms=int(time.time() * 1000)), workflow_path=str(wf))
        self.assertEqual(paths, [])
        self.assertTrue(any(executor.NO_OUTPUT_MARKER in str(line) for line in lines))


if __name__ == "__main__":
    unittest.main()
