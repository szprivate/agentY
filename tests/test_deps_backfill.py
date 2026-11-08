"""Older files get their dependencies recorded in the background. Run: python -m unittest tests.test_deps_backfill"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import dependencies as dp
from agenty_core.utils import provenance as pv
from src.utils import deps_backfill as bf


class Backfill(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inp, self.out = self.tmp / "input", self.tmp / "output"
        self.inp.mkdir()
        self.out.mkdir()
        dp.set_roots({"input": str(self.inp), "output": str(self.out), "temp": str(self.tmp / "temp")})
        (self.inp / "ref.png").write_bytes(b"ref" * 500)
        g = {"1": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
             "2": {"class_type": "KSampler", "inputs": {"latent_image": ["1", 0], "seed": 1}},
             "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "r"}}}
        self.render = self.out / "r_00001.png"
        self.render.write_bytes(b"render" * 300)
        pv.write(self.render, g, output_node="3")
        # As it was before dependency tracking: the graph is stored, no record, no reverse list.
        sc = pv.sidecar_path(self.render)
        doc = json.loads(sc.read_text())
        doc["provenance"].pop("deps", None)
        sc.write_text(json.dumps(doc))
        if dp.derived_path(self.inp / "ref.png").exists():
            dp.derived_path(self.inp / "ref.png").unlink()

    def tearDown(self):
        dp.set_roots(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_records_older_files(self):
        with mock.patch.object(bf, "STATE_FILE", self.tmp / "state.json"):
            got = bf.run(delay=0)
        self.assertEqual(got["recorded"], 1)
        self.assertEqual([os.path.basename(i["path"]) for i in dp.derived_of(self.inp / "ref.png")], ["r_00001.png"])
        self.assertTrue((self.tmp / "state.json").is_file())

    def test_switch(self):
        with mock.patch.dict(os.environ, {"AGENTY_DEPS_BACKFILL": "0"}):
            self.assertFalse(bf.enabled())
        with mock.patch.dict(os.environ, {"AGENTY_DEPS_BACKFILL": "1"}):
            self.assertTrue(bf.enabled())


if __name__ == "__main__":
    unittest.main()
