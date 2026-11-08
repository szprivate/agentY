"""What Magnific says of a finished creation is kept beside the downloaded file (the sidecar's
``generator`` key, tags "magnific" + the model) - the viewer's Dependencies tab reads it from there."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.utils import magnific_watch as mw
from src.utils import output_tags as ot

# A creations_get body as Magnific's MCP answers it (key:value text; this one seen on ws26, 2026-11-08).
BODY = """identifier: ubCCT8eQLD
family: 20077247-e51d-41b6-a2e6-4467b3a5e770
tool: video-generator
kind: video
status: completed
createdAt: "2026-07-24T19:57:08+00:00"
url: "https://pikaso.cdnpk.net/private/x/ubCCT8eQLD.mp4?token=SECRET123&exp=1"
metadata:
  prompt: apply the exact motion of the reference video to the reference image.
  mode: pro-2.0
  duration: 4
  resolution: 480p
  api: bytedance
  mediaCollection[0]:
    url: "https://pikaso.cdnpk.net/private/x/audio.mp3?token=SECRET456"
  credits: 580
webUrl: "https://www.magnific.com/app/creation/ubCCT8eQLD?utm_source=mcp&amp;utm_medium=ai_connector"
"""


class MagnificRecordTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_the_metadata_block_and_the_record(self):
        md = mw.parse_metadata_block(BODY)
        self.assertEqual(md, {"prompt": "apply the exact motion of the reference video to the reference image.",
                              "mode": "pro-2.0", "duration": "4", "resolution": "480p", "api": "bytedance",
                              "credits": "580"})
        rec = mw.creation_record("ubCCT8eQLD", mw._parse_kv_text(BODY), BODY)
        self.assertEqual((rec["service"], rec["model"], rec["provider"], rec["credits"], rec["tool"]),
                         ("magnific", "pro-2.0", "bytedance", "580", "video-generator"))
        self.assertEqual(rec["created"], "2026-07-24T19:57:08+00:00")
        self.assertTrue(rec["web_url"].startswith("https://www.magnific.com/app/creation/ubCCT8eQLD?utm_source=mcp&utm"))
        self.assertNotIn("SECRET", json.dumps(rec))

    def test_tokens_are_masked_in_the_log(self):
        out = mw._mask_tokens(BODY)
        self.assertNotIn("SECRET123", out)
        self.assertNotIn("SECRET456", out)
        self.assertIn("token=<masked>", out)

    def test_the_sidecar_keeps_it_beside_the_role_and_the_provenance(self):
        f = self.dir / "magnific_ubCCT8eQLD.mp4"
        f.write_bytes(b"\x00")
        rec = mw.creation_record("ubCCT8eQLD", mw._parse_kv_text(BODY), BODY)
        self.assertTrue(ot.write_generator(f, rec))
        doc = ot.read_sidecar(f)
        self.assertEqual(doc["generator"]["model"], "pro-2.0")
        self.assertEqual(doc["tags"], ["magnific", "pro-2.0"])
        sc = ot.sidecar_path(f)
        doc["provenance"] = {"v": 1}
        sc.write_text(json.dumps(doc))
        self.assertTrue(ot.write_sidecar(f, "hero shot"))       # a role written later keeps both
        doc = ot.read_sidecar(f)
        self.assertEqual((doc["role"], doc["generator"]["provider"], doc["provenance"]), ("hero shot", "bytedance", {"v": 1}))
        self.assertEqual(doc["tags"], ["magnific", "pro-2.0"])
        self.assertFalse(ot.write_generator(self.dir / "nope.mp4", rec))
        self.assertFalse(ot.write_generator(f, {}))


if __name__ == "__main__":
    unittest.main()
