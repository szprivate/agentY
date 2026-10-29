"""A prompt enhancer's model does not make a template "not installed".

The official LTX 2.5 templates carry a Gemma-4 E2B prompt enhancer behind a
switch that is off. Its model counted as one the template needs, so on a
machine with every LTX 2.5 file the render uses, the catalog said
"[models not installed]" and the researcher built from an LTX 2.0 template
instead (CX15, night_20261029).

    python -m unittest discover -s tests -t .
"""

import unittest

from src.utils.template_pick import optional_model_files


def _loader(nid, name, links):
    return {"id": nid, "type": "CLIPLoader", "outputs": [{"name": "CLIP", "links": links}],
            "properties": {"models": [{"name": name, "directory": "text_encoders"}]}}


class OnlyForAPromptHelper(unittest.TestCase):

    def test_the_enhancer_model_is_optional_and_the_text_encoder_is_not(self):
        sub = {"nodes": [_loader(1, "gemma4-12b-with-proj.safetensors", [10, 11]),
                         _loader(2, "gemma4_e2b_it.safetensors", [12]),
                         {"id": 3, "type": "CLIPTextEncode"},
                         {"id": 4, "type": "TextGenerateLTX2Prompt"}],
               "links": [{"id": 10, "origin_id": 1, "target_id": 3},
                         {"id": 11, "origin_id": 1, "target_id": 4},
                         [12, 2, 0, 4, 0, "CLIP"]]}
        wf = {"nodes": [], "definitions": {"subgraphs": [sub]}}
        self.assertEqual(optional_model_files(wf), {"gemma4_e2b_it.safetensors"})

    def test_nothing_to_read_is_nothing_optional(self):
        self.assertEqual(optional_model_files({}), set())
        self.assertEqual(optional_model_files(None), set())


if __name__ == "__main__":
    unittest.main()
