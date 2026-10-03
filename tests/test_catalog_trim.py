"""An earlier turn's workflow catalog leaves the history; the turn reading it keeps it.

A catalog is ~140k characters and would otherwise ride along in every model call
for the whole history window. Setting ``trim_old_workflow_catalog`` (on by default)
turns it off again.
"""

import json
import unittest
from unittest import mock

from src.pipeline import Pipeline

CATALOG = json.dumps({f"template_{i}": "x" * 60 for i in range(300)})


def _turn(use_id, name, result):
    return [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": use_id, "name": name, "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": use_id, "status": "success",
                                                     "content": [{"text": result}]}}]},
    ]


class OldCatalogs(unittest.TestCase):

    def history(self):
        return (_turn("c1", "get_workflow_catalog", CATALOG)
                + _turn("s1", "get_node_schema", "y" * 5000)
                + _turn("c2", "get_workflow_catalog", '{"error": "offline"}'))

    def test_the_catalog_becomes_a_note(self):
        msgs = self.history()
        self.assertEqual(Pipeline._shrink_old_catalogs(msgs), 1)
        note = json.loads(msgs[1]["content"][0]["toolResult"]["content"][0]["text"])["note"]
        self.assertIn("(300 templates)", note)
        self.assertIn("Call get_workflow_catalog again", note)
        # Other results, short ones and the tool calls themselves are untouched.
        self.assertEqual(len(msgs[3]["content"][0]["toolResult"]["content"][0]["text"]), 5000)
        self.assertIn("offline", msgs[5]["content"][0]["toolResult"]["content"][0]["text"])
        self.assertEqual(msgs[0]["content"][0]["toolUse"]["name"], "get_workflow_catalog")
        # A second pass finds nothing left to do.
        self.assertEqual(Pipeline._shrink_old_catalogs(msgs), 0)

    def test_the_setting_keeps_it(self):
        msgs = self.history()
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"trim_old_workflow_catalog": False}):
            self.assertEqual(Pipeline._shrink_old_catalogs(msgs), 0)
        self.assertEqual(msgs[1]["content"][0]["toolResult"]["content"][0]["text"], CATALOG)


if __name__ == "__main__":
    unittest.main()
