"""POST /agentY/comp_farm: the viewer's comp farm request handed to core's deadline_farm.submit_comp, its
refusals answered as {ok: false, error}."""
import unittest
from unittest import mock

from route_client import authorised_client


class CompFarmRoute(unittest.TestCase):
    def test_submits_and_refuses(self):
        from agenty_core.tools import deadline_farm as df
        c = authorised_client()
        with mock.patch.object(df, "submit_comp", return_value={"ok": True, "job_id": "j9", "tasks": 3}) as sub:
            r = c.post("/agentY/comp_farm", json={"comp": "//s/a.comp.json", "frames": "0-9"})
        self.assertEqual((r.status_code, r.get_json()["job_id"]), (200, "j9"))
        sub.assert_called_once_with({"comp": "//s/a.comp.json", "frames": "0-9"})
        with mock.patch.object(df, "submit_comp", side_effect=df.FarmError("Deadline not configured", "config")):
            r = c.post("/agentY/comp_farm", json={})
        self.assertEqual((r.status_code, r.get_json()["ok"], r.get_json()["error_kind"]), (422, False, "config"))


if __name__ == "__main__":
    unittest.main()
