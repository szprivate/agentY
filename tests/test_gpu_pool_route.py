"""GET / POST /agentY/pool: the viewer's Farm space handed to core's gpu_pool, its refusals answered as
{ok: false, error}, and finished renders delivered to the viewer."""
import unittest
from unittest import mock

from route_client import authorised_client


class GpuPoolRoute(unittest.TestCase):
    def test_state_actions_and_refusals(self):
        from agenty_core.utils import gpu_pool as gp
        c = authorised_client()
        with mock.patch.object(gp, "state", return_value={"machines": [], "jobs": [], "keys": {"vast": False}}):
            r = c.get("/agentY/pool")
        self.assertEqual((r.status_code, r.get_json()["ok"], r.get_json()["keys"]), (200, True, {"vast": False}))
        with mock.patch.object(gp, "action", return_value={"job": {"id": "j1"}, "state": {}}) as act:
            r = c.post("/agentY/pool", json={"action": "submit", "batch": 4})
        self.assertEqual((r.status_code, r.get_json()["job"]["id"]), (200, "j1"))
        act.assert_called_once_with({"action": "submit", "batch": 4})
        with mock.patch.object(gp, "action", side_effect=gp.PoolError("No VAST_API_KEY is set")):
            r = c.post("/agentY/pool", json={"action": "offers"})
        self.assertEqual((r.status_code, r.get_json()["ok"]), (422, False))
        self.assertIn("VAST_API_KEY", r.get_json()["error"])

    def test_finished_renders_go_to_the_viewer_under_farm(self):
        from src.utils import agentY_server as srv
        from agenty_core.utils import gpu_pool as gp
        authorised_client().get("/agentY/pool")
        self.assertIn(srv._deliver_farm_output, gp.listeners)
        with mock.patch.object(srv, "_stage_into_comfy_input", return_value="agentY/000_a.png"), \
                mock.patch.object(srv, "_deliver_elsewhere") as de:
            srv._deliver_farm_output({"name": "cars"}, {"i": 2}, ["/x/000_a.png", "/x/000_b.mp4"])
        self.assertEqual([c.args[1] for c in de.call_args_list], ["image", "video"])
        self.assertEqual(de.call_args_list[0].args[3], "Farm: cars #3")


if __name__ == "__main__":
    unittest.main()
