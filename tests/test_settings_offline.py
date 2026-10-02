"""Settings can be read and saved while the host is not running.

The settings page shows files in this checkout, but could only be opened while
the host answered — exactly when a setting that stops it starting cannot be
fixed. ``python -m src.utils.settings_offline`` is what the ComfyUI extension
runs instead. It must call the same functions as the routes, and print one JSON
object on its last line whatever a library prints before it.

    python -m unittest discover -s tests
"""

import io
import json
import unittest
from unittest import mock

from src.utils import agentY_server as srv
from src.utils import settings_offline as off


def _run(argv, stdin=b""):
    out = io.StringIO()
    fake_in = mock.Mock()
    fake_in.buffer = io.BytesIO(stdin)
    with mock.patch.object(off.sys, "stdout", out), mock.patch.object(off.sys, "stdin", fake_in), \
            mock.patch.object(off.os, "chdir"), mock.patch.object(off, "_load_env"):
        code = off.main(["settings_offline"] + argv)
    return code, json.loads(out.getvalue().strip().splitlines()[-1])


class OfflineSettingsTest(unittest.TestCase):

    def test_get_is_the_routes_payload_plus_the_mcp_config(self):
        with mock.patch.object(srv, "settings_payload", return_value={"env": {"K": "masked"}}), \
                mock.patch("src.tools.mcp_tools.load_mcp_config", return_value={"servers": {"a": {}}}), \
                mock.patch("src.tools.mcp_tools.mcp_status", return_value={"a": {"state": "disabled"}}):
            code, out = _run(["get"])
        self.assertEqual(code, 0)
        self.assertTrue(out["offline"])
        self.assertEqual(out["settings"], {"env": {"K": "masked"}})
        self.assertEqual(out["mcp"]["config"], {"servers": {"a": {}}})

    def test_a_broken_mcp_file_does_not_take_the_settings_with_it(self):
        with mock.patch.object(srv, "settings_payload", return_value={}), \
                mock.patch("src.tools.mcp_tools.load_mcp_config", side_effect=ValueError("bad json")):
            code, out = _run(["get"])
        self.assertEqual(code, 0)
        self.assertFalse(out["mcp"]["ok"])

    def test_save_goes_through_the_routes_writer_and_saves_mcp_when_sent(self):
        body = {"env": {"A": "1"}, "settings": {"x": 1}, "mcp_config": {"servers": {}}}
        with mock.patch.object(srv, "settings_save", return_value={"env_updated": ["A"]}) as save, \
                mock.patch("src.tools.mcp_tools.save_mcp_config") as save_mcp:
            code, out = _run(["save"], json.dumps(body).encode())
        save.assert_called_once_with(body)
        save_mcp.assert_called_once_with({"servers": {}})
        self.assertEqual((code, out["env_updated"], out["mcp_saved"]), (0, ["A"], True))

    def test_save_without_mcp_leaves_mcp_alone(self):
        with mock.patch.object(srv, "settings_save", return_value={}), \
                mock.patch("src.tools.mcp_tools.save_mcp_config") as save_mcp:
            _run(["save"], b'{"env": {}}')
        save_mcp.assert_not_called()

    def test_a_failure_is_one_json_line_and_a_nonzero_exit(self):
        with mock.patch.object(srv, "settings_payload", side_effect=RuntimeError("no toml")):
            code, out = _run(["get"])
        self.assertEqual(code, 1)
        self.assertIn("no toml", out["error"])
        self.assertEqual(_run(["frobnicate"])[0], 1)


class RouteAndHelperShareOneWriter(unittest.TestCase):

    def test_settings_save_writes_each_part_and_skips_masked_keys(self):
        with mock.patch.object(srv, "_update_env_file") as env, \
                mock.patch.object(srv, "_update_settings_file", return_value=["x"]) as st, \
                mock.patch.object(srv, "_save_pricing_config") as pr:
            got = srv.settings_save({"env": {"A": "new", "B": srv._SECRET_MASK},
                                     "settings": {"x": 1}, "pricing": {"m": {}}})
        env.assert_called_once_with({"A": "new"})
        st.assert_called_once_with({"x": 1})
        pr.assert_called_once_with({"m": {}})
        self.assertEqual(got, {"env_updated": ["A"], "settings_updated": ["x"], "pricing_updated": True})

    def test_an_empty_save_writes_nothing(self):
        with mock.patch.object(srv, "_update_env_file") as env, \
                mock.patch.object(srv, "_update_settings_file") as st, \
                mock.patch.object(srv, "_save_pricing_config") as pr:
            self.assertEqual(srv.settings_save({}), {})
        for m in (env, st, pr):
            m.assert_not_called()


if __name__ == "__main__":
    unittest.main()
