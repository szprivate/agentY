"""Media made outside a ComfyUI save node reaches the canvas and the viewer.

An MCP tool's render or a script's output is read from the tool result at the one
hook every agent has, handed to the turn's output sink, and - once the host has
staged it - sent to the viewer's agentY tab and, for a turn nobody streams (a
shot), announced on the notification bus so the open panel drops it. Nothing here
touches ComfyUI: the viewer transport and the sink are stubs.

    python -m unittest tests.test_media_sink
"""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import media_sink as ms
from src.utils import notify_bus


def _touch(folder, name, age=0.0):
    path = os.path.join(folder, name)
    with open(path, "wb") as fh:
        fh.write(b"x")
    if age:
        t = time.time() - age
        os.utime(path, (t, t))
    return path


class Fresh(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        for bucket in (ms._intent, ms._seen, ms._viewer_sent, ms._dropped):
            bucket.clear()
        ms._viewer_off_until = 0.0
        self.sink = mock.Mock()
        p = mock.patch.object(ms, "_publish", self.sink)
        p.start()
        self.addCleanup(p.stop)


class ReadingResults(Fresh):
    def test_an_mcp_render_is_published_under_its_server(self):
        started = time.time()
        mp4 = _touch(self.dir, "final cut.mp4")
        result = {"status": "success", "content": [{"text": f"Rendered. Output: {mp4}"}]}
        self.assertEqual(ms.observe("remotion__render_media", result, started), [mp4])
        self.sink.assert_called_once_with(mp4)
        self.assertEqual(ms.take(mp4), {"group": "remotion", "label": "render_media"})

    def test_the_same_file_twice_is_published_once(self):
        mp4 = _touch(self.dir, "a.mp4")
        ms.observe("remotion__render_media", {"content": [{"text": mp4}]}, time.time())
        ms.observe("remotion__render_media", {"content": [{"text": mp4}]}, time.time())
        self.assertEqual(self.sink.call_count, 1)

    def test_a_file_the_tool_only_echoed_is_not_its_output(self):
        started = time.time()
        old = _touch(self.dir, "input.png", age=3600)
        self.assertEqual(ms.observe("remotion__render_media", {"content": [{"text": old}]}, started), [])
        self.sink.assert_not_called()

    def test_tools_that_publish_their_own_files_are_left_alone(self):
        png = _touch(self.dir, "a.png")
        for name in ("view_image", "web_search_images", "magnific__image_generate", "execute_workflow"):
            self.assertEqual(ms.observe(name, {"content": [{"text": png}]}, time.time()), [], name)

    def test_run_script_output_counts(self):
        png = _touch(self.dir, "plot.png")
        self.assertEqual(ms.observe("run_script", {"content": [{"text": f"wrote {png}"}]}, time.time()), [png])
        self.assertEqual(ms.take(png)["group"], "script")

    def test_a_media_url_from_an_mcp_tool_is_downloaded_first(self):
        saved = _touch(self.dir, "dl.mp4")
        with mock.patch("agenty_core.utils.media_refs.download", return_value=saved) as dl:
            out = ms.observe("fal__video", {"content": [{"text": "https://cdn.x.io/v/out.mp4"}]}, time.time())
        self.assertEqual(out, [saved])
        self.assertEqual(dl.call_args.args[0], "https://cdn.x.io/v/out.mp4")

    def test_urls_are_not_fetched_for_run_script(self):
        with mock.patch("agenty_core.utils.media_refs.download") as dl:
            ms.observe("run_script", {"content": [{"text": "see https://x.io/a.png"}]}, time.time())
        dl.assert_not_called()


class TheHook(Fresh):
    def test_before_and_after_events_publish_what_the_call_wrote(self):
        hook = ms.MediaSinkHookProvider()
        use = {"name": "remotion__render_still", "toolUseId": "t1"}
        hook._before(SimpleNamespace(tool_use=use))
        png = _touch(self.dir, "still.png")
        hook._after(SimpleNamespace(tool_use=use, exception=None,
                                    result={"content": [{"text": png}]}))
        self.sink.assert_called_once_with(png)

    def test_a_failed_call_publishes_nothing(self):
        hook = ms.MediaSinkHookProvider()
        use = {"name": "remotion__render_still", "toolUseId": "t2"}
        hook._before(SimpleNamespace(tool_use=use))
        png = _touch(self.dir, "still.png")
        hook._after(SimpleNamespace(tool_use=use, exception=RuntimeError("x"),
                                    result={"content": [{"text": png}]}))
        self.sink.assert_not_called()


class TheViewer(Fresh):
    def setUp(self):
        super().setUp()
        self.client = mock.Mock()
        self.client.post.return_value = {"ok": True, "shown": 1}
        p = mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=self.client)
        p.start()
        self.addCleanup(p.stop)

    def test_a_file_is_pushed_once_into_the_agenty_tab(self):
        png = _touch(self.dir, "a.png")
        self.assertTrue(ms.push_to_viewer(png, "remotion", "still", wait=True))
        self.assertFalse(ms.push_to_viewer(png, "remotion", "still", wait=True))
        path, = self.client.post.call_args.args
        body = self.client.post.call_args.kwargs["json_data"]
        self.assertEqual(path, "/bepic/push")
        self.assertEqual((body["paths"], body["group"], body["label"]), ([png], "remotion", "still"))
        self.assertTrue(body["tab_name"])

    def test_a_viewer_without_the_route_is_not_asked_again(self):
        self.client.post.side_effect = RuntimeError("404 Not Found")
        self.assertFalse(ms.push_to_viewer(_touch(self.dir, "a.png"), "g", wait=True))
        self.assertFalse(ms.push_to_viewer(_touch(self.dir, "b.png"), "g", wait=True))
        self.assertEqual(self.client.post.call_count, 1)

    def test_it_can_be_switched_off(self):
        with mock.patch.dict(os.environ, {"AGENTY_VIEWER_PUSH": "0"}):
            self.assertFalse(ms.push_to_viewer(_touch(self.dir, "a.png"), "g", wait=True))
        self.client.post.assert_not_called()


class FromTheHost(Fresh):
    """``_deliver_elsewhere`` in the server, with the transports stubbed."""

    def setUp(self):
        super().setUp()
        from src.utils import agentY_server as srv
        self.srv = srv
        self.push = mock.Mock(return_value=True)
        for target, value in ((ms, "push_to_viewer"),):
            p = mock.patch.object(target, value, self.push)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(srv, "_viewer_copy", side_effect=lambda path, staged: "IN/" + str(staged))
        p.start()
        self.addCleanup(p.stop)
        self.since = notify_bus.snapshot()["seq"]

    def events(self):
        return notify_bus.snapshot(self.since)["events"]

    def test_a_shot_turn_announces_its_file_for_the_canvas_once(self):
        png = _touch(self.dir, "shot1.png")
        self.srv._deliver_elsewhere(png, "image", "agentY/shot1.png", "", None, "shots", "t")
        self.srv._deliver_elsewhere(png, "image", "agentY/shot1.png", "", None, "shots", "t")
        got = self.events()
        self.assertEqual(len(got), 1)
        out = got[0]["output"]
        self.assertEqual((got[0]["kind"], out["path"], out["filename"], out["show"]),
                         ("media", png, "agentY/shot1.png", False))

    def test_a_panel_turn_leaves_the_canvas_to_its_own_stream(self):
        png = _touch(self.dir, "lead.png")
        self.srv._deliver_elsewhere(png, "image", "agentY/lead.png", "", None, "panel", "t")
        self.assertEqual(self.events(), [])

    def test_only_a_file_with_a_note_goes_to_the_viewer(self):
        saved = _touch(self.dir, "save_node.png")
        other = _touch(self.dir, "mcp.mp4")
        ms.note(other, "remotion", "render_media")
        self.srv._deliver_elsewhere(saved, "image", "agentY/save_node.png", "", None, "panel", "t")
        self.push.assert_not_called()
        self.srv._deliver_elsewhere(other, "video", "agentY/mcp.mp4", "", None, "panel", "t")
        self.push.assert_called_once_with("IN/agentY/mcp.mp4", "remotion", "render_media")

    def test_a_running_shot_keeps_the_panels_poll_going(self):
        from src.utils import turn_bus
        shot = SimpleNamespace(origin="shots")
        with mock.patch.object(turn_bus, "active", return_value=[shot]):
            from tests.route_client import authorised_client
            client = authorised_client()
            self.assertGreaterEqual(client.get("/agentY/notifications?since=0").get_json()["pending"], 1)


if __name__ == "__main__":
    unittest.main()
