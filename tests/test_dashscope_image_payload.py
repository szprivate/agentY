"""Images sent to DashScope stay small and are not re-sent turn after turn.

DashScope fetches every inline image from its own storage before the model
reads it; several multi-MB PNGs in one request made that fetch time out with
``invalid_parameter_error`` "Download multimodal file timed out", and a 400 was
never retried. Three things now hold:

* embedded images are JPEG (PNG only with real transparency), and for DashScope
  no longer than 1280 px (src/utils/vision_payload.py);
* between turns, images of all but the latest turn leave the history
  (Pipeline._strip_old_images);
* that error is retried, but only before anything has streamed
  (src/utils/transient_retry.py).

    PYTHONPATH=.:tests python -m unittest tests/test_dashscope_image_payload.py
"""
import asyncio
import io
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from src.pipeline import Pipeline
from src.utils import transient_retry, vision_payload


def _png(w, h, alpha=False):
    img = Image.new("RGBA" if alpha else "RGB", (w, h), (200, 30, 30, 128 if alpha else 255))
    # Noise, so the PNG is not trivially small.
    img.putdata([((x * 7) % 256, (x * 13) % 256, (x * 3) % 256) + ((128,) if alpha else ())
                 for x in range(w * h)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _size(data):
    return Image.open(io.BytesIO(data)).size


class Payload(unittest.TestCase):
    def setUp(self):
        p = mock.patch("src.utils.settings.load_settings", return_value={"llm": {}})
        p.start()
        self.addCleanup(p.stop)

    def test_dashscope_gets_jpeg_capped_at_1280(self):
        data, fmt = vision_payload.for_model(_png(1500, 900), "png", "dashscope,qwen3.8-max")
        self.assertEqual(fmt, "jpeg")
        self.assertEqual(max(_size(data)), 1280)

    def test_claude_gets_jpeg_at_the_usual_edge(self):
        data, fmt = vision_payload.for_model(_png(1500, 900), "png", "claude,claude-sonnet-5-5")
        self.assertEqual(fmt, "jpeg")
        self.assertEqual(max(_size(data)), 1500)

    def test_transparency_keeps_png(self):
        data, fmt = vision_payload.for_model(_png(300, 200, alpha=True), "png", "dashscope,qwen3.8-max")
        self.assertEqual(fmt, "png")

    def test_setting_overrides_the_edge(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"llm": {"embed_max_edge": 512}}):
            data, _ = vision_payload.for_model(_png(1000, 1000), "png", "claude,x")
        self.assertEqual(max(_size(data)), 512)

    def test_build_content_embeds_jpeg_for_dashscope(self):
        from src.utils import agentY_server as srv
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frame.png")
            with open(path, "wb") as fh:
                fh.write(_png(1600, 900))
            with mock.patch.object(srv, "_orchestrator_spec", return_value="dashscope,qwen3.8-max"):
                blocks = srv._build_content("look", [path], embed_images=True)
        image = [b for b in blocks if "image" in b][0]["image"]
        self.assertEqual(image["format"], "jpeg")
        self.assertEqual(max(_size(image["source"]["bytes"])), 1280)
        self.assertIn(path, blocks[0]["text"])


def _img():
    return {"image": {"format": "jpeg", "source": {"bytes": b"x"}}}


def _history():
    return [
        {"role": "user", "content": [{"text": "turn 1 /a.png"}, _img()]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "analyze_image", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "/b.png"}, _img()]}}]},
        {"role": "assistant", "content": [{"text": "done"}]},
        {"role": "user", "content": [{"text": "turn 2 /c.png"}, _img()]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t2", "name": "analyze_image", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t2", "content": [{"text": "/d.png"}, _img()]}}]},
        {"role": "assistant", "content": [{"text": "done"}]},
    ]


def _images(msg):
    n = 0
    for b in msg["content"]:
        if "image" in b:
            n += 1
        if "toolResult" in b:
            n += sum("image" in c for c in b["toolResult"]["content"])
    return n


class OldImages(unittest.TestCase):
    def test_only_the_latest_turn_keeps_its_images(self):
        msgs = _history()
        self.assertEqual(Pipeline._strip_old_images(msgs, keep_turns=1), 2)
        self.assertEqual([_images(m) for m in msgs], [0, 0, 0, 0, 1, 0, 1, 0])
        self.assertIn("earlier turn", msgs[0]["content"][1]["text"])
        self.assertEqual(msgs[0]["content"][0]["text"], "turn 1 /a.png")   # paths stay

    def test_zero_drops_all_and_negative_keeps_all(self):
        msgs = _history()
        self.assertEqual(Pipeline._strip_old_images(msgs, keep_turns=0), 4)
        msgs = _history()
        self.assertEqual(Pipeline._strip_old_images(msgs, keep_turns=-1), 0)

    def test_setting_is_read(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"llm": {"history_image_turns": 2}}):
            self.assertEqual(Pipeline._strip_old_images(_history()), 0)


class _Timeout(Exception):
    pass


def _model(failures, then_events=("a", "b"), fail_after_start=False):
    calls = {"n": 0}

    class Base:
        async def stream(self, *args, **kwargs):
            calls["n"] += 1
            if fail_after_start:
                yield "a"
                raise _Timeout("Download multimodal file timed out")
            if calls["n"] <= failures:
                raise _Timeout('data: {"error":{"code":"invalid_parameter_error",'
                               '"message":"Download multimodal file timed out"}}')
            for e in then_events:
                yield e

    class M(transient_retry.TransientRetryMixin, Base):
        retry_delays = (0, 0)

    return M(), calls


def _collect(model):
    async def run():
        return [e async for e in model.stream([])]
    return asyncio.run(run())


class Retry(unittest.TestCase):
    def test_the_timeout_is_retried_until_it_goes_through(self):
        m, calls = _model(failures=2)
        self.assertEqual(_collect(m), ["a", "b"])
        self.assertEqual(calls["n"], 3)

    def test_it_gives_up_after_the_retries(self):
        m, calls = _model(failures=5)
        with self.assertRaises(_Timeout):
            _collect(m)
        self.assertEqual(calls["n"], 3)

    def test_other_errors_are_not_retried(self):
        class Base:
            async def stream(self, *a, **k):
                raise ValueError("Unexpected item type in content.")
                yield  # pragma: no cover

        class M(transient_retry.TransientRetryMixin, Base):
            retry_delays = (0, 0)

        with self.assertRaises(ValueError):
            _collect(M())

    def test_nothing_is_repeated_once_streaming_started(self):
        m, calls = _model(failures=0, fail_after_start=True)
        with self.assertRaises(_Timeout):
            _collect(m)
        self.assertEqual(calls["n"], 1)

    def test_the_real_model_class_has_the_retry(self):
        from src.utils import usage_capture
        cls = usage_capture.capturing_openai_model()
        self.assertTrue(issubclass(cls, transient_retry.TransientRetryMixin))


if __name__ == "__main__":
    unittest.main()
