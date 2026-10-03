"""The settings page never waits for the model lists once one exists.

Listing asks every vendor's /models, and on Windows a refused connection to an
Ollama that isn't running costs a second per address; with Ollama absent the
list was kept for 20 s, so nearly every opening of the settings took ~4 s.
"""

import socket
import threading
import time
import unittest
from unittest import mock

from src.utils import agentY_server as srv


class ModelListCache(unittest.TestCase):

    def setUp(self):
        saved = dict(srv._MODEL_CACHE)
        self.addCleanup(lambda: (srv._MODEL_CACHE.clear(), srv._MODEL_CACHE.update(saved)))

    def test_a_stale_list_is_answered_at_once_and_refreshed_behind(self):
        srv._MODEL_CACHE.update(groups={"Old": []}, ts=time.time() - 3600, ttl=20)
        started, release = threading.Event(), threading.Event()

        def slow():
            started.set()
            release.wait(5)
            srv._MODEL_CACHE.update(groups={"New": []}, ts=time.time(), ttl=300)
            return srv._MODEL_CACHE["groups"]

        with mock.patch.object(srv, "_list_models_now", side_effect=slow):
            t = time.time()
            self.assertEqual(srv._available_models(), {"Old": []})
            self.assertLess(time.time() - t, 0.5)
            self.assertTrue(started.wait(2))
            # A second caller meanwhile neither waits nor starts another refresh.
            self.assertEqual(srv._available_models(), {"Old": []})
            release.set()
            for _ in range(100):
                if srv._MODEL_CACHE["groups"] == {"New": []} and not srv._MODEL_REFRESH.locked():
                    break
                time.sleep(0.02)
        self.assertEqual(srv._available_models(), {"New": []})

    def test_the_first_call_has_to_wait(self):
        srv._MODEL_CACHE.clear()
        with mock.patch.object(srv, "_list_models_now", return_value={"A": []}) as now:
            self.assertEqual(srv._available_models(), {"A": []})
        now.assert_called_once()


class QuickPortCheck(unittest.TestCase):

    def test_nothing_listening_is_known_quickly(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        t = time.time()
        self.assertFalse(srv._port_open(f"http://localhost:{port}"))
        self.assertLess(time.time() - t, 1.0)

    def test_a_listener_is_found(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        self.addCleanup(s.close)
        self.assertTrue(srv._port_open(f"http://localhost:{s.getsockname()[1]}"))

    def test_remote_hosts_are_left_to_the_request(self):
        self.assertTrue(srv._port_open("http://gpu-box.example:11434"))


class OllamaOverIPv4(unittest.TestCase):
    """Windows tries "localhost" as IPv6 first; Ollama listens on IPv4 only, and
    every request waited ~2 s for the refusal (the memory embedder: every turn)."""

    def setUp(self):
        from src.utils import settings
        self.settings = settings
        settings._LOCALHOST_SEEN.clear()
        self.addCleanup(settings._LOCALHOST_SEEN.clear)

    def test_localhost_becomes_127_when_it_listens_there(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        self.addCleanup(s.close)
        port = s.getsockname()[1]
        self.assertEqual(self.settings._ipv4_for_localhost(f"http://localhost:{port}"),
                         f"http://127.0.0.1:{port}")

    def test_anything_else_is_left_alone(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertEqual(self.settings._ipv4_for_localhost(f"http://localhost:{port}"),
                         f"http://localhost:{port}")
        self.assertEqual(self.settings._ipv4_for_localhost("http://gpu-box:11434"), "http://gpu-box:11434")
        self.assertEqual(self.settings._ipv4_for_localhost("http://127.0.0.1:11434"), "http://127.0.0.1:11434")


if __name__ == "__main__":
    unittest.main()
