"""The listener thread must not die silently (host alive, port closed)."""
import unittest
from unittest import mock

from src.utils import agentY_server as srv


class _Dying:
    def __init__(self, exc):
        self.exc = exc
        self.closed = False

    def serve_forever(self):
        raise self.exc

    def server_close(self):
        self.closed = True


class ServeResilientTest(unittest.TestCase):
    def setUp(self):
        srv._server_failed = ""
        p = mock.patch.object(srv, "_host_log")
        self.log = p.start()
        self.addCleanup(p.stop)

    def test_rebinds_then_gives_up_with_a_reason(self):
        made = []

        def make(host, port, app, threaded):
            made.append(_Dying(OSError("boom")))
            return made[-1]

        srv._serve_resilient("127.0.0.1", 1, None, make=make, max_restarts=2, pause=0)
        self.assertEqual(len(made), 3)
        self.assertTrue(all(m.closed for m in made))
        self.assertIn("died 3 times", srv.server_failure())

    def test_bind_failure_is_retried_not_swallowed(self):
        calls = []

        def make(host, port, app, threaded):
            calls.append(1)
            raise OSError(10048, "address in use")

        srv._serve_resilient("127.0.0.1", 1, None, make=make, max_restarts=1, pause=0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(srv.server_failure())


if __name__ == "__main__":
    unittest.main()
