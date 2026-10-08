"""A finished stream must not leave a connection the browser thinks it can reuse.

The host runs on Werkzeug's server, which answers one request per connection.
The stream responses used to say ``Connection: keep-alive``; the browser then
sent its next request down the finished turn's connection, where it was
swallowed and never answered. A handful of turns later all six of the browser's
connections to the host were gone and the panel hung "after 'Orchestrator
finished'".
"""
import re
import unittest
from pathlib import Path

from tests.route_client import authorised_client

SERVER = (Path(__file__).resolve().parent.parent / "src" / "utils" / "agentY_server.py"
          ).read_text(encoding="utf-8")


class StreamsCloseTheirConnection(unittest.TestCase):
    def test_a_run_stream_says_close(self):
        resp = authorised_client().get("/agentY/runs/nosuchrun/stream")
        self.assertEqual(resp.mimetype, "text/event-stream")
        self.assertEqual(resp.headers.get("Connection"), "close")
        resp.close()

    def test_no_response_promises_keep_alive(self):
        self.assertEqual(re.findall(r"""headers\[["']Connection["']\]\s*=\s*["']keep-alive""", SERVER), [])

    def test_every_stream_goes_through_the_one_wrapper(self):
        # A stream built by hand would not carry the header.
        self.assertEqual(SERVER.count('mimetype="text/event-stream"'), 1)


if __name__ == "__main__":
    unittest.main()
