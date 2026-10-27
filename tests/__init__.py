"""Makes this directory a package, and keeps it importable from inside itself.

Two things are needed for `python -m unittest discover -s tests` to work here, and
both are accidents of history rather than design:

* This file has to exist at all. A dependency (literalai) ships a top-level
  `tests` package into site-packages, and a REGULAR package there beats a
  namespace one wherever it sits on sys.path — so without this, discovery finds
  literalai's tests and reports this directory as "not importable".
* Sixteen test modules import their shared harness as `from pipeline_stub import
  ...`, which only resolves when this directory is itself on sys.path. That is
  true when a file is run directly and false under discovery, so the line below
  makes it true either way rather than rewriting sixteen imports to say
  `tests.pipeline_stub`.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)


def _keep_the_suite_off_a_dead_comfyui():
    """The unit suite must not wait on a ComfyUI that is not running.

    Plenty of code under test reaches ComfyUI (schema lookups, dir resolution)
    without the test meaning it to. With nothing listening on the configured
    address, every such call is a real connection attempt (seconds each on
    Windows, and one full-suite run hung for twenty minutes in them). So: probe
    once with a 1 s timeout, and when nobody answers, make requests to that
    address fail at once with ConnectionError. Requests to any other address, and
    tests that stub ``requests`` themselves, are untouched. With a live ComfyUI
    nothing changes.
    """
    import socket
    from urllib.parse import urlparse

    try:
        from agenty_core.utils.comfyui_client import ComfyUIClient
        url = ComfyUIClient._load_base_url()
    except Exception:  # noqa: BLE001 — no core, nothing to guard
        return
    parsed = urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return                                   # alive: leave everything alone
    except OSError:
        pass

    import requests
    from requests.adapters import HTTPAdapter
    real_send = HTTPAdapter.send

    def send(self, request, *args, **kwargs):
        target = urlparse(request.url)
        if (target.hostname, target.port or 80) == (host, port):
            raise requests.ConnectionError(
                f"ComfyUI at {host}:{port} is not running (unit suite is offline-safe)")
        return real_send(self, request, *args, **kwargs)

    HTTPAdapter.send = send


_keep_the_suite_off_a_dead_comfyui()
