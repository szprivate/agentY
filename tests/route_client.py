"""A Flask test client that gets past the host's admission guard.

Every request to the agent host now carries a session token (see
:mod:`src.utils.api_guard`), so a bare ``_build_app().test_client()`` gets a 403
and the test reads as though the route is broken.

The knowledge of how to authenticate lives here and only here. Spreading a header
through forty route tests would mean forty places to update the next time it
changes, and — worse — forty tests quietly asserting a security detail they are
not about. What each of those files is about is its route; this is the door.

Note that these clients go *through* the guard rather than around it. Refusal is
tested in ``tests/test_server_admission.py``; what is tested here is that an
authorised caller still gets the behaviour the route always had.
"""

from __future__ import annotations

from src.utils import api_guard

# Werkzeug's test client sends `Host: localhost`, which the guard already accepts
# as a local name, and no Origin header, which is what a non-browser caller looks
# like. So the token is the only thing missing.
TEST_TOKEN = "test-session-token"


def install_test_token(token: str = TEST_TOKEN) -> str:
    """Pin the host's token so tests do not depend on a file being writable."""
    api_guard._token = token
    return token


def _client_class():
    """A FlaskClient that adds the token to every request.

    Subclassed rather than configured: Flask's ``test_client`` does not take
    default headers, and adding them at each call site is what this module
    exists to avoid.
    """
    from flask.testing import FlaskClient

    class TokenClient(FlaskClient):
        def open(self, *args, **kwargs):
            headers = kwargs.get("headers") or {}
            # A test that sets the header itself is testing the header; leave it.
            if api_guard.TOKEN_HEADER not in {str(k) for k in dict(headers)}:
                headers = dict(headers)
                headers[api_guard.TOKEN_HEADER] = TEST_TOKEN
                kwargs["headers"] = headers
            return super().open(*args, **kwargs)

    return TokenClient


def authorised_client(app=None):
    """A test client for *app* (or a freshly built one) that the guard admits."""
    if app is None:
        from src.utils.agentY_server import _build_app
        app = _build_app()
    install_test_token()
    app.test_client_class = _client_class()
    return app.test_client()
