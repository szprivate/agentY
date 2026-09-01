"""The guard as the real app applies it, not as its rules describe themselves.

test_api_guard.py checks the decision function. This checks that the decision is
actually consulted — that the ``before_request`` hook exists, that the endpoints
are behind it, and that ``GET /agentY/settings`` no longer answers with the
contents of ``.env``.

That distinction is the whole point of the file. The vulnerability was never a
wrong rule; it was a correct-looking server with no rule at all, and a unit test
of a policy nobody had wired up would have passed just as happily.
"""

import unittest

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:  # pragma: no cover
    HAVE_FLASK = False

from src.utils import api_guard

# A recognisable stand-in. If any of this reaches a response body, the assertion
# that finds it names the exact thing that leaked.
FAKE_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-THIS-MUST-NEVER-BE-SENT",
    "HF_TOKEN": "hf_THIS-MUST-NEVER-BE-SENT",
    "DASHSCOPE_BASE_URL": "https://dashscope.example/v1",
    "SLACK_ALLOWED_USERS": "U123,U456",
    "OPENAI_API_KEY": "",
}

PANEL_ORIGIN = "http://127.0.0.1:8188"
HOST_HEADER = "127.0.0.1:5001"


@unittest.skipUnless(HAVE_FLASK, "Flask is not installed")
class ServerAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.utils import agentY_server as srv
        cls.srv = srv
        cls._real_read_env = srv._read_env_file
        cls._real_note_ages = srv._note_key_ages
        srv._read_env_file = staticmethod(lambda: dict(FAKE_ENV))
        srv._note_key_ages = staticmethod(lambda: [])
        srv.set_bound_port(5001)
        api_guard._token = "test-token"
        cls.app = srv._build_app()
        cls.app.config["TESTING"] = True

    @classmethod
    def tearDownClass(cls):
        cls.srv._read_env_file = cls._real_read_env
        cls.srv._note_key_ages = cls._real_note_ages
        api_guard._token = None

    def get(self, path, origin=PANEL_ORIGIN, token="test-token", **kw):
        headers = {"Host": HOST_HEADER}
        if origin:
            headers["Origin"] = origin
        if token:
            headers[api_guard.TOKEN_HEADER] = token
        with self.app.test_client() as c:
            return c.get(path, headers=headers, **kw)

    # ── the leak ────────────────────────────────────────────────────────────
    def test_the_settings_endpoint_no_longer_returns_the_keys(self):
        body = self.get("/agentY/settings").get_data(as_text=True)
        self.assertNotIn("sk-ant-THIS-MUST-NEVER-BE-SENT", body)
        self.assertNotIn("hf_THIS-MUST-NEVER-BE-SENT", body)

    def test_it_still_says_which_keys_are_set(self):
        """Masked, not omitted. The panel has to show that a key IS configured,
        or every visit looks like a fresh install with nothing filled in."""
        data = self.get("/agentY/settings").get_json()
        self.assertEqual(data["env"]["ANTHROPIC_API_KEY"], self.srv._SECRET_MASK)
        self.assertEqual(data["env"]["OPENAI_API_KEY"], "", "an unset key stays empty")

    def test_settings_that_are_not_credentials_still_round_trip(self):
        """The UI has to be able to edit and re-save these, so they cannot be
        masked — and neither is a secret."""
        env = self.get("/agentY/settings").get_json()["env"]
        self.assertEqual(env["DASHSCOPE_BASE_URL"], "https://dashscope.example/v1")
        self.assertEqual(env["SLACK_ALLOWED_USERS"], "U123,U456")

    def test_a_website_gets_nothing(self):
        res = self.get("/agentY/settings", origin="https://evil.example")
        self.assertEqual(res.status_code, 403)
        self.assertNotIn("sk-ant", res.get_data(as_text=True))

    def test_the_response_is_not_readable_by_everyone(self):
        """`Access-Control-Allow-Origin: *` is what made the leak reachable.

        A reflected origin lets the panel read its own responses and leaves every
        other page with a CORS error — the browser enforces that for us, which is
        the only reason any of this works.
        """
        res = self.get("/agentY/health")
        self.assertNotEqual(res.headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), PANEL_ORIGIN)
        self.assertIn("Origin", res.headers.get("Vary", ""),
                      "without Vary a cached response can be replayed to another origin")

    def test_a_refused_origin_gets_no_cors_header_at_all(self):
        res = self.get("/agentY/settings", origin="https://evil.example")
        self.assertIsNone(res.headers.get("Access-Control-Allow-Origin"))

    # ── the token ───────────────────────────────────────────────────────────
    def test_a_local_script_without_the_token_is_refused(self):
        # No Origin header at all: a curl on this machine, which the Origin rule
        # cannot see. This is the caller the token exists for.
        res = self.get("/agentY/settings", origin=None, token=None)
        self.assertEqual(res.status_code, 403)

    def test_health_answers_without_a_token(self):
        res = self.get("/agentY/health", origin=None, token=None)
        self.assertEqual(res.status_code, 200)

    def test_the_preflight_is_answered(self):
        """A preflight carries no token; refusing it would report an auth
        decision as a CORS failure."""
        with self.app.test_client() as c:
            res = c.options("/agentY/settings", headers={
                "Host": HOST_HEADER, "Origin": PANEL_ORIGIN,
                "Access-Control-Request-Method": "POST"})
        self.assertIn(res.status_code, (200, 204))
        self.assertIn(api_guard.TOKEN_HEADER,
                      res.headers.get("Access-Control-Allow-Headers", ""),
                      "the browser will not send a header we did not permit")

    # ── the approval channel ────────────────────────────────────────────────
    def test_the_permission_endpoint_is_reachable(self):
        """The panel long-polls this while the agent thread sits blocked in a
        tool. If it were unreachable, every gated tool would wait out its timeout
        and decline — a working agent that silently refuses to do anything."""
        res = self.get("/agentY/permission")
        self.assertEqual(res.status_code, 200)
        self.assertIn("request", res.get_json())

    def test_the_approval_channel_is_behind_the_guard(self):
        """It has to be. A page that could answer these could approve the
        commands it had just asked the agent to run."""
        for path in ("/agentY/permission", "/agentY/permission/reply"):
            with self.subTest(path=path):
                res = self.get(path, origin="https://evil.example")
                self.assertEqual(res.status_code, 403)

    def test_answering_a_question_nobody_asked(self):
        with self.app.test_client() as c:
            res = c.post("/agentY/permission/reply",
                         headers={"Host": HOST_HEADER, "Origin": PANEL_ORIGIN,
                                  api_guard.TOKEN_HEADER: "test-token"},
                         json={"permission_id": "nope", "allowed": True})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["expired"])

    # ── the pages we serve ourselves ────────────────────────────────────────
    def test_a_viewer_page_carries_the_token_it_cannot_be_sent(self):
        res = self.get("/agentY/log_viewer", origin=None, token=None)
        if res.status_code == 404:
            self.skipTest("log_viewer.html is not present in this checkout")
        body = res.get_data(as_text=True)
        self.assertIn("test-token", body)
        self.assertIn(api_guard.TOKEN_HEADER, body)


@unittest.skipUnless(HAVE_FLASK, "Flask is not installed")
class RefusalLogging(unittest.TestCase):
    """A refused panel does not fail once — it polls.

    Every panel in the sidebar retries on a timer, so a token that cannot be
    authenticated produces several identical refusals a second. Logging each one
    buries the startup banner, the key-age warning and every real error under a
    wall of the same line, which is how a useful message becomes noise.
    """

    def setUp(self):
        from src.utils import agentY_server as srv
        self.srv = srv
        srv._refusals.clear()
        self.logged = []
        self._real = srv.logger.warning
        srv.logger.warning = lambda *a, **k: self.logged.append(a)

    def tearDown(self):
        self.srv.logger.warning = self._real
        self.srv._refusals.clear()

    def test_a_repeated_refusal_is_logged_once(self):
        for _ in range(50):
            self.srv._log_refusal("GET", "/agentY/threads", "missing token")
        self.assertEqual(len(self.logged), 1)

    def test_a_different_refusal_is_still_reported(self):
        """Deduping must be per problem, or the second thing to go wrong is
        silent because the first one is still happening."""
        self.srv._log_refusal("GET", "/agentY/threads", "missing token")
        self.srv._log_refusal("GET", "/agentY/settings", "cross-site request")
        self.assertEqual(len(self.logged), 2)

    def test_it_says_how_often_when_it_speaks_again(self):
        import time
        self.srv._log_refusal("GET", "/agentY/threads", "missing token")
        for _ in range(9):
            self.srv._log_refusal("GET", "/agentY/threads", "missing token")
        # Wind the clock back past the quiet period rather than sleeping a minute.
        self.srv._refusals[("/agentY/threads", "missing token")][0] = (
            time.time() - self.srv._REFUSAL_QUIET_SECONDS - 1)
        self.srv._log_refusal("GET", "/agentY/threads", "missing token")
        self.assertEqual(len(self.logged), 2)
        self.assertIn(10, self.logged[1], "the repeat count is the useful part")


class TheAccessLog(unittest.TestCase):
    """Polls must not scroll the console away.

    The session token made this worse rather than better: a custom header makes
    even a GET a non-simple request, so the browser sends a CORS preflight first
    and every poll became TWO access-log lines. The exact line below is one a user
    watched repeat until the agent's own output was unreadable.
    """

    def setUp(self):
        import logging
        from src.utils.agentY_server import _QuietPollFilter
        self.filter = _QuietPollFilter()
        self.record = lambda msg: logging.LogRecord(
            "werkzeug", logging.INFO, "", 0, msg, (), None)

    def quiet(self, line):
        return not self.filter.filter(self.record(line))

    def test_the_preflight_for_a_long_poll_is_silenced(self):
        self.assertTrue(self.quiet(
            '127.0.0.1 - - [01/Sep/2026 09:52:36] "OPTIONS /agentY/permission HTTP/1.1" 200 -'))

    def test_the_long_poll_itself_is_silenced(self):
        for line in ('127.0.0.1 - - [01/Sep/2026 09:52:36] "GET /agentY/permission?wait=25 HTTP/1.1" 200 -',
                     '127.0.0.1 - - [01/Sep/2026 09:52:36] "GET /agentY/health HTTP/1.1" 200 -',
                     '127.0.0.1 - - [01/Sep/2026 09:52:36] "GET /agentY/canvas_probe?wait=20 HTTP/1.1" 200 -'):
            with self.subTest(line=line):
                self.assertTrue(self.quiet(line))

    def test_an_answer_is_not_a_poll_and_is_still_logged(self):
        """Someone approving a command is an event, not background noise."""
        self.assertFalse(self.quiet(
            '127.0.0.1 - - [01/Sep/2026 09:52:36] "POST /agentY/permission/reply HTTP/1.1" 200 -'))

    def test_a_failing_poll_is_still_logged(self):
        """The whole point of quieting the successes: a 403 or a 500 on the same
        path is exactly the line worth seeing, and it is no longer buried."""
        self.assertFalse(self.quiet(
            '127.0.0.1 - - [01/Sep/2026 09:52:36] "GET /agentY/permission HTTP/1.1" 403 -'))
        self.assertFalse(self.quiet(
            '127.0.0.1 - - [01/Sep/2026 09:52:36] "GET /agentY/health HTTP/1.1" 500 -'))

    def test_a_real_request_is_never_touched(self):
        self.assertFalse(self.quiet(
            '127.0.0.1 - - [01/Sep/2026 09:52:36] "POST /agentY/chat HTTP/1.1" 200 -'))


@unittest.skipUnless(HAVE_FLASK, "Flask is not installed")
class MaskingHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.utils import agentY_server as srv
        cls.srv = srv

    def test_only_credentials_are_masked(self):
        out = self.srv._masked_env(FAKE_ENV)
        self.assertEqual(out["HF_TOKEN"], self.srv._SECRET_MASK)
        self.assertEqual(out["DASHSCOPE_BASE_URL"], "https://dashscope.example/v1")

    def test_the_mask_carries_nothing_of_the_value(self):
        """Not even the last few characters.

        The field name already says which credential it is, so a partial reveal
        buys nothing and costs the one thing masking is for.
        """
        mask = self.srv._masked_env({"HF_TOKEN": "hf_abcdefghijklmnop"})["HF_TOKEN"]
        for chunk in ("hf_", "abcd", "mnop"):
            self.assertNotIn(chunk, mask)

    def test_a_save_cannot_write_the_mask_over_a_real_key(self):
        """The failure this prevents is silent and unrecoverable: the mask in
        .env, the key gone, and nothing to say so until the next 401."""
        kept = self.srv._drop_masked({
            "HF_TOKEN": self.srv._SECRET_MASK,
            "ANTHROPIC_API_KEY": "sk-ant-a-real-new-value",
        })
        self.assertNotIn("HF_TOKEN", kept)
        self.assertEqual(kept["ANTHROPIC_API_KEY"], "sk-ant-a-real-new-value")

    def test_clearing_a_field_still_works(self):
        # An empty string means "remove this key" and must survive the filter,
        # or a key can never be deleted from the UI.
        self.assertEqual(self.srv._drop_masked({"HF_TOKEN": ""}), {"HF_TOKEN": ""})

    def test_no_env_block_at_all(self):
        self.assertEqual(self.srv._drop_masked(None), {})


if __name__ == "__main__":
    unittest.main()
