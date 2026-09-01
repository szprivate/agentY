"""Who the agent host lets in.

The interesting cases here cannot be produced by hand. A browser will not let you
forge an ``Origin``, and the request that mattered — a page on the open web reading
``.env`` out of a server on localhost — looked, from the machine it happened on,
exactly like the panel working. So the rules are pure functions and every refusal
below is a test rather than a thing to go and try.

The regression these guard is real and was live: ``Access-Control-Allow-Origin: *``
on a host that answers ``GET /agentY/settings`` with every API key in plaintext.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils import api_guard


def call(**over):
    """A request that should be allowed, with *over* substituted in.

    Written this way round on purpose: each test names only the field it is about,
    so a test that fails is a test about that field.
    """
    args = dict(
        method="GET", path="/agentY/settings",
        host_header="127.0.0.1:5001", origin="http://127.0.0.1:8188",
        token="good", expected_token="good",
        agent_port=5001, comfyui_url="http://127.0.0.1:8188",
    )
    args.update(over)
    return api_guard.verdict(**args)


class TheCrossSiteHole(unittest.TestCase):
    """The specific attack this module was written to close."""

    def test_a_website_cannot_read_the_settings(self):
        ok, why = call(origin="https://evil.example", token="", expected_token="good")
        self.assertFalse(ok)
        self.assertIn("evil.example", why)

    def test_a_website_cannot_drive_the_chat(self):
        # The other half: the agent holds run_script, so putting words in its
        # mouth is as good as a shell.
        ok, _ = call(method="POST", path="/agentY/chat", origin="https://evil.example")
        self.assertFalse(ok)

    def test_a_stolen_token_still_needs_the_right_origin(self):
        """Origin is checked even when the token is correct.

        The two checks answer different questions and neither substitutes for the
        other: a token can leak into a log, and an origin cannot be forged by a
        page. Requiring both means one failure is not a compromise.
        """
        ok, _ = call(origin="https://evil.example", token="good", expected_token="good")
        self.assertFalse(ok)

    def test_the_panel_itself_is_allowed(self):
        ok, why = call()
        self.assertTrue(ok, why)


class TheOriginRule(unittest.TestCase):
    """Same hostname as the request was addressed to, on a port we serve."""

    def test_a_lan_install_works_without_configuration(self):
        """The panel builds the backend URL from its own location.hostname.

        So on a LAN both sides are whatever name the user typed, and the hostnames
        match with nothing configured. Hard-coding 127.0.0.1 instead would have
        broken every install that is not on the same machine.
        """
        ok, why = call(host_header="192.168.1.5:5001",
                       origin="http://192.168.1.5:8188")
        self.assertTrue(ok, why)

    def test_a_stranger_on_the_same_host_but_a_foreign_port_is_refused(self):
        ok, why = call(origin="http://127.0.0.1:31337")
        self.assertFalse(ok)
        self.assertIn("31337", why)

    def test_our_own_port_is_allowed_for_the_pages_we_serve(self):
        # The log/memory viewers are served by this host, so their fetches carry
        # our own origin.
        ok, why = call(origin="http://127.0.0.1:5001")
        self.assertTrue(ok, why)

    def test_an_explicitly_allowed_origin_wins(self):
        ok, why = call(origin="http://studio.example:8188",
                       host_header="studio.example:5001",
                       allowed_hosts=("studio.example",),
                       allowed_origins=("http://studio.example:8188",))
        self.assertTrue(ok, why)

    def test_no_origin_header_is_not_a_cross_site_request(self):
        """curl, the launcher's own health check, and same-origin navigations.

        Absence of Origin is not evidence of innocence — which is what the token
        is for — but it is not evidence of an attack either, and refusing it would
        break every non-browser caller for no gain.
        """
        ok, why = call(origin="")
        self.assertTrue(ok, why)

    def test_the_check_can_be_switched_off(self):
        ok, _ = call(origin="https://evil.example", check_origin=False)
        self.assertTrue(ok)


class DnsRebinding(unittest.TestCase):
    """A name the attacker controls, pointed at 127.0.0.1.

    Rebinding defeats the Origin rule by making both sides agree: the page is at
    ``http://evil.example:8188`` and the request is addressed to
    ``evil.example:5001``, so the hostnames match. Only the Host check sees it.
    """

    def test_a_matching_pair_of_attacker_names_is_still_refused(self):
        ok, why = call(host_header="evil.example:5001",
                       origin="http://evil.example:8188")
        self.assertFalse(ok)
        self.assertIn("evil.example", why)

    def test_ip_literals_and_localhost_are_fine(self):
        for host in ("127.0.0.1:5001", "localhost:5001", "192.168.1.5:5001",
                     "[::1]:5001"):
            with self.subTest(host=host):
                ok, why = call(host_header=host, origin="")
                self.assertTrue(ok, why)

    def test_a_real_name_can_be_allowed_on_purpose(self):
        ok, why = call(host_header="studio.local:5001", origin="",
                       allowed_hosts=("studio.local",))
        self.assertTrue(ok, why)

    def test_is_local_hostname(self):
        for good in ("127.0.0.1", "localhost", "::1", "10.0.0.4", "[::1]:5001"):
            self.assertTrue(api_guard.is_local_hostname(good), good)
        for bad in ("evil.example", "studio.local", "", "example.com:80"):
            self.assertFalse(api_guard.is_local_hostname(bad), bad)


class TheToken(unittest.TestCase):
    """What stops a caller that has no browser to be honest for it."""

    def test_a_script_without_the_token_is_refused(self):
        # No Origin, correct Host — a curl on this machine. This is precisely the
        # caller the Origin rule cannot see.
        ok, why = call(origin="", token="")
        self.assertFalse(ok)
        self.assertIn("token", why)

    def test_a_wrong_token_is_refused(self):
        ok, _ = call(origin="", token="not-it")
        self.assertFalse(ok)

    def test_health_is_reachable_without_one(self):
        """Otherwise a token fault is indistinguishable from a dead host.

        The panel decides "is the host up" from this endpoint before it has
        anything else, and "the host is down" is the single most misleading thing
        it could say about a host that is running.
        """
        ok, why = call(path="/agentY/health", origin="", token="")
        self.assertTrue(ok, why)

    def test_the_viewer_pages_are_reachable_without_one(self):
        # A top-level navigation cannot send a header. The token is injected into
        # the HTML instead, and the Origin rule is what stops another page from
        # fetching them to read it out.
        for page in ("/agentY/log_viewer", "/agentY/memory_viewer",
                     "/agentY/project_memory_viewer"):
            with self.subTest(page=page):
                ok, why = call(path=page, origin="", token="")
                self.assertTrue(ok, why)

    def test_the_data_behind_the_viewers_is_not_public(self):
        """The pages are public; what they read is not.

        Listing the viewer PAGES as public is safe only because their data
        endpoints are separate paths. If /agentY/memory had been public too, the
        exemption would have handed over the memory store.
        """
        for path in ("/agentY/memory", "/agentY/project_memory",
                     "/agentY/message_history", "/agentY/settings"):
            with self.subTest(path=path):
                ok, _ = call(path=path, origin="", token="")
                self.assertFalse(ok)

    def test_preflight_is_never_refused(self):
        """A preflight cannot carry a token.

        Refusing it would make the browser report a CORS error for what is really
        an auth decision — the most confusing possible way to say no.
        """
        ok, why = call(method="OPTIONS", path="/agentY/chat",
                       origin="https://evil.example", token="")
        self.assertTrue(ok, why)

    def test_the_requirement_can_be_switched_off(self):
        ok, _ = call(origin="", token="", require_token=False)
        self.assertTrue(ok)

    def test_no_token_configured_means_no_token_check(self):
        # Nothing to compare against: fail open on THIS check rather than lock the
        # user out of a host that could not write its token file.
        ok, _ = call(origin="", token="", expected_token="")
        self.assertTrue(ok)


class TheTokenFile(unittest.TestCase):
    def test_it_is_written_owner_only(self):
        with TemporaryDirectory() as tmp:
            api_guard._token = None
            token = api_guard.session_token(Path(tmp))
            path = api_guard.token_path(Path(tmp))
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8").strip(), token)
            if hasattr(path, "stat"):
                self.assertEqual(path.stat().st_mode & 0o077, 0,
                                 "the token file must not be group/world readable")

    def test_it_is_stable_within_a_process(self):
        with TemporaryDirectory() as tmp:
            api_guard._token = None
            first = api_guard.session_token(Path(tmp))
            self.assertEqual(first, api_guard.session_token(Path(tmp)))

    def test_a_restart_keeps_the_same_token(self):
        """The fix for "the tab is always older than the token".

        A ComfyUI tab reads the token once, at page load, and the host is
        restarted far more often than the tab is reloaded. Minting per process
        therefore invalidated every open panel on every restart, for no gain: the
        file is owner-only and sits beside .env, so anyone who can read one can
        read every API key anyway.
        """
        with TemporaryDirectory() as tmp:
            api_guard._token = None
            first = api_guard.session_token(Path(tmp))
            api_guard._token = None                  # what a restart looks like
            self.assertEqual(api_guard.session_token(Path(tmp)), first)

    def test_a_malformed_token_file_is_replaced_not_trusted(self):
        """The value is spliced into a <script> for the viewer pages, so a file
        holding a quote or a tag would be a way to run script in that page."""
        with TemporaryDirectory() as tmp:
            for junk in ('"; alert(1); //', "", "short", "x" * 500, "has spaces"):
                with self.subTest(junk=junk):
                    api_guard.token_path(Path(tmp)).write_text(junk, encoding="utf-8")
                    api_guard._token = None
                    token = api_guard.session_token(Path(tmp))
                    self.assertNotEqual(token, junk)
                    self.assertRegex(token, r"^[A-Za-z0-9_-]{16,128}$")

    def test_an_env_override_wins_over_the_file(self):
        """For a panel and a host on different machines, where the file cannot be
        the channel and somebody carries the secret across by hand."""
        import os
        with TemporaryDirectory() as tmp:
            api_guard._token = None
            api_guard.session_token(Path(tmp))       # write one to the file
            api_guard._token = None
            os.environ["AGENTY_SESSION_TOKEN"] = "carried-across-by-hand"
            try:
                self.assertEqual(api_guard.session_token(Path(tmp)),
                                 "carried-across-by-hand")
            finally:
                os.environ.pop("AGENTY_SESSION_TOKEN", None)

    def test_clearing_it_really_does_rotate(self):
        """Kept for the case deletion was always right for: revoking a token you
        believe somebody else has seen."""
        with TemporaryDirectory() as tmp:
            api_guard._token = None
            first = api_guard.session_token(Path(tmp))
            api_guard.clear_token_file(Path(tmp))
            api_guard._token = None
            self.assertNotEqual(api_guard.session_token(Path(tmp)), first)

    def test_clearing_removes_it(self):
        with TemporaryDirectory() as tmp:
            api_guard._token = None
            api_guard.session_token(Path(tmp))
            api_guard.clear_token_file(Path(tmp))
            self.assertFalse(api_guard.token_path(Path(tmp)).exists())

    def test_clearing_a_missing_file_is_not_an_error(self):
        with TemporaryDirectory() as tmp:
            api_guard.clear_token_file(Path(tmp))     # must not raise

    def tearDown(self):
        api_guard._token = None


class ShutdownLeavesTheTokenAlone(unittest.TestCase):
    """The regression that would bring the whole problem back.

    Deleting the token on the way out is superficially tidy and was exactly what
    made every restart lock out every open ComfyUI tab. A source check because the
    alternative is starting a host, stopping it, and asserting about a file — a
    slow test for a one-line mistake.
    """

    def test_the_shutdown_handler_does_not_delete_it(self):
        from pathlib import Path as P
        source = (P(__file__).resolve().parents[1] / "src" / "agenty_ui_server.py"
                  ).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in source.splitlines()
                          if not ln.lstrip().startswith("#"))
        self.assertNotIn("clear_token_file", code,
                         "deleting the token at shutdown invalidates every open panel")


class InjectingTheTokenIntoAPage(unittest.TestCase):
    def test_it_lands_inside_head(self):
        out = api_guard.inject_token("<html><head><title>x</title></head></html>", "abc123")
        self.assertIn("abc123", out)
        self.assertLess(out.index("abc123"), out.index("<title>"),
                        "the token must be defined before anything on the page runs")

    def test_a_page_without_a_head_still_gets_it(self):
        out = api_guard.inject_token("<p>hi</p>", "abc123")
        self.assertTrue(out.lstrip().startswith("<script"))

    def test_it_installs_the_header_on_fetch(self):
        # The viewer HTML files are not modified for this; the shim is what makes
        # their existing same-origin fetches carry the token.
        out = api_guard.inject_token("<head></head>", "abc123")
        self.assertIn(api_guard.TOKEN_HEADER, out)
        self.assertIn("window.fetch", out)

    def test_a_hostile_token_cannot_break_out_of_the_script(self):
        """Belt and braces: token_urlsafe cannot produce these, but nothing here
        should depend on that being true forever."""
        out = api_guard.inject_token("<head></head>", '"; alert(1); //</script>')
        self.assertNotIn("alert(1)", out)
        self.assertNotIn("</script><", out.replace("</script>", "", 1) + "<")

    def test_no_token_leaves_the_page_alone(self):
        self.assertEqual(api_guard.inject_token("<head></head>", ""), "<head></head>")


class ThePanelsCopyOfThePublicList(unittest.TestCase):
    """The sidebar mirrors PUBLIC_PATHS, and the two must not drift.

    The panel needs the list to tell "recovered" from "still refused": every
    public path keeps answering 200 throughout a total lockout, so treating one of
    those as proof of recovery clears the warning on the next heartbeat and leaves
    the panel silently broken again — which is exactly what it did.

    A path added to the server's list and not the panel's is that bug, back.
    """

    def _panel_source(self):
        from pathlib import Path as P
        here = P(__file__).resolve().parents[2]
        for base in (here / "ComfyUI" / "custom_nodes", here):
            js = base / "agentY-comfyuiConnect" / "web" / "agent_backend.js"
            if js.exists():
                return js.read_text(encoding="utf-8")
        self.skipTest("the agentY-comfyuiConnect extension is not beside this checkout")

    def test_the_two_lists_match(self):
        import re
        source = self._panel_source()
        block = re.search(r"const PUBLIC_PATHS = \[(.*?)\];", source, re.S)
        self.assertIsNotNone(block, "the panel no longer declares PUBLIC_PATHS")
        panel = set(re.findall(r'"([^"]+)"', block.group(1)))
        self.assertEqual(panel, set(api_guard.PUBLIC_PATHS))

    def test_the_panel_uses_the_same_header_name(self):
        self.assertIn(f'"{api_guard.TOKEN_HEADER}"', self._panel_source())


if __name__ == "__main__":
    unittest.main()
