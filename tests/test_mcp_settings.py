"""Adding an MCP server from Settings: paste what the server's page says, test it.

Adding one used to mean writing config/mcp.json by hand in a textarea. What people
have instead is whatever the server's README shows: an address, a start command,
or a JSON block in some client's dialect. The parser turns any of those into
agentY's entry, and a key in the paste goes to .env, never into mcp.json. The
Test button then connects once, with keys typed but not yet saved, and leaves the
orchestrator's live connections alone.

    python -m unittest discover -s tests
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from src.tools import mcp_tools as mt
from src.utils.mcp_import import McpImportError, parse_mcp_snippet


class AnAddressTest(unittest.TestCase):

    def test_a_url_is_an_http_server_named_for_its_host(self):
        out = parse_mcp_snippet("https://mcp.notion.com/mcp")
        self.assertEqual(out["servers"], {"notion": {
            "enabled": True, "transport": "http", "url": "https://mcp.notion.com/mcp",
            "auth": "none"}})
        self.assertEqual(out["secrets"], {})

    def test_an_sse_path_is_an_sse_server(self):
        out = parse_mcp_snippet("https://mcp.deepwiki.com/sse")
        self.assertEqual(out["servers"]["deepwiki"]["transport"], "sse")

    def test_an_address_with_no_key_says_where_a_key_would_go(self):
        notes = " ".join(parse_mcp_snippet("https://mcp.notion.com/mcp")["notes"])
        self.assertIn("API key", notes)

    def test_a_name_already_taken_is_not_overwritten(self):
        out = parse_mcp_snippet("https://mcp.magnific.com", existing=["magnific"])
        self.assertEqual(list(out["servers"]), ["magnific_2"])


class ACommandTest(unittest.TestCase):

    def test_npx_is_a_local_server_named_for_its_package(self):
        out = parse_mcp_snippet("npx -y @modelcontextprotocol/server-filesystem C:/Users/me/Documents")
        self.assertEqual(out["servers"], {"filesystem": {
            "enabled": True, "transport": "stdio", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/Documents"],
            "auth": "none"}})

    def test_a_key_in_front_of_the_command_goes_to_env_not_the_config(self):
        out = parse_mcp_snippet("GITHUB_PERSONAL_ACCESS_TOKEN=ghp_abc123 npx -y @modelcontextprotocol/server-github")
        entry = out["servers"]["github"]
        self.assertEqual(entry["env"], {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"})
        self.assertEqual(out["secrets"], {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_abc123"})

    def test_uvx_packages_lose_their_mcp_server_prefix(self):
        self.assertEqual(list(parse_mcp_snippet("uvx mcp-server-fetch")["servers"]), ["fetch"])

    def test_the_claude_mcp_add_line_many_readmes_print(self):
        out = parse_mcp_snippet("claude mcp add --transport http linear https://mcp.linear.app/mcp")
        self.assertEqual(out["servers"]["linear"]["url"], "https://mcp.linear.app/mcp")
        self.assertEqual(out["servers"]["linear"]["transport"], "http")

    def test_claude_mcp_add_with_a_command_and_an_env_key(self):
        out = parse_mcp_snippet("claude mcp add context7 -e CONTEXT7_API_KEY=ctx_9 -- npx -y @upstash/context7-mcp")
        entry = out["servers"]["context7"]
        self.assertEqual(entry["command"], "npx")
        self.assertEqual(entry["env"], {"CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"})
        self.assertEqual(out["secrets"], {"CONTEXT7_API_KEY": "ctx_9"})


class AJsonBlockTest(unittest.TestCase):

    def test_claude_desktop_mcpservers_with_a_token(self):
        out = parse_mcp_snippet('''{"mcpServers": {"github": {"command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_real", "LOG_LEVEL": "info"}}}}''')
        entry = out["servers"]["github"]
        self.assertEqual(entry["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"], "${GITHUB_PERSONAL_ACCESS_TOKEN}")
        self.assertEqual(entry["env"]["LOG_LEVEL"], "info", "only secrets move to .env")
        self.assertEqual(out["secrets"], {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_real"})

    def test_a_placeholder_is_a_value_to_fill_in_not_a_secret(self):
        out = parse_mcp_snippet('''{"mcpServers": {"brave": {"command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "<YOUR_API_KEY>"}}}}''')
        self.assertEqual(out["secrets"], {})
        self.assertEqual(out["missing"], ["BRAVE_API_KEY"])
        self.assertEqual(out["servers"]["brave"]["env"], {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
                         "the JSON's own name wins over one guessed from the package")

    def test_vs_code_servers_with_an_input_prompt(self):
        out = parse_mcp_snippet('''{
          // VS Code's mcp.json
          "servers": {"stripe": {"type": "http", "url": "https://mcp.stripe.com",
                                 "headers": {"Authorization": "Bearer ${input:stripe_key}"},},
          }}''')
        entry = out["servers"]["stripe"]
        self.assertEqual(entry["auth"], "header")
        self.assertEqual(entry["headers"], {"Authorization": "Bearer ${MCP_STRIPE_API_KEY}"})
        self.assertEqual(out["missing"], ["MCP_STRIPE_API_KEY"])

    def test_a_literal_bearer_token_moves_to_env_and_keeps_its_scheme(self):
        out = parse_mcp_snippet('''{"acme": {"url": "https://tools.acme.io/mcp",
            "headers": {"Authorization": "Bearer sk-live-1", "X-Team": "art"}}}''')
        entry = out["servers"]["acme"]
        self.assertEqual(entry["headers"], {"Authorization": "Bearer ${MCP_ACME_API_KEY}", "X-Team": "art"})
        self.assertEqual(out["secrets"], {"MCP_ACME_API_KEY": "sk-live-1"})

    def test_a_bare_fragment_copied_out_of_a_bigger_block(self):
        out = parse_mcp_snippet('"context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},')
        self.assertEqual(list(out["servers"]), ["context7"])

    def test_several_servers_at_once(self):
        out = parse_mcp_snippet('''{"mcpServers": {"a": {"url": "https://a.example.com/mcp"},
                                                   "b": {"command": "uvx", "args": ["mcp-server-time"]}}}''')
        self.assertEqual(sorted(out["servers"]), ["a", "b"])


class NotAServerTest(unittest.TestCase):

    def test_nothing_pasted(self):
        with self.assertRaises(McpImportError):
            parse_mcp_snippet("   ")

    def test_json_with_no_server_in_it(self):
        with self.assertRaises(McpImportError):
            parse_mcp_snippet('{"theme": "dark"}')

    def test_broken_json_says_where(self):
        with self.assertRaises(McpImportError) as ctx:
            parse_mcp_snippet('{"mcpServers": {"a": {"url": }}}')
        self.assertIn("line", str(ctx.exception))


class TestButtonTest(unittest.TestCase):
    """One connection from the form, saved or not."""

    def setUp(self):
        os.environ.pop("MCP_DEMO_API_KEY", None)
        self.addCleanup(os.environ.pop, "MCP_DEMO_API_KEY", None)

    def test_a_typed_key_is_there_for_the_connection_and_gone_after(self):
        seen = {}
        client = mock.Mock()

        def connect(name, sc, interactive):
            seen["key"] = os.environ.get("MCP_DEMO_API_KEY")
            seen["interactive"] = interactive
            return client, [SimpleNamespace(tool_name="demo_search"), SimpleNamespace(tool_name="demo_fetch")]

        with mock.patch.object(mt, "_connect", connect):
            res = mt.test_server("demo", {"transport": "http", "url": "https://x"},
                                 {"MCP_DEMO_API_KEY": "sk-1"})
        self.assertEqual(res, {"ok": True, "tools": 2, "names": ["search", "fetch"]})
        self.assertEqual(seen, {"key": "sk-1", "interactive": False},
                         "a test never opens a browser")
        self.assertNotIn("MCP_DEMO_API_KEY", os.environ)
        client.stop.assert_called_once()

    def test_tool_names_are_the_servers_own(self):
        """A live MCPClient names a tool `<server>__<tool>`, two underscores."""
        with mock.patch.object(mt, "_connect",
                               lambda *a, **k: (mock.Mock(), [SimpleNamespace(tool_name="demo__fetch")])):
            res = mt.test_server("demo", {"transport": "http", "url": "https://x"})
        self.assertEqual(res["names"], ["fetch"])

    def test_the_live_connections_are_left_alone(self):
        live = {"magnific": object()}
        with mock.patch.dict(mt._CLIENTS, live, clear=True), \
             mock.patch.object(mt, "_connect", lambda *a, **k: (mock.Mock(), [])):
            mt.test_server("magnific", {"transport": "http", "url": "https://x"})
            self.assertIs(mt._CLIENTS["magnific"], live["magnific"])

    def test_a_browser_sign_in_server_with_no_token_says_so(self):
        def connect(*a, **k):
            raise mt._AuthRequired()
        with mock.patch.object(mt, "_connect", connect):
            res = mt.test_server("demo", {"transport": "http", "url": "https://x", "auth": "oauth"})
        self.assertFalse(res["ok"])
        self.assertTrue(res["needs_auth"])

    def test_a_401_buried_in_an_exception_group_reads_as_wanting_credentials(self):
        def connect(*a, **k):
            try:
                raise RuntimeError("HTTP 401 Unauthorized")
            except RuntimeError as inner:
                raise ExceptionGroup("task group", [inner]) from None
        with mock.patch.object(mt, "_connect", connect):
            res = mt.test_server("demo", {"transport": "http", "url": "https://x"})
        self.assertEqual(res["error"], "HTTP 401 Unauthorized")
        self.assertTrue(res["needs_auth"])


if __name__ == "__main__":
    unittest.main()
