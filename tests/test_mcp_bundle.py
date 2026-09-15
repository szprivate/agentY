"""Installing an MCP bundle (.mcpb) from Settings.

A bundle is a zip with a manifest.json and a local MCP server in it. Installing
one means: open it (a signed one has a signature block after the zip), say what
it is and what it will run, ask for its settings, unpack it where nothing can
escape the bundle folder, and resolve its start command exactly as the MCPB
reference implementation does. What comes out is an ordinary stdio server
entry, with any key the user typed kept out of it and returned for .env.

    python -m unittest discover -s tests
"""

import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from src.tools import mcp_tools as mt
from src.utils import mcp_bundle as mb

MANIFEST = {
    "manifest_version": "0.3", "name": "file-helper", "display_name": "File Helper",
    "version": "1.2.0", "description": "Reads files.", "author": {"name": "Ada"},
    "server": {"type": "node", "entry_point": "server/index.js", "mcp_config": {
        "command": "node",
        "args": ["${__dirname}/server/index.js", "${user_config.roots}", "--verbose=${user_config.verbose}"],
        "env": {"API_KEY": "${user_config.api_key}", "HOME_DIR": "${HOME}"}}},
    "user_config": {
        "api_key": {"type": "string", "title": "API key", "sensitive": True, "required": True},
        "roots": {"type": "directory", "title": "Folders", "multiple": True, "default": ["${HOME}/Documents"]},
        "verbose": {"type": "boolean", "default": False},
        "limit": {"type": "number", "title": "Limit", "min": 1, "max": 10},
    },
    "tools": [{"name": "read_file"}, {"name": "list_dir"}],
}


def bundle(manifest=None, files=None, wrap="", signed=False, extra=()):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(wrap + "manifest.json", json.dumps(manifest or MANIFEST))
        for name, data in (files or {"server/index.js": "console.log('v1')"}).items():
            zf.writestr(wrap + name, data)
        for name, data in extra:
            zf.writestr(name, data)
    data = buf.getvalue()
    if signed:
        sig = b"0\x82fake-pkcs7-signature"
        data += b"MCPB_SIG_V1" + len(sig).to_bytes(4, "little") + sig + b"MCPB_SIG_END"
    return data


class _Root(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "mcp_bundles"
        self.enterContext(mock.patch.object(mb, "BUNDLES_ROOT", self.root))
        # This machine's PATH is not under test: node is "installed", at 25.4.0.
        self.enterContext(mock.patch.object(mb.shutil, "which", lambda cmd: "/usr/bin/" + cmd))
        self.enterContext(mock.patch.object(mb, "_version_of", lambda path: "25.4.0"))

    def stage(self, data, **kw):
        return mb.stage_upload(io.BytesIO(data), "file-helper.mcpb", **kw)


class InspectTest(_Root):

    def test_it_says_what_it_is_and_what_it_will_run(self):
        out = mb.inspect(self.stage(bundle()))
        b = out["bundle"]
        self.assertEqual((b["display_name"], b["version"], b["author"], b["server_type"]),
                         ("File Helper", "1.2.0", "Ada", "node"))
        self.assertTrue(b["command"].startswith("node <bundle folder>/server/index.js"), b["command"])
        self.assertEqual(b["tools"], ["read_file", "list_dir"])
        self.assertEqual(out["name"], "file_helper")
        self.assertEqual([f["key"] for f in out["user_config"]], ["api_key", "roots", "verbose", "limit"])
        self.assertTrue(out["user_config"][0]["sensitive"])
        self.assertEqual(out["user_config"][1]["default"], [str(Path.home()) + "/Documents"],
                         "a default is shown resolved, not as ${HOME}")
        self.assertFalse(b["signed"])
        self.assertEqual(out["problems"], [])

    def test_a_signed_bundle_opens_and_says_it_is_signed(self):
        out = mb.inspect(self.stage(bundle(signed=True)))
        self.assertTrue(out["bundle"]["signed"])
        self.assertEqual(out["bundle"]["version"], "1.2.0")

    def test_a_bundle_zipped_inside_a_folder(self):
        self.assertEqual(mb.inspect(self.stage(bundle(wrap="file-helper/")))["name"], "file_helper")

    def test_a_file_that_is_not_a_bundle(self):
        with self.assertRaises(mb.BundleError):
            mb.inspect(self.stage(b"not a zip at all"))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "hi")
        with self.assertRaises(mb.BundleError):
            mb.inspect(self.stage(buf.getvalue()))

    def test_a_platform_it_does_not_support_blocks_the_install(self):
        other = "linux" if mb._platform() != "linux" else "darwin"
        token = self.stage(bundle({**MANIFEST, "compatibility": {"platforms": [other]}}))
        self.assertTrue(mb.inspect(token)["problems"])
        with self.assertRaises(mb.BundleError):
            mb.install(token, "file_helper", {"api_key": "k"})
        self.assertFalse((self.root / "file_helper").exists())

    def test_a_missing_runtime_is_a_warning_not_a_refusal(self):
        with mock.patch.object(mb.shutil, "which", lambda cmd: None):
            out = mb.inspect(self.stage(bundle()))
        self.assertEqual(out["problems"], [])
        self.assertTrue(any("`node`" in w and "Node.js" in w for w in out["warnings"]), out["warnings"])

    def test_an_old_runtime_is_named_with_both_versions(self):
        out = mb.inspect(self.stage(bundle({**MANIFEST, "compatibility": {"runtimes": {"node": ">=30"}}})))
        self.assertTrue(any(">=30" in w and "25.4.0" in w for w in out["warnings"]), out["warnings"])

    def test_reinstalling_offers_the_installed_name(self):
        out = mb.inspect(self.stage(bundle()), servers={"files": {"bundle": {"name": "file-helper"}}})
        self.assertEqual((out["name"], out["replaces"]), ("files", "files"))

    def test_a_name_in_use_gets_a_suffix(self):
        self.assertEqual(mb.inspect(self.stage(bundle()), taken=["file_helper"])["name"], "file_helper_2")

    def test_an_upload_over_the_limit_leaves_nothing_behind(self):
        with self.assertRaises(mb.BundleError):
            self.stage(bundle(), limit=10)
        self.assertEqual(list((self.root / ".staging").glob("*.mcpb")), [])


class InstallTest(_Root):

    def test_the_entry_is_resolved_as_the_reference_implementation_does(self):
        out = mb.install(self.stage(bundle()), "file_helper",
                         {"api_key": "sk-1", "roots": ["C:/a", "C:/b"], "verbose": True})
        target = self.root / "file_helper"
        entry = out["server"]
        self.assertEqual(entry["command"], "node")
        self.assertEqual(entry["args"], [f"{target}/server/index.js", "C:/a", "C:/b", "--verbose=true"],
                         "a multi-value setting that is a whole argument becomes several")
        self.assertEqual(entry["env"], {"API_KEY": "${MCP_FILE_HELPER_API_KEY}", "HOME_DIR": str(Path.home())})
        self.assertEqual(out["secrets"], {"MCP_FILE_HELPER_API_KEY": "sk-1"}, "the key goes to .env, not the entry")
        self.assertEqual(entry["cwd"], str(target))
        self.assertEqual(entry["bundle"], {"name": "file-helper", "version": "1.2.0",
                                           "dir": "config/mcp_bundles/file_helper", "signed": False})
        self.assertEqual((target / "server" / "index.js").read_text(), "console.log('v1')")
        self.assertTrue((target / mb.MARKER).is_file())
        self.assertEqual(list((self.root / ".staging").iterdir()), [], "the upload is discarded")

    def test_defaults_fill_what_was_not_entered(self):
        entry = mb.install(self.stage(bundle()), "file_helper", {"api_key": "k"})["server"]
        self.assertEqual(entry["args"][1:], [str(Path.home()) + "/Documents", "--verbose=false"])

    def test_a_required_setting_left_empty_installs_nothing(self):
        with self.assertRaises(mb.BundleError) as ctx:
            mb.install(self.stage(bundle()), "file_helper", {"api_key": ""})
        self.assertIn("API key", str(ctx.exception))
        self.assertFalse((self.root / "file_helper").exists())

    def test_a_number_outside_its_range(self):
        with self.assertRaises(mb.BundleError) as ctx:
            mb.install(self.stage(bundle()), "file_helper", {"api_key": "k", "limit": 50})
        self.assertIn("at most 10", str(ctx.exception))

    def test_this_platforms_override_wins(self):
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["server"]["mcp_config"]["platform_overrides"] = {
            mb._platform(): {"command": "node-special", "env": {"EXTRA": "1"}}}
        entry = mb.install(self.stage(bundle(manifest)), "file_helper", {"api_key": "k"})["server"]
        self.assertEqual(entry["command"], "node-special")
        self.assertEqual(entry["env"]["EXTRA"], "1")
        self.assertIn("API_KEY", entry["env"], "an override's env is merged over the base, not instead")

    def test_a_path_that_leaves_the_bundle_is_refused(self):
        token = self.stage(bundle(extra=[("../evil.txt", "x")]))
        with self.assertRaises(mb.BundleError):
            mb.install(token, "file_helper", {"api_key": "k"})
        self.assertFalse((self.root.parent / "evil.txt").exists())
        self.assertFalse((self.root / "file_helper").exists())
        self.assertEqual([p.name for p in self.root.iterdir() if p.name != ".staging"], [],
                         "no half-unpacked folder is left")

    def test_reinstalling_replaces_the_files(self):
        mb.install(self.stage(bundle()), "file_helper", {"api_key": "k"})
        mb.install(self.stage(bundle(files={"server/index.js": "console.log('v2')"})), "file_helper", {"api_key": "k"})
        self.assertEqual((self.root / "file_helper" / "server" / "index.js").read_text(), "console.log('v2')")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [".staging", "file_helper"])

    def test_a_folder_agentY_did_not_make_is_never_replaced(self):
        mine = self.root / "file_helper"
        mine.mkdir(parents=True)
        (mine / "notes.txt").write_text("keep me")
        with self.assertRaises(mb.BundleError):
            mb.install(self.stage(bundle()), "file_helper", {"api_key": "k"})
        self.assertEqual((mine / "notes.txt").read_text(), "keep me")

    def test_an_expired_upload(self):
        with self.assertRaises(mb.BundleError):
            mb.install("0" * 24, "file_helper", {})


class PruneTest(_Root):

    def test_only_unused_bundles_agentY_installed_are_removed(self):
        mb.install(self.stage(bundle()), "a", {"api_key": "k"})
        mb.install(self.stage(bundle()), "b", {"api_key": "k"})
        (self.root / "handmade").mkdir()
        removed = mb.prune_unreferenced({"a": {"bundle": {"dir": "config/mcp_bundles/a"}}})
        self.assertEqual(removed, ["b"])
        self.assertTrue((self.root / "a").is_dir())
        self.assertTrue((self.root / "handmade").is_dir())

    def test_saving_the_config_removes_a_deleted_servers_files(self):
        mb.install(self.stage(bundle()), "a", {"api_key": "k"})
        config = Path(self.root.parent) / "mcp.json"
        with mock.patch.object(mt, "_config_path", lambda: config):
            mt.save_mcp_config({"servers": {}})
        self.assertFalse((self.root / "a").exists())


class VersionTest(unittest.TestCase):

    def test_the_constraints_manifests_use(self):
        for have, want, expected in [
            ("25.4.0", ">=16.0.0", True), ("3.12.11", ">=3.10.0 <4", True), ("3.9.1", ">=3.10", False),
            ("3.12.0", ">=3.8,<4.0", True), ("4.0.0", ">=3.8,<4.0", False), ("3.12.0", "^3.9", True),
            ("4.1.0", "^3.9", False), ("3.10.4", "~3.10", True), ("3.11.0", "~3.10", False),
            ("18.2.0", "18", True), ("20.1.0", "^18 || ^20", True),
        ]:
            with self.subTest(have=have, want=want):
                self.assertIs(mb._satisfies(have, want), expected)

    def test_an_unreadable_constraint_is_not_a_verdict(self):
        self.assertIsNone(mb._satisfies("1.0.0", "latest"))


class StdioLaunchTest(unittest.TestCase):

    def test_references_in_args_expand_and_the_bundle_folder_is_the_working_directory(self):
        with mock.patch.dict(os.environ, {"MCP_T_KEY": "v"}), mock.patch("mcp.stdio_client") as client:
            launch = mt._transport_callable({"transport": "stdio", "command": "node",
                                             "args": ["--key=${MCP_T_KEY}"], "cwd": "/bundle"}, None)
            launch()
        params = client.call_args.args[0]
        self.assertEqual(params.args, ["--key=v"])
        self.assertEqual(str(params.cwd).replace("\\", "/"), "/bundle")


if __name__ == "__main__":
    unittest.main()
