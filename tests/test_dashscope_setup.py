"""The installer's DashScope endpoint step: plan detection, the URL it writes, the PAYG key
offer, and the same call in both installers. No real Alibaba calls: the probe is injected."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agenty_core import dashscope_endpoints as ep  # noqa: E402
from scripts import dashscope_setup as ds  # noqa: E402

CRLF = chr(13) + chr(10)
PLAN, PAY = "sk-sp-secret1", "sk-secret2"


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = Path(self.tmp.name) / ".env"
        self.out = []

    def go(self, key, interactive=False, answers=(), probe=None, embedder=False, extra=""):
        raw = "HF_TOKEN=abc" + CRLF + "DASHSCOPE_API_KEY=" + key + CRLF + extra
        self.env.write_bytes(raw.encode())
        it = iter(answers)
        ds.run(self.env, interactive, embedder, ask=lambda _p: next(it, ""), say=self.out.append,
               probe_fn=probe or (lambda u, k: False))
        return "\n".join(self.out)

    def test_plan_key_detected_token_plan_singapore(self):
        text = self.go(PLAN, probe=lambda u, k: "token-plan.ap-southeast-1" in u)
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_BASE_URL"), ep.endpoint_for(ep.TOKEN_PLAN, "singapore"))
        self.assertIn("Token Plan", text)
        self.assertIn("automated scripts", text)          # the terms, quoted
        self.assertIn("DASHSCOPE_PAYG_API_KEY", text)
        self.assertNotIn(PLAN, text)

    def test_payg_key_detected_region(self):
        self.go(PAY, probe=lambda u, k: u.startswith("https://dashscope.aliyuncs.com"))
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_BASE_URL"), ep.endpoint_for(ep.PAYG, "beijing"))
        self.assertNotIn("Alibaba restricts", "\n".join(self.out))

    def test_env_file_otherwise_untouched_and_crlf_kept(self):
        self.go(PAY, probe=lambda u, k: True)
        raw = self.env.read_bytes().decode()
        self.assertTrue(raw.startswith("HF_TOKEN=abc\r\nDASHSCOPE_API_KEY=sk-secret2\r\n"))
        self.assertTrue(raw.endswith("\r\n"))

    def test_non_interactive_undetected_keeps_default(self):
        self.go(PAY)
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_BASE_URL"), "")

    def test_non_interactive_plan_key_never_asks_or_writes_payg(self):
        self.go(PLAN)
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_PAYG_API_KEY"), "")

    def test_interactive_asks_plan_and_region_when_undetected(self):
        # plan 2 = Token Plan, region 2 = Beijing, then skip the PAYG key
        self.go(PLAN, interactive=True, answers=["2", "2", ""])
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_BASE_URL"), ep.endpoint_for(ep.TOKEN_PLAN, "beijing"))
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_PAYG_API_KEY"), "")

    def test_interactive_default_for_plan_key_is_token_plan(self):
        self.go(PLAN, interactive=True, answers=["", "", ""])
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_BASE_URL"), ep.endpoint_for(ep.TOKEN_PLAN, "singapore"))

    def test_payg_key_offer_writes_key_and_region(self):
        def probe(u, k):
            return ("token-plan.ap-southeast-1" in u) if k == PLAN else u.startswith("https://dashscope-us.")
        self.go(PLAN, interactive=True, answers=["sk-payg"], probe=probe)
        # the probe got the PAYG key "sk-payg", which maps to the US endpoint
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_PAYG_API_KEY"), "sk-payg")
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_PAYG_BASE_URL"), ep.endpoint_for(ep.PAYG, "us"))

    def test_existing_url_is_kept_and_payg_key_not_reasked(self):
        text = self.go(PLAN, interactive=True, answers=["sk-new"],
                       extra="DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1\r\n"
                             "DASHSCOPE_PAYG_API_KEY=sk-have\r\n")
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_BASE_URL"), "https://coding.dashscope.aliyuncs.com/v1")
        self.assertEqual(ds.get_env(self.env, "DASHSCOPE_PAYG_API_KEY"), "sk-have")
        self.assertIn("Coding Plan", text)

    def test_no_key_does_nothing(self):
        self.env.write_text("HF_TOKEN=abc\n", encoding="utf-8")
        ds.run(self.env, True, ask=lambda _p: self.fail("asked"), say=self.out.append)
        self.assertEqual(self.out, [])
        self.assertEqual(self.env.read_text(), "HF_TOKEN=abc\n")


class InstallerParity(unittest.TestCase):
    def test_both_installers_call_the_same_step_for_llm_provider_and_embedder(self):
        ps1 = (ROOT / "install_agent.ps1").read_text(encoding="utf-8")
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        for text in (ps1, sh):
            self.assertIn("dashscope_setup.py", text)
            self.assertIn("--non-interactive", text)
            self.assertIn("--embedder", text)
        self.assertIn('if ($p.id -eq "dashscope") { Set-DashScopeEndpoint', ps1)
        self.assertIn('-eq "dashscope"', ps1)
        self.assertIn('[ "${P_ID[$i]}" = "dashscope" ] && set_dashscope_endpoint', sh)
        self.assertIn('[ "$pick" = "dashscope" ] && set_dashscope_endpoint "$env_file" embedder-only', sh)
        self.assertIn("Set-DashScopeEndpoint -EnvFile $EnvFile -EmbedderOnly", ps1)


class Wiring(unittest.TestCase):
    def test_settings_catalog_lists_endpoints_and_payg_fields(self):
        from src.utils import agentY_server as srv
        ds_svc = next(s for s in srv._services() if s["id"] == "dashscope")
        self.assertEqual({f["env"] for f in ds_svc["fields"]}, {"DASHSCOPE_PAYG_API_KEY", "DASHSCOPE_PAYG_BASE_URL"})
        self.assertEqual({e["url"] for e in ds_svc["endpoints"]}, set(ep.ENDPOINTS.values()))
        self.assertIn("DASHSCOPE_PAYG_API_KEY", srv._KNOWN_ENV_KEYS)

    def test_detect_route_function_never_returns_the_key(self):
        from unittest import mock
        from src.utils import agentY_server as srv
        with mock.patch.object(ep, "probe", lambda u, k: "coding-intl" in u):
            out = srv.detect_dashscope(PLAN)
        self.assertTrue(out["ok"])
        self.assertEqual(out["plan"], ep.CODING_PLAN)
        self.assertNotIn(PLAN, str(out))
        self.assertIn("warning", out)

    def test_embedder_uses_payg_key_over_plan_key(self):
        from unittest import mock
        from src.utils import memory
        env = {"DASHSCOPE_API_KEY": PLAN, "DASHSCOPE_PAYG_API_KEY": PAY,
               "DASHSCOPE_PAYG_BASE_URL": ep.endpoint_for(ep.PAYG, "beijing")}
        with mock.patch.dict("os.environ", env):
            self.assertEqual(memory._dashscope_embedder_credentials(),
                             (PAY, ep.endpoint_for(ep.PAYG, "beijing"), ""))

    def test_embedder_refuses_plan_key_alone_naming_the_fix(self):
        from unittest import mock
        from src.utils import memory
        with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": PLAN, "DASHSCOPE_PAYG_API_KEY": ""}):
            key, _base, why = memory._dashscope_embedder_credentials()
        self.assertEqual(key, "")
        self.assertIn("DASHSCOPE_PAYG_API_KEY", why)


if __name__ == "__main__":
    unittest.main()
