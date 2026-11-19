"""The farm machine image's helper node (deploy/farm-image/agenty_farm_helper): model downloads and the token."""
import asyncio
import importlib.util
import io
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:  # pragma: no cover
    web = None

HELPER = Path(__file__).resolve().parents[1] / "deploy" / "farm-image" / "agenty_farm_helper" / "__init__.py"


def load_helper(models_dir: Path, token: str, storage: str = ""):
    """Import the helper against stand-ins for ComfyUI's folder_paths and PromptServer."""
    fp = types.ModuleType("folder_paths")
    fp.folder_names_and_paths = {"diffusion_models": ([str(models_dir / "diffusion_models")], set()),
                                 "loras": ([str(models_dir / "loras")], set())}
    fp.get_folder_paths = lambda f: fp.folder_names_and_paths[f][0]
    srv = types.ModuleType("server")
    app = web.Application()
    routes = web.RouteTableDef()

    @routes.get("/system_stats")
    async def stats(request):
        return web.json_response({"ok": True})
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(app=app, routes=routes))
    with mock.patch.dict(sys.modules, {"folder_paths": fp, "server": srv}), \
            mock.patch.dict(os.environ, {"COMFY_TOKEN": token, "FARM_MODELS_DIR": storage}):
        spec = importlib.util.spec_from_file_location("agenty_farm_helper_under_test", HELPER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    app.add_routes(routes)
    return mod, app


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@unittest.skipIf(web is None, "aiohttp not installed")
class FarmHelper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_app(self, app, fn):
        async def go():
            async with TestClient(TestServer(app)) as c:
                return await fn(c)
        return asyncio.run(go())

    def test_without_the_token_nothing_answers(self):
        _, app = load_helper(self.models, "tok")

        async def fn(c):
            no = await c.get("/system_stats")
            yes = await c.get("/system_stats", headers={"Authorization": "Bearer tok"})
            wrong = await c.get("/farm/fetch", headers={"Authorization": "Bearer nope"})
            browser = await c.get("/system_stats?token=tok")
            return no.status, yes.status, wrong.status, browser.status, browser.cookies.get("agenty_farm")
        no, yes, wrong, browser, cookie = self.run_app(app, fn)
        self.assertEqual((no, yes, wrong, browser), (401, 200, 401, 200))
        self.assertEqual(cookie.value, "tok")

    def test_a_model_is_downloaded_into_its_folder(self):
        mod, app = load_helper(self.models, "")
        data = b"x" * 5000
        with mock.patch.object(mod.urllib.request, "urlopen", lambda req, timeout=60: FakeResponse(data)):
            async def fn(c):
                r = await c.post("/farm/fetch", json={"items": [
                    {"folder": "diffusion_models", "name": "moss/m.safetensors", "url": "https://hf.co/x/m.safetensors"},
                    {"folder": "loras", "name": "../evil.safetensors", "url": "https://hf.co/e"},
                    {"folder": "nope", "name": "a.safetensors", "url": "https://hf.co/a"},
                    {"folder": "loras", "name": "b.safetensors", "url": "http://insecure/b"}]})
                first = await r.json()
                for _ in range(50):
                    st = await (await c.get("/farm/fetch")).json()
                    if all(i["state"] in ("done", "failed") for i in st["items"]):
                        break
                    await asyncio.sleep(0.1)
                return first, st
            first, st = self.run_app(app, fn)
        states = [i["state"] for i in first["items"]]
        self.assertEqual(states[1:], ["failed", "failed", "failed"])
        done = [i for i in st["items"] if i["name"] == "moss/m.safetensors"][0]
        self.assertEqual((done["state"], done["bytes"], done["total"]), ("done", 5000, 5000))
        self.assertEqual((self.models / "diffusion_models" / "moss" / "m.safetensors").read_bytes(), data)
        self.assertFalse(any(p.name.endswith(".part") for p in self.models.rglob("*")))

    def test_a_file_already_there_is_done_at_once(self):
        mod, app = load_helper(self.models, "")
        (self.models / "loras").mkdir(parents=True)
        (self.models / "loras" / "have.safetensors").write_bytes(b"abc")

        async def fn(c):
            r = await c.post("/farm/fetch", json={"items": [
                {"folder": "loras", "name": "have.safetensors", "url": "https://hf.co/h"}]})
            return await r.json()
        out = self.run_app(app, fn)
        self.assertEqual(out["items"][0]["state"], "done")
        self.assertEqual(out["items"][0]["total"], 3)

    def test_with_storage_downloads_land_on_it_and_are_listed(self):
        vol = self.models / "vol"
        mod, app = load_helper(self.models, "", storage=str(vol / "models"))
        data = b"y" * 300
        with mock.patch.object(mod.urllib.request, "urlopen", lambda req, timeout=60: FakeResponse(data)):
            async def fn(c):
                await c.post("/farm/fetch", json={"items": [
                    {"folder": "loras", "name": "s.safetensors", "url": "https://hf.co/s"}]})
                for _ in range(50):
                    st = await (await c.get("/farm/fetch")).json()
                    if st["items"][0]["state"] in ("done", "failed"):
                        break
                    await asyncio.sleep(0.1)
                return await (await c.get("/farm/models")).json()
            out = self.run_app(app, fn)
        self.assertEqual((vol / "models" / "loras" / "s.safetensors").read_bytes(), data)
        self.assertEqual(out["files"], [{"folder": "loras", "name": "s.safetensors", "bytes": 300}])


if __name__ == "__main__":
    unittest.main()
