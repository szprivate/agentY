"""The memory embedder can be chosen, and switching it keeps every memory.

Without Ollama the default embedder cannot run, and an embedder is not a chat
model, so each provider's own embedding model is offered with the key agentY
already has. Vectors from two embedders cannot be compared — not even two of the
same size — so a switch re-embeds the store from its texts.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mem0.vector_stores.faiss import FAISS

from src.utils import memory as mem


class _FakeEmbedder:
    def __init__(self, dims, fail=False):
        self.dims, self.fail, self.calls = dims, fail, 0

    def embed(self, text, action=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("401 Incorrect API key")
        return [float((hash(text) >> i) & 1) for i in range(self.dims)]


def _cfg(store, provider="openai", model="text-embedding-v4", dims=1024, base="https://ds/v1"):
    c = {"model": model, "embedding_dims": dims}
    if provider == "openai":
        c["openai_base_url"] = base
    else:
        c["ollama_base_url"] = "http://127.0.0.1:11434"
    return {"embedder": {"provider": provider, "config": c},
            "vector_store": {"provider": "faiss", "config": {
                "collection_name": "agenty_memory", "path": str(store), "embedding_model_dims": dims}}}


class Switching(unittest.TestCase):

    def setUp(self):
        self.store = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.store, True)
        old = FAISS(collection_name="agenty_memory", path=str(self.store), embedding_model_dims=768)
        self.payloads = {
            "id-a": {"data": "User prefers 1024x1024 portraits.", "user_id": "learnings_global",
                     "source": "memory_write", "created_at": "2026-07-01T00:00:00+00:00"},
            "id-b": {"data": "Wan 2.2 A14B uses the Wan 2.1 VAE.", "user_id": "learnings_global",
                     "created_at": "2026-08-01T00:00:00+00:00"},
        }
        old.insert(vectors=[[0.1] * 768, [0.2] * 768], payloads=list(self.payloads.values()),
                   ids=list(self.payloads))
        old_fp = mem._fingerprint(_cfg(self.store, "ollama", "nomic-embed-text", 768))
        (self.store / "embedder.json").write_text(json.dumps(old_fp), encoding="utf-8")

    def test_a_new_embedder_re_embeds_everything_and_keeps_it(self):
        fake = _FakeEmbedder(1024)
        with mock.patch("mem0.utils.factory.EmbedderFactory.create", return_value=fake):
            mem._match_index_to_embedder(_cfg(self.store))
        fresh = FAISS(collection_name="agenty_memory", path=str(self.store), embedding_model_dims=1024)
        self.assertEqual(fresh.index.d, 1024)
        self.assertEqual(fresh.index.ntotal, 2)
        self.assertEqual(fresh.docstore, self.payloads)
        self.assertEqual(sorted(fresh.index_to_id.values()), ["id-a", "id-b"])
        self.assertEqual(len(list(self.store.glob("backup-embedder-*/agenty_memory.faiss"))), 1)
        self.assertEqual(json.loads((self.store / "embedder.json").read_text())["model"], "text-embedding-v4")
        # Done once: the same embedder again changes nothing.
        with mock.patch("mem0.utils.factory.EmbedderFactory.create") as again:
            mem._match_index_to_embedder(_cfg(self.store))
        again.assert_not_called()

    def test_same_size_different_embedder_still_re_embeds(self):
        fake = _FakeEmbedder(768)
        with mock.patch("mem0.utils.factory.EmbedderFactory.create", return_value=fake):
            mem._match_index_to_embedder(_cfg(self.store, model="gemini-embedding-001", dims=768,
                                              base="https://gemini/"))
        self.assertEqual(fake.calls, 3)      # the trial, then both memories

    def test_an_embedder_that_fails_touches_nothing(self):
        before = (self.store / "agenty_memory.faiss").read_bytes()
        with mock.patch("mem0.utils.factory.EmbedderFactory.create", return_value=_FakeEmbedder(1024, fail=True)):
            with self.assertRaises(RuntimeError):
                mem._match_index_to_embedder(_cfg(self.store))
        self.assertEqual((self.store / "agenty_memory.faiss").read_bytes(), before)
        self.assertEqual(list(self.store.glob("backup-embedder-*")), [])
        self.assertEqual(json.loads((self.store / "embedder.json").read_text())["provider"], "ollama")

    def test_an_index_without_a_record_is_taken_as_built_by_the_current_embedder(self):
        (self.store / "embedder.json").unlink()
        with mock.patch("mem0.utils.factory.EmbedderFactory.create") as create:
            mem._match_index_to_embedder(_cfg(self.store, "ollama", "nomic-embed-text", 768))
        create.assert_not_called()
        self.assertEqual(json.loads((self.store / "embedder.json").read_text())["model"], "nomic-embed-text")

    def test_an_index_of_the_wrong_size_without_a_record_is_re_embedded(self):
        (self.store / "embedder.json").unlink()
        with mock.patch("mem0.utils.factory.EmbedderFactory.create", return_value=_FakeEmbedder(1024)):
            mem._match_index_to_embedder(_cfg(self.store))
        self.assertEqual(FAISS(collection_name="agenty_memory", path=str(self.store),
                               embedding_model_dims=1024).index.d, 1024)


class Choosing(unittest.TestCase):

    def test_a_preset_builds_its_embedder_with_the_agents_endpoint(self):
        with mock.patch.object(mem, "_get", side_effect=lambda env, *path, default="": (
                "dashscope" if path[-1:] == ("preset",) else default)), \
             mock.patch.object(mem, "_provider_base_url", return_value="https://ws-x/compatible-mode/v1"), \
             mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}):
            cfg = mem._build_config()
        e = cfg["embedder"]
        self.assertEqual(e["provider"], "openai")
        self.assertEqual(e["config"]["model"], "text-embedding-v4")
        self.assertEqual(e["config"]["embedding_dims"], 1024)
        self.assertEqual(e["config"]["openai_base_url"], "https://ws-x/compatible-mode/v1")
        self.assertEqual(e["config"]["api_key"], "k")
        self.assertEqual(cfg["vector_store"]["config"]["embedding_model_dims"], 1024)

    def test_the_local_preset_runs_in_process_and_keeps_its_model_out_of_the_repo(self):
        with mock.patch.object(mem, "_get", side_effect=lambda env, *path, default="": (
                "local" if path[-1:] == ("preset",) else default)),              mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FASTEMBED_CACHE_PATH", None)
            cfg = mem._build_config()
            cache = os.environ.get("FASTEMBED_CACHE_PATH", "")
        e = cfg["embedder"]
        self.assertEqual(e["provider"], "fastembed")
        self.assertEqual(e["config"]["model"], "nomic-ai/nomic-embed-text-v1.5-Q")
        self.assertEqual(e["config"]["embedding_dims"], 768)
        self.assertEqual(Path(cache), mem.LOCAL_MODELS_DIR)
        self.assertEqual(mem.LOCAL_MODELS_DIR.relative_to(mem._PROJECT_ROOT).parts[0], "models")
        import subprocess
        ignored = subprocess.run(["git", "check-ignore", "-q", str(mem.LOCAL_MODELS_DIR / "x.onnx")],
                                 cwd=str(mem._PROJECT_ROOT))
        self.assertEqual(ignored.returncode, 0, "models/embeddings must be gitignored")

    def test_choices_say_what_is_missing(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g"}, clear=False), \
             mock.patch.object(mem, "_ollama_reachable", return_value=False):
            os.environ.pop("OPENAI_API_KEY", None)
            by = {c["id"]: c for c in mem.embedder_choices()}
        self.assertTrue(by["gemini"]["available"])
        self.assertFalse(by["ollama"]["available"])
        self.assertIn("not running", by["ollama"]["why"])
        self.assertIn("OPENAI_API_KEY", by["openai"]["why"])
        self.assertIn("local", by)
        with mock.patch.object(mem, "_fastembed_installed", return_value=False):
            local = {c["id"]: c for c in mem.embedder_choices()}["local"]
        self.assertFalse(local["available"])
        self.assertIn("fastembed", local["why"])


if __name__ == "__main__":
    unittest.main()
