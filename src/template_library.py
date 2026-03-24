"""Remote ComfyUI workflow-template retrieval and caching.

This module provides a lightweight retrieval layer over the public
``Comfy-Org/workflow_templates`` repository. It fetches ``templates/index.json``,
flattens it into searchable template metadata, ranks candidates against a user
brief, and can cache selected workflow JSON files locally.

The goal is not to mirror the entire upstream repository into the workspace.
Instead, it supports a retrieval-augmented selection flow where the agent sees
only the most relevant remote templates, then downloads the chosen workflow on
demand.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


_INDEX_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/"
    "workflow_templates/main/templates/index.json"
)
_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/"
    "workflow_templates/main/templates/{name}.json"
)
_REQUEST_TIMEOUT = 30
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class RemoteTemplateCandidate:
    """Searchable metadata for one remote workflow template."""

    name: str
    title: str
    description: str
    media_type: str
    tags: list[str]
    models: list[str]
    input_count: int
    open_source: bool
    usage: int
    score: float = 0.0


class RemoteTemplateLibrary:
    """Retrieval and caching facade for upstream ComfyUI workflow templates."""

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._index_cache_file = os.path.join(cache_dir, "remote_template_index.json")
        self._template_cache_dir = os.path.join(cache_dir, "remote_templates")

    def _ensure_cache_dirs(self) -> None:
        """Create local cache directories on demand."""
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._template_cache_dir, exist_ok=True)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Split free text into lowercase search tokens."""
        return {
            token
            for token in _TOKEN_RE.findall(text.lower())
            if len(token) > 1
        }

    def _load_cached_index(self) -> list[dict[str, Any]]:
        """Load the locally cached upstream index, if available."""
        if not os.path.exists(self._index_cache_file):
            return []
        try:
            with open(self._index_cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save_index_cache(self, data: list[dict[str, Any]]) -> None:
        """Persist the fetched upstream index locally."""
        self._ensure_cache_dirs()
        with open(self._index_cache_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def fetch_index(self) -> list[dict[str, Any]]:
        """Return the upstream template index, using cache as fallback."""
        try:
            response = requests.get(_INDEX_URL, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                self._save_index_cache(data)
                return data
        except (requests.RequestException, ValueError):
            pass
        return self._load_cached_index()

    def list_templates(self) -> list[RemoteTemplateCandidate]:
        """Flatten ``index.json`` into searchable template metadata."""
        flattened: list[RemoteTemplateCandidate] = []
        for category in self.fetch_index():
            if not isinstance(category, dict):
                continue
            templates = category.get("templates", [])
            if not isinstance(templates, list):
                continue
            for template in templates:
                if not isinstance(template, dict):
                    continue
                name = str(template.get("name") or "").strip()
                description = str(template.get("description") or "").strip()
                if not name or not description:
                    continue
                io_block = template.get("io") or {}
                inputs = io_block.get("inputs") if isinstance(io_block, dict) else []
                flattened.append(
                    RemoteTemplateCandidate(
                        name=name,
                        title=str(template.get("title") or name),
                        description=description,
                        media_type=str(template.get("mediaType") or "image"),
                        tags=[str(tag) for tag in template.get("tags", [])],
                        models=[str(model) for model in template.get("models", [])],
                        input_count=len(inputs) if isinstance(inputs, list) else 0,
                        open_source=bool(template.get("openSource", False)),
                        usage=int(template.get("usage", 0) or 0),
                    )
                )
        return flattened

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        media_type: str | None = "image",
        prompt_only: bool = False,
    ) -> list[RemoteTemplateCandidate]:
        """Return the top remote templates matching *query*."""
        query_tokens = self._tokenize(query)
        candidates: list[RemoteTemplateCandidate] = []

        for template in self.list_templates():
            if media_type and template.media_type != media_type:
                continue
            if prompt_only and template.input_count > 0:
                continue

            name_tokens = self._tokenize(template.name)
            title_tokens = self._tokenize(template.title)
            description_tokens = self._tokenize(template.description)
            tag_tokens = self._tokenize(" ".join(template.tags))
            model_tokens = self._tokenize(" ".join(template.models))

            exact_phrase_bonus = 0.0
            query_text = query.lower()
            if query_text and query_text in template.description.lower():
                exact_phrase_bonus += 4.0
            if query_text and query_text in template.title.lower():
                exact_phrase_bonus += 6.0

            overlap = lambda tokens: len(query_tokens & tokens)
            score = 0.0
            score += overlap(name_tokens) * 5.0
            score += overlap(title_tokens) * 6.0
            score += overlap(description_tokens) * 4.0
            score += overlap(tag_tokens) * 2.0
            score += overlap(model_tokens) * 1.5
            score += exact_phrase_bonus
            score += min(math.log10(template.usage + 1), 4.0)
            if template.open_source:
                score += 0.5
            if template.input_count == 0:
                score += 1.0

            if score <= 0:
                continue

            candidates.append(
                RemoteTemplateCandidate(
                    name=template.name,
                    title=template.title,
                    description=template.description,
                    media_type=template.media_type,
                    tags=template.tags,
                    models=template.models,
                    input_count=template.input_count,
                    open_source=template.open_source,
                    usage=template.usage,
                    score=score,
                )
            )

        candidates.sort(
            key=lambda candidate: (-candidate.score, -candidate.usage, candidate.name)
        )
        return candidates[:limit]

    def ensure_template_cached(self, name: str) -> str:
        """Download a remote workflow JSON into the local cache if needed."""
        self._ensure_cache_dirs()
        target_path = os.path.join(self._template_cache_dir, f"{name}.json")
        if os.path.exists(target_path):
            return target_path

        response = requests.get(
            _TEMPLATE_URL.format(name=name),
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        with open(target_path, "w", encoding="utf-8") as fh:
            fh.write(response.text)
        return target_path
