"""Turn what an MCP server's page says to paste into agentY server entries.

Adding a server used to mean hand-writing config/mcp.json in a textarea: knowing
that agentY calls the transport ``transport`` where a README says ``type``, that a
key belongs in .env behind a ``${VAR}`` reference, and that the whole block sits
under ``"servers"``. What people actually have is one of three things, copied from
the server's own page:

* its **address**: ``https://mcp.example.com/mcp``;
* the **command** that starts it: ``npx -y @modelcontextprotocol/server-github``,
  sometimes with ``KEY=value`` in front, or written as ``claude mcp add …``;
* its **JSON block**, in whichever client's dialect the README chose:
  ``mcpServers`` (Claude Desktop, Cursor), ``servers`` (VS Code), a bare
  ``"name": {…}`` fragment, or one server object.

:func:`parse_mcp_snippet` accepts any of them and returns the servers in agentY's
shape. A literal key or token in the paste never reaches mcp.json: it comes back
in ``secrets`` for the settings modal to write to .env, and the entry references
it as ``${VAR}``. A placeholder (``<YOUR_TOKEN>``, VS Code's ``${input:…}``) is
not a secret; its variable comes back in ``missing`` instead.
"""
from __future__ import annotations

import json
import re
import shlex
from urllib.parse import urlparse

_SECRET_NAME = re.compile(r"key|token|secret|passw|\bpat\b|_pat$|auth|credential", re.I)
_REF = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_INPUT_REF = re.compile(r"\$\{input:[^}]*\}")
_PLACEHOLDER = re.compile(
    r"^<[^>]*>$|^\[[^\]]*\]$|^\{[^}]*\}$|your[_\- ]|^x{3,}$|^\*+$|^\.{3}$|^…$"
    r"|replace|changeme|^todo$", re.I)
_SCHEME = re.compile(r"^(Bearer|Token|Basic|Bot)\s+(.+)$", re.I)
_URL = re.compile(r"^https?://\S+$", re.I)
_WRAPPERS = ("mcpServers", "servers", "mcp_servers", "context_servers")
_RUNNERS = {"npx", "bunx", "pnpx", "uvx", "pipx", "yarn", "pnpm", "node", "python",
            "python3", "py", "deno", "bun", "docker", "podman", "uv"}
_RUNNER_WORDS = {"dlx", "run", "exec", "tool", "x", "-y", "--yes", "-i", "--rm", "-m"}
_TWO_PART_TLD = {"co", "com", "org", "net", "ac", "gov", "edu"}


class McpImportError(ValueError):
    """What was pasted is not something a server entry can be made from."""


def parse_mcp_snippet(text: str, existing=()) -> dict:
    """Read a pasted address, command or JSON block into agentY server entries.

    Returns ``{"servers": {name: entry}, "secrets": {VAR: value},
    "missing": [VAR, …], "notes": [str, …]}``. A name already in *existing* gets a
    ``_2``-style suffix, so an import never overwrites a configured server.
    """
    text = str(text or "").strip()
    if not text:
        raise McpImportError("Paste the server's address, its start command, or its JSON config.")
    acc = _Accumulator({str(n) for n in existing or ()})
    if text[:1] in "{[\"":
        for name, raw in _entries(_load_json(text)):
            acc.add(name, raw)
    elif _URL.match(text):
        acc.add("", {"url": text})
    else:
        acc.add_command(text)
    if not acc.servers:
        raise McpImportError("No MCP server found in that. Expected an address, a start "
                             "command, or a JSON block with a url or a command.")
    return acc.result()


# ── reading the paste ────────────────────────────────────────────────────────

def _load_json(text: str):
    """JSON as README authors write it: sometimes a bare ``"name": {…}`` fragment,
    sometimes JSONC with comments and trailing commas (VS Code's settings)."""
    cleaned = re.sub(r"(?m)^\s*//.*$", "", text)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    candidates = [text, cleaned]
    if cleaned.lstrip().startswith('"'):
        candidates.append("{" + cleaned.strip().rstrip(",") + "}")
    error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            error = error or exc
    raise McpImportError(f"That looks like JSON but does not parse: {error.msg} "
                         f"(line {error.lineno}).") from None


def _is_server(d) -> bool:
    return isinstance(d, dict) and any(d.get(k) for k in ("url", "serverUrl", "httpUrl", "command"))


def _entries(data) -> list:
    """``[(name, raw_server), …]`` from any of the JSON dialects."""
    if isinstance(data, list):
        return [(str(d.get("name") or ""), d) for d in data if _is_server(d)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("mcp"), dict):
        return _entries(data["mcp"])
    for key in _WRAPPERS:
        value = data.get(key)
        if isinstance(value, dict):
            return [(str(k), v) for k, v in value.items() if _is_server(v)]
        if isinstance(value, list):
            return _entries(value)
    if _is_server(data):
        return [(str(data.get("name") or ""), data)]
    return [(str(k), v) for k, v in data.items() if _is_server(v)]


def _split(text: str) -> list:
    """A command line into words. Windows paths keep their backslashes."""
    try:
        words = shlex.split(text, posix="\\" not in text)
    except ValueError as exc:
        raise McpImportError(f"Could not read that command: {exc}.") from None
    return [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]


# ── building entries ─────────────────────────────────────────────────────────

class _Accumulator:
    def __init__(self, taken: set):
        self.taken = set(taken)
        self.servers: dict = {}
        self.secrets: dict = {}
        self.missing: list = []

    def add(self, name: str, raw: dict) -> None:
        url = raw.get("url") or raw.get("serverUrl") or raw.get("httpUrl")
        kind = str(raw.get("type") or raw.get("transport") or "").lower()
        if raw.get("command"):
            command = str(raw["command"])
            args = [str(a) for a in (raw.get("args") or [])]
            if not args and " " in command.strip():
                command, *args = _split(command)
            entry = {"enabled": True, "transport": "stdio", "command": command,
                     "args": args, "auth": "none"}
            base = name or _name_from_command(command, args)
        elif url:
            url = str(url)
            sse = kind == "sse" or (not kind and urlparse(url).path.rstrip("/").endswith("/sse"))
            entry = {"enabled": True, "transport": "sse" if sse else "http", "url": url,
                     "auth": "none"}
            base = name or _name_from_url(url)
        else:
            return
        if raw.get("disabled") is True or raw.get("enabled") is False:
            entry["enabled"] = False
        final = self._unique(_slug(base))
        env = raw.get("env")
        if entry["transport"] == "stdio" and isinstance(env, dict) and env:
            entry["env"] = {str(k): self._value(str(k), v, str(k)) for k, v in env.items()}
        headers = raw.get("headers")
        if entry["transport"] != "stdio" and isinstance(headers, dict) and headers:
            entry["headers"] = self._headers(final, headers)
            entry["auth"] = "header"
        if str(raw.get("auth") or "").lower() == "oauth":
            entry["auth"] = "oauth"
        self.servers[final] = entry

    def add_command(self, text: str) -> None:
        words = _split(text)
        env: dict = {}
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            key, value = words.pop(0).split("=", 1)
            env[key] = value
        if len(words) >= 3 and [w.lower() for w in words[:3]] == ["claude", "mcp", "add"]:
            self._claude_add(words[3:], env)
            return
        if words:
            raw = {"command": words[0], "args": words[1:]}
            if env:
                raw["env"] = env
            self.add("", raw)

    def _claude_add(self, words: list, env: dict) -> None:
        """``claude mcp add [flags] <name> <url | -- command args…>``, the line many
        READMEs now print instead of JSON."""
        transport, headers, name, rest = "", {}, "", []
        i = 0
        while i < len(words):
            word = words[i]
            nxt = words[i + 1] if i + 1 < len(words) else None
            if word == "--":
                rest = words[i + 1:]
                break
            if word in ("--transport", "-t") and nxt:
                transport, i = nxt, i + 2
                continue
            if word in ("--env", "-e") and nxt and "=" in nxt:
                key, value = nxt.split("=", 1)
                env[key] = value
                i += 2
                continue
            if word in ("--header", "-H") and nxt and ":" in nxt:
                key, value = nxt.split(":", 1)
                headers[key.strip()] = value.strip()
                i += 2
                continue
            if word in ("--scope", "-s") and nxt:
                i += 2
                continue
            if word.startswith("-"):
                i += 1
                continue
            if not name:
                name = word
            else:
                rest.append(word)
            i += 1
        if len(rest) == 1 and _URL.match(rest[0]):
            raw: dict = {"url": rest[0], "type": transport}
            if headers:
                raw["headers"] = headers
        elif rest:
            raw = {"command": rest[0], "args": rest[1:]}
            if env:
                raw["env"] = env
        else:
            return
        self.add(name, raw)

    def _headers(self, server: str, headers: dict) -> dict:
        secret_headers = [h for h in headers if _SECRET_NAME.search(str(h))]
        out = {}
        for header, value in headers.items():
            header = str(header)
            text = "" if value is None else str(value)
            if header not in secret_headers:
                out[header] = text
                continue
            var = f"MCP_{_env_name(server)}_API_KEY"
            if len(secret_headers) > 1:
                var = f"MCP_{_env_name(server)}_{_env_name(header)}"
            scheme = _SCHEME.match(text.strip())
            if scheme:
                out[header] = f"{scheme.group(1)} " + self._value("key", scheme.group(2), var)
            else:
                out[header] = self._value("key", text, var)
        return out

    def _value(self, key: str, value, var: str) -> str:
        """One env or header value: a literal secret moves to .env, a placeholder
        becomes a variable to fill in, anything else stays as written."""
        text = "" if value is None else str(value)
        var = _env_name(var)
        if _INPUT_REF.search(text):
            self._need(var)
            return _INPUT_REF.sub("${%s}" % var, text)
        if _REF.search(text) or not _SECRET_NAME.search(key):
            return text
        if not text.strip() or _PLACEHOLDER.search(text.strip()):
            self._need(var)
            return "${%s}" % var
        self.secrets[var] = text
        return "${%s}" % var

    def _need(self, var: str) -> None:
        if var not in self.missing:
            self.missing.append(var)

    def _unique(self, base: str) -> str:
        name, n = base, 2
        while name in self.taken or name in self.servers:
            name, n = f"{base}_{n}", n + 1
        self.taken.add(name)
        return name

    def result(self) -> dict:
        notes = []
        if self.missing:
            notes.append("Fill in " + ", ".join(self.missing) + " before testing.")
        if any(e["transport"] != "stdio" and e["auth"] == "none" for e in self.servers.values()):
            notes.append("No key came with it. If Test says the server wants credentials, "
                         "choose Sign-in ▸ API key or Browser sign-in.")
        return {"servers": self.servers, "secrets": self.secrets,
                "missing": list(self.missing), "notes": notes}


# ── names ────────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "server"


def _env_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(text).upper()).strip("_") or "MCP"


def _name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in ("localhost", "0.0.0.0", "::1") or re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
        segments = [p for p in parsed.path.split("/") if p and p.lower() not in ("mcp", "sse")]
        return segments[0] if segments else "local"
    parts = [p for p in host.split(".") if p not in ("www", "mcp", "api", "server", "remote")]
    if len(parts) >= 3 and parts[-2] in _TWO_PART_TLD and len(parts[-1]) == 2:
        parts = parts[:-2]
    elif len(parts) > 1:
        parts = parts[:-1]
    return parts[-1] if parts else "server"


def _name_from_command(command: str, args: list) -> str:
    """``npx -y @modelcontextprotocol/server-github`` → ``github``;
    ``uvx mcp-server-fetch`` → ``fetch``; ``docker run … ghcr.io/x/github-mcp-server``
    → ``github``."""
    word = re.split(r"[\\/]", command)[-1].lower()
    word = re.sub(r"\.(exe|cmd|bat)$", "", word)
    if word in _RUNNERS:
        package = ""
        for arg in args:
            if arg.startswith("-") or arg.lower() in _RUNNER_WORDS or re.match(r"^[A-Z0-9_]+(=.*)?$", arg):
                continue
            package = arg
            break
        word = package or word
    word = word.lstrip("@")
    word = word.split("@")[0]
    word = re.split(r"[\\/:]", word.rstrip("/\\"))[-1]
    word = re.sub(r"\.(py|js|ts|mjs|cjs)$", "", word)
    word = re.sub(r"^(mcp[-_]server[-_]|server[-_]|mcp[-_])", "", word)
    word = re.sub(r"([-_]mcp[-_]server|[-_]mcp|[-_]server)$", "", word)
    return word or "server"
