"""Run a snippet in an ``agentY python`` node and read back what it produced.

The node (agentY-comfyuiConnect) runs Python inside ComfyUI, which is the one
place a snippet can take a canvas node's output — an IMAGE tensor, a loaded
video — as an argument, and hand a value or a file on to the graph. The agent
uses it for code the user wants ON the canvas; `run_script` stays the tool for
code that only the agent needs to run.

Only the part of the canvas the snippet reads from is submitted: the python
node plus every node upstream of its wired inputs. Submitting the whole canvas
would also run the user's save nodes and generations.
"""

from __future__ import annotations

import time
from pathlib import Path

NODE_TYPE = "AgentYPython"
# Fixed outputs on the node (connect: _N_PY_OUT), and the id the node gets in
# the submitted prompt — out of the way of any canvas id.
N_OUTPUTS = 4
RUN_NODE_ID = "990001"
MAX_INPUTS = 20


class PythonNodeError(Exception):
    """A request that cannot be run as asked; the message says what to change."""


def parse_inputs(inputs, base_prompt: dict | None) -> list[tuple[str, int]]:
    """``[(node_id, output_slot), …]`` for ``in0``, ``in1``, … in order.

    Each entry names a canvas node and one of its outputs: ``{"node_id": "12",
    "output": 1}``, or just ``"12"`` for its first output. (No ``"12:1"``
    shorthand: a node inside a subgraph is called ``"5:3"``.) The node must be in
    the graph captured this turn, since that is what gets submitted.
    """
    wired: list[tuple[str, int]] = []
    for i, item in enumerate(inputs or []):
        if isinstance(item, dict):
            nid = str(item.get("node_id", "")).strip()
            slot = item.get("output", item.get("slot", 0))
        else:
            nid, slot = str(item).strip(), 0
        try:
            slot = int(slot or 0)
        except (TypeError, ValueError):
            raise PythonNodeError(f"in{i}: output {slot!r} is not a slot number.") from None
        if not nid:
            raise PythonNodeError(f"in{i}: name the canvas node it comes from (node_id).")
        if not isinstance(base_prompt, dict) or nid not in base_prompt:
            raise PythonNodeError(
                f"in{i}: node {nid} is not in the graph captured this turn — use an id "
                "from the [CANVAS GRAPH] block (the whole graph must be shared).")
        wired.append((nid, slot))
    if len(wired) > MAX_INPUTS:
        raise PythonNodeError(f"at most {MAX_INPUTS} inputs, not {len(wired)}.")
    return wired


def upstream(base_prompt: dict, roots: list[str]) -> dict:
    """The nodes *roots* depend on, *roots* included, from an API-format prompt."""
    keep: dict = {}
    stack = list(roots)
    while stack:
        nid = str(stack.pop())
        if nid in keep or nid not in base_prompt:
            continue
        node = base_prompt[nid]
        keep[nid] = node
        for value in ((node or {}).get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                stack.append(str(value[0]))
    return keep


def build_prompt(code: str, wired: list[tuple[str, int]], base_prompt: dict | None,
                 title: str = "") -> dict:
    """An API prompt: the python node reading *wired*, plus what those need."""
    import copy

    prompt = copy.deepcopy(upstream(base_prompt or {}, [n for n, _ in wired]))
    prompt[RUN_NODE_ID] = {
        "class_type": NODE_TYPE,
        "inputs": {"code": code,
                   **{f"inputs.in{i}": [nid, slot] for i, (nid, slot) in enumerate(wired)}},
        "_meta": {"title": title or "agentY python"},
    }
    return prompt


def read_result(entry: dict) -> dict:
    """``{"outputs": [lines], "files": [records]}`` from the run's history entry."""
    out = ((entry or {}).get("outputs") or {}).get(RUN_NODE_ID) or {}
    files = []
    for key in ("images", "videos", "audio", "gifs"):
        for record in out.get(key) or []:
            if isinstance(record, dict) and record.get("filename"):
                files.append(record)
    return {"outputs": [str(x) for x in (out.get("text") or [])], "files": files}


def error_of(entry: dict) -> str | None:
    """The snippet's error, if the run failed."""
    status = (entry or {}).get("status") or {}
    if status.get("status_str") != "error":
        return None
    for kind, data in status.get("messages") or []:
        if kind == "execution_error" and isinstance(data, dict):
            return str(data.get("exception_message") or "the run failed").strip()
    return "the run failed"


def run(prompt: dict, *, timeout: float = 300.0, poll: float = 0.5) -> dict:
    """Submit *prompt* and wait for it; returns the history entry.

    Polled rather than streamed: the run usually finishes in well under a
    second, and a node that saves no file never gives the streaming helper the
    output files it waits for.
    """
    from agenty_core.utils.comfyui_client import describe_node_errors, get_client

    client = get_client()
    payload: dict = {"prompt": prompt}
    if client.api_key:
        payload["extra_data"] = {"api_key_comfy_org": client.api_key}
    try:
        res = client.post("/prompt", json_data=payload)
    except Exception as exc:  # noqa: BLE001 — a 400 carries the reason in its message
        raise PythonNodeError(f"ComfyUI refused the run: {exc}") from exc
    if not isinstance(res, dict) or "prompt_id" not in res:
        detail = res.get("node_errors") if isinstance(res, dict) else None
        raise PythonNodeError(
            "ComfyUI refused the run: "
            + (describe_node_errors(detail) if detail else str((res or {}).get("error") or res)))
    if res.get("node_errors"):
        raise PythonNodeError("ComfyUI refused the run: " + describe_node_errors(res["node_errors"]))
    pid = res["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hist = client.get(f"/history/{pid}")
        entry = hist.get(pid) if isinstance(hist, dict) else None
        status = (entry or {}).get("status") or {}
        if status.get("completed") or status.get("status_str") == "error":
            return entry
        time.sleep(poll)
    raise PythonNodeError(f"the run did not finish within {int(timeout)} s.")


def local_paths(files: list[dict], resolve) -> list[str]:
    """Absolute paths for history file records, via *resolve* (the executor's)."""
    paths = []
    for rec in files:
        try:
            path = resolve(rec["filename"], rec.get("subfolder", ""), rec.get("type", "output"))
        except Exception:  # noqa: BLE001
            continue
        if path and Path(path).exists():
            paths.append(str(path))
    return paths
