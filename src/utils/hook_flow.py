"""Flow control on the canvas: loops and parallel branches of a hook pipeline.

Two nodes mark a loop — ``agentY loop start`` and ``agentY loop break`` — and
every hook wired between them is its body. The break carries the condition in
the user's own words ("the dancer's pose matches the reference"), a cap on the
rounds, and which outputs go on to whatever follows.

Neither node does work, so the rest of the system never sees them:
:func:`plan` takes the hooks as the canvas reports them and hands back the work
hooks wired as if the flow nodes were plain wire, plus what those nodes said.
Chains stay chains, QA scoping and the keep switch read what they always read.

Hooks that share no wire are separate branches; :func:`branches` finds them, and
a run with more than one is worked on concurrently (see ``parallel_lines``).

Pure data in, pure data out: nothing here touches ComfyUI or a model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

LOOP_START = "loop_start"
LOOP_BREAK = "loop_break"
_FLOW = {LOOP_START, LOOP_BREAK}

DEFAULT_ROUNDS = 3
MAX_ROUNDS = 10

# What goes on from a loop when it ends.
FORWARD_BEST = "best"
FORWARD_PASSING = "all that pass"
FORWARD_ALL = "all"
_FORWARD = (FORWARD_BEST, FORWARD_PASSING, FORWARD_ALL)


def _purpose(hook: dict) -> str:
    return str((hook or {}).get("purpose") or "").strip().lower()


def is_loop_start(hook: dict) -> bool:
    return _purpose(hook) == LOOP_START


def is_loop_break(hook: dict) -> bool:
    return _purpose(hook) == LOOP_BREAK


def is_flow(hook: dict) -> bool:
    return _purpose(hook) in _FLOW


_REVIEW = {"human_review", "human review", "review", "halt", "pause",
           "check_in", "check-in", "checkin"}


def _is_review(hook: dict) -> bool:
    """A stop for the person to choose at (canvas_hooks._is_review, kept in step)."""
    return _purpose(hook) in _REVIEW


_REVIEW = {"human_review", "human review", "review", "halt", "pause",
           "check_in", "check-in", "checkin"}


def _is_review(hook: dict) -> bool:
    """A stop for the person to choose at (canvas_hooks._is_review, kept in step)."""
    return _purpose(hook) in _REVIEW


def _hid(hook: dict) -> str:
    return str((hook or {}).get("hook_node_id") or "")


def _via_ids(hook: dict) -> list[str]:
    """Hooks this one follows by way of real nodes (canvas_hooks.link_through_nodes)."""
    return [str(i) for i in (hook.get("via_hook_ids") or []) if i is not None]


def _prev_ids(hook: dict, direct_only: bool = False) -> list[str]:
    ids = [str(i) for i in (hook.get("prev_hook_ids") or []) if i is not None]
    if not direct_only:
        ids += [i for i in _via_ids(hook) if i not in ids]
    one = hook.get("prev_hook_id")
    if one is not None and str(one) not in ids:
        ids.insert(0, str(one))
    return ids


def clamp_rounds(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_ROUNDS
    return max(1, min(MAX_ROUNDS, n))


def forward_mode(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in _FORWARD else FORWARD_BEST


@dataclass
class Loop:
    """One loop: the hooks between a start and a break, and how it ends."""
    break_id: str
    start_id: str = ""
    members: list = field(default_factory=list)   # work-hook ids, in run order
    condition: str = ""
    max_rounds: int = DEFAULT_ROUNDS
    forward: str = FORWARD_BEST
    title: str = ""
    # A review hook in the body: the PERSON judges this loop, not the QA agent.
    review_id: str = ""
    # A review hook in the body: the PERSON judges this loop, not the QA agent.
    review_id: str = ""

    def name(self) -> str:
        return self.title or f"loop {self.break_id}"


@dataclass
class Flow:
    """A hook pipeline with its flow nodes taken out of the wiring."""
    hooks: list = field(default_factory=list)     # work hooks, rewired
    loops: list = field(default_factory=list)
    problems: list = field(default_factory=list)  # things the user should fix

    def loop_of(self, hook_id) -> Loop | None:
        hid = str(hook_id)
        return next((lp for lp in self.loops if hid in lp.members), None)

    def loop(self, break_id) -> Loop | None:
        bid = str(break_id)
        return next((lp for lp in self.loops if lp.break_id == bid or lp.start_id == bid), None)


def _resolve_through(hook_id: str, by_id: dict, seen: set | None = None) -> list[str]:
    """The work hooks a link from *hook_id* really comes from."""
    seen = seen if seen is not None else set()
    if hook_id in seen:
        return []
    seen.add(hook_id)
    hook = by_id.get(hook_id)
    if hook is None:
        return []
    if not is_flow(hook):
        return [hook_id]
    out: list[str] = []
    for pid in _prev_ids(hook):
        for rid in _resolve_through(pid, by_id, seen):
            if rid not in out:
                out.append(rid)
    return out


def _flow_anchors(hook_id: str, by_id: dict, seen: set | None = None) -> list:
    """Real nodes wired into the flow nodes a link from *hook_id* passes through."""
    seen = seen if seen is not None else set()
    hook = by_id.get(hook_id)
    if hook is None or not is_flow(hook) or hook_id in seen:
        return []
    seen.add(hook_id)
    out = [a for a in (hook.get("anchors") or []) if isinstance(a, dict)]
    for pid in _prev_ids(hook):
        out += _flow_anchors(pid, by_id, seen)
    return out


def _ancestors(hook_id: str, by_id: dict, stop_at_start: bool = True) -> tuple[list[str], str]:
    """Hooks upstream of *hook_id* up to the nearest loop start: (ids, start_id)."""
    found: list[str] = []
    start = ""
    stack = list(_prev_ids(by_id.get(hook_id) or {}))
    seen: set = set()
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in by_id:
            continue
        seen.add(pid)
        hook = by_id[pid]
        if is_loop_start(hook) and stop_at_start:
            start = start or pid
            continue
        found.append(pid)
        stack.extend(_prev_ids(hook))
    return found, start


def plan(hooks: list | None) -> Flow:
    """Split *hooks* into work hooks (rewired past the flow nodes) and loops."""
    raw = [h for h in (hooks or []) if isinstance(h, dict)]
    by_id = {_hid(h): h for h in raw if _hid(h)}
    flow = Flow()
    if not any(is_flow(h) for h in raw):
        flow.hooks = raw
        return flow

    order = [_hid(h) for h in raw]
    for brk in (h for h in raw if is_loop_break(h)):
        bid = _hid(brk)
        upstream, start = _ancestors(bid, by_id)
        members = [i for i in order if i in set(upstream) and not is_flow(by_id[i])]
        loop = Loop(break_id=bid, start_id=start, members=members,
                    condition=" ".join(str(brk.get("condition") or brk.get("directive") or "").split()),
                    max_rounds=clamp_rounds(brk.get("max_rounds")),
                    forward=forward_mode(brk.get("forward")),
                    title=str(brk.get("title") or "").strip())
        if loop.title.lower() in ("", "agenty loop break"):
            loop.title = ""
        loop.review_id = next((i for i in reversed(members) if _is_review(by_id[i])), "")
        loop.review_id = next((i for i in reversed(members) if _is_review(by_id[i])), "")
        if not members:
            flow.problems.append(f"{loop.name()}: nothing is wired between the loop start and the "
                                 "loop break, so there is nothing to repeat.")
            continue
        if not start:
            flow.problems.append(f"{loop.name()}: no loop start is wired upstream of it, so the "
                                 "loop takes every stage that leads into the break.")
        if not loop.condition:
            flow.problems.append(f"{loop.name()}: the break has no condition, so it ends when the "
                                 "QA briefings on its stages pass.")
        flow.loops.append(loop)
    used_starts = {lp.start_id for lp in flow.loops}
    for h in raw:
        if is_loop_start(h) and _hid(h) not in used_starts:
            flow.problems.append(f"loop start {_hid(h)} has no loop break downstream of it — "
                                 "it does nothing.")

    for h in raw:
        if is_flow(h):
            continue
        h = copy.deepcopy(h)
        prev: list[str] = []
        extra: list = []
        for pid in _prev_ids(h, direct_only=True):
            for rid in _resolve_through(pid, by_id):
                if rid not in prev:
                    prev.append(rid)
            extra += _flow_anchors(pid, by_id)
        # Reached by way of real nodes: an order, not a value that is read.
        via: list[str] = []
        for pid in _via_ids(h):
            for rid in _resolve_through(pid, by_id):
                if rid not in prev and rid not in via and rid != _hid(h):
                    via.append(rid)
        h["via_hook_ids"] = via
        links = []
        for link in (h.get("prev_links") or []):
            if not isinstance(link, dict):
                continue
            src = str(link.get("from_hook_id"))
            if src in by_id and is_flow(by_id[src]):
                links += [{**link, "from_hook_id": rid, "from_output_slot": 0}
                          for rid in _resolve_through(src, by_id)]
            else:
                links.append(link)
        h["prev_hook_ids"] = prev
        h["prev_hook_id"] = prev[0] if prev else None
        h["prev_links"] = links
        if extra:
            have = {str(a.get("node_id")) for a in (h.get("anchors") or []) if isinstance(a, dict)}
            h["anchors"] = list(h.get("anchors") or []) + [
                a for a in extra if str(a.get("node_id")) not in have]
            if not h.get("anchor_node_id") and h["anchors"]:
                first = h["anchors"][0]
                h["anchor_node_id"] = first.get("node_id")
                h["anchor_type"] = first.get("type")
                h["anchor_title"] = first.get("title")
                h["anchor_widgets"] = first.get("widgets") or {}
        # A target that is a flow node is not an input anyone fills.
        h["targets"] = [t for t in (h.get("targets") or [])
                        if not (isinstance(t, dict) and str(t.get("node_id")) in by_id
                                and is_flow(by_id[str(t.get("node_id"))]))]
        flow.hooks.append(h)
    return flow


def branches(hooks: list | None) -> list[list[str]]:
    """Work hooks grouped into branches that share no wire, in canvas order."""
    work = [h for h in (hooks or []) if isinstance(h, dict) and _hid(h)]
    ids = [_hid(h) for h in work]
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for h in work:
        for pid in _prev_ids(h):
            if pid in parent:
                parent[find(pid)] = find(_hid(h))
    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ── a loop while it runs ────────────────────────────────────────────────────

def new_state(loop: Loop) -> dict:
    return {"break_id": loop.break_id, "round": 0, "max_rounds": loop.max_rounds,
            "rounds": [], "finished": False, "outcome": ""}


def _passing(results: list) -> list:
    return [r for r in results if r.get("passed")]


def best_of(results: list) -> dict | None:
    """The candidate to carry on with: passing before failing, then by score."""
    if not results:
        return None
    return max(results, key=lambda r: (bool(r.get("passed")), -len(r.get("missed") or []),
                                       float(r.get("score") or 0.0)))


def choose(results: list, forward: str) -> list[str]:
    """Which files leave the loop."""
    if forward == FORWARD_ALL:
        return [r["path"] for r in results]
    if forward == FORWARD_PASSING:
        passing = _passing(results)
        if passing:
            return [r["path"] for r in passing]
    top = best_of(results)
    return [top["path"]] if top else []


def record_round(state: dict, results: list, loop: Loop) -> dict:
    """Close a round with its judged *results*; says whether the loop goes on.

    A round ends the loop when a candidate meets everything, or when the rounds
    are used up. Otherwise the answer carries what was missed, which is what the
    next round has to fix.
    """
    state["round"] += 1
    n = state["round"]
    state["rounds"].append({"round": n, "results": results})
    done = bool(results) and any(r.get("passed") for r in results)
    out_of_rounds = n >= state["max_rounds"]
    everything = [r for rnd in state["rounds"] for r in rnd["results"]]
    verdict = {"round": n, "max_rounds": state["max_rounds"], "finished": done or out_of_rounds,
               "condition_met": done}
    if done:
        state.update(finished=True, outcome="met")
        verdict["forward"] = choose(results, loop.forward)
    elif out_of_rounds:
        state.update(finished=True, outcome="out_of_rounds")
        # Nothing met the condition: the best attempt of ANY round goes on, not
        # merely the last one.
        top = best_of(everything)
        verdict["forward"] = ([r["path"] for r in everything] if loop.forward == FORWARD_ALL
                              else [top["path"]] if top else [])
    else:
        top = best_of(results)
        missed: list[str] = []
        for r in results:
            for m in (r.get("missed") or []):
                if m not in missed:
                    missed.append(m)
        verdict["missed"] = missed[:8]
        verdict["closest"] = top["path"] if top else ""
        verdict["rounds_left"] = state["max_rounds"] - n
    return verdict


# ── what the agent is told ──────────────────────────────────────────────────

def _label(hook: dict) -> str:
    title = " ".join(str(hook.get("title") or "").split())
    if title.lower() in ("", "agenty hook"):
        title = " ".join(str(hook.get("directive") or "").split())[:50]
    return f'hook {_hid(hook)} "{title}"' if title else f"hook {_hid(hook)}"


def loop_lines(flow: Flow) -> list[str]:
    if not flow.loops and not flow.problems:
        return []
    by_id = {_hid(h): h for h in flow.hooks}
    lines: list[str] = []
    if flow.loops:
        lines.append(
            "\nLOOPS — the stages listed under a loop are its BODY: they repeat until the "
            "loop's condition is met or its rounds are used up. One round = run every stage "
            "of the body in order, then call loop_check(break_node_id, outputs=[the files "
            "the LAST stage of the body produced this round]). You do not judge the "
            "condition yourself: loop_check has a separate QA agent judge each output "
            "against the condition and the QA briefings on those stages, and answers with "
            "one of two things.\n"
            "  • `finished: false` — `missed` says what the judge objected to and "
            "`closest` is the best attempt so far. Change what the objections are about "
            "(the prompt, a setting, the input passed between stages — start from "
            "`closest` when the fix is a refinement of it), run the body again and call "
            "loop_check again. Never re-run a round unchanged.\n"
            "  • `finished: true` — the loop is over, whether the condition was met "
            "(`condition_met`) or the rounds ran out. `forward` names the file(s) that "
            "leave the loop: those, and only those, are the input of the stage wired "
            "after the break. Do not start another round, and say in your report how "
            "many rounds it took and whether the condition was met.\n"
            "Say one line in chat as each round starts (\"Round 2 of 3: …what you are "
            "changing\").")
        for lp in flow.loops:
            cond = f'finished when: "{lp.condition}"' if lp.condition \
                else "finished when the QA briefings on its stages pass"
            if lp.review_id:
                cond = (f'finished when: "{lp.condition}"' if lp.condition
                        else "finished when the user approves")
                lines.append(
                    f"- LOOP (break node {lp.break_id}"
                    + (f', "{lp.title}"' if lp.title else "")
                    + f") — JUDGED BY THE USER at review hook {lp.review_id}, not by "
                      f"loop_check; {cond}; at most {lp.max_rounds} round(s). One round "
                      f"= run the body's stages, call halt_for_review({lp.review_id}) "
                      "and END the turn. Their reply decides: a change they ask for is "
                      "the next round (redo the stage with it, halt again); `continue` "
                      "or an approval ends the loop, and what they kept is what goes on "
                      "to the stage after the break. Never call loop_check for this "
                      "loop, and never judge the condition yourself.")
            else:
                lines.append(f"- LOOP (break node {lp.break_id}"
                             + (f', "{lp.title}"' if lp.title else "")
                             + f") — {cond}; at most {lp.max_rounds} round(s); "
                               f"forwards: {lp.forward}.")
            for i, hid in enumerate(lp.members, 1):
                lines.append(f"    body stage {i}: {_label(by_id.get(hid, {'hook_node_id': hid}))}")
    for problem in flow.problems:
        lines.append(f"- LOOP WIRING NOTE — tell the user: {problem}")
    return lines


def parallel_lines(flow: Flow, is_work=None) -> list[str]:
    """The branches of this run that can be worked on at the same time."""
    work = [h for h in flow.hooks if (is_work(h) if is_work else True)]
    groups = [g for g in branches(flow.hooks)
              if any(_hid(h) in g for h in work)]
    if len(groups) < 2:
        return []
    by_id = {_hid(h): h for h in flow.hooks}
    lines = [
        f"\nPARALLEL BRANCHES — this pipeline has {len(groups)} branches that share no "
        "wire, so none waits for another. Work on them AT THE SAME TIME, as the lead of "
        "one conversation per branch:\n"
        "  1. Write the shared notes once (look, models, naming, the QA criteria that "
        "apply to all) and call start_shot(name, briefing) once per branch, all in the "
        "same step. A branch's briefing is self-contained: its stages in order with "
        "their directives, the input files, its loop (condition, rounds) and QA "
        "criteria if it has them, and what to report back — the workflow file of every "
        "stage and the output file(s) it forwards.\n"
        "  2. The branch conversations build and run; they do NOT touch the canvas. "
        "You are woken when they report.\n"
        "  3. When a branch reports, put each of its workflows into the canvas "
        "yourself with insert_workflow_into_canvas(workflow_path, hook_node_id) and "
        "name the forwarded outputs with forward_outputs. When every branch has "
        "reported, give the user one summary.\n"
        "A branch with a single cheap stage (a text hook, one parameter) is not worth "
        "a conversation: do those yourself while the others run."]
    for n, group in enumerate(groups, 1):
        names = "; ".join(_label(by_id[i]) for i in group if i in by_id)
        lines.append(f"- branch {n}: {names}")
    return lines
