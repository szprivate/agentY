"""Hold the researcher's template pick to what the request asks for - by the template's graph.

Three small seams the pipeline calls (all thin over :mod:`agenty_core.utils.template_fit`
and :mod:`agenty_core.tools.local_templates`):

* :func:`annotate_catalog` - the catalog the researcher reads gets a marker on each
  template that cannot do the job ("NOT FOR THIS REQUEST: it draws a storyboard
  first (8 panels) ...") or only needs a setting changed ("adjust: resolution is
  720p, the request asks 480p");
* :func:`reject_reason` - a pick that is structurally wrong is handed back to the
  researcher with the reason (bounded retries, then build_new), unless the user
  named the template themselves;
* :func:`saved_jobs_block` - the machine-local templates that match the request,
  offered first.

Nothing here raises: a template that cannot be read is simply not judged.
"""
from __future__ import annotations

import re

_LEAF = re.compile(r"^(?P<lead>\s+- )(?P<name>[A-Za-z0-9_.\-]+): (?P<hint>.*)$")


def _short(text: str, n: int = 150) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def annotate_catalog(block: str, request: str) -> str:
    """*block* with ``[NOT FOR THIS REQUEST: ...]`` / ``[adjust: ...]`` after leaf lines that mismatch."""
    if not block or not str(request or "").strip():
        return block
    try:
        from agenty_core.utils import template_fit
    except Exception:  # noqa: BLE001
        return block
    out = []
    for line in block.split("\n"):
        m = _LEAF.match(line)
        if m:
            try:
                verdict = template_fit.check_template(request, m.group("name"))
            except Exception:  # noqa: BLE001
                verdict = None
            if verdict and verdict["verdict"] == "reject":
                line += f" [NOT FOR THIS REQUEST: {_short(template_fit.one_line(verdict))}]"
            elif verdict and verdict["verdict"] == "demote":
                line += f" [adjust: {_short(template_fit.one_line(verdict), 110)}]"
        out.append(line)
    return "\n".join(out)


def reject_reason(name: str, request: str, user_named: bool = False) -> str:
    """Why template *name* is wrong for *request* (``""`` when it is not, or the user chose it)."""
    if user_named or not name or str(name).lower() in ("build_new", "none"):
        return ""
    try:
        from agenty_core.utils import template_fit
        verdict = template_fit.check_template(request, name)
    except Exception:  # noqa: BLE001
        return ""
    if verdict and verdict["verdict"] == "reject":
        return template_fit.one_line(verdict)
    return ""


def adjustments(name: str, request: str) -> list:
    """What to set on the picked template so it matches the request (resolution, audio, ...)."""
    try:
        from agenty_core.utils import template_fit
        verdict = template_fit.check_template(request, name)
    except Exception:  # noqa: BLE001
        return []
    return list((verdict or {}).get("adjust") or [])


def names_template(request: str, name: str) -> bool:
    """Did the user write the template's name out, as a whole name?"""
    stem = name[:-5] if str(name).endswith(".json") else str(name)
    return len(stem) >= 8 and bool(
        re.search(r"(?<![\w.-])" + re.escape(stem.lower()) + r"(?![\w-])", str(request or "").lower()))


def saved_jobs_block(request: str) -> str:
    """The prompt section naming matching machine-local templates, or ""."""
    try:
        from agenty_core.tools import local_templates
        from src.pipeline import Pipeline
        # an explicit "local" ask excludes every partner-API job, as it does for templates
        return local_templates.catalog_block(request, local_only=bool(
            Pipeline._LOCAL_RE.search(Pipeline._norm_request(request))))
    except Exception:  # noqa: BLE001
        return ""
