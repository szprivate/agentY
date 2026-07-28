"""Lift per-role model pins in settings.local.json into the new model tiers.

Before tiers, every role had to be pinned individually, so a typical
``settings.local.json`` carries a dozen ``llm.pipeline.*`` entries that mostly say
the same thing several times over. Those entries still work — an explicit role
always beats its tier — but they keep the tier selectors inert, so the settings UI
shows six tier dropdowns that appear to do nothing.

This lifts the repetition up one level: for each tier, the most common value among
its roles becomes the tier, and every role already holding that value is cleared to
"inherit". Roles that genuinely differ (a code-specialist model for ``coder``, a
stronger judge for ``qa_checker``) keep their explicit pin.

The effective model for every role is unchanged — this is a rewrite of how the same
answer is stored, not a change to any answer. Run with ``--dry-run`` first; a
timestamped ``.bak`` is written before anything is saved.

    python scripts/migrate_model_tiers.py --dry-run
    python scripts/migrate_model_tiers.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import _ROLE_TIERS  # noqa: E402


def plan(local: dict) -> tuple[dict, dict, list[str]]:
    """Return ``(tiers, remaining_overrides, notes)`` for *local*'s pipeline block."""
    pipeline = dict(((local.get("llm") or {}).get("pipeline") or {}))
    existing_tiers = dict(((local.get("llm") or {}).get("tiers") or {}))
    notes: list[str] = []

    by_tier: dict[str, list[tuple[str, str]]] = {}
    for role, value in pipeline.items():
        value = str(value or "").strip()
        if not value:
            continue
        tier = _ROLE_TIERS.get(role)
        if tier is None:
            notes.append(f"{role}: not a known role — left as an override")
            continue
        by_tier.setdefault(tier, []).append((role, value))

    tiers = dict(existing_tiers)
    keep: dict[str, str] = {r: str(v).strip() for r, v in pipeline.items() if str(v or "").strip()}
    for tier, entries in sorted(by_tier.items()):
        if existing_tiers.get(tier):
            notes.append(f"{tier}: already set to {existing_tiers[tier]!r} — left alone")
            continue
        winner, count = Counter(v for _r, v in entries).most_common(1)[0]
        tiers[tier] = winner
        lifted = [r for r, v in entries if v == winner]
        for role in lifted:
            keep.pop(role, None)
        kept = [f"{r}={v}" for r, v in entries if v != winner]
        notes.append(
            f"{tier} = {winner!r}  (from {count}/{len(entries)} role(s): {', '.join(sorted(lifted))})"
            + (f"; keeping override {', '.join(sorted(kept))}" if kept else "")
        )
    return tiers, keep, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show the change, write nothing")
    ap.add_argument("--path", default="", help="settings.local.json to migrate")
    args = ap.parse_args()

    from src.utils.settings import local_path

    path = Path(args.path) if args.path else local_path()
    if not path.is_file():
        print(f"nothing to migrate — {path} does not exist")
        return 0
    local = json.loads(path.read_text(encoding="utf-8"))

    tiers, keep, notes = plan(local)
    if not notes:
        print("nothing to lift — no per-role model pins found")
        return 0

    print(f"{path}\n")
    for note in notes:
        print("  " + note)
    before = dict(((local.get("llm") or {}).get("pipeline") or {}))
    dropped = sorted(set(before) - set(keep))
    print(f"\n  tiers set     : {len(tiers)}")
    print(f"  overrides kept: {len(keep)}" + (f" ({', '.join(sorted(keep))})" if keep else ""))
    print(f"  overrides now inheriting: {len(dropped)}" + (f" ({', '.join(dropped)})" if dropped else ""))

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    backup = path.with_suffix(f".json.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(path, backup)
    llm = local.setdefault("llm", {})
    llm["tiers"] = tiers
    llm["pipeline"] = keep
    path.write_text(json.dumps(local, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwritten. backup: {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
