"""Fit the ranking weights to the reviews you have actually answered.

    .venv/Scripts/python.exe scripts/fit_fitness_weights.py            # report only
    .venv/Scripts/python.exe scripts/fit_fitness_weights.py --write    # install if better
    .venv/Scripts/python.exe scripts/fit_fitness_weights.py --force    # install regardless

Every time a `review` hook is answered, agentY writes down which outputs you kept
and which you deleted, with each one's measured features
(:mod:`src.utils.preference_log`). This reads that log, fits the weights in
:mod:`src.utils.fitness` to it, and reports whether the result is any good.

**It will not install weights that lose to the hand-set ones.** A slice of the
labels is held out, both weight vectors are scored on data neither has seen, and
``config/fitness_weights.json`` is written only if the fit wins by a real margin.
A learned model quietly worse than the guess it replaced is the characteristic
failure of this kind of work, and measuring is the only defence.

Nothing here is automatic. It runs when you run it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.fitness import DEFAULT_WEIGHTS, FEATURES, WEIGHTS_FILE  # noqa: E402
from src.utils.fitness_fit import (MIN_MARGIN, MIN_SLATES, active_keys,  # noqa: E402
                                   evaluate, fit, split)
from src.utils.preference_log import (LOG_PATH, read_events, slates,  # noqa: E402
                                      summary)


def _table(name_a: str, a: dict, name_b: str, b: dict) -> str:
    rows = [f"{'':<14}{name_a:>14}{name_b:>14}"]
    for key, label in (("pair_accuracy", "pair accuracy"),
                       ("top1_accuracy", "top-1 accuracy")):
        rows.append(f"{label:<14}{a[key]:>14.3f}{b[key]:>14.3f}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=None,
                    help=f"preference log to read (default: {LOG_PATH})")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"where to write the weights (default: {WEIGHTS_FILE})")
    ap.add_argument("--write", action="store_true",
                    help="install the fitted weights IF they beat the defaults")
    ap.add_argument("--force", action="store_true",
                    help="install them even if they do not. For experiments only.")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of reviews kept back for the comparison (default 0.3)")
    args = ap.parse_args()

    events = read_events(args.log)
    rows = slates(events)
    print(f"log:      {args.log or LOG_PATH}")
    print(f"labels:   {summary(args.log)}")
    if not rows:
        print("\nNothing to fit yet. Answer a review hook with some outputs removed "
              "and the labels will appear here.")
        return 0

    keys = active_keys(rows)
    print(f"features: {', '.join(keys) if keys else '(none vary — nothing to learn)'}")
    unused = [k for k in FEATURES if k not in keys]
    if unused:
        print(f"          keeping the default for: {', '.join(unused)}")
    if not keys:
        return 0

    train, test = split(rows, holdout=args.holdout)
    print(f"split:    {len(train)} to fit, {len(test)} held back\n")

    learned = fit(train, keys=keys)
    on = test or train
    where = "held-out reviews" if test else "the training reviews (too few to hold any back)"
    base_eval, new_eval = evaluate(on, DEFAULT_WEIGHTS), evaluate(on, learned)
    print(f"Measured on {len(on)} {where}:")
    print(_table("hand-set", base_eval, "fitted", new_eval))

    print("\nweights:")
    for k in FEATURES:
        a, b = DEFAULT_WEIGHTS.get(k, 0.0), learned.get(k, 0.0)
        mark = "" if abs(a - b) < 1e-9 else "   <-- moved"
        print(f"  {k:<12} {a:>7.3f}  ->  {b:>7.3f}{mark}")

    gain = new_eval["pair_accuracy"] - base_eval["pair_accuracy"]
    enough = len(rows) >= MIN_SLATES
    better = gain > MIN_MARGIN
    print()
    if not enough:
        print(f"NOT installing: {len(rows)} reviews is under the {MIN_SLATES} this "
              "asks for. The hand-set weights stay until there is real evidence.")
    elif not better:
        print(f"NOT installing: the fit gains {gain:+.3f} on held-out pair accuracy, "
              f"which is not more than the {MIN_MARGIN} margin. The defaults are "
              "still the better bet.")
    else:
        print(f"The fit gains {gain:+.3f} on held-out pair accuracy.")

    install = args.force or (args.write and enough and better)
    if not install:
        if args.write and not (enough and better):
            print("(--force would install them anyway.)")
        elif not args.write:
            print("Report only. Pass --write to install them when they qualify.")
        return 0

    target = args.out or WEIGHTS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "weights": {k: learned.get(k, DEFAULT_WEIGHTS.get(k, 0.0)) for k in FEATURES},
        "fitted_from": {"reviews": len(rows), "features": keys,
                        "holdout": args.holdout, "forced": bool(args.force)},
        "held_out": {"hand_set": base_eval, "fitted": new_eval},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {target}. src/utils/fitness.py loads it from now on; "
          "delete the file to go back to the hand-set weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
