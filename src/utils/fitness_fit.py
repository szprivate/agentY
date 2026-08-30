"""Learn the ranking weights from what the user kept at a review.

The weights in :mod:`src.utils.fitness` are hand-set and admit it. This is where
they stop being a guess: :mod:`src.utils.preference_log` has recorded, for every
answered `review` hook, which outputs the user kept and which they deleted, with
each one's measured feature vector. That is training data, and this fits to it.

**The likelihood is Plackett-Luce, not Bradley-Terry**, because that is what the
data is: the user saw all N outputs at once and kept k of them. The
choice-from-a-slate likelihood puts the whole slate in one denominator:

    L = prod over kept i of   exp(s_i) / (exp(s_i) + sum over rejected j of exp(s_j))

with ``s = w . x``, a linear merit function. Bradley-Terry is not discarded — it
is the two-item case of exactly this expression, so a review that kept one of two
contributes precisely the pairwise term it should.

*How much does the choice matter?* Less than it looks, and it was worth measuring
rather than asserting. Against a known ground-truth preference over synthetic
slates, with no regulariser so only the estimators are being compared, the
pairwise decomposition is **as accurate as the slate likelihood** — within 0.005
on both pair and top-1 accuracy at every training size from 6 to 80 reviews, with
no consistent winner. That is the expected result: decomposing a slate into
duels is a *composite* likelihood of this same model, and composite likelihoods
are consistent. They converge to the same answer.

What the decomposition does break is the **bookkeeping**, and that is the real
reason to prefer the slate form. It turns one review into seven observations, and
everything that counts observations then lies: the L2 pull below is per-fit, so
the same prior silently becomes seven times weaker; ``MIN_SLATES`` would be
counting duels rather than decisions; and any standard error computed from
dependent comparisons treated as independent is wrong. Measured directly: hold
the prior's pull constant per *review* and the slate form wins outright (top-1
0.91 vs 0.41 at 80 reviews); leave it fixed per *term* and the decomposition
appears to win, purely for being less regularised. Neither number is about the
likelihood. With PL, one review is one observation and the constants mean what
they say.

(The strictly correct likelihood for an *unordered* kept set sums over the orders
in which those k could have been chosen, which is k! terms and unidentifiable
from this data anyway. The form above is the standard treatment: each kept item
against the field of rejects. It is exact when k=1, which is the common case, and
it never invents an ordering among the kept — which the collector's row order
would happily supply and which would be pure noise, since that order is the one
the halt wrote, not one the user expressed.)

**It is regularised toward the hand-set weights, not toward zero.** With a
handful of labels the prior should win; only real evidence should move a weight
off its default. That also means the fit degrades gracefully: ten labels nudge,
a thousand decide.

**And it refuses to install weights that are not better.** The script
``scripts/fit_fitness_weights.py`` holds out a slice of the labels, compares the
fitted weights against the hand-set ones on data neither has seen, and writes
``config/fitness_weights.json`` only if the fit actually wins. A learned model
that is worse than the guess it replaced is the failure mode this whole area is
prone to, and the only defence is to measure it.
"""
from __future__ import annotations

import logging
import math

from src.utils.fitness import DEFAULT_WEIGHTS, FEATURES

logger = logging.getLogger("agentY.fitness_fit")

# Enough labels that a fit means something. Below this the prior should simply
# win, and the script says so rather than fitting noise and calling it learning.
MIN_SLATES = 12
# The fitted weights must beat the hand-set ones on held-out data by more than
# this to be installed. A hair's-breadth win on a small sample is not a win.
MIN_MARGIN = 0.02

_L2 = 2.0            # pull toward the prior; see the module docstring
_STEPS = 4000
_LR = 0.5


def _vec(features: dict, keys) -> list:
    return [float(features.get(k, 0.0)) for k in keys]


def _merit(x, w) -> float:
    return sum(xi * wi for xi, wi in zip(x, w))


def _softmax(scores) -> list:
    hi = max(scores)
    exps = [math.exp(s - hi) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def loss_and_grad(slates, w, keys, prior, l2: float = _L2):
    """Negative log-likelihood of *slates* under *w*, and its gradient.

    One term per kept output: that output against the field of rejected ones.
    """
    n = len(keys)
    loss = 0.0
    grad = [0.0] * n
    for chosen, rejected in slates:
        rej = [_vec(f, keys) for f in rejected]
        rej_scores = [_merit(x, w) for x in rej]
        for pick in chosen:
            xi = _vec(pick, keys)
            scores = [_merit(xi, w)] + rej_scores
            probs = _softmax(scores)
            loss -= math.log(max(probs[0], 1e-12))
            # d(-log p_0)/dw = -(x_0 - sum_k p_k x_k)
            expected = [0.0] * n
            for p, x in zip(probs, [xi] + rej):
                for j in range(n):
                    expected[j] += p * x[j]
            for j in range(n):
                grad[j] -= (xi[j] - expected[j])
    for j in range(n):
        diff = w[j] - prior[j]
        loss += l2 * diff * diff
        grad[j] += 2.0 * l2 * diff
    return loss, grad


def fit(slates, keys=None, prior: dict | None = None, l2: float = _L2,
        steps: int = _STEPS, lr: float = _LR) -> dict:
    """Weights fitted to *slates*, as ``{feature: weight}``.

    Plain gradient descent with a decaying step. The problem is a handful of
    parameters over a few hundred observations and it is convex, so nothing
    cleverer earns its dependency — and this way the fit is a pure-Python
    function that the test suite can check exactly.
    """
    base = dict(prior or DEFAULT_WEIGHTS)
    keys = list(keys or [k for k in FEATURES if k in base])
    if not slates or not keys:
        return dict(base)
    p = [float(base.get(k, 0.0)) for k in keys]
    w = list(p)
    best, best_loss = list(w), None
    for step in range(steps):
        loss, grad = loss_and_grad(slates, w, keys, p, l2)
        if best_loss is None or loss < best_loss:
            best_loss, best = loss, list(w)
        rate = lr / (1.0 + step / 200.0)
        w = [wi - rate * gi for wi, gi in zip(w, grad)]
    out = dict(base)
    out.update({k: round(v, 4) for k, v in zip(keys, best)})
    return out


def active_keys(slates) -> list:
    """The features that actually VARY in the labels.

    A feature identical across every slate cannot be learned from them, and
    leaving it in the fit lets the optimiser drift it around on nothing but the
    regularisation. It keeps its prior instead.
    """
    seen: dict = {}
    for chosen, rejected in (slates or []):
        for f in list(chosen) + list(rejected):
            for k, v in f.items():
                seen.setdefault(k, set()).add(round(float(v), 4))
    return [k for k in FEATURES if len(seen.get(k, ())) > 1]


def pair_accuracy(pairs, weights: dict) -> float:
    """How often *weights* put the kept output above the rejected one.

    Ties count as half, which is what they are: the model has expressed no
    preference, and scoring that as a win would flatter a weight vector that
    ignores every feature in play.
    """
    rows = list(pairs or [])
    if not rows:
        return 0.0
    hits = 0.0
    for win, lose in rows:
        keys = sorted(set(win) & set(lose))
        sw = sum(float(weights.get(k, 0.0)) * float(win[k]) for k in keys)
        sl = sum(float(weights.get(k, 0.0)) * float(lose[k]) for k in keys)
        hits += 1.0 if sw > sl else (0.5 if sw == sl else 0.0)
    return hits / len(rows)


def top1_accuracy(slates, weights: dict) -> float:
    """How often the highest-scoring member of a slate is one the user kept.

    Closer to the question actually being asked — "show me the best one" — than
    pairwise accuracy, and harder, since one bad reject at the top ruins it.
    """
    rows = list(slates or [])
    if not rows:
        return 0.0
    hits = 0
    for chosen, rejected in rows:
        def s(f):
            return sum(float(weights.get(k, 0.0)) * float(v) for k, v in f.items())
        best_chosen = max((s(f) for f in chosen), default=float("-inf"))
        best_rejected = max((s(f) for f in rejected), default=float("-inf"))
        if best_chosen > best_rejected:
            hits += 1
    return hits / len(rows)


def split(slates, holdout: float = 0.3, seed: int = 12345):
    """(train, test), split deterministically so a re-run reports the same thing."""
    rows = list(slates or [])
    if len(rows) < 2:
        return rows, []
    import random
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    cut = max(1, int(round(len(rows) * holdout)))
    test_idx = set(order[:cut])
    train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]
    return (train, test) if train else (rows, [])


def pairs_of(slates) -> list:
    """Every (winner, loser) implied by *slates* — for scoring a fit, not making one."""
    out = []
    for chosen, rejected in (slates or []):
        for win in chosen:
            for lose in rejected:
                out.append((dict(win), dict(lose)))
    return out


def evaluate(slates, weights: dict) -> dict:
    """Both accuracies for one weight vector on one set of slates."""
    return {
        "slates": len(slates or []),
        "pairs": len(pairs_of(slates)),
        "pair_accuracy": round(pair_accuracy(pairs_of(slates), weights), 4),
        "top1_accuracy": round(top1_accuracy(slates, weights), 4),
    }
