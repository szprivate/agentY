"""Phase 3 - cluster workflows into types by fingerprint similarity.

Similarity is a weighted sum of per-signal Jaccard overlaps (node classes,
typed connection patterns, local-cluster signatures, spine roles). Weights come
from config so any signal can be re-weighted or dropped.

Clustering is threshold-based agglomerative with *average linkage* and needs no
pre-specified cluster count: every workflow starts in its own cluster, and the
two most-similar clusters are merged while their average pairwise similarity is
at least the threshold. Average linkage (rather than single linkage) avoids
"chaining" distinct types together through one borderline pair.

Determinism: inputs are processed in sorted order and ties are broken by the
lexicographically smallest member-index tuple, so the same inputs and threshold
always yield the same clusters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from .fingerprint import Fingerprint


def jaccard(a: FrozenSet, b: FrozenSet) -> float:
    """Jaccard overlap of two sets; two empty sets are treated as identical (1.0)."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def similarity_breakdown(
    fa: Fingerprint, fb: Fingerprint, weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """Return (combined_similarity, per-signal Jaccard) for two fingerprints.

    The combined score is the weighted average of per-signal Jaccard values,
    normalized by the sum of the weights of the signals actually in play.
    """
    sets_a = fa.signal_sets()
    sets_b = fb.signal_sets()
    per_signal: Dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for signal, weight in weights.items():
        if weight <= 0 or signal not in sets_a:
            continue
        sa, sb = sets_a[signal], sets_b[signal]
        # The catalog category is "unknown" (empty) for custom workflows. When it
        # is unknown on either side the signal is undefined for this pair, so it
        # is excluded from the weighted average (neutral) rather than scored 0 or
        # treated as a match. Structural signals are never empty for real graphs.
        if signal == "category" and (not sa or not sb):
            continue
        j = jaccard(sa, sb)
        per_signal[signal] = j
        weighted_sum += weight * j
        weight_total += weight
    combined = weighted_sum / weight_total if weight_total else 0.0
    return combined, per_signal


def pairwise_matrix(
    fps: List[Fingerprint], weights: Dict[str, float]
) -> Dict[Tuple[int, int], Tuple[float, Dict[str, float]]]:
    """Compute the upper-triangular similarity matrix keyed by (i, j), i < j."""
    matrix: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            matrix[(i, j)] = similarity_breakdown(fps[i], fps[j], weights)
    return matrix


# --------------------------------------------------------------------------- #
# Description-based similarity (TF-IDF cosine over catalog descriptions)
# --------------------------------------------------------------------------- #
# Generic words that do not help distinguish workflow intent. Domain words that
# DO discriminate (image, video, audio, model names) are deliberately kept.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "from", "with", "and", "or", "via", "for",
    "in", "on", "by", "up", "is", "are", "it", "its", "this", "that", "as",
    "at", "into", "using", "use", "uses", "used", "plus", "per", "while",
    "also", "than", "then", "your", "you", "blueprint", "subgraph", "output",
    "outputs", "input", "inputs", "node", "nodes", "takes", "produces",
    "generate", "generates", "generating", "creates", "create", "creating",
    "supports", "support", "based", "local", "api", "optional", "single", "one",
}
_WORD = re.compile(r"[a-z0-9.]+")


def _tokenize(text: str) -> List[str]:
    tokens = []
    for raw in _WORD.findall((text or "").lower()):
        tok = raw.strip(".")
        if len(tok) >= 2 and tok not in _STOPWORDS and not tok.isdigit():
            tokens.append(tok)
    return tokens


def description_matrix(
    texts: List[str],
) -> Dict[Tuple[int, int], Tuple[float, Dict[str, float]]]:
    """TF-IDF cosine similarity over per-workflow description texts.

    Rare, distinctive terms (canny, wan2.2, ltx, relight) dominate the score
    while generic words contribute little, so workflows that *read* alike group
    together. Returns the same (score, per_signal) matrix shape as the
    structural path so the rest of the pipeline is unchanged."""
    docs = [_tokenize(t) for t in texts]
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    vecs: List[Tuple[Dict[str, float], float]] = []
    for d in docs:
        tf = Counter(d)
        vec = {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vecs.append((vec, norm))

    matrix: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]] = {}
    for i in range(n):
        vi, ni = vecs[i]
        for j in range(i + 1, n):
            vj, nj = vecs[j]
            small, large = (vi, vj) if len(vi) <= len(vj) else (vj, vi)
            dot = sum(val * large.get(t, 0.0) for t, val in small.items())
            cos = dot / (ni * nj)
            matrix[(i, j)] = (round(cos, 6), {"description": round(cos, 6)})
    return matrix


def shared_terms(texts: List[str], members: List[int]) -> List[str]:
    """Tokens common to all members' description texts (for the report)."""
    if not members:
        return []
    sets = [set(_tokenize(texts[m])) for m in members]
    return sorted(set.intersection(*sets)) if sets else []


def _pair_sim(matrix, i: int, j: int) -> float:
    return matrix[(i, j)][0] if i < j else matrix[(j, i)][0]


@dataclass
class Cluster:
    members: List[int]                       # fingerprint indices, sorted
    cohesion: float = 0.0                     # mean intra-cluster similarity


def agglomerate(
    fps: List[Fingerprint],
    matrix: Dict[Tuple[int, int], Tuple[float, Dict[str, float]]],
    threshold: float,
) -> List[Cluster]:
    """Average-linkage agglomerative clustering at the given similarity threshold."""
    clusters: List[List[int]] = [[i] for i in range(len(fps))]

    def avg_linkage(ca: List[int], cb: List[int]) -> float:
        total = 0.0
        for x in ca:
            for y in cb:
                total += _pair_sim(matrix, x, y)
        return total / (len(ca) * len(cb))

    while len(clusters) > 1:
        best_score = -1.0
        best_pair: Tuple[int, int] = (-1, -1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                score = avg_linkage(clusters[a], clusters[b])
                # Deterministic tie-break: prefer the pair whose merged member
                # set is lexicographically smallest.
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12
                    and best_pair != (-1, -1)
                    and sorted(clusters[a] + clusters[b])
                    < sorted(clusters[best_pair[0]] + clusters[best_pair[1]])
                ):
                    best_score = score
                    best_pair = (a, b)
        if best_score < threshold or best_pair == (-1, -1):
            break
        a, b = best_pair
        merged = sorted(clusters[a] + clusters[b])
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)]
        clusters.append(merged)

    result: List[Cluster] = []
    for members in clusters:
        members = sorted(members)
        cohesion = _mean_intra(members, matrix)
        result.append(Cluster(members=members, cohesion=cohesion))

    # Sort by size desc, then by the members' fingerprint names for stable order.
    result.sort(key=lambda c: (-len(c.members), [fps[m].name for m in c.members]))
    return result


def _mean_intra(members: List[int], matrix) -> float:
    if len(members) < 2:
        return 1.0
    total = 0.0
    count = 0
    for a_idx in range(len(members)):
        for b_idx in range(a_idx + 1, len(members)):
            total += _pair_sim(matrix, members[a_idx], members[b_idx])
            count += 1
    return total / count if count else 1.0


def shared_signals(
    fps: List[Fingerprint], members: List[int]
) -> Dict[str, List]:
    """Explain *why* members grouped: classes and connection patterns common to
    all members of the cluster. Used in the human-readable clustering report."""
    if not members:
        return {"shared_classes": [], "shared_connections": []}
    class_sets = [fps[m].class_set for m in members]
    conn_sets = [fps[m].connection_set for m in members]
    shared_classes = sorted(set.intersection(*[set(s) for s in class_sets])) if class_sets else []
    shared_conns = (
        sorted(set.intersection(*[set(s) for s in conn_sets])) if conn_sets else []
    )
    return {
        "shared_classes": shared_classes,
        "shared_connections": [list(c) for c in shared_conns],
    }
