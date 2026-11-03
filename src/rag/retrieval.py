"""Retrieval policies. Each function has one explicit intervention boundary."""

from __future__ import annotations

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _scored_documents(question: str, documents: list[dict]) -> list[dict]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([question, *[doc["text"] for doc in documents]])
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    return [
        {**doc, "score": round(float(score), 6)}
        for doc, score in zip(documents, scores)
    ]


def retrieve(question: str, documents: list[dict], k: int = 5) -> list[dict]:
    return sorted(
        _scored_documents(question, documents),
        key=lambda doc: (-doc["score"], doc["id"]),
    )[:k]


def root_aware_retrieve(
    question: str,
    documents: list[dict],
    k: int = 5,
    *,
    use_reliability: bool = False,
    expose_conflict: bool = False,
) -> list[dict]:
    """Collapse roots before top-k; metadata switches remain visible ablations."""
    roots: dict[str, dict] = {}
    root_stances: dict[str, set[str]] = defaultdict(set)
    for doc in _scored_documents(question, documents):
        root_stances[doc["root_id"]].add(doc["stance"])
        weighted = (
            doc["score"] * doc["reliability"] if use_reliability else doc["score"]
        )
        candidate = {**doc, "provenance_score": weighted}
        current = roots.get(doc["root_id"])
        if (
            current is None
            or candidate["provenance_score"] > current["provenance_score"]
        ):
            roots[doc["root_id"]] = candidate
    collapsed = []
    for root, doc in roots.items():
        conflict = expose_conflict and len(root_stances[root]) > 1
        collapsed.append(
            {
                **doc,
                "root_conflict": conflict,
                "stance": "unknown" if conflict else doc["stance"],
            }
        )
    return sorted(collapsed, key=lambda doc: (-doc["provenance_score"], doc["id"]))[:k]


def provenance_retrieve(question: str, documents: list[dict], k: int = 5) -> list[dict]:
    """Compatibility name for the metadata-assisted pilot policy."""
    return root_aware_retrieve(
        question, documents, k, use_reliability=True, expose_conflict=True
    )


def post_topk_collapse(question: str, documents: list[dict], k: int = 5) -> list[dict]:
    """Collapse only the lexical top-k, preserving V0's variable context budget."""
    selected: dict[str, dict] = {}
    for document in retrieve(question, documents, k):
        selected.setdefault(document["root_id"], document)
    return list(selected.values())


def _observed_polarity(document: dict) -> str:
    """Conservative text-only polarity feature for synthetic conflict preservation."""
    text = document["text"].lower()
    refuting_markers = ("does not support", "not implemented", "is a joke", "satire")
    if any(marker in text for marker in refuting_markers):
        return "refutes"
    return "supports" if "support" in text else "unknown"


def _fill_by_group(
    question: str, documents: list[dict], k: int, group_key
) -> list[dict]:
    ranked = sorted(
        _scored_documents(question, documents),
        key=lambda doc: (-doc["score"], doc["id"]),
    )
    selected, seen_groups, selected_ids = [], set(), set()
    for document in ranked:
        key = group_key(document)
        if key not in seen_groups:
            selected.append(document)
            seen_groups.add(key)
            selected_ids.add(document["id"])
            if len(selected) == k:
                return selected
    for document in ranked:
        if document["id"] not in selected_ids:
            selected.append(document)
            if len(selected) == k:
                break
    return selected


def root_aware_fill(question: str, documents: list[dict], k: int = 5) -> list[dict]:
    """Use one document per root first, then fill the fixed context budget."""
    return _fill_by_group(question, documents, k, lambda document: document["root_id"])


def conflict_preserving_fill(
    question: str, documents: list[dict], k: int = 5
) -> list[dict]:
    """Reserve separate slots for text-observed opposing claims within one root."""
    return _fill_by_group(
        question,
        documents,
        k,
        lambda document: (document["root_id"], _observed_polarity(document)),
    )


def ablation_strategies(
    question: str, documents: list[dict], k: int = 5
) -> dict[str, list[dict]]:
    return {
        "lexical_topk": retrieve(question, documents, k),
        "root_collapse": root_aware_retrieve(question, documents, k),
        "source_weighted": root_aware_retrieve(
            question, documents, k, use_reliability=True
        ),
        "metadata_assisted": provenance_retrieve(question, documents, k),
    }


def v1_retrieval_strategies(
    question: str, documents: list[dict], k: int = 5
) -> dict[str, list[dict]]:
    """Fixed-budget, conflict-preserving retrieval controls for the V1 study."""
    return {
        "lexical_topk": retrieve(question, documents, k),
        "post_topk_collapse": post_topk_collapse(question, documents, k),
        "root_aware_fill": root_aware_fill(question, documents, k),
        "conflict_preserving_fill": conflict_preserving_fill(question, documents, k),
    }


def answer(selected: list[dict], provenance_aware: bool) -> str:
    if not selected:
        return "unknown"
    weights = defaultdict(float)
    for doc in selected:
        weights[doc["stance"]] += doc["reliability"] if provenance_aware else 1.0
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return (
        "unknown" if len(ranked) > 1 and ranked[0][1] == ranked[1][1] else ranked[0][0]
    )
