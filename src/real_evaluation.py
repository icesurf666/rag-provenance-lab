"""Validation and statistics for a preregistered, manually adjudicated real corpus."""
from __future__ import annotations

import math
import random
from collections import Counter

REQUIRED_DOCUMENT_KEYS = {"id", "url", "canonical_url", "archive_url", "retrieved_at", "content_hash", "published_date", "role", "root_id", "stance", "claim_relation", "dependency_evidence", "annotations", "adjudication"}
VALID_ROLES = {"primary", "dependent", "independent", "debunk"}
VALID_STANCES = {"supports", "refutes", "unknown"}
DEPENDENCY_EVIDENCE = {"primary_record", "explicit_citation", "syndication_record", "editor_confirmation"}
CLAIM_RELATIONS = {"supports", "refutes", "irrelevant"}


def validate_real_corpus(corpus: dict) -> list[str]:
    """Return all validation errors; an empty list is the publication gate."""
    errors = []
    if not corpus.get("items"):
        return ["real corpus contains no admissible items"]
    seen_items = set()
    for item in corpus.get("items", []):
        item_id = item.get("id")
        if not item_id or item_id in seen_items:
            errors.append(f"invalid or duplicate item id: {item_id}")
        seen_items.add(item_id)
        docs = item.get("documents", [])
        if item.get("expected_stance") not in VALID_STANCES:
            errors.append(f"{item_id}: invalid expected stance")
        if not any(doc.get("role") == "primary" for doc in docs):
            errors.append(f"{item_id}: no primary document")
        if not any(doc.get("role") == "dependent" for doc in docs):
            errors.append(f"{item_id}: no documented dependent document")
        for doc in docs:
            missing = REQUIRED_DOCUMENT_KEYS - set(doc)
            if missing:
                errors.append(f"{item_id}/{doc.get('id', '?')}: missing {sorted(missing)}")
                continue
            if doc["role"] not in VALID_ROLES or doc["stance"] not in VALID_STANCES:
                errors.append(f"{item_id}/{doc['id']}: invalid role or stance")
            if doc["claim_relation"] not in CLAIM_RELATIONS:
                errors.append(f"{item_id}/{doc['id']}: invalid claim relation")
            if doc["dependency_evidence"] not in DEPENDENCY_EVIDENCE:
                errors.append(f"{item_id}/{doc['id']}: dependency lacks admissible evidence")
            if doc["role"] == "dependent" and doc["dependency_evidence"] == "primary_record":
                errors.append(f"{item_id}/{doc['id']}: dependent document needs explicit dependency evidence")
            if not all(isinstance(doc[key], str) and doc[key].strip() for key in ("url", "canonical_url", "archive_url", "retrieved_at", "content_hash", "published_date")):
                errors.append(f"{item_id}/{doc['id']}: incomplete source snapshot")
            annotations = doc["annotations"]
            annotators = {annotation.get("annotator_id") for annotation in annotations if annotation.get("annotator_id")}
            if len(annotations) < 2 or len(annotators) < 2:
                errors.append(f"{item_id}/{doc['id']}: requires two independent annotations")
            adjudication = doc["adjudication"]
            required_adjudication = {"adjudicator_id", "role", "stance", "root_id", "claim_relation", "resolution_note"}
            if not isinstance(adjudication, dict) or required_adjudication - set(adjudication):
                errors.append(f"{item_id}/{doc['id']}: incomplete adjudication")
            elif any(adjudication[key] != doc[key] for key in ("role", "stance", "root_id", "claim_relation")):
                errors.append(f"{item_id}/{doc['id']}: adjudication does not match final label")
    return errors


def paired_summary(rows: list[dict], baseline: str, intervention: str, metric: str = "correct") -> dict:
    """Pair by item ID; no independent-sample test on repeated model outputs."""
    pairs = {}
    for row in rows:
        if row.get("strategy") in {baseline, intervention}:
            pairs.setdefault(row["task"], {})[row["strategy"]] = bool(row.get(metric))
    complete = [value for value in pairs.values() if baseline in value and intervention in value]
    improvements = sum(not pair[baseline] and pair[intervention] for pair in complete)
    regressions = sum(pair[baseline] and not pair[intervention] for pair in complete)
    return {"n": len(complete), "baseline_rate": sum(p[baseline] for p in complete) / len(complete) if complete else 0, "intervention_rate": sum(p[intervention] for p in complete) / len(complete) if complete else 0, "improvements": improvements, "regressions": regressions}


def bootstrap_delta_ci(rows: list[dict], baseline: str, intervention: str, repetitions: int = 10_000, seed: int = 20261101) -> tuple[float, float]:
    paired = {}
    for row in rows:
        if row.get("strategy") in {baseline, intervention}:
            paired.setdefault(row["task"], {})[row["strategy"]] = int(bool(row.get("correct")))
    values = [pair[intervention] - pair[baseline] for pair in paired.values() if baseline in pair and intervention in pair]
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions))
    return estimates[int(.025 * repetitions)], estimates[int(.975 * repetitions) - 1]


def exact_sign_test_pvalue(improvements: int, regressions: int) -> float:
    """Two-sided exact binomial test for paired discordant outcomes."""
    n = improvements + regressions
    if n == 0:
        return 1.0
    k = min(improvements, regressions)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def citation_entailment(answer_stance: str, citation_ids: list[str], documents: list[dict]) -> dict:
    """Score whether cited source labels actually entail a non-unknown answer."""
    by_id = {document["id"]: document for document in documents}
    cited = [by_id[citation_id] for citation_id in citation_ids if citation_id in by_id]
    if answer_stance == "unknown":
        return {"scorable": False, "citation_precision": None, "primary_cited": any(doc["role"] == "primary" for doc in cited)}
    matching = [doc for doc in cited if doc.get("claim_relation") == answer_stance]
    return {
        "scorable": True,
        "citation_precision": len(matching) / len(cited) if cited else 0.0,
        "primary_cited": any(doc["role"] == "primary" and doc.get("claim_relation") == answer_stance for doc in cited),
    }
