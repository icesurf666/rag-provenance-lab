"""Non-negotiable V2 admission and paired-budget checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED_CASE = {"id", "question", "documents", "adjudication"}
REQUIRED_DOCUMENT = {
    "id", "url", "canonical_url", "retrieved_at", "content_hash", "role",
    "root_id", "claim_relation", "dependency_evidence", "annotations",
}
FORBIDDEN_PROMPT_FIELDS = {"expected_stance", "warranted_answer", "factual_truth", "adjudication"}


def case_errors(case: dict) -> list[str]:
    errors = []
    missing = REQUIRED_CASE - set(case)
    if missing:
        return [f"{case.get('id', '?')}: missing case fields {sorted(missing)}"]
    for document in case["documents"]:
        absent = REQUIRED_DOCUMENT - set(document)
        if absent:
            errors.append(f"{case['id']}/{document.get('id', '?')}: missing {sorted(absent)}")
        if len({item.get('annotator_id') for item in document.get('annotations', [])}) < 2:
            errors.append(f"{case['id']}/{document.get('id', '?')}: needs two independent annotations")
        if not document.get("dependency_evidence"):
            errors.append(f"{case['id']}/{document.get('id', '?')}: missing dependency evidence")
    return errors


def no_leakage(prompt_payload: dict) -> list[str]:
    exposed = FORBIDDEN_PROMPT_FIELDS & set(prompt_payload)
    return [f"prompt contains forbidden fields: {sorted(exposed)}"] if exposed else []


def paired_budget_errors(rows: list[dict]) -> list[str]:
    by_key: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        key = row["case_id"], row["model"], row["order_seed"]
        by_key.setdefault(key, {})[row["condition"]] = row
    errors = []
    for key, pair in by_key.items():
        baseline, intervention = pair.get("primary_only"), pair.get("root_aware")
        if not baseline or not intervention:
            errors.append(f"{key}: missing paired conditions")
            continue
        for field in ("document_budget", "prompt_eval_count"):
            if baseline.get(field) != intervention.get(field):
                errors.append(f"{key}: unequal {field}")
    return errors


def manifest_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_file(path: Path) -> list[str]:
    corpus = json.loads(path.read_text())
    return [error for case in corpus.get("items", []) for error in case_errors(case)]
