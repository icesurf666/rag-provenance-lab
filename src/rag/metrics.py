"""Inspectable retrieval metrics and summaries for provenance-RAG ablations."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean


def task_category(task_id: str) -> str:
    """Recover the generator category without adding a redundant fixture field."""
    return task_id.rsplit("_", 1)[0]


def reference_record_recalled(documents: list[dict]) -> bool:
    """Whether context retained the primary record or the satire-context record."""
    return any(document.get("role") in {"primary", "satire"} for document in documents)


def root_concentration(documents: list[dict]) -> float:
    """Largest source-root share of selected context, from 0.0 to 1.0."""
    if not documents:
        return 0.0
    root_counts: dict[str, int] = defaultdict(int)
    for document in documents:
        root_counts[document["root_id"]] += 1
    return max(root_counts.values()) / len(documents)


def retrieval_audit(documents: list[dict]) -> dict:
    """Return serializable context diagnostics for one retrieved set."""
    return {
        "selected_ids": [document["id"] for document in documents],
        "returned_documents": len(documents),
        "approximate_context_tokens": sum(
            len(document.get("text", "").split()) for document in documents
        ),
        "unique_roots": len({document["root_id"] for document in documents}),
        "root_concentration": round(root_concentration(documents), 6),
        "reference_record_recalled": reference_record_recalled(documents),
    }


def fixture_contradiction_coverage(documents: list[dict]) -> bool | None:
    """Benchmark-only check: did a selected context retain both labelled stances?"""
    stances = {document.get("stance") for document in documents}
    return {"supports", "refutes"}.issubset(stances)


def summarize(records: list[dict]) -> list[dict]:
    """Aggregate correctness and retrieval diagnostics by split, policy, and category."""
    grouped: dict[tuple[int | None, str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record.get("context_k"),
            record["split"],
            record["strategy"],
            record["category"],
        )
        grouped[key].append(record)
    summary = []
    for (context_k, split, strategy, category), rows in sorted(grouped.items()):
        result = {
            "split": split,
            "strategy": strategy,
            "category": category,
            "n": len(rows),
            "accuracy": round(mean(row["correct"] for row in rows), 6),
            "mean_returned_documents": round(
                mean(row["returned_documents"] for row in rows), 6
            ),
            "mean_approximate_context_tokens": round(
                mean(row["approximate_context_tokens"] for row in rows), 6
            ),
            "mean_unique_roots": round(mean(row["unique_roots"] for row in rows), 6),
            "mean_root_concentration": round(
                mean(row["root_concentration"] for row in rows), 6
            ),
        }
        applicable = [row for row in rows if row["reference_record_applicable"]]
        result["reference_record_recall_applicable"] = (
            round(mean(row["reference_record_recalled"] for row in applicable), 6)
            if applicable
            else None
        )
        conflict_rows = [row for row in rows if row["conflict_applicable"]]
        result["fixture_contradiction_coverage"] = (
            round(
                mean(row["fixture_contradiction_coverage"] for row in conflict_rows), 6
            )
            if conflict_rows
            else None
        )
        if context_k is not None:
            result["context_k"] = context_k
        summary.append(result)
    return summary
