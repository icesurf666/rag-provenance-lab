"""Benchmark orchestration and serializable experiment records."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .generation import llm_answer, validate_completion
from .metrics import fixture_contradiction_coverage, retrieval_audit, task_category
from .retrieval import (
    ablation_strategies,
    answer,
    provenance_retrieve,
    retrieve,
    v1_retrieval_strategies,
)


def _load(path: Path | None) -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[2]
    path = path or root / "data/benchmarks/provenance_rag_synthetic.json"
    return root, json.loads(path.read_text())


def _strategy_cases(benchmark: dict, k: int):
    """Yield every task/policy pair once for deterministic and LLM runners."""
    for task in benchmark["tasks"]:
        policies = ablation_strategies(task["question"], task["documents"], k)
        yield from ((task, strategy, docs) for strategy, docs in policies.items())


def _policy_record(task: dict, strategy: str, docs: list[dict]) -> dict:
    category = task_category(task["id"])
    return {
        "task": task["id"],
        "category": category,
        "split": task.get("split", "unspecified"),
        "lexical_focus": task.get("lexical_focus", "unspecified"),
        "strategy": strategy,
        "expected": task["expected_stance"],
        "reference_record_applicable": category != "conflicting_root",
        "conflict_applicable": category == "conflicting_root",
        "fixture_contradiction_coverage": fixture_contradiction_coverage(docs),
        **retrieval_audit(docs),
    }


def run(benchmark_path: Path | None = None, k: int = 5) -> list[dict]:
    _, benchmark = _load(benchmark_path)
    records = []
    for task in benchmark["tasks"]:
        baseline, aware = (
            retrieve(task["question"], task["documents"], k),
            provenance_retrieve(task["question"], task["documents"], k),
        )
        base_answer, aware_answer = answer(baseline, False), answer(aware, True)
        records.append(
            {
                "task": task["id"],
                "expected": task["expected_stance"],
                "baseline_answer": base_answer,
                "provenance_answer": aware_answer,
                "baseline_correct": base_answer == task["expected_stance"],
                "provenance_correct": aware_answer == task["expected_stance"],
                "baseline_unique_roots": len({doc["root_id"] for doc in baseline}),
                "provenance_unique_roots": len({doc["root_id"] for doc in aware}),
                "baseline_docs": baseline,
                "provenance_docs": aware,
            }
        )
    return records


def run_ablation(benchmark_path: Path, k: int = 5) -> list[dict]:
    _, benchmark = _load(benchmark_path)
    records = []
    for task, strategy, docs in _strategy_cases(benchmark, k):
        stance = answer(docs, strategy in {"source_weighted", "metadata_assisted"})
        records.append(
            {
                **_policy_record(task, strategy, docs),
                "answer": stance,
                "correct": stance == task["expected_stance"],
                "root_conflict_exposed": strategy == "metadata_assisted",
            }
        )
    return records


def run_ablation_sweep(benchmark_path: Path, k_values: list[int]) -> list[dict]:
    """Run named ablations across fixed context budgets without mixing their rows."""
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    records = []
    for k in k_values:
        records.extend(
            {**record, "context_k": k} for record in run_ablation(benchmark_path, k)
        )
    return records


def run_ablation_v1(benchmark_path: Path, k: int = 5) -> list[dict]:
    """Evaluate fair-budget retrieval controls without reliability or oracle signals."""
    _, benchmark = _load(benchmark_path)
    records = []
    for task in benchmark["tasks"]:
        for strategy, docs in v1_retrieval_strategies(
            task["question"], task["documents"], k
        ).items():
            stance = answer(docs, False)
            records.append(
                {
                    **_policy_record(task, strategy, docs),
                    "answer": stance,
                    "correct": stance == task["expected_stance"],
                    "root_conflict_exposed": False,
                }
            )
    return records


def run_llm(
    model: str,
    endpoint: str,
    benchmark_path: Path | None = None,
    k: int = 5,
    seed: int = 42,
) -> list[dict]:
    _, benchmark = _load(benchmark_path)
    records = []
    for task, strategy, docs in _strategy_cases(benchmark, k):
        completion = llm_answer(task["question"], docs, model, endpoint, seed)
        stance, citations, valid = validate_completion(
            completion["parsed"], {doc["id"] for doc in docs}
        )
        records.append(
            {
                **_policy_record(task, strategy, docs),
                "seed": seed,
                "model_stance": stance,
                "correct": stance == task["expected_stance"],
                "citations": citations,
                "citation_ids_valid": valid,
                "retrieved": docs,
                "completion": completion,
            }
        )
    return records


def run_llm_v1(
    model: str,
    endpoint: str,
    benchmark_path: Path,
    k: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Run fixed-budget V1 retrievals through an LLM without oracle conflict flags."""
    _, benchmark = _load(benchmark_path)
    records = []
    for task in benchmark["tasks"]:
        policies = v1_retrieval_strategies(task["question"], task["documents"], k)
        for strategy, docs in policies.items():
            ordered = list(docs)
            random.Random(f"{seed}:{task['id']}:{strategy}").shuffle(ordered)
            completion = llm_answer(task["question"], ordered, model, endpoint, seed)
            stance, citations, valid = validate_completion(
                completion["parsed"], {doc["id"] for doc in ordered}
            )
            records.append(
                {
                    **_policy_record(task, strategy, ordered),
                    "seed": seed,
                    "evidence_order": [doc["id"] for doc in ordered],
                    "model_stance": stance,
                    "correct": stance == task["expected_stance"],
                    "citations": citations,
                    "citation_ids_valid": valid,
                    "retrieved": ordered,
                    "completion": completion,
                }
            )
    return records
