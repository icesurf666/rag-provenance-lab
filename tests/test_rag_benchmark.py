import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from generate_benchmark import build
from rag.metrics import retrieval_audit, summarize
from rag_benchmark import (
    ablation_strategies,
    provenance_retrieve,
    run,
    run_ablation,
    run_ablation_sweep,
    run_ablation_v1,
    validate_completion,
)


def test_duplicate_dilution_has_expected_failure_and_recovery():
    result = next(row for row in run() if row["task"] == "false_claim_dilution")
    assert result["baseline_answer"] == "supports"
    assert result["provenance_answer"] == "refutes"
    assert result["baseline_unique_roots"] == 1
    assert result["provenance_unique_roots"] == 2


def test_conflicting_single_root_becomes_unknown():
    docs = [
        {
            "id": "a",
            "root_id": "same",
            "stance": "supports",
            "reliability": 0.2,
            "text": "storm replication support",
        },
        {
            "id": "b",
            "root_id": "same",
            "stance": "refutes",
            "reliability": 0.2,
            "text": "storm replication does not support",
        },
    ]
    result = provenance_retrieve("storm replication support", docs)
    assert len(result) == 1
    assert result[0]["stance"] == "unknown"
    assert result[0]["root_conflict"] is True


def test_generated_benchmark_has_balanced_60_task_design():
    tasks = build()["tasks"]
    assert len(tasks) == 60
    for prefix in (
        "false_dilution",
        "true_repetition",
        "conflicting_root",
        "satire_context",
    ):
        assert sum(task["id"].startswith(prefix) for task in tasks) == 15


def test_ablation_strategies_are_separate_and_named():
    docs = [
        {
            "id": "a",
            "root_id": "one",
            "role": "primary",
            "stance": "supports",
            "reliability": 1,
            "text": "supports polar export",
        }
    ]
    strategies = ablation_strategies("polar export", docs)
    assert list(strategies) == [
        "lexical_topk",
        "root_collapse",
        "source_weighted",
        "metadata_assisted",
    ]


def test_completion_schema_requires_known_stance_and_exact_ids():
    assert validate_completion({"stance": "supports", "citations": ["a"]}, {"a"}) == (
        "supports",
        ["a"],
        True,
    )
    assert validate_completion({"stance": "maybe", "citations": []}, {"a"}) == (
        "invalid",
        [],
        False,
    )
    assert validate_completion({"stance": "supports", "citations": "a"}, {"a"}) == (
        "invalid",
        [],
        False,
    )


def test_deterministic_ablation_emits_all_policies():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.json"
        path.write_text(json.dumps(build()))
        rows = run_ablation(path)
    assert len(rows) == 240
    assert {row["strategy"] for row in rows} == {
        "lexical_topk",
        "root_collapse",
        "source_weighted",
        "metadata_assisted",
    }
    assert {row["category"] for row in rows} == {
        "false_dilution",
        "true_repetition",
        "conflicting_root",
        "satire_context",
    }
    assert all("reference_record_recalled" in row for row in rows)


def test_retrieval_audit_measures_roots_and_reference_record():
    docs = [
        {"id": "p", "root_id": "primary", "role": "primary"},
        {"id": "d1", "root_id": "mirror", "role": "derivative"},
        {"id": "d2", "root_id": "mirror", "role": "derivative"},
    ]
    assert retrieval_audit(docs) == {
        "selected_ids": ["p", "d1", "d2"],
        "returned_documents": 3,
        "approximate_context_tokens": 0,
        "unique_roots": 2,
        "root_concentration": 0.666667,
        "reference_record_recalled": True,
    }


def test_summary_keeps_failure_modes_separate():
    rows = [
        {
            "split": "test",
            "strategy": "lexical_topk",
            "category": "false_dilution",
            "correct": False,
            "reference_record_recalled": False,
            "reference_record_applicable": True,
            "conflict_applicable": False,
            "fixture_contradiction_coverage": False,
            "returned_documents": 1,
            "approximate_context_tokens": 10,
            "unique_roots": 1,
            "root_concentration": 1.0,
        },
        {
            "split": "test",
            "strategy": "lexical_topk",
            "category": "false_dilution",
            "correct": True,
            "reference_record_recalled": True,
            "reference_record_applicable": True,
            "conflict_applicable": False,
            "fixture_contradiction_coverage": False,
            "returned_documents": 2,
            "approximate_context_tokens": 20,
            "unique_roots": 2,
            "root_concentration": 0.5,
        },
    ]
    assert summarize(rows) == [
        {
            "split": "test",
            "strategy": "lexical_topk",
            "category": "false_dilution",
            "n": 2,
            "accuracy": 0.5,
            "mean_returned_documents": 1.5,
            "mean_approximate_context_tokens": 15,
            "mean_unique_roots": 1.5,
            "mean_root_concentration": 0.75,
            "reference_record_recall_applicable": 0.5,
            "fixture_contradiction_coverage": None,
        }
    ]


def test_k_sweep_tags_each_ablation_record():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.json"
        path.write_text(json.dumps(build()))
        rows = run_ablation_sweep(path, [3, 5])
    assert len(rows) == 480
    assert {row["context_k"] for row in rows} == {3, 5}


def test_v1_controls_keep_budget_except_when_corpus_is_smaller():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.json"
        path.write_text(json.dumps(build()))
        rows = run_ablation_v1(path, k=5)
    root_fill = [row for row in rows if row["strategy"] == "root_aware_fill"]
    assert all(
        row["returned_documents"]
        == min(
            5,
            len(
                next(task for task in build()["tasks"] if task["id"] == row["task"])[
                    "documents"
                ]
            ),
        )
        for row in root_fill
    )
    assert any(
        row["strategy"] == "post_topk_collapse" and row["returned_documents"] < 5
        for row in rows
    )
