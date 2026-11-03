import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from real_evaluation import bootstrap_delta_ci, citation_entailment, exact_sign_test_pvalue, paired_summary, validate_real_corpus


def test_real_corpus_gate_rejects_empty_template():
    assert validate_real_corpus({"items": []}) == ["real corpus contains no admissible items"]
    errors = validate_real_corpus({"items": [{"id": "x", "expected_stance": "supports", "documents": []}]})
    assert any("no primary" in error for error in errors)
    assert any("no documented dependent" in error for error in errors)


def test_paired_statistics_use_item_pairs():
    rows = [
        {"task":"a", "strategy":"base", "correct":False}, {"task":"a", "strategy":"new", "correct":True},
        {"task":"b", "strategy":"base", "correct":True}, {"task":"b", "strategy":"new", "correct":False},
    ]
    summary = paired_summary(rows, "base", "new")
    assert summary["n"] == 2 and summary["improvements"] == 1 and summary["regressions"] == 1
    assert bootstrap_delta_ci(rows, "base", "new", repetitions=100, seed=1)[0] <= 0
    assert exact_sign_test_pvalue(1, 1) == 1.0


def test_citation_entailment_requires_matching_source_relation():
    docs = [{"id":"p", "role":"primary", "claim_relation":"supports"}, {"id":"d", "role":"dependent", "claim_relation":"refutes"}]
    score = citation_entailment("supports", ["p", "d"], docs)
    assert score["citation_precision"] == .5 and score["primary_cited"] is True
