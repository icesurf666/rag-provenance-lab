"""Audit actual document and prompt-token budgets from saved Ollama V1 outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(rows: list[dict]) -> dict:
    by_model_task: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_model_task[row["model"], row["task"]][row["strategy"]] = row
    comparisons = []
    for (model, task), conditions in sorted(by_model_task.items()):
        base, added = conditions.get("primary_only"), conditions.get("primary_plus_dependent")
        if not base or not added:
            continue
        comparisons.append({
            "model": model,
            "task": task,
            "baseline_documents": len(base["retrieved"]),
            "dependent_documents": len(added["retrieved"]),
            "baseline_prompt_tokens": base["ollama"].get("prompt_eval_count"),
            "dependent_prompt_tokens": added["ollama"].get("prompt_eval_count"),
        })
    unequal_documents = [row for row in comparisons if row["baseline_documents"] != row["dependent_documents"]]
    unequal_tokens = [row for row in comparisons if row["baseline_prompt_tokens"] != row["dependent_prompt_tokens"]]
    return {
        "audit": "V1 real-pilot paired budget audit",
        "pairs": len(comparisons),
        "unequal_document_pairs": len(unequal_documents),
        "unequal_prompt_token_pairs": len(unequal_tokens),
        "fixed_budget_pass": not unequal_documents and not unequal_tokens,
        "conclusion": "V1 citation and stance comparisons are descriptive only; adding the dependent page changed both provenance composition and context budget.",
        "pairs_detail": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    report = audit(read_rows(root / "data/public/v1/v1_raw_model_responses.jsonl"))
    output = args.output or root / "reports/V1_PAIRED_BUDGET_AUDIT.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "pairs_detail"}, indent=2))
