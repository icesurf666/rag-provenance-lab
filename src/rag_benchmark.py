"""CLI and backwards-compatible public API for the provenance-RAG benchmark.

Core behaviour lives in :mod:`rag`; keeping this facade thin makes the command
line stable for reports, while retrieval and generation remain independently
testable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import (
    ablation_strategies,
    answer,
    conflict_preserving_fill,
    llm_answer,
    post_topk_collapse,
    provenance_retrieve,
    retrieve,
    root_aware_fill,
    root_aware_retrieve,
    run,
    run_ablation,
    run_ablation_sweep,
    run_ablation_v1,
    run_llm,
    run_llm_v1,
    summarize,
    v1_retrieval_strategies,
    validate_completion,
)

__all__ = [
    "ablation_strategies",
    "answer",
    "conflict_preserving_fill",
    "llm_answer",
    "post_topk_collapse",
    "provenance_retrieve",
    "retrieve",
    "root_aware_fill",
    "root_aware_retrieve",
    "run",
    "run_ablation",
    "run_ablation_sweep",
    "run_ablation_v1",
    "run_llm",
    "run_llm_v1",
    "summarize",
    "v1_retrieval_strategies",
    "validate_completion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the provenance-RAG benchmark.")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k-sweep", type=int, nargs="+")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--v1", action="store_true")
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--endpoint", default="http://localhost:11434/api/generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.ollama:
        benchmark = (
            args.benchmark or root / "data/benchmarks/provenance_rag_60_synthetic.json"
        )
        records = (
            run_llm_v1(args.model, args.endpoint, benchmark, args.k, args.seed)
            if args.v1
            else run_llm(args.model, args.endpoint, args.benchmark, args.k, args.seed)
        )
        default_output = (
            root / "data/processed/rag_llm_v1_results.json"
            if args.v1
            else root / "data/processed/rag_llm_results.json"
        )
    elif args.ablations:
        benchmark = (
            args.benchmark or root / "data/benchmarks/provenance_rag_60_synthetic.json"
        )
        if args.v1 and args.k_sweep:
            raise SystemExit("--v1 cannot be combined with --k-sweep")
        records = (
            run_ablation_v1(benchmark, args.k)
            if args.v1
            else run_ablation_sweep(benchmark, args.k_sweep)
            if args.k_sweep
            else run_ablation(benchmark, args.k)
        )
        default_output = (
            root / "data/processed/rag_ablation_sweep.json"
            if args.k_sweep
            else root / "data/processed/rag_ablation_v1.json"
            if args.v1
            else root / "data/processed/rag_ablation_results.json"
        )
    else:
        if args.k_sweep or args.v1:
            raise SystemExit("--k-sweep and --v1 require --ablations")
        records = run(args.benchmark, args.k)
        default_output = root / "data/processed/rag_benchmark_results.json"
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))
    if args.summary_output:
        if not args.ablations:
            raise SystemExit("--summary-output requires --ablations")
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summarize(records), indent=2))
        print(f"wrote summary to {args.summary_output}")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
