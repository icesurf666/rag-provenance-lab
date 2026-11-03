"""Small, auditable primitives for the provenance-RAG benchmark."""

from .generation import llm_answer, validate_completion
from .metrics import retrieval_audit, root_concentration, summarize, task_category
from .retrieval import (
    ablation_strategies,
    answer,
    conflict_preserving_fill,
    post_topk_collapse,
    provenance_retrieve,
    retrieve,
    root_aware_fill,
    root_aware_retrieve,
    v1_retrieval_strategies,
)
from .runner import (
    run,
    run_ablation,
    run_ablation_sweep,
    run_ablation_v1,
    run_llm,
    run_llm_v1,
)

__all__ = [
    "ablation_strategies",
    "answer",
    "conflict_preserving_fill",
    "llm_answer",
    "post_topk_collapse",
    "provenance_retrieve",
    "retrieval_audit",
    "retrieve",
    "root_aware_fill",
    "root_aware_retrieve",
    "root_concentration",
    "run",
    "run_ablation",
    "run_ablation_sweep",
    "run_ablation_v1",
    "run_llm",
    "run_llm_v1",
    "summarize",
    "task_category",
    "v1_retrieval_strategies",
    "validate_completion",
]
