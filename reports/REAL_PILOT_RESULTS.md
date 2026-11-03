# Real pilot results

The pilot uses 15 real NASA-to-Phys.org syndicated chains. Gemma `gemma3:4b` and Qwen `qwen2.5:7b-instruct` ran through Ollama 0.40.3 with temperature 0, seed 20261103, and a 160-token output ceiling. The released raw outputs are in [`data/public/v1/v1_raw_model_responses.jsonl`](../data/public/v1/v1_raw_model_responses.jsonl).

| Model | Retrieval condition | Accuracy | Valid citation IDs | Mean documents |
|---|---|---:|---:|---:|
| Gemma | primary-only | 15/15 | 3/15 | 1.0 |
| Gemma | primary + dependent | 15/15 | 6/15 | 2.0 |
| Gemma | root-aware | 15/15 | 3/15 | 1.0 |
| Gemma | conflict-preserving root-aware | 15/15 | 3/15 | 1.0 |
| Qwen | primary-only | 15/15 | 12/15 | 1.0 |
| Qwen | primary + dependent | 15/15 | 10/15 | 2.0 |
| Qwen | root-aware | 15/15 | 12/15 | 1.0 |
| Qwen | conflict-preserving root-aware | 15/15 | 12/15 | 1.0 |

Every condition answered the one-sided support questions correctly. That is an expected ceiling effect, not evidence that duplicates help. Root-aware and conflict-preserving root-aware select the NASA primary in every case because all documents have one root. Their paired accuracy delta against primary-only is 0/15 for both models.

The secondary signal is citation behavior. Gemma produced valid citation IDs in 20% of primary-only runs and 40% with the dependent copy. Qwen fell from 80% to 66.7% when the dependent copy was added. These are small, exploratory counts. They are not statistical claims.

## Paired-budget audit

The recorded comparison is not fixed-budget. Across all 30 model-case pairs, `primary-only` supplied one document and `primary + dependent` supplied two. Ollama's saved `prompt_eval_count` also changed in all 30 pairs. The added dependent page is therefore a demonstrated context-budget confound. It may contribute to the observed citation differences, but the V1 run cannot establish it as their only cause. The result remains as a descriptive pilot finding; it is not a causal estimate of provenance handling. See [`V1_PAIRED_BUDGET_AUDIT.json`](V1_PAIRED_BUDGET_AUDIT.json).

False-claim adoption, contradiction coverage, and conflict-specific behavior are not estimable in this slice: it contains no false-claim or contradictory-root items. Primary reference recall is 100% by construction in all four conditions, and root concentration is 1.0.

The corpus answers a narrow engineering question: can we preserve a documented provenance relation and run paired RAG comparisons over it? Yes. It does not yet answer when duplicate sources cause a model to adopt a false claim.
