# V1 reproducibility status

## Public endpoints checked

- Article: `https://www.pkazantsev.com/writing/rag-source-roots/` returned HTTP 200 on 2026-11-03.
- The retired research-lab route `https://www.pkazantsev.com/research/rag-provenance/` returned HTTP 404 on 2026-11-03. The article is the public research entry point.

## Public-safe artifact boundary

The public release includes metadata, hashes, provenance assertions, machine annotations, run manifests, raw model responses, code, and generated figures. It excludes retrieved third-party HTML and full extracted page text. The public data directory is `data/public/v1/`.

## Recorded V1 environment

- Ollama client 0.40.3; the server was unavailable during this reproducibility audit, so a current server model digest could not be obtained.
- Recorded model tags: `gemma3:4b` and `qwen2.5:7b-instruct`.
- Generation settings: temperature 0, seed 20261103, output ceiling 160.
- Raw model responses: `data/public/v1/v1_raw_model_responses.jsonl`.

## Fixed-budget audit

The paired audit covers 30 model-case pairs. All 30 change document count and Ollama `prompt_eval_count` when a dependent page is added. The V1 real pilot therefore does **not** meet a fixed document or token budget. Its stance and citation comparisons remain retained as descriptive results. The observed context difference is a demonstrated confound; it is not evidence that it was the only cause of any observed citation difference.
