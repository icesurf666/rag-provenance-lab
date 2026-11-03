# Verifying RAG Provenance Lab V1 artifacts

V1 is a snapshot-backed, machine-adjudicated pilot of 15 documented NASA-to-Phys.org syndication chains. It is a provenance-pipeline validation, not evidence that source-root retrieval improves RAG accuracy on the web.

## Artifact verification

`data/public/v1/` contains 15 cases with public URLs, canonical URLs, retrieval timestamps, HTTP status, SHA-256 content hashes, source roles, documented dependency evidence, machine annotations, a run manifest, and 120 raw model responses. It deliberately omits third-party HTML and extracted page text.

The published V1 artifacts can be verified without reconstructing source pages. Check the manifest structure, response count, SHA-256 digests, budget-audit result, and deterministic unit tests with the commands below.

## Environment

- Python 3.10+
- Ollama 0.40.3 used for the recorded V1 run
- `gemma3:4b` and `qwen2.5:7b-instruct` as recorded model tags
- temperature `0`, seed `20261103`, output ceiling `160`

Model tags are not immutable digests. The run manifest records this limitation. A future rerun must record `ollama show <model> --modelfile`, the Ollama server version, and the local model digest before execution.

## Verify the published artifacts

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
shasum -a 256 data/public/v1/*
PYTHONPATH=src pytest -q
```

The saved V1 outputs are artifacts of the recorded local environment, not a promise that a later model tag resolves identically.

## End-to-end reproduction

This public release does not support end-to-end reproduction of the original V1 run. It omits third-party HTML, extracted page text, recollection tooling, and the local annotation workflow. Those omissions prevent redistribution of publisher content and mean that the release supports artifact verification only.

An end-to-end rerun requires independently retrieving the allowlisted public URLs, recording fresh snapshots and hashes, completing independent annotations and adjudication, pinning model digests, and then running the paired evaluation. A later rerun is a new dated experiment, not a byte-for-byte replay of V1.

## V1 paired-budget status

`reports/V1_PAIRED_BUDGET_AUDIT.json` evaluates all 30 model-case pairs between `primary_only` and `primary_plus_dependent`. Every pair changes from one document to two, and every saved prompt-token count changes. The V1 citation differences are descriptive. They cannot isolate the effect of source dependence from the additional context.

V2 blocks evaluation unless both the document budget and the model-observed prompt-token budget are equal for each paired comparison.
