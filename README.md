# RAG Provenance Lab

Experimental tooling for evaluating whether source-provenance-aware retrieval changes model behavior under a fixed evidence budget.

## Scope

V1 is a 15-case, machine-adjudicated pilot over documented NASA-to-Phys.org syndication chains. It validates collection, provenance storage, paired execution, and result retention. It does not estimate web-wide prevalence or establish an accuracy benefit from source-root retrieval.

The saved V1 comparison is not fixed-budget: adding a dependent document changed both document count and observed prompt-token count in all 30 model-case pairs. Its citation results are descriptive. See [`reports/V1_PAIRED_BUDGET_AUDIT.json`](reports/V1_PAIRED_BUDGET_AUDIT.json).

## Repository layout

- `data/benchmarks/`: synthetic fixtures and schema contracts.
- `data/public/v1/`: public URLs, hashes, annotations, run manifest, and 120 raw model responses. Third-party HTML and extracted text are excluded.
- `src/rag/`: retrieval, generation, metrics, and runner modules.
- `src/v2_gates.py`: admission, leakage, and paired-budget checks for V2.
- `tests/`: deterministic unit tests.
- `reports/`: protocol and recorded V1 audit artifacts.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
PYTHONPATH=src pytest -q
shasum -a 256 data/public/v1/*
```

For environment details, artifact checksums, model settings, and the boundary between public metadata and local source snapshots, read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## License

MIT. Source pages remain subject to their publishers' terms; this repository distributes metadata and hashes, not third-party page copies.
