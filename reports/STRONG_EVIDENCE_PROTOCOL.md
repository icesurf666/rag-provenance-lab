# Strong-evidence protocol: dependent sources in RAG

## Claim under test

In this fixed RAG pipeline, adding **documented dependent** sources to a corpus changes retrieval and answer quality relative to a primary-only corpus; root-aware handling changes that effect. This does not test the Dead Internet Theory, web-wide prevalence, or model-training contamination.

## Corpus gate

No item enters the real corpus unless it contains:

- a public primary record;
- at least one dependent document with explicit citation, syndication evidence, or editor confirmation;
- URL, canonical URL, archive URL, publication date, retrieval timestamp, and content hash;
- two independent annotations for stance, role, root ID, and claim relation, plus a named adjudication record with a resolution note;
- a permitted local excerpt sufficient to support the label.

`src/real_evaluation.py:validate_real_corpus()` blocks evaluation when these fields are absent.

## Counterfactual conditions

For each item and fixed query/model/prompt:

1. **Primary-only:** primary evidence and genuinely independent sources.
2. **Primary + dependent:** same corpus plus documented dependent sources.
3. **Root-aware:** condition 2 with pre-registered source-root policy.

All three outputs are paired by item. The primary condition must be generated before looking at answer outcomes.

## Metrics and decision rule

Primary: correct stance. Secondary: primary-source recall@k, citation entailment precision, primary-source citation rate, exact citation validity, abstention precision/recall, source-root diversity, and false-support rate. `citation_entailment()` scores citations against adjudicated claim-relation labels. Report paired accuracy delta with bootstrap 95% CI and a two-sided exact sign test on discordant pairs. Do not report a p-value if fewer than 30 independent held-out items survive the corpus gate.

## Freeze before collecting outcomes

Record model digest, Ollama version, prompt hash, retriever settings, `k`, seed(s), hardware, source policy, exclusions, and analysis script hash. Split items into development and held-out test before tuning any policy.
