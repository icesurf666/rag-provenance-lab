"""Generate a deterministic synthetic provenance-RAG benchmark.

All records are fictional. The generator makes the benchmark inspectable and avoids
hand-editing dozens of near-identical fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

NAMES = [
    "Atlas", "Borealis", "Cirrus", "Drift", "Ember", "Fjord", "Gale", "Harbor", "Ion", "Juniper",
    "Kestrel", "Lumen", "Mistral", "Nimbus", "Orbit", "Praxis", "Quartz", "Ridge", "Solace", "Tundra",
]
FEATURES = [
    "quasar mode", "polar export", "storm replication", "ember sync", "fjord routing", "gale cache",
    "harbor mirror", "ion archive", "juniper relay", "kestrel stream", "lumen index", "mistral queue",
    "nimbus lease", "orbit ledger", "praxis gateway", "quartz backup", "ridge filter", "solace bridge",
    "tundra snapshot", "aurora merge",
]


def mirrors(prefix: str, name: str, feature: str, stance: str, count: int, root: str, lexical_focus: str) -> list[dict]:
    verb = "supports" if stance == "supports" else "does not support"
    return [{
        "id": f"{prefix}_mirror_{i:02}", "root_id": root, "publisher": f"{name} Mirror {i}",
        "role": "derivative", "stance": stance, "reliability": .3,
        "text": (f"Does the fictional {name} service support {feature}? This guide says {name} {verb} {feature}."
                 if lexical_focus == "mirrors" else f"Guide: {name} {verb} {feature}."),
    } for i in range(1, count + 1)]


def task(kind: str, index: int) -> dict:
    name, feature = NAMES[index % len(NAMES)], FEATURES[index % len(FEATURES)]
    prefix = f"{kind}_{index:02}"
    question = f"Does the fictional {name} service support {feature}?"
    lexical_focus = "primary" if index % 2 else "mirrors"
    split = "dev" if index <= 10 else "test"
    if kind == "false_dilution":
        primary_text = (f"Does the fictional {name} service support {feature}? No. {name} Specification v1 says {feature} is not implemented." if lexical_focus == "primary" else f"{name} Specification v1: {name} does not support {feature}. Requests must be rejected.")
        docs = [{"id":f"{prefix}_primary", "root_id":f"{prefix}_spec", "publisher":f"{name} specification", "role":"primary", "stance":"refutes", "reliability":1.0, "text":primary_text}]
        return {"id":prefix, "split":split, "lexical_focus":lexical_focus, "question":question, "expected_stance":"refutes", "documents":docs + mirrors(prefix, name, feature, "supports", 10, f"{prefix}_mirror_root", lexical_focus)}
    if kind == "true_repetition":
        docs = [{"id":f"{prefix}_primary", "root_id":f"{prefix}_spec", "publisher":f"{name} documentation", "role":"primary", "stance":"supports", "reliability":1.0, "text":f"{name} documentation: {feature} is supported in all current plans."}]
        return {"id":prefix, "split":split, "lexical_focus":lexical_focus, "question":question, "expected_stance":"supports", "documents":docs + mirrors(prefix, name, feature, "supports", 8, f"{prefix}_mirror_root", lexical_focus)}
    if kind == "conflicting_root":
        docs = [
            {"id":f"{prefix}_forum_yes", "root_id":f"{prefix}_forum", "publisher":f"{name} forum", "role":"discussion", "stance":"supports", "reliability":.2, "text":f"A forum participant says {name} may support {feature}."},
            {"id":f"{prefix}_forum_no", "root_id":f"{prefix}_forum", "publisher":f"{name} forum", "role":"discussion", "stance":"refutes", "reliability":.2, "text":f"Another forum participant says {name} does not support {feature}."},
        ]
        return {"id":prefix, "split":split, "lexical_focus":lexical_focus, "question":question, "expected_stance":"unknown", "documents":docs}
    # Satire-context task: a primary label says the apparent feature announcement is a joke.
    docs = [{"id":f"{prefix}_satire", "root_id":f"{prefix}_satire_source", "publisher":f"{name} satirical release", "role":"satire", "stance":"refutes", "reliability":.8, "text":f"Satire notice: the claim that {name} supports {feature} is a joke, not a product announcement."}]
    return {"id":prefix, "split":split, "lexical_focus":lexical_focus, "question":question, "expected_stance":"refutes", "documents":docs + mirrors(prefix, name, feature, "supports", 7, f"{prefix}_mirror_root", lexical_focus)}


def build() -> dict:
    tasks = []
    for kind in ("false_dilution", "true_repetition", "conflicting_root", "satire_context"):
        tasks.extend(task(kind, i) for i in range(1, 16))
    return {
        "name": "Provenance RAG 60-task synthetic benchmark", "seed": 20261101,
        "warning": "Every document is synthetic and fictional. This is a controlled mechanism test, not evidence about the web.",
        "tasks": tasks,
    }


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "data/benchmarks/provenance_rag_60_synthetic.json"
    destination.write_text(json.dumps(build(), indent=2))
    print(f"wrote {len(build()['tasks'])} tasks to {destination}")
